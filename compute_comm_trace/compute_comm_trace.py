# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from pathlib import Path

import pandas as pd

from nsys_recipe import log
from nsys_recipe.data_service import DataService
from nsys_recipe.lib import args, cuda, data_utils, helpers, nvtx, overlap, recipe
from nsys_recipe.lib.args import Option
from nsys_recipe.log import logger


class ComputeCommTimeUtilMap(recipe.Recipe):
    @staticmethod
    def _mapper_func(report_path, parsed_args):
        service = DataService(report_path)

        table_column_dict = {
            "StringIds": None,
            "CUPTI_ACTIVITY_KIND_RUNTIME": [
                "correlationId",
                "globalTid",
                "start",
                "end",
            ],
            "CUPTI_ACTIVITY_KIND_KERNEL": [
                "correlationId",
                "globalPid",
                "start",
                "end",
                "deviceId",
                "shortName",
                "streamId",
            ],
        }

        df_dict = service.read_tables(table_column_dict)
        nccl_df = service.get_nccl_table()

        if df_dict is None or nccl_df is None:
            return None

        kernel_df = df_dict["CUPTI_ACTIVITY_KIND_KERNEL"]

        data_utils.filter_by_time_range(kernel_df, parsed_args.start, parsed_args.end)
        kernel_df = data_utils.replace_id_with_value(
            df_dict["CUPTI_ACTIVITY_KIND_KERNEL"],
            df_dict["StringIds"],
            "shortName",
            "name",
        )

        cuda_df = cuda.combine_runtime_gpu_dfs(
            df_dict["CUPTI_ACTIVITY_KIND_RUNTIME"], kernel_df
        )

        if nccl_df.empty or cuda_df.empty:
            logger.info(
                f"{report_path} was successfully processed, but no data was found."
            )
            return None

        nccl_gpu_df = nvtx.project_nvtx_onto_gpu(nccl_df, cuda_df)
        if nccl_gpu_df.empty:
            logger.info(
                f"{report_path} does not contain any NCCL data that can be projected onto the GPU."
            )
            return None

        # Merge the kernel dataframe with the projected NCCL dataframe.
        # If a kernel exists in both dataframes, it is an NCCL kernel.
        # Otherwise, it is a compute kernel.
        merged_df = kernel_df.merge(
            nccl_gpu_df, on=["start", "end", "pid"], how="left", indicator="merged"
        )

        kernel_grouped = merged_df.groupby(["pid", "deviceId"])
        results = []

        for _, group_df in kernel_grouped:
            nccl_group_df = group_df[group_df["merged"] == "both"].reset_index(
                drop=True
            )

            compute_group_df = group_df[group_df["merged"] == "left_only"].reset_index(
                drop=True
            )

            # Communication - communication overlap.
            nccl_group_df["Communication Sum"] = overlap.calculate_overlap_sum(
                nccl_group_df, divisor=parsed_args.divisor
            )

            # Communication - compute overlap.
            nccl_group_df["Compute Sum"] = overlap.calculate_overlap_sum(
                nccl_group_df, compute_group_df, divisor=parsed_args.divisor
            )

            # Compute - communication overlap.
            compute_group_df["Communication Sum"] = overlap.calculate_overlap_sum(
                compute_group_df, nccl_group_df, divisor=parsed_args.divisor
            )

            # Compute - compute overlap.
            compute_group_df["Compute Sum"] = overlap.calculate_overlap_sum(
                compute_group_df, divisor=parsed_args.divisor
            )

            results.extend([nccl_group_df, compute_group_df])

        name_dict = {
            "name": "Name",
            "start": "Start",
            "end": "End",
            "pid": "PID",
            "deviceId": "DeviceID",
            "Communication Sum": "Communication Sum",
            "Compute Sum": "Compute Sum",
            "streamId": "StreamID",  
        }

        df = pd.concat(results, ignore_index=True).rename(columns=name_dict)[
            name_dict.values()
        ]
        df = data_utils.add_time_range_mask(df,parsed_args.subtimerange)
        filename = Path(report_path).stem

        return filename, df

    @log.time("Mapper")
    def mapper_func(self, context):
        return context.wait(
            context.map(
                self._mapper_func,
                self._parsed_args.input,
                parsed_args=self._parsed_args,
            )
        )

    def reducer_func(self, mapper_res):
        filtered_res = helpers.filter_none(mapper_res)
        # Sort by file name.
        filtered_res = sorted(filtered_res, key=lambda x: x[0])
        filenames, trace_dfs = zip(*filtered_res)

        files_df = pd.DataFrame({"File": filenames}).rename_axis("Rank")
        files_df.to_parquet(self.add_output_file("files.parquet"))

        trace_dfs = [df.assign(Rank=rank) for rank, df in enumerate(trace_dfs)]
        trace_df = pd.concat(trace_dfs)

        trace_df["Duration"] = trace_df["End"] - trace_df["Start"]
        trace_df["Communication Overlap"] = (
            trace_df["Communication Sum"] / trace_df["Duration"] * 100
        )
        trace_df["Compute Overlap"] = (
            trace_df["Compute Sum"] / trace_df["Duration"] * 100
        )
        trace_df["Exclude Compute Overlap Duration"] = (
            trace_df["Duration"] - trace_df["Compute Sum"]
        )

        rank_trace_df = (
            trace_df[
                [
                    "Name",
                    "Start",
                    "End",
                    "PID",
                    "DeviceID",
                    "Communication Overlap",
                    "Compute Overlap",
                    "Exclude Compute Overlap Duration",
                    "Rank",
                    "StreamID",
                    "SubRange",
                ]
            ]
            .set_index("Name")
            .round(1)
        )
        rank_trace_df.to_parquet(self.add_output_file("rank_trace.parquet"))

        trace_gdf = trace_df.groupby("Name")
        duration = trace_gdf["Duration"].sum()
        comm_sum = trace_gdf["Communication Sum"].sum()
        compute_sum = trace_gdf["Compute Sum"].sum()
        
        stream_gdf = trace_df.groupby(["StreamID","SubRange"])
        stream_duration = stream_gdf["Duration"].sum()
        stream_comm_sum = stream_gdf["Communication Sum"].sum()
        stream_compute_sum = stream_gdf["Compute Sum"].sum()
        stream_exclude_compute_sum = stream_gdf["Exclude Compute Overlap Duration"].sum()
        
        '''
        stream_duration = stream_gdf["Duration"].sum().groupby(level=0).sum()
        stream_subrange_duration = stream_gdf["Duration"].sum()
        stream_comm_sum = stream_gdf["Communication Sum"].sum().groupby(level=0).sum()
        stream_subrange_comm_sum = stream_gdf["Communication Sum"].sum()
        stream_compute_sum = stream_gdf["Compute Sum"].sum().groupby(level=0).sum()
        stream_subrange_compute_sum = stream_gdf["Compute Sum"].sum()
        stream_exclude_compute_sum = stream_gdf["Exclude Compute Overlap Duration"].sum().groupby(level=0).sum()
        stream_subrange_exclude_compute_sum = stream_gdf["Exclude Compute Overlap Duration"].sum()
        '''
        
        grouped_trace_df = pd.DataFrame(
            {
                "Count": trace_gdf.size(),
                "Communication Overlap": comm_sum / duration * 100,
                "Compute Overlap": compute_sum / duration * 100,
            }
        ).round(1)
        grouped_stream_df = pd.DataFrame(
            {
                "Duration": stream_duration,
                "Communication Overlap Duration": stream_comm_sum,
                "Communication Overlap Percent": stream_comm_sum / stream_duration * 100,
                "Compute Overlap Duration": stream_compute_sum,
                "Compute Overlap Percent": stream_compute_sum / stream_duration * 100,
                "Exclude Compute Overlap Duration": stream_exclude_compute_sum,
            }
        ).round(1)

        grouped_trace_df.to_parquet(self.add_output_file("grouped_trace.parquet"))
        grouped_stream_df.to_parquet(self.add_output_file("grouped_stream.parquet"))

        if self._parsed_args.csv:
            files_df.to_csv(self.add_output_file("files.csv"))
            rank_trace_df.to_csv(self.add_output_file("rank_trace.csv"))
            grouped_trace_df.to_csv(self.add_output_file("grouped_trace.csv"))
            grouped_stream_df.to_csv(self.add_output_file("grouped_stream.csv"))

    def save_notebook(self):
        self.create_notebook("trace.ipynb")
        self.add_notebook_helper_file("nsys_display.py")

    def save_analysis_file(self):
        self._analysis_dict.update(
            {
                "EndTime": str(datetime.now()),
                "InputFiles": self._parsed_args.input,
                "Outputs": self._output_files,
            }
        )
        self.create_analysis_file()

    def run(self, context):
        super().run(context)

        mapper_res = self.mapper_func(context)
        self.reducer_func(mapper_res)

        self.save_notebook()
        self.save_analysis_file()

    @classmethod
    def get_argument_parser(cls):
        parser = super().get_argument_parser()

        parser.add_recipe_argument(Option.INPUT, required=True)
        parser.add_recipe_argument(Option.OUTPUT)
        parser.add_recipe_argument(Option.FORCE_OVERWRITE)
        parser.add_recipe_argument(Option.START)
        parser.add_recipe_argument(Option.END)
        parser.add_recipe_argument(Option.CSV)
        parser.add_recipe_argument(Option.SUBTIMERANGE)
        parser.add_recipe_argument(
            "--divisor",
            type=args.process_integer(1),
            help="Break down the computation of the overlapping kernel truth table."
            " This increases the computation time but helps reduce memory usage.",
            default=100,
        )
        return parser
