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
from functools import reduce
import pandas as pd

from nsys_recipe import log
from nsys_recipe.data_service import DataService
from nsys_recipe.lib import helpers, nccl, overlap, recipe
from nsys_recipe.lib.args import Option
from nsys_recipe.lib.table_config import CompositeTable
from nsys_recipe.log import logger

class NcclGpuTimeUtilMap(recipe.Recipe):
    @staticmethod
    def _mapper_func(report_path, parsed_args):
        service = DataService(report_path, parsed_args)

        service.queue_custom_table(CompositeTable.CUDA_KERNEL)

        df_dict = service.read_queued_tables()
        if df_dict is None:
            return None

        kernel_df = df_dict[CompositeTable.CUDA_KERNEL]
        err_msg = service.filter_and_adjust_time(kernel_df)
        if err_msg is not None:
            logger.error(f"{report_path}: {err_msg}")
            return None

        if kernel_df.empty:
            logger.info(
                f"{report_path}: Report was successfully processed, but no data was found."
            )
            return None

        kernel_df = nccl.add_kernel_type(kernel_df)
        nccl_kernel_df = kernel_df[kernel_df["type"] == "nccl"]
        compute_kernel_df = kernel_df[kernel_df["type"] == "compute"]

        if nccl_kernel_df.empty:
            logger.info(f"{report_path}: Report does not contain any NCCL or DeepEP kernels. in this case, if you want to see kernel overlap matrix, you can use kernel_overlap_trace recipe.")
            return None

        kernel_grouped = kernel_df.groupby(["pid", "deviceId"])
        results = []
        results_merge = []

        # merge by type
        type_merge_df = overlap.merge_by_type(kernel_df)
        type_merge_df['overlap_sum'] = overlap.calculate_overlap_sum(type_merge_df)
        comm_stream_type_merge_df = overlap.merge_by_type(nccl_kernel_df, streamid=True)
        comm_stream_type_merge_df['overlap_sum'] = overlap.calculate_overlap_sum(comm_stream_type_merge_df, type_merge_df[type_merge_df['shortName'] == 'compute'])
        type_merge_df = pd.concat([type_merge_df, comm_stream_type_merge_df], ignore_index=True)
        # merge by name
        nccl_df_merge = overlap.merge_overlapping_ranges_by_name(nccl_kernel_df, self_overlapped_duration=parsed_args.self_overlap)
        compute_df_merge = overlap.merge_overlapping_ranges_by_name(compute_kernel_df, self_overlapped_duration=parsed_args.self_overlap)
        # calculate overlap matrix
        comm_comm_matrix = overlap.calculate_overlap_sum_matrix(nccl_df_merge)
        comm_compute_matrix = overlap.calculate_overlap_sum_matrix(nccl_df_merge, compute_df_merge)
        compute_comm_matrix = overlap.calculate_overlap_sum_matrix(compute_df_merge, nccl_df_merge)
        compute_compute_matrix = overlap.calculate_overlap_sum_matrix(compute_df_merge)

        # calculate group trace
        nccl_df_merge['Communication Sum'] = overlap.calculate_overlap_sum(nccl_df_merge)
        nccl_df_merge['Compute Sum'] = overlap.calculate_overlap_sum(nccl_df_merge, compute_df_merge)
        compute_df_merge['Communication Sum'] = overlap.calculate_overlap_sum(compute_df_merge, nccl_df_merge)
        compute_df_merge['Compute Sum'] = overlap.calculate_overlap_sum(compute_df_merge)
        results_merge.extend([nccl_df_merge, compute_df_merge])

        # post process matrix
        concat_comm = pd.concat([comm_comm_matrix, comm_compute_matrix.drop(columns=['original_duration'])], axis=1).reset_index().sort_values(by='original_duration', ascending=False)
        concat_comm["type"] = "comm"
        concat_compute = pd.concat([compute_comm_matrix, compute_compute_matrix.drop(columns=['original_duration'])], axis=1).reset_index().sort_values(by='original_duration', ascending=False)
        concat_compute["type"] = "compute"      
        desired_order = ['shortName','original_duration']+list(concat_comm['shortName'])+list(concat_compute['shortName']) + ['type']


        comm_compute_matrix_final = pd.concat([
            concat_comm.reindex(columns=desired_order,fill_value=0), 
            concat_compute.reindex(columns=desired_order,fill_value=0)
        ], ignore_index=True)

        for _, group_df in kernel_grouped:
            nccl_group_df = group_df[group_df["type"] == "nccl"].reset_index(drop=True)
            compute_group_df = group_df[group_df["type"] == "compute"].reset_index(
                drop=True
            )
            # Communication - communication overlap.    if in a period of time, same kernel name has more than 1 instance, it will be calculated multiple times.
            nccl_group_df["Communication Sum"] = overlap.calculate_overlap_sum(
                nccl_group_df
            )

            # Communication - compute overlap.
            nccl_group_df["Compute Sum"] = overlap.calculate_overlap_sum(
                nccl_group_df, compute_group_df
            )

            # Compute - communication overlap.
            compute_group_df["Communication Sum"] = overlap.calculate_overlap_sum(
                compute_group_df, nccl_group_df
            )

            # Compute - compute overlap.
            compute_group_df["Compute Sum"] = overlap.calculate_overlap_sum(
                compute_group_df
            )

            results.extend([nccl_group_df, compute_group_df])


        name_dict = {
            "shortName": "Name",
            "start": "Start",
            "end": "End",
            "pid": "PID",
            "deviceId": "DeviceID",
            "Communication Sum": "Communication Sum",
            "Compute Sum": "Compute Sum",
        }

        name_dict_merge = {
            "shortName": "Name",
            "start": "Start",
            "end": "End",
            "Communication Sum": "Communication Sum",
            "Compute Sum": "Compute Sum",
        }

        df = pd.concat(results, ignore_index=True).rename(columns=name_dict)[
            name_dict.values()
        ]
        df_merge = pd.concat(results_merge, ignore_index=True).rename(columns=name_dict_merge)[
            name_dict_merge.values()
        ]
        filename = Path(report_path).stem

        return filename, df, comm_compute_matrix_final, df_merge, type_merge_df

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
        filenames, trace_dfs, comm_compute_final_dfs, name_merge_dfs, type_merge_dfs = zip(*filtered_res)

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
                    "Rank",
                ]
            ]
            .set_index("Name")
            .round(1)
        )
        rank_trace_df.to_parquet(self.add_output_file("rank_trace.parquet"))

        #type merge df
        type_merge_dfs = [df.assign(Rank=rank) for rank, df in enumerate(type_merge_dfs)]
        type_merge_df = pd.concat(type_merge_dfs)
        type_merge_df["Duration"] = type_merge_df["end"] - type_merge_df["start"]
        type_merge_df["shortName"] = type_merge_df["shortName"].replace("nccl", "Communication")
        type_merge_df = type_merge_df.groupby(["streamId","shortName"])
        type_merge_df_duration = type_merge_df["Duration"].sum()
        type_merge_df_overlap_sum = type_merge_df["overlap_sum"].sum()
        total_duration = type_merge_df_duration.loc["all"].sum() - type_merge_df_overlap_sum.loc[("all","compute")]
        exposed_compute_ratio = 100*type_merge_df_duration.loc[("all","compute")]/total_duration
        exposed_ratio = (type_merge_df_duration - type_merge_df_overlap_sum)/total_duration * 100
        exposed_ratio.loc[('all','compute')] = exposed_compute_ratio
        grouped_type_merge_df = pd.DataFrame(
            {
                "Duration": type_merge_df_duration,
                "Overlapped Duration": type_merge_df_overlap_sum,
                "Overlapped Percentage": type_merge_df_overlap_sum / type_merge_df_duration * 100,
                "Exposed Percentage": exposed_ratio,
            }
        ).round(1)
        grouped_type_merge_df = grouped_type_merge_df.reset_index(level='shortName')
        grouped_type_merge_df.index = grouped_type_merge_df.index.astype(str)
        grouped_type_merge_df.to_parquet(self.add_output_file("grouped_type_merge_df.parquet"))

        # name merge df
        name_merge_dfs = [df.assign(Rank=rank) for rank, df in enumerate(name_merge_dfs)]
        name_merge_df = pd.concat(name_merge_dfs)
        name_merge_df["Duration"] = name_merge_df["End"] - name_merge_df["Start"]
        name_merge_df = name_merge_df.groupby("Name")
        name_merge_df_duration = name_merge_df["Duration"].sum()
        name_merge_df_compute_sum = name_merge_df["Compute Sum"].sum()
        name_merge_df_comm_sum = name_merge_df["Communication Sum"].sum()
        grouped_name_merge_df = pd.DataFrame(
            {
                "Duration": name_merge_df_duration,
                "Overlapped Compute Duration": name_merge_df_compute_sum,
                "Overlapped Compute Percentage": name_merge_df_compute_sum / name_merge_df_duration * 100,
                "Overlapped Communication Duration": name_merge_df_comm_sum,
                "Overlapped Communication Percentage": name_merge_df_comm_sum / name_merge_df_duration * 100,
            }
        ).round(1)

        grouped_name_merge_df.to_parquet(self.add_output_file("grouped_name_merge.parquet"))

        shortNamelist = comm_compute_final_dfs[0]['shortName']
        classlist = comm_compute_final_dfs[0]['type']
        comm_compute_final_dfs = [df.drop(columns=['shortName','type']) for rank,df in enumerate(comm_compute_final_dfs)]
        comm_compute_final_df = reduce(lambda left, right: left.add(right, fill_value=0), comm_compute_final_dfs)
        comm_compute_final_df.iloc[:, 1:] = comm_compute_final_df.iloc[:, 1:].div(comm_compute_final_df.iloc[:, 0], axis=0)
        comm_compute_final_df = pd.concat([shortNamelist, comm_compute_final_df, classlist], axis=1)
        comm_compute_final_df.to_parquet(self.add_output_file("comm_compute_final_df.parquet"))

        trace_gdf = trace_df.groupby("Name")
        duration = trace_gdf["Duration"].sum()
        comm_sum = trace_gdf["Communication Sum"].sum()
        compute_sum = trace_gdf["Compute Sum"].sum()

        grouped_trace_df = pd.DataFrame(
            {
                "Count": trace_gdf.size(),
                "Communication Overlap": comm_sum / duration * 100,
                "Compute Overlap": compute_sum / duration * 100,
            }
        ).round(1)

        grouped_trace_df.to_parquet(self.add_output_file("grouped_trace.parquet"))



        if self._parsed_args.csv:
            files_df.to_csv(self.add_output_file("files.csv"))
            rank_trace_df.to_csv(self.add_output_file("rank_trace.csv"))
            grouped_trace_df.to_csv(self.add_output_file("grouped_trace.csv"))
            comm_compute_final_df.to_csv(self.add_output_file("comm_compute_final_df.csv"))
            grouped_name_merge_df.to_csv(self.add_output_file("grouped_name_merge.csv"))
            grouped_type_merge_df.to_csv(self.add_output_file("grouped_type_merge_df.csv"))

    def save_notebook(self):
        self.create_notebook("trace.ipynb")
        self.add_notebook_helper_file("nsys_display.py")

    def save_analysis_file(self):
        self._analysis_dict.update(
            {
                "EndTime": str(datetime.now()),
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
        parser.add_recipe_argument(Option.START)
        parser.add_recipe_argument(Option.END)
        parser.add_recipe_argument(Option.CSV)
        parser.add_recipe_argument(
            "--self-overlap",
            action="store_true",
            help="Enable self-overlap calculation."
        )
        filter_group = parser.recipe_group.add_mutually_exclusive_group()
        parser.add_argument_to_group(filter_group, Option.FILTER_TIME)
        parser.add_argument_to_group(filter_group, Option.FILTER_NVTX)

        return parser
