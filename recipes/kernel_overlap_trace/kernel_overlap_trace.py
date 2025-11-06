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

        kernel_grouped = kernel_df.groupby(["pid", "deviceId"])
        results = []
        results_merge = []
        # merge by name

        kernel_df_merge = overlap.merge_overlapping_ranges_by_name(kernel_df, self_overlapped_duration=parsed_args.self_overlap)
        kernel_matrix = overlap.calculate_overlap_sum_matrix(kernel_df_merge)
        kernel_matrix = kernel_matrix.sort_values(by='original_duration', ascending=False).reset_index()     
        desired_order = ['shortName','original_duration']+list(kernel_matrix['shortName'])
        kernel_matrix_final = kernel_matrix.reindex(columns=desired_order,fill_value=0)
        filename = Path(report_path).stem

        return filename, kernel_matrix_final

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
        filenames, kernel_matrix_finals = zip(*filtered_res)

        files_df = pd.DataFrame({"File": filenames}).rename_axis("Rank")
        files_df.to_parquet(self.add_output_file("files.parquet"))


        shortNamelist = kernel_matrix_finals[0]['shortName']
        kernel_matrix_finals = [df.drop(columns=['shortName']) for rank,df in enumerate(kernel_matrix_finals)]
        kernel_matrix_final = reduce(lambda left, right: left.add(right, fill_value=0), kernel_matrix_finals)
        kernel_matrix_final.iloc[:, 1:] = kernel_matrix_final.iloc[:, 1:].div(kernel_matrix_final.iloc[:, 0], axis=0)
        kernel_matrix_final = pd.concat([shortNamelist, kernel_matrix_final], axis=1)
        kernel_matrix_final.to_parquet(self.add_output_file("kernel_matrix_final.parquet"))

        if self._parsed_args.csv:
            files_df.to_csv(self.add_output_file("files.csv"))
            kernel_matrix_final.to_csv(self.add_output_file("kernel_matrix_final.csv"))

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
