
from datetime import datetime
from pathlib import Path

import pandas as pd
from nsys_recipe.lib import data_utils, helpers, pace, recipe, summary

from nsys_recipe.lib.args import Option
#from nsys_recipe.lib.loader import TableConfig


from nsys_recipe import log
from nsys_recipe.data_service import DataService
from nsys_recipe.log import logger


class NewMetricOverlap(recipe.Recipe):
    @staticmethod
    def mapper_func(report_path, parsed_args):
        service = DataService(report_path)
        # sqlite_file = helpers.nsysrep_to_sqlite(nsysrep)
        # if sqlite_file is None:
        #     return None
        
        name_column = 'demangledName'
        table_column_dict = {
            'TARGET_INFO_SESSION_START_TIME': None,
            'StringIds': None,
            'CUPTI_ACTIVITY_KIND_KERNEL': [name_column, 'start', 'end', 'deviceId']
        }

        # df_dict = loader.read_sqlite_tables(
        #     sqlite_file,
        #     table_column_dict,
        #     parsed_args.start,
        #     parsed_args.end
        # )
        df_dict = service.read_tables(table_column_dict)
        if df_dict is None:
            return None


        kernel_df = df_dict["CUPTI_ACTIVITY_KIND_KERNEL"]
        data_utils.filter_by_time_range(kernel_df, parsed_args.start, parsed_args.end)

        kernel_df = data_utils.replace_id_with_value(
            kernel_df, df_dict["StringIds"], name_column
        )
        if kernel_df.empty:
            logger.info(
                f"{report_path} was successfully processed, but no data was found."
            )
            return None
        
        # range_df = loader.replace_id_with_value(
        #     df_dict['CUPTI_ACTIVITY_KIND_KERNEL'],
        #     df_dict['StringIds'],
        #     name_column
        # )

        
        # range_df["kernelType"] = range_df[name_column].apply(NewMetricOverlap.getTypeFromKernelName)
        kernel_df["kernelType"] = kernel_df[name_column].apply(NewMetricOverlap.getTypeFromKernelName)

        computeTimeline = []
        ncclTimeline = []
        for index, row in kernel_df.iterrows():
            if row["kernelType"] == "NCCL":
                ncclTimeline.append((row["start"], row["end"]))
            else:
                computeTimeline.append((row["start"], row["end"]))
        
        timeDict, allTimeSplit = NewMetricOverlap.calculateOverlap(computeTimeline, ncclTimeline)

        # Create df based on the time split(array of tuples) of current rank
        allTimeDf = pd.DataFrame(allTimeSplit, columns=['start', 'end', 'OverlapType'])
        allTimeDf['duration'] = allTimeDf['end'] - allTimeDf['start']

        # # print(timeDict)
        # _totalTime = 0
        # for key in timeDict.keys():
        #     _totalTime += int(timeDict[key])
        # for key in timeDict.keys():
        #     print(key, timeDict[key]/1e9, "s", f"{timeDict[key]/_totalTime*100:.2f}%")

        # range_df[name_column] = range_df["Type"]

        filename = Path(report_path).stem
        session_start = pace.get_session_start_time(df_dict['TARGET_INFO_SESSION_START_TIME'])
        pace_df, stats_df = pace.compute_pace_stats_dfs(kernel_df, name_column)
        # pace_df, stats_df = pace.compute_pace_stats_dfs(allTimeDf, "OverlapType")
        return pace.PaceInfo(filename, allTimeDf, None, session_start)
        # return pace.PaceInfo(filename, pace_df, stats_df, session_start)


    # ncclKernel_AllGather_RING_LL_Sum_int8_t
    # Output expected to be NCCL
    @classmethod
    def getTypeFromKernelName(self, kername):
        if kername.startswith("ncclKernel_"):
            return "NCCL"
        if kername.startswith("ncclDevKernel_"):
            return "NCCL"
        if kername.startswith("ampere"):
            return "MATH_SPEC"
        if kername.startswith("cutlass"):
            return "MATH"
        if kername.startswith("at::native::"):
            return "MATH"
        if kername.startswith("void at::native::"):
            return "MATH"
        if kername.startswith("void transformer_engine::"):
            return "MATH"
        if kername.startswith("void <unnamed>::elementwise_kernel"):
            return "MATH"
        if kername.startswith("void fmha"):
            return "MATH"
        if kername.startswith("CudaCodeGen::kernel"):
            return "MATH"
        if "cublasLt" in kername:
            return "MATH"
        if kername.startswith("nvjpeg::"):
            return "NVJPEG"
        if kername.startswith("dali::"):
            return "DALI"

        return "MATH"
    
    @classmethod
    def mergeTimeRange(self, times):
        # times could be empty, for debugging
        if len(times) == 0:
            print("Zero length encountered")
            return None
        saved = list(times[0])
        for st, en in sorted([sorted(t) for t in times]):
            if st <= saved[1]:
                saved[1] = max(saved[1], en)
            else:
                yield tuple(saved)
                saved[0] = st
                saved[1] = en
        yield tuple(saved)

    @classmethod
    def calculateOverlap(self, computeTimeline, ncclTimeline):
        # Sort and merge the timeline by start time
        sortedNccl = [x for x in NewMetricOverlap.mergeTimeRange(ncclTimeline) if x is not None]
        sortedCompute = [x for x in NewMetricOverlap.mergeTimeRange(computeTimeline) if x is not None]

        idxNccl = 0
        idxCompute = 0
        timeDict = {"Overlap":0, "Communicate": 0, "Compute": 0, "Empty": 0}

        # Collect all time points(start or end)
        allTimeSplit = []
        flattenedTime = [i for tup in sortedNccl for i in tup]
        flattenedTime.extend([i for tup in sortedCompute for i in tup])
        flattenedTime = sorted(flattenedTime)
        StartTime, EndTime = flattenedTime[0], flattenedTime[-1]
        for idx in range(0, len(flattenedTime)-1):
            curStart = flattenedTime[idx]
            curEnd = flattenedTime[idx+1]
            if idxNccl < len(sortedNccl) and sortedNccl[idxNccl][0] <= curStart and curEnd <= sortedNccl[idxNccl][1]:
                # In NCCL seg
                if idxCompute < len(sortedCompute) and sortedCompute[idxCompute][0] <= curStart and curEnd <= sortedCompute[idxCompute][1]:
                    allTimeSplit.append((curStart, curEnd, "Overlap"))
                    timeDict["Overlap"] += curEnd - curStart
                else:
                    allTimeSplit.append((curStart, curEnd, "Communicate"))
                    timeDict["Communicate"] += curEnd - curStart
            elif idxCompute < len(sortedCompute) and sortedCompute[idxCompute][0] <= curStart and curEnd <= sortedCompute[idxCompute][1]:
                allTimeSplit.append((curStart, curEnd, "Compute"))
                timeDict["Compute"] += curEnd - curStart
            else:
                allTimeSplit.append((curStart, curEnd, "Empty"))
                timeDict["Empty"] += curEnd - curStart

            # Move the index to next if the time span end is the end item of current item.
            if idxNccl < len(sortedNccl) and sortedNccl[idxNccl][1] == curEnd:
                idxNccl += 1
            if idxCompute < len(sortedCompute) and sortedCompute[idxCompute][1] == curEnd:
                idxCompute += 1
        
        return timeDict, allTimeSplit

    def reducer_func(self, mapper_res):
        pace_infos = helpers.filter_none(mapper_res)
        pace_infos = sorted(pace_infos, key=lambda x: x.filename)

        filenames, pace_dfs, stats_dfs, session_starts = zip(*pace_infos)
        pace.apply_time_offset(session_starts, pace_dfs)

        rank_file_df = pd.DataFrame(filenames, columns=['File'])
        rank_file_df.to_parquet(self.add_output_file('files.parquet'), index=False)

        for p in pace_infos:
            name, type_df, _, session_start = p
            type_df.to_parquet(self.add_output_file(f'type_{name}.parquet'), index=False)

    def save_notebook(self):
        self.create_notebook('pace.ipynb')
        self.add_notebook_helper_file('nsys_pres.py')

    def save_analysis_file(self):
        self._analysis_dict.update({
            'EndTime': str(datetime.now()),
            'InputReports': self._parsed_args.input,
            'Outputs': self._output_files
        })
        self.create_analysis_file()

    def run(self, context):
        super().run(context)

        mapper_res = context.wait(context.map(
            self.mapper_func,
            self._parsed_args.input,
            parsed_args=self._parsed_args
        ))
        self.reducer_func(mapper_res)

        self.save_notebook()
        self.save_analysis_file()

    @classmethod
    def get_argument_parser(cls):
        parser = super().get_argument_parser()

        # mutually_exclusive_group = parser.recipe_group.add_mutually_exclusive_group(required=True)
        # parser.add_argument_to_group(mutually_exclusive_group, Option.REPORT_DIR)
        # parser.add_argument_to_group(mutually_exclusive_group, Option.INPUT)

        parser.add_recipe_argument(Option.INPUT, required=True)
        parser.add_recipe_argument(Option.OUTPUT)
        parser.add_recipe_argument(Option.FORCE_OVERWRITE)
        parser.add_recipe_argument(
            "--name",
            type=str,
            help="Name of the kernel used as delineator between iterations",
            required=True)
        parser.add_recipe_argument(Option.START)
        parser.add_recipe_argument(Option.END)

        return parser
