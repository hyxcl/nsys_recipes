# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys

import numpy as np
import pandas as pd


def _get_time_cols(df):
    if "start" in df.columns:
        if "end" in df.columns:
            # Time range.
            return ("start", "end")
        else:
            # Point in time.
            return ("start", "start")

    if "timestamp" in df.columns:
        # Point in time.
        return ("timestamp", "timestamp")
    elif "rawTimestamp" in df.columns:
        # Point in time.
        return ("rawTimestamp", "rawTimestamp")
    else:
        raise NotImplementedError()


def filter_by_time_range(
    dfs,
    start_time=0,
    end_time=sys.maxsize,
    start_col=None,
    end_col=None,
    strict_start=None,
    strict_end=None,
):
    """Filter the dataframe(s) to retain only events that start
    or end within the given range.

    Parameters
    ----------
    dfs : list of dataframes or dataframe
        Dataframes to filter.
    start_time : int
        Start time of the desired range.
    end_time : int
        End time of the desired range.
    start_col : str
        Name of the column that contains the start time.
    end_col : str
        Name of the column that contains the end time.
    strict_start : int
        If specified we apply strict boundaries for the start time. We discard all events that start
        before strict_start.
    strict_end : int
        If specified we apply strict boundaries for the end time. We discard all events that end
        after strict_end.
    """
    if not isinstance(dfs, list):
        dfs = [dfs]

    for df in dfs:
        if df.empty:
            continue

        default_start_col, default_end_col = _get_time_cols(df)
        start_col = start_col or default_start_col
        end_col = end_col or default_end_col

        mask = pd.Series(True, index=df.index)
        if end_time is not None:
            mask &= df[start_col] <= end_time
        if strict_end is not None:
            mask &= df[end_col] <= strict_end
        if start_time is not None:
            mask &= df[end_col] >= start_time
        if strict_start is not None:
            mask &= df[start_col] >= strict_start

        df.drop(df[~mask].index, inplace=True)

def add_time_range_mask(
    dfs,
    subtimerange=None,
):
    if not isinstance(dfs, list):
        dfs = [dfs]

    for df in dfs:
        if df.empty:
            continue
        if subtimerange == None:
            continue
        df['SubRange']=0
        for i in range(len(subtimerange)):
            '''
            mask = pd.Series(True, index=df.index)
            mask &= df['Start'] <= subtimerange[i][1]
            mask &= df['End'] >= subtimerange[i][0]
            colname = f"Subrange_{i}"
            df[colname] = mask
            '''
            mask = pd.Series(True, index=df.index)
            mask &= df['Start'] <= subtimerange[i][1]
            mask &= df['End'] >= subtimerange[i][0]
            df.loc[mask == True, 'SubRange'] = i+1
            
    return df


def apply_time_offset(dfs, session_offset, start_col=None, end_col=None):
    """Synchronize session start times.

    Parameters
    ----------
    dfs : list of dataframes or dataframe
        Dataframes to filter.
    session_offset : int
        Offset of the session time
    """
    if not isinstance(dfs, list):
        dfs = [dfs]

    for df in dfs:
        if df.empty:
            continue

        default_start_col, default_end_col = _get_time_cols(df)
        start_col = start_col or default_start_col
        end_col = end_col or default_end_col

        df.loc[:, start_col] += session_offset

        if start_col != end_col:
            df.loc[:, end_col] += session_offset


def compute_session_duration(
    analysis_df, target_info_df, min_session, disable_alignment
):
    profile_duration = analysis_df.at[0, "duration"]

    if disable_alignment:
        return 0, profile_duration

    session_time = target_info_df.at[0, "utcEpochNs"]
    session_offset = session_time - min_session
    profile_duration += session_offset

    return session_offset, profile_duration


def replace_id_with_value(main_df, str_df, id_column, value_col_name=None):
    """Replace the values in 'id_column' of 'main_df' with the corresponding
    string value stored in 'str_df'.

    Parameters
    ----------
    main_df : dataframe
        Dataframe containing 'id_column'.
    str_df : dataframe
        Dataframe 'StringId' that maps IDs to string values.
    id_column : str
        Name of the column that should be replaced with the corresponding
        string values.
    value_col_name : str
        Name of the column that contains the string value of 'id_column'.
        If not specified, the 'id_column' will be retained as the column name.
    """
    renamed_str_df = str_df.rename(columns={"id": id_column})
    merged_df = main_df.merge(renamed_str_df, on=id_column, how="left")

    # Drop the original 'id_column' column.
    merged_df = merged_df.drop(columns=[id_column])
    # Rename the 'value' column.
    value_col_name = value_col_name or id_column
    return merged_df.rename(columns={"value": value_col_name})


def add_cols_from_global_pid(df):
    # <Hardware ID:8><VM ID:8><Process ID:24><ThreadID:24>
    global_id = df["globalPid"]

    if "pid" not in df:
        df["pid"] = (global_id >> np.array(24)) & 0x00FFFFFF


def add_cols_from_global_tid(df):
    # <Hardware ID:8><VM ID:8><Process ID:24><ThreadID:24>
    global_id = df["globalTid"]

    if "pid" not in df:
        df["pid"] = (global_id >> np.array(24)) & 0x00FFFFFF
    if "tid" not in df:
        df["tid"] = global_id & 0x00FFFFFF


def add_cols_from_type_id(df):
    # <Hardware ID:8><VM ID:8><Source ID:16><Event tag:24><GPU ID:8>
    type_id = df["typeId"]

    if "gpuId" not in df:
        df["gpuId"] = type_id & 0xFF


def decompose_bit_fields(df):
    if "globalPid" in df:
        add_cols_from_global_pid(df)
    elif "globalTid" in df:
        add_cols_from_global_tid(df)
    elif "typeId" in df:
        add_cols_from_type_id(df)
