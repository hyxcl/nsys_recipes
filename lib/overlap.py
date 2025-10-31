# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import defaultdict

import numpy as np
import pandas as pd
import time

def group_overlapping_ranges(range_df):
    """Assign unique group identifiers to overlapping ranges.

    Parameters
    ----------
    range_df : dataframe
        DataFrame containing ranges with 'start' and 'end' columns.

    Returns
    -------
    result : series
        Series containing group identifiers for each range.
    """
    df = range_df.sort_values("start")
    cumulative_max_end = df["end"].cummax()
    groups = (df["start"] > cumulative_max_end.shift()).cumsum()
    return groups.reindex(range_df.index)


def consolidate_ranges(range_df):
    """Consolidate overlapping time ranges.

    For each set of overlapping ranges, only the earliest start time and latest
    end time will be retained.

    Parameters
    ----------
    range_df : dataframe
        DataFrame containing ranges with 'start' and 'end' columns.

    Returns
    -------
    result : dataframe
        DataFrame with overlapping ranges consolidated, where the 'start' column
        contains the minimum start value and the 'end' column contains the
        maximum end value for each group.
    """
    groups = group_overlapping_ranges(range_df)
    return range_df.groupby(groups).agg({"start": "min", "end": "max"})


def process_overlapping_ranges(df1, df2, process_func, fully_contained=False):
    """Process overlapping ranges between two dataframes.

    Parameters
    ----------
    df1 : dataframe
        Dataframe containing ranges with 'start' and 'end' columns. If
        'fully_contained' is True, this is checked to see if its ranges
        fully contain the ranges from 'df2'.
    df2 : dataframe
        Dataframe containing ranges with 'start' and 'end' columns. If
        'fully_contained' is True, this is checked to see if its ranges
        are fully contained within the ranges from 'df1'.
    process_func : callable
        Function to process overlapping ranges. It takes two arguments:
        - df1_index: index of a row from 'df1' that overlaps with 'df2_row'.
        - df2_row: itertuples iterator representing a row from 'df2', containing
            the index, start, and end values.
    fully_contained : bool
        Whether to check if the ranges are fully contained within each other.
        Fully contained ranges must have their start and end values within the
        start and end values of the containing range, with the end being
        exclusive.
    """
    if df1 is None or df1.empty or df2 is None or df2.empty:
        return

    df2_time_df = pd.DataFrame(
        data={"start": df2["start"], "end": df2["end"], "shortName": df2["shortName"]}
    ).sort_values("start")

    df1_active_indices = set()
    df1_start_df = pd.DataFrame(data={"time": df1["start"]}).sort_values("time")
    df1_end_df = pd.DataFrame(data={"time": df1["end"]}).sort_values("time")

    df2_iter = iter(df2_time_df.itertuples())
    df1_start_iter = iter(df1_start_df.itertuples())
    df1_end_iter = iter(df1_end_df.itertuples())

    df2_row = next(df2_iter)
    df1_start_row = next(df1_start_iter)
    df1_end_row = next(df1_end_iter)

    while True:
        should_include_range = False
        if df1_start_row is not None:
            if fully_contained:
                should_include_range = df1_start_row.time <= df2_row.start
            else:
                should_include_range = df1_start_row.time < df2_row.end

        if should_include_range:
            df1_active_indices.add(df1_start_row.Index)

            try:
                df1_start_row = next(df1_start_iter)
            except StopIteration:
                df1_start_row = None
        elif df1_end_row.time <= df2_row.start:
            df1_active_indices.remove(df1_end_row.Index)

            try:
                df1_end_row = next(df1_end_iter)
            except StopIteration:
                break
        else:
            for index in df1_active_indices:
                # Check if the end of the range is contained, as only the start
                # was checked in the first condition. If the end is not
                # contained, skip the current 'df2' range.
                if fully_contained and df2_row.end > df1_end_df.loc[index, "time"]:
                    continue

                process_func(index, df2_row)

            try:
                df2_row = next(df2_iter)
            except StopIteration:
                break


def map_overlapping_ranges(df1, df2, key_df="df2", fully_contained=False):
    """Map overlapping ranges between two dataframes.

    Parameters
    ----------
    df1 : dataframe
        Dataframe containing ranges with 'start' and 'end' columns. If
        'fully_contained' is True, this is checked to see if its ranges
        fully contain the ranges from 'df2'.
    df2 : dataframe
        Dataframe containing ranges with 'start' and 'end' columns. If
        'fully_contained' is True, this is checked to see if its ranges
        are fully contained within the ranges from 'df1'.
    key_df : str
        Whether indices of 'df1' or 'df2' should be used as the key of the
        resulting mapping. Must be either 'df1' or 'df2'.
    fully_contained : bool
        Whether to check if the ranges are fully contained within each other.
        Fully contained ranges must have their start and end values within the
        start and end values of the containing range, with the end being
        exclusive.

    Returns
    -------
    overlap_map : dict
        Dictionary that maps indices of the 'key_df' to the indices of the
        corresponding ranges in the other dataframe.
    """
    if key_df != "df1" and key_df != "df2":
        raise ValueError("key_df must be either 'df1' or 'df2'.")

    overlap_map = defaultdict(set)

    def process_func(df1_index, df2_row):
        if key_df == "df1":
            overlap_map[df1_index].add(df2_row.Index)
        else:
            overlap_map[df2_row.Index].add(df1_index)

    process_overlapping_ranges(df1, df2, process_func, fully_contained)
    return overlap_map


def calculate_overlapping_ranges(df1, df2=None, overlapping_shortName=False):
    """Calculate the overlapping ranges between two dataframes.

    Parameters
    ----------
    df1 : dataframe
        DataFrame containing ranges to calculate the overlap from, with 'start'
        and 'end' columns.
    df2 : dataframe, optional
        DataFrame containing ranges to calculate the overlap with, with 'start'
        and 'end' columns. If not provided, the function calculates the
        overlap within df1.

    Returns
    -------
    result : dataframe
        DataFrame containing overlapping ranges, with the following columns:
        - start: start position of the overlap.
        - end: end position of the overlap.
        - original_index: index of the original row in df1.
        These ranges may not exactly match the original ranges, as they could
        be created by combining start and end values from different ranges.
    """
    overlap_map = defaultdict(set)

    def process_func(df1_index, df2_row):
        overlap_map[df1_index].add((df2_row.Index, df2_row.start, df2_row.end, df2_row.shortName))

    if df2 is None:
        process_overlapping_ranges(df1, df1, process_func)
    else:
        process_overlapping_ranges(df1, df2, process_func)

    results = []

    for df1_row in df1.itertuples():
        if df1_row.Index not in overlap_map:
            continue

        indices, starts, ends, shortNames = zip(*overlap_map[df1_row.Index])
        indices = np.array(indices)
        starts_array = np.array(starts)
        ends_array = np.array(ends)
        shortNames_array = np.array(shortNames)

        # We don't want to consider the overlap between the same range
        # instances.
        if df2 is None:
            non_self_mask = indices != df1_row.Index
            starts_array = starts_array[non_self_mask]
            ends_array = ends_array[non_self_mask]
            shortNames_array = shortNames_array[non_self_mask]

        overlap_start = np.maximum(df1_row.start, starts_array)
        overlap_end = np.minimum(df1_row.end, ends_array)
        overlap_duration = overlap_end - overlap_start

        valid_overlap_mask = overlap_duration > 0
        overlap_start = overlap_start[valid_overlap_mask]
        overlap_end = overlap_end[valid_overlap_mask]
        shortNames_array = shortNames_array[valid_overlap_mask]

        if overlapping_shortName:
            results.extend(
                zip(overlap_start, overlap_end, [df1_row.Index] * len(overlap_start), shortNames_array)
            )
            columns = ["start", "end", "original_index", "overlapped_shortName"]
        else:
            results.extend(
                zip(overlap_start, overlap_end, [df1_row.Index] * len(overlap_start))
            )
            columns = ["start", "end", "original_index"]

    return pd.DataFrame(results, columns=columns)


def calculate_overlap_sum(df1, df2=None, consolidate=True):
    """Calculate the sum of overlapping durations between two dataframes.

    Parameters
    ----------
    df1 : dataframe
        DataFrame containing ranges to calculate the overlap from, with 'start'
        and 'end' columns.
    df2 : dataframe, optional
        DataFrame containing ranges to calculate the overlap with, with 'start'
        and 'end' columns. If not provided, the function calculates the
        overlap within df1.
    consolidate : bool, optional
        Whether to consolidate overlapping ranges. If True, only one overlap is
        considered for each range.

    Returns
    -------
    result : series
        Series containing the sum of overlapping durations for each row in df1.
        Non overlapping ranges will have a sum of 0.
    """
    overlap_df = calculate_overlapping_ranges(df1, df2)

    if consolidate:
        overlap_df = (
            overlap_df.assign(groups=group_overlapping_ranges(overlap_df))
            .groupby(["original_index", "groups"])
            .agg({"start": "min", "end": "max"})
        )

    overlap_df["duration"] = overlap_df["end"] - overlap_df["start"]
    total_duration = overlap_df.groupby("original_index")["duration"].sum()
    return total_duration.reindex(df1.index, fill_value=0.0).astype(float).round(1)


def merge_overlapping_ranges_by_name(df, self_overlapped_duration=False):
    """Merge overlapping ranges for the same shortName. and the duration in the finnal column is the sum of the duration of the overlapping ranges. 

    example:
    df1:
       shortName  start  end
    0  allgather      1    4
    1  allreduce      2    3
    2  broadcast      6    8
    3  allgather      7   30
    4  allgather     10   15
    5  allgather     15   20
    
        shortName  start  end  self_overlapped_duration  
    0  allgather      1    4       0.0                 
    1  allgather      7   30      10.0                 
    2  allreduce      2    3       0.0                  
    3  broadcast      6    8       0.0                  

    Parameters
    ----------
    df : dataframe
        DataFrame containing ranges with 'shortName', 'start', and 'end' columns.
    
    Returns
    -------
    result : dataframe
        DataFrame with overlapping ranges merged for each shortName.
    """
    
    final_df=pd.DataFrame()
    result_rows = []
    for name, group in df.groupby('shortName'):
        # Use group_overlapping_ranges to assign group identifiers
        group_with_groups = group.assign(groups=group_overlapping_ranges(group))

        # Group by the assigned groups and consolidate ranges
        consolidated = group_with_groups.groupby('groups').agg({
            'shortName': 'first',
            'start': 'min',
            'end': 'max'
        }).reset_index()
        
        # Calculate overlapped duration for each consolidated group
        if self_overlapped_duration:          
            for group_id, consolidated_group in consolidated.groupby('groups'):  
                original_ranges = group_with_groups[group_with_groups['groups'] == group_id]         
                    # Calculate total overlapped duration
                overlapped_duration = calculate_self_overlapped_duration(original_ranges)     
                result_rows.append({
                    'shortName': name,
                    'start': consolidated_group.iloc[0]['start'],
                    'end': consolidated_group.iloc[0]['end'],
                    'self_overlapped_duration': overlapped_duration
                })
        else:
            consolidated['self_overlapped_duration'] = 0.0
            result_rows.extend(consolidated.drop(columns=['groups']).to_dict(orient='records'))
    final_df = pd.DataFrame(result_rows)
    return final_df

def merge_by_type(df, streamid=False):
    """Merge overlapping ranges for the same type.

    """
    
    final_df=pd.DataFrame()
    result_rows = []
    for name, group in df.groupby('type'):
        if streamid:
            for stream_id, stream_group in group.groupby('streamId'):
                stream_group_with_groups = stream_group.assign(groups=group_overlapping_ranges(stream_group))
                stream_consolidated = stream_group_with_groups.groupby('groups').agg({
                    'type': 'first',
                    'start': 'min',
                    'end': 'max' ,
                    'streamId': 'first'
                }).reset_index()
                result_rows.extend(stream_consolidated.drop(columns=['groups']).to_dict(orient='records'))
        else:
            group_with_groups = group.assign(groups=group_overlapping_ranges(group))    
            consolidated = group_with_groups.groupby('groups').agg({
                'type': 'first',
                'start': 'min',
                'end': 'max' ,
            }).reset_index()
            consolidated['streamId'] = 'all'
            result_rows.extend(consolidated.drop(columns=['groups']).to_dict(orient='records'))
    final_df = pd.DataFrame(result_rows)
    final_df.rename(columns={'type': 'shortName'}, inplace=True)
    return final_df


def calculate_self_overlapped_duration(ranges_df):
    """Calculate the total overlapped duration within a group of ranges.
    example, in merge_overlapping_ranges_by_name, the self_overlapped_duration is the sum of the duration of the selfoverlapping ranges. the self_overlapped_duration is calculated by this function.

    df1:
       shortName  start  end
    0  allgather      1    4
    1  allreduce      2    3
    2  broadcast      6    8
    3  allgather      7   30
    4  allgather     10   15
    5  allgather     15   20
    
        shortName  start  end  self_overlapped_duration  
    0  allgather      1    4       0.0                 
    1  allgather      7   30      10.0                 
    2  allreduce      2    3       0.0                  
    3  broadcast      6    8       0.0          
    
    Parameters
    ----------
    ranges_df : dataframe
        DataFrame containing ranges with 'start' and 'end' columns.
    
    Returns
    -------
    duration : float
        Total overlapped duration.
    """
    if len(ranges_df) <= 1:
        return 0.0  # No overlap possible with single range
    
    # Sort by start time
    ranges_sorted = ranges_df.sort_values('start').reset_index(drop=True)
    
    # Use a sweep line algorithm to find overlapped regions
    events = []
    for _, row in ranges_sorted.iterrows():
        events.append((row['start'], 'start'))
        events.append((row['end'], 'end'))
    
    events.sort()
    
    overlapped_duration = 0
    active_count = 0
    overlap_start = None
    
    for time, event_type in events:
        if event_type == 'start':
            active_count += 1
            if active_count == 2 and overlap_start is None:
                overlap_start = time
        else:  # end
            active_count -= 1
            if active_count == 1 and overlap_start is not None:
                overlapped_duration += time - overlap_start
                overlap_start = None
    
    return overlapped_duration

def calculate_overlap_sum_matrix(df1, df2=None, consolidate=True):
    """Calculate the sum of overlapping durations between two dataframes.

    """
    overlap_df = calculate_overlapping_ranges(df1, df2, overlapping_shortName=True)
    # If original_index and overlapped_shortName are both the same, sum them
    overlap_df["duration"] = overlap_df["end"] - overlap_df["start"]
    overlap_df = overlap_df.groupby(["original_index", "overlapped_shortName"])["duration"].sum().reset_index()
    df1["original_duration"] = df1["end"] - df1["start"]
    # Create a pivot table where overlapped_shortName becomes columns
    pivot_df = overlap_df.pivot_table(
        index="original_index", 
        columns="overlapped_shortName", 
        values="duration", 
        aggfunc="sum", 
        fill_value=0.0
    )
    # Ensure all shortName values from df2 are included as columns
    if df2 is not None:
        all_df2_shortnames = df2['shortName'].unique()
        for shortname in all_df2_shortnames:
            if shortname not in pivot_df.columns:
                pivot_df[shortname] = 0.0
        # Reorder columns to match df2 shortName order
        pivot_df = pivot_df.reindex(columns=all_df2_shortnames, fill_value=0.0)
    
    if df2 is None:
        pivot_gruped_df1 = df1.reset_index().pivot_table(
            index="index", 
            columns="shortName", 
            values="self_overlapped_duration", 
            aggfunc="sum", 
            fill_value=0.0
        )
        pivot_df = pivot_df.add(pivot_gruped_df1, fill_value=0.0)
    # Merge pivot_df with df1 (original_index in pivot_df corresponds to df1 index)
    merged_df = df1.drop(columns=['start', 'end', 'self_overlapped_duration']).join(pivot_df.astype(float).round(1)).fillna(0)

    final_df = merged_df.groupby("shortName").sum()
    
    return final_df
