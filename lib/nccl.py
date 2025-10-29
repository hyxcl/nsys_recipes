# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pandas as pd


def add_kernel_type(kernel_df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column named "type" that will either be 'nccl' or 'compute'."""
    type_df = kernel_df.assign(type="compute")
    type_df.loc[kernel_df["shortName"].str.startswith("nccl"), "type"] = "nccl"
    type_df.loc[kernel_df["demangledName"].str.startswith("void deep_ep"), "type"] = "nccl"
    return type_df
