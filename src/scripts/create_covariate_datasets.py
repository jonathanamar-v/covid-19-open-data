#!/usr/bin/env python
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple
import pandas as pd
from tqdm.notebook import tqdm

# Add our library utils to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.concurrent import process_map
from lib.constants import SRC
from lib.io import read_file

URL = "https://storage.googleapis.com/covid19-open-data/v3/aggregated.csv.gz"
LAWATLAS_URL = URL.replace("aggregated.csv.gz", "lawatlas-emergency-declarations.csv")
LAWATLAS = read_file(LAWATLAS_URL)
READ_OPTS = dict(na_values=[""], keep_default_na=False)
DATA_COLUMNS = pd.read_csv(URL, nrows=0, **READ_OPTS).columns
FEATURES = list(DATA_COLUMNS) + list(LAWATLAS.columns)

COLS_IDX = ["location_key", "date", "population"]
COLS_EPI = ["new_confirmed", "new_deceased", "new_hospitalized"]

COVARIATE_GROUPS = {
    "mobility": [col for col in FEATURES if col.startswith("mobility")],
    # "emergency_declarations": [col for col in FEATURES if col.startswith("lawatlas")],
    "search_trends": [
        "search_trends_fever",
        "search_trends_cough",
        "search_trends_common_cold",
        "search_trends_fatigue",
        "search_trends_shortness_of_breath",
        "search_trends_sputum",
        "search_trends_myalgia",
        "search_trends_chills",
        "search_trends_dizziness",
        "search_trends_headache",
        "search_trends_sore_throat",
        "search_trends_nausea",
        "search_trends_vomiting",
        "search_trends_diarrhea",
    ],
    "weather": [
        "minimum_temperature_celsius",
        "maximum_temperature_celsius",
        "average_temperature_celsius",
        "relative_humidity",
        "rainfall_mm",
    ],
    "government_response": [
        "school_closing",
        "workplace_closing",
        "cancel_public_events",
        "restrictions_on_gatherings",
        "public_transport_closing",
        "stay_at_home_requirements",
        "restrictions_on_internal_movement",
        "international_travel_controls",
        "income_support",
        "debt_relief",
        "fiscal_measures",
        "international_support",
        "public_information_campaigns",
        "testing_policy",
        "contact_tracing",
        "emergency_investment_in_healthcare",
        "investment_in_vaccines",
        "stringency_index",
    ],
    "static": [
        "latitude",
        "longitude",
        "area_sq_km",
        "elevation_m",
        "population",
        "population_male",
        "population_female",
        "population_age_00_09",
        "population_age_10_19",
        "population_age_20_29",
        "population_age_30_39",
        "population_age_40_49",
        "population_age_50_59",
        "population_age_60_69",
        "population_age_70_79",
        "population_age_80_and_older",
        "life_expectancy",
    ],
}


def get_data(columns: List[str]) -> pd.DataFrame:
    usecols = [col for col in columns if col in DATA_COLUMNS]
    return pd.read_csv(URL, usecols=usecols, **READ_OPTS).merge(LAWATLAS, how="left")[columns]


def convert_to_timeseries(data: pd.DataFrame, static_columns: List[str] = None) -> pd.DataFrame:
    print("Converting data to time series format")

    records = {}
    suffixes = set()

    # Make sure that location_key is part of the static columns
    static_columns = set((static_columns or []) + ["location_key"])
    tqdm_opts = dict(total=len(data), desc="Converting to time series")
    for _, row in tqdm(data.iterrows(), **tqdm_opts):
        key = row["location_key"]
        suffix = row["date"].replace("-", "_")
        suffixes.add(suffix)

        # Make sure that an entry for this location key exists in the outputs
        if key not in records:
            records[key] = {col: row[col] for col in static_columns}

        # Copy all the columns into the output using the date as the suffix
        for col, val in row.to_dict().items():
            if col != "date" and col not in static_columns:
                records[key][f"{col}_{suffix}"] = val

    suffixes = list(sorted(suffixes))
    return suffixes, pd.DataFrame(records.values())


def create_rolling_aggregates(
    data: pd.DataFrame, suffixes: List[str], window_size: int = 21
) -> pd.DataFrame:
    print("Creating rolling aggregates for each record")

    subsets = []
    suffix_template = "YYYY_mm_dd"

    # Keep track of columns which do not need rolling
    static_columns = [
        col for col in data.columns if not any(col.endswith(suffix) for suffix in suffixes)
    ]

    # Iterate over each of the starting indices
    tqdm_opts = dict(desc="Creating rolling aggregates")
    for idx in tqdm(range(len(suffixes) - window_size), **tqdm_opts):

        # Only keep the columns between starting index up to window size
        suffixes_subset = [suffixes[i] for i in range(idx, idx + window_size)]
        keep_columns = [
            col for col in data.columns if any(col.endswith(suffix) for suffix in suffixes_subset)
        ]

        # Rename the date suffixes to step_1, step_2, step_3...
        suffix_builder = lambda col: f"{col[: -len(suffix_template) - 1]}_step_" + str(
            1 + suffixes_subset.index(col[-len(suffix_template) :])
        )
        column_adapter = {col: suffix_builder(col) for col in keep_columns}
        subset = data[static_columns + keep_columns].rename(columns=column_adapter)

        # Add the starting date for each subset as a column
        subset["date"] = suffixes_subset[0].replace("_", "-")
        subsets.append(subset)

    return pd.concat(subsets)


def aggregate_predict_values(
    data: pd.DataFrame,
    target_columns: List[str],
    split_train_predict: Tuple[int, int] = (14, 7),
    aggregate_function: Callable = sum,
) -> pd.DataFrame:
    print("Aggregating the predicted values")

    window_size = sum(split_train_predict)
    suffixes_predict = [f"step_{i + 1}" for i in range(split_train_predict[0], window_size)]

    # Aggregate the target columns for all the predict values
    for column in target_columns:
        agg_columns = [f"{column}_{suffix}" for suffix in suffixes_predict]
        data[column] = data.loc[:, agg_columns].apply(aggregate_function, axis=1)

    # Remove all rolling columns outside of the training range
    drop_columns = [
        col for col in data.columns if any(col.endswith(suffix) for suffix in suffixes_predict)
    ]
    data = data.drop(columns=drop_columns)

    return data


def format_training_data(
    data: pd.DataFrame,
    target_columns: List[str],
    static_columns: List[str] = None,
    split_train_predict: Tuple[int, int] = (14, 7),
    aggregate_function: Callable = sum,
) -> pd.DataFrame:
    suffixes, data = convert_to_timeseries(data, static_columns=static_columns)

    window_size = sum(split_train_predict)
    data = create_rolling_aggregates(data, suffixes, window_size=window_size)

    return aggregate_predict_values(
        data,
        target_columns,
        split_train_predict=split_train_predict,
        aggregate_function=aggregate_function,
    )


def preprocess_data(
    data: pd.DataFrame, cov_name: str, cols_cov: List[str], split_train_predict: Tuple[int, int]
) -> pd.DataFrame:

    # Convert epi columns to relative with respect to population
    for column in COLS_EPI:
        data[column] = 100_000 * data[column] / data["population"]

    # Remove population unless it's one of the covariates
    if "population" not in cols_cov:
        data = data.drop(columns=["population"])

    # Drop records which have no covariate data
    data = data.dropna(subset=cols_cov, how="all")

    # Format the data for training purposes
    static_columns = cols_cov if cov_name == "static" else []
    return format_training_data(
        data, COLS_EPI, static_columns=static_columns, split_train_predict=split_train_predict
    )


def slice_data_by_target_output(data: pd.DataFrame, col_target: str) -> pd.DataFrame:
    target_label = col_target.replace("new_", "")

    # Remove all the targets that are not part of this prediction
    data = data.drop(columns=[col for col in COLS_EPI if col != col_target])

    # Remove all rows that have a null target
    data = data.dropna(subset=[col_target])

    # Rename target label to make it easier to recognize
    data = data.rename(columns={col_target: f"target_{target_label}"})

    return data


def create_covariate_datasets(
    cov_name: str, output_path: Path, split_train_predict: Tuple[int, int]
):
    cols_cov = COVARIATE_GROUPS[cov_name]

    # Download the data using only the necessary columns
    print("Downloading data...")
    data = get_data(COLS_IDX + COLS_EPI + cols_cov)
    print("Data downloaded")

    # Preprocess the dataset by normalizing w.r.t population and creating training records
    data = preprocess_data(data, cov_name, cols_cov, split_train_predict=split_train_predict)

    # Slice the dataset for each of the desired prediction targets
    for target in tqdm(COLS_EPI, desc="Slicing by target label"):
        df = slice_data_by_target_output(data, target)

        # Save to local storage
        target_label = target.replace("new_", "")
        fname = f"{cov_name}_{target_label}_training_data.csv"
        df.to_csv(output_path / fname, index=False)


def main(output_path: Path, split_train_predict: Tuple[int, int]):
    wrap_func = create_covariate_datasets
    map_func = partial(wrap_func, output_path=output_path, split_train_predict=split_train_predict)
    for _ in process_map(map_func, COVARIATE_GROUPS.keys(), desc="Processing covariate groups"):
        pass


if __name__ == "__main__":
    output_root = SRC / ".." / "output" / "covariates"
    output_root.mkdir(exist_ok=True, parents=True)

    argparser = ArgumentParser()
    argparser.add_argument("--train-window-size", type=int, default=14)
    argparser.add_argument("--predict-window-size", type=int, default=7)
    args = argparser.parse_args()

    main(output_root, (args.train_window_size, args.predict_window_size))
