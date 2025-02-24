# %%
import pickle
import numpy as np
import pandas as pd
import re
from typing import Optional, List, Dict, Tuple
from fuzzywuzzy import process
from geopy.distance import geodesic
from config import dir_config


# Utility Functions
class StationNameCleaner:
    """
    Provides functionality to clean and fuzzy-match station names.
    """

    @staticmethod
    def clean_station_name_strings(station_name: Optional[str]) -> Optional[str]:
        if isinstance(station_name, str):
            return re.sub(r"\s*(v\.|amh\.|vá\.|id\.)\s*", "", station_name)
        return station_name

    @staticmethod
    def fuzzy_match_stations(
        not_found_stops: List[str],
        stops_df: pd.DataFrame,
        stops_to_skip: Optional[List[str]] = None,
        threshold: int = 80,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        stop_names = stops_df["stop_name"].tolist()
        mapping_dict, not_found_dict = {}, {}
        stops_to_skip = stops_to_skip or []

        for stop in not_found_stops:
            normalized_stop = stop.strip().lower()
            if normalized_stop in map(str.lower, stops_to_skip):
                continue
            match, score = process.extractOne(stop, stop_names)
            if score >= threshold:
                mapping_dict[stop] = match
            else:
                not_found_dict[stop] = match

        return mapping_dict, not_found_dict


# Class to handle Sequence Assignment
class SequenceAssigner:
    def __init__(self, group_id: str = "train_number"):
        self.group_id = group_id

    def assign_sequence_id(
        self, df: pd.DataFrame, station_col: str = "stop_name"
    ) -> pd.DataFrame:
        df = df.copy()
        df["station_concat"] = df.groupby(self.group_id)[station_col].transform(
            lambda x: "->".join(x)
        )
        df_unique = df.drop_duplicates(subset=["station_concat"]).copy()
        df_unique["sequence_id"] = df_unique.groupby(self.group_id).cumcount() + 1

        df = df.merge(
            df_unique[[self.group_id, "station_concat", "sequence_id"]],
            on=[self.group_id, "station_concat"],
            how="left",
        )
        df.drop(columns=["station_concat"], inplace=True)
        return df


# Class to Process Schedules
class ScheduleProcessor:
    def __init__(self, g, vertex_df, stops_df):
        self.graph = g
        self.vertex_df = vertex_df
        self.stops_df = stops_df

    def load_schedules(self, filepath: str) -> pd.DataFrame:
        schedules_df = pd.read_parquet(filepath).rename(
            columns={
                "Km": "distance_km",
                "Station": "stop_name",
                "Scheduled Arrival": "scheduled_arrival",
                "Scheduled Departure": "scheduled_departure",
                "Estimated Arrival": "estimated_arrival",
                "Estimated Departure": "estimated_departure",
                "Platform": "platform",
                "Status": "status",
            }
        )
        schedules_df = schedules_df[schedules_df["distance_km"].str.len() > 0]
        return schedules_df

    def preprocess_schedules(
        self, schedules_df: pd.DataFrame, group_id: str = "train_number"
    ) -> pd.DataFrame:
        assigner = SequenceAssigner(group_id=group_id)
        schedules_df = assigner.assign_sequence_id(schedules_df)
        schedules_df.drop(columns=["sequence_id"], inplace=True)
        schedules_df["observation_sequence_id"] = range(len(schedules_df))

        # Format date and time columns
        date = schedules_df["elvira_id"].str.split("_").str[1]
        schedules_df.insert(
            2, "date", pd.to_datetime(date, format="%y%m%d", errors="coerce")
        )
        schedule_time_cols = [
            "scheduled_arrival",
            "scheduled_departure",
            "estimated_arrival",
            "estimated_departure",
        ]
        for col in schedule_time_cols:
            schedules_df[col] = pd.to_datetime(
                schedules_df["date"].astype(str) + " " + schedules_df[col],
                format="%Y-%m-%d %H:%M",
                errors="coerce",
            )

        # Calculate time and distance differences
        for col in schedule_time_cols:
            diff_col = f"{col}_diff"
            schedules_df[diff_col] = schedules_df.groupby("elvira_id")[col].diff()
        schedules_df["distance_km"] = schedules_df["distance_km"].astype(float)
        schedules_df["distance_diff"] = schedules_df.groupby("elvira_id")[
            "distance_km"
        ].diff()

        schedules_df["delay_minutes"] = (
            schedules_df["estimated_arrival"] - schedules_df["scheduled_arrival"]
        ).dt.seconds / 60
        return schedules_df

    def clean_station_names(
        self, schedules_df: pd.DataFrame, stops_to_skip: List[str] = None
    ) -> pd.DataFrame:
        schedules_df["stop_name"] = (
            schedules_df["stop_name"].str.split("/").str[0].str.strip()
        )
        schedules_df["stop_name"] = schedules_df["stop_name"].apply(
            StationNameCleaner.clean_station_name_strings
        )

        merged_schedules = schedules_df.merge(
            self.stops_df, on="stop_name", how="left", indicator=True
        )
        not_found_stops = (
            merged_schedules.loc[merged_schedules["_merge"] == "left_only"]["stop_name"]
            .unique()
            .tolist()
        )

        mapping_dict, not_found_dict = StationNameCleaner.fuzzy_match_stations(
            not_found_stops, self.stops_df, stops_to_skip
        )
        schedules_df["stop_name"] = schedules_df["stop_name"].replace(mapping_dict)

        schedules_df = schedules_df[
            ~schedules_df["stop_name"].isin(stops_to_skip or [])
        ]
        mapping_dict_display_string = "\n".join(
            [key + " -> " + value for key, value in mapping_dict.items()]
        )
        not_found_dict_display_string = "\n".join(
            [key + " -> " + value for key, value in not_found_dict.items()]
        )
        print("The following stations will be renamed based on fuzzy search:")
        print(mapping_dict_display_string)
        print()
        print(
            "The following stations were not found in the stops_df. They are not mapped"
        )
        print(not_found_dict_display_string)
        print()
        print("The following stations are manually skipped from the fuzzy mapping:")
        print("\n".join(stops_to_skip))
        return schedules_df

    def calculate_shortest_paths(self, schedules_df: pd.DataFrame) -> pd.DataFrame:
        schedules_df = schedules_df.merge(self.vertex_df, on="stop_name", how="left")
        schedules_df["next_vertex_id"] = schedules_df.groupby("elvira_id")[
            "vertex_id"
        ].shift(-1)
        schedules_df["prev_vertex_id"] = schedules_df.groupby("elvira_id")[
            "vertex_id"
        ].shift(1)
        schedules_df["consecutive_vertex_ids"] = (
            schedules_df["vertex_id"].astype(str)
            + "->"
            + schedules_df["next_vertex_id"].astype(str)
        )
        schedules_df["shortest_path"] = schedules_df[
            ["vertex_id", "next_vertex_id"]
        ].apply(
            lambda row: self.graph.get_shortest_path(
                int(row["vertex_id"]), int(row["next_vertex_id"])
            )
            if pd.notna(row["next_vertex_id"])
            else [row["vertex_id"]],
            axis=1,
        )
        schedules_df = schedules_df[
            schedules_df["shortest_path"].apply(lambda x: len(x) != 0)
        ]
        schedules_df["shortest_path_stops"] = schedules_df["shortest_path"].apply(
            lambda x: len(x) - 1
        )
        return schedules_df

    def interpolate_schedules(self, schedules_df: pd.DataFrame) -> pd.DataFrame:
        exploded_df = (
            schedules_df[schedules_df["shortest_path_stops"] > 1]
            .explode("shortest_path")
            .set_index("shortest_path")
        )
        exploded_df.update(self.vertex_df.set_index("vertex_id"))
        exploded_df.drop(columns=["vertex_id", "next_vertex_id"], inplace=True)
        exploded_df.index.name = "vertex_id"
        exploded_df.reset_index(inplace=True)
        exploded_df["next_vertex_id"] = exploded_df.groupby(
            ["elvira_id", "consecutive_vertex_ids"]
        )["vertex_id"].shift(-1)

        exploded_df["stop_count"] = (
            exploded_df.groupby(["elvira_id", "consecutive_vertex_ids"])[
                "vertex_id"
            ].cumcount()
            + 1
        )

        exploded_df[["next_stop_lat", "next_stop_lon"]] = exploded_df.groupby(
            ["elvira_id", "consecutive_vertex_ids"]
        )[["stop_lat", "stop_lon"]].shift(1)

        exploded_df["distance_diff"] = exploded_df.apply(
            lambda row: geodesic(
                (row["stop_lat"], row["stop_lon"]),
                (row["next_stop_lat"], row["next_stop_lon"]),
            ).kilometers
            if pd.notna(row["next_stop_lat"]) and pd.notna(row["next_stop_lon"])
            else 0,
            axis=1,
        )

        exploded_df["distance_km"] = (
            exploded_df.groupby(["elvira_id", "consecutive_vertex_ids"])[
                "distance_diff"
            ].cumsum()
            + exploded_df["distance_km"]
        )
        exploded_df["total_distance"] = exploded_df.groupby(
            ["elvira_id", "consecutive_vertex_ids"]
        )["distance_diff"].transform("sum")

        schedule_time_cols = [
            "scheduled_arrival",
            "scheduled_departure",
            "estimated_arrival",
            "estimated_departure",
        ]
        for time_col in schedule_time_cols:
            exploded_df[f"{time_col}_diff"] = (
                exploded_df["distance_diff"] / exploded_df["total_distance"]
            ) * exploded_df[f"{time_col}_diff"]
            exploded_df[time_col] = (
                exploded_df.groupby(["elvira_id", "consecutive_vertex_ids"])[
                    f"{time_col}_diff"
                ].cumsum()
                + exploded_df[time_col]
            )
        exploded_df["delay_minutes"] = (
            (
                exploded_df["estimated_arrival"] - exploded_df["scheduled_arrival"]
            ).dt.seconds
            / 60
        ).round(0)

        exploded_df.drop(
            columns=["stop_count", "next_stop_lat", "next_stop_lon", "total_distance"],
            inplace=True,
        )

        schedules_df = schedules_df.merge(
            exploded_df,
            on=["elvira_id", "consecutive_vertex_ids"],
            how="left",
            suffixes=("", "_interpolated"),
        )

        for col in schedules_df.columns:
            if col.endswith("_interpolated"):
                original_col = col.replace("_interpolated", "")
                schedules_df[original_col] = np.where(
                    schedules_df[col].notna(),
                    schedules_df[col],
                    schedules_df[original_col],
                )

        schedules_df.drop(
            columns=[
                col for col in schedules_df.columns if col.endswith("_interpolated")
            ],
            inplace=True,
        )

        schedules_df["consecutive_vertex_ids"] = (
            schedules_df["prev_vertex_id"].astype(str)
            + "->"
            + schedules_df["vertex_id"].astype(str)
        )

        return schedules_df

    def run_pipeline(
        self, filepath: str, stops_to_skip: List[str] = None
    ) -> pd.DataFrame:
        schedules_df = self.load_schedules(filepath)
        schedules_df = self.preprocess_schedules(schedules_df)
        schedules_df = self.clean_station_names(schedules_df, stops_to_skip)
        schedules_df = self.calculate_shortest_paths(schedules_df)
        schedules_df = self.interpolate_schedules(schedules_df)
        return schedules_df


def align_schedule_to_graph():
    with open(
        dir_config.full_transit_graphs_dir / "gtfs_graph_undirected.pkl", "rb"
    ) as f:
        g = pickle.load(f)

    g = g.as_directed(mode="mutual")
    vertex_df = g.get_vertex_dataframe().reset_index(names="vertex_id")
    stops_df = pd.read_csv(dir_config.mav_gtfs_dir / "stops.txt")

    processor = ScheduleProcessor(g, vertex_df, stops_df)
    schedules_df = processor.run_pipeline(
        dir_config.mav_scrape_dir / "parsed_train_schedules.parquet",
        stops_to_skip=["Kondoroskert", "Látókép"],
    )
    schedules_df.to_parquet(
        dir_config.working_dir / "interpolated_train_schedules.parquet", index=False
    )


# %%
# Execution
if __name__ == "__main__":
    align_schedule_to_graph()

# %%
