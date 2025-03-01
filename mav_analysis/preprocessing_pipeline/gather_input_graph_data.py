# %%
from mav_analysis.preprocessing_pipeline.physical_routes_fetcher.fetch_shape_routes import (
    RouteStopGatherer,
)
import pandas as pd

# from pathlib import Path
import datetime
from mav_analysis.config import dir_config
import pytz


# %%
class SaveGatheredStops:
    def __init__(self):
        self.route_stop_gatherer = RouteStopGatherer(
            by_train_collection=True
        )  # directly fetched shapes are useless
        self.stops_df = pd.read_csv(dir_config.mav_gtfs_dir / "stops.txt")
        self.stop_times_df = pd.read_csv(dir_config.mav_gtfs_dir / "stop_times.txt")
        self.stop_lines_df = None

    def gather_stops(
        self, distance_threshold_meters=500, distance_limit=100, interpolate_points=True
    ):
        self.stop_lines_df = self.route_stop_gatherer.gather_stops_for_lines(
            distance_threshold_meters=distance_threshold_meters,
            interpolation_distance_limit=distance_limit,
            interpolate_points=interpolate_points,
        )


def gather_and_save_graph_building_inputs(
    distance_threshold_meters=500,
    interpolation_distance_limit=100,
    graph_input_dir=dir_config.input_graph_data_dir,
):
    gatherer = RouteStopGatherer(by_train_collection=True)
    datetime_now = datetime.datetime.now(pytz.utc)
    # gather stops for lines
    stop_lines_df = gatherer.gather_stops_for_lines(
        distance_threshold_meters=distance_threshold_meters,
        interpolation_distance_limit=interpolation_distance_limit,
        interpolate_points=True,
    )
    stop_lines_df_uninterpolated = gatherer.gather_stops_for_lines(
        distance_threshold_meters=distance_threshold_meters,
        interpolation_distance_limit=interpolation_distance_limit,
        interpolate_points=False,
    )

    # assign current datetime to the dataframes
    stop_lines_df["creation_date"] = datetime_now
    stop_lines_df_uninterpolated["creation_date"] = datetime_now

    # save gathered stops
    stop_lines_df.to_parquet(graph_input_dir / "stop_lines_df.parquet")
    stop_lines_df_uninterpolated.to_parquet(
        graph_input_dir / "stop_lines_uninterpolate_df.parquet"
    )


# %%
if __name__ == "__main__":
    gather_and_save_graph_building_inputs()
# %%
