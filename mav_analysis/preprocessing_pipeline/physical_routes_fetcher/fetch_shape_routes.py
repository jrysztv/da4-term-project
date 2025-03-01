# %%
import pandas as pd
from mav_analysis.preprocessing_pipeline.physical_routes_fetcher.utils.shape_interpolator import (
    ShapeInterpolator,
)
from mav_analysis.preprocessing_pipeline.physical_routes_fetcher.utils.shapes_route_assign import (
    RouteStopAnalyzer,
)
from mav_analysis.preprocessing_pipeline.physical_routes_fetcher.utils.shapes_block_request import (
    ShapesRequest,
)
from mav_analysis.config import dir_config


# ---------------------------------------------
# Main Class: RouteStopGatherer
# ---------------------------------------------
class RouteStopGatherer:
    def __init__(
        self,
        by_train_collection=True,
        train_collection_path=dir_config.mav_scrape_dir / "parsed_train_shapes.parquet",
        stops_df_path=dir_config.mav_gtfs_dir / "stops.txt",
    ):
        self.by_train_collection = by_train_collection
        self.train_collection_path = train_collection_path
        self.stops_df = pd.read_csv(stops_df_path)
        self.route_df = None
        self.load_route_df()

    @staticmethod
    def fetch_newest_parsed_train_shapes():
        # the filename format is f"{filename}_{datetime_now}.parquet"
        return sorted(dir_config.mav_scrape_dir.glob("parsed_train_shapes_*.parquet"))[
            -1
        ]

    def load_route_df(self, by_train_collection=None, train_collection_path=None):
        if by_train_collection is not None:
            self.by_train_collection = by_train_collection
        if train_collection_path is not None:
            self.train_collection_path = train_collection_path

        if self.by_train_collection:
            route_df = pd.read_parquet(self.train_collection_path)
            route_df = route_df.rename(columns={"lat": "stop_lat", "lon": "stop_lon"})
        else:
            shapes_requester = ShapesRequest()
            route_df = shapes_requester.request_and_extract().rename(
                columns={"lat": "stop_lat", "lon": "stop_lon"}
            )
        route_df["observation_sequence_id"] = range(len(route_df))

        self.route_df = route_df
        return route_df

    @staticmethod
    def drop_duplicated_shapes(route_df, group_id="elvira_id"):
        """
        Example de-duplication method.
        It groups by `group_id`, collects lat/lon in a list, and removes duplicates.
        """
        route_df_copy = route_df.copy()
        route_df_copy["stop_lat_lon"] = list(
            zip(route_df_copy["stop_lat"], route_df_copy["stop_lon"])
        )
        grouped_route_df = route_df_copy.groupby(group_id)["stop_lat_lon"].apply(list)
        unique_sequences = grouped_route_df.drop_duplicates()

        route_df_copy = unique_sequences.reset_index()[[group_id]].merge(
            route_df_copy.drop(columns="stop_lat_lon"), on=group_id, how="left"
        )
        route_df_copy.sort_values(by=["observation_sequence_id"], inplace=True)
        return route_df_copy

    def gather_stops_for_lines(
        self,
        distance_threshold_meters=500,
        group_id="elvira_id",
        interpolate_points=True,
        interpolation_distance_limit=200,  # used if interpolate_points=True
        prefilter_duplicates=True,
    ):
        """
        Main pipeline entry point.
        :param distance_threshold_meters:       Buffer distance to determine if a stop is encountered.
        :param group_id:                        The column to group shapes by.
        :param interpolate_points:              If True, run the shape interpolation pipeline.
        :param interpolation_distance_limit:    The threshold for inserting intermediate points (meters).
        :param prefilter_duplicates:            If True, deduplicate shapes before processing.
        """
        stops_df = self.stops_df.copy()

        # Optionally remove duplicates before continuing
        if prefilter_duplicates:
            self.route_df = self.drop_duplicated_shapes(
                self.route_df, group_id=group_id
            )

        # ------------------------------------------------------
        # OPTIONAL LOGIC: Preprocessing with interpolation here
        # ------------------------------------------------------
        if interpolate_points:
            # Insert a sequence column if your route_df doesn't have one
            # so that we can keep track of the row order within each group.
            self.route_df = self.route_df.copy()

            shape_interpolator = ShapeInterpolator(
                self.route_df,
                group_id=group_id,
                distance_limit=interpolation_distance_limit,
            )
            self.route_df = shape_interpolator.interpolate_shapes()

        # Now run the existing logic to get encountered stops
        analyzer = RouteStopAnalyzer(
            self.route_df,
            stops_df,
            lat_col="stop_lat",
            lon_col="stop_lon",
            stop_id_col="stop_id",
        )

        grouped_encountered_stops_df = analyzer.stop_sequence_by_shape_grouped(
            group_id=group_id,
            distance_threshold_meters=distance_threshold_meters,
            buffer_degrees=0.005,
        )

        # for debug
        self.group_encountered_stops_df = grouped_encountered_stops_df
        return grouped_encountered_stops_df.merge(stops_df.dropna(axis=1), on="stop_id")


# %%
if __name__ == "__main__":
    # Example usage
    gatherer = RouteStopGatherer(by_train_collection=True)

    # 1) Without interpolation
    stop_lines_block_df = gatherer.gather_stops_for_lines(
        distance_threshold_meters=1000, group_id="elvira_id", interpolate_points=False
    )

    # 2) With interpolation
    stop_lines_block_df_interpolated = gatherer.gather_stops_for_lines(
        distance_threshold_meters=500,
        group_id="elvira_id",
        interpolate_points=True,  # <--- interpolation of points
        interpolation_distance_limit=200,
    )
