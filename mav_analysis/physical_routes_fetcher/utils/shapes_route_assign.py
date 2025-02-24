# %%
import pandas as pd
import numpy as np


class RouteStopAnalyzer:
    def __init__(
        self, route_df, stops_df, lat_col="lat", lon_col="lon", stop_id_col="id"
    ):
        self.route_df = route_df
        self.stops_df = stops_df
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.stop_id_col = stop_id_col

    @staticmethod
    def haversine_distance_matrix(route, stops):
        """
        Compute the Haversine distance in meters between each route point and each stop.

        route: NumPy array of shape (N, 2) -> [[lat0, lon0], [lat1, lon1], ..., [latN, lonN]]
        stops: NumPy array of shape (M, 2) -> [[lat0, lon0], [lat1, lon1], ..., [latM, lonM]]

        Returns: dist, a NumPy array of shape (N, M),
                 where dist[i, j] is the distance (meters)
                 between route[i] and stops[j].
        """
        # Earth’s radius (mean) in meters
        R = 6371000.0

        # Convert degrees to radians
        route_rad = np.radians(route)  # shape (N, 2)
        stops_rad = np.radians(stops)  # shape (M, 2)

        # Split into lat/lon components
        route_lat = route_rad[:, 0].reshape(-1, 1)  # shape (N,1)
        route_lon = route_rad[:, 1].reshape(-1, 1)
        stops_lat = stops_rad[:, 0].reshape(1, -1)  # shape (1,M)
        stops_lon = stops_rad[:, 1].reshape(1, -1)

        # Haversine formula
        dlat = stops_lat - route_lat  # shape (N, M)
        dlon = stops_lon - route_lon  # shape (N, M)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(route_lat) * np.cos(stops_lat) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        dist = R * c  # shape (N, M)
        return dist

    def build_bounding_box(self, group_df, buffer_degrees):
        min_lat = group_df[self.lat_col].min() - buffer_degrees
        max_lat = group_df[self.lat_col].max() + buffer_degrees
        min_lon = group_df[self.lon_col].min() - buffer_degrees
        max_lon = group_df[self.lon_col].max() + buffer_degrees
        return min_lat, max_lat, min_lon, max_lon

    def filter_stops_by_bounding_box(self, min_lat, max_lat, min_lon, max_lon):
        bbox_mask = (
            (self.stops_df[self.lat_col] >= min_lat)
            & (self.stops_df[self.lat_col] <= max_lat)
            & (self.stops_df[self.lon_col] >= min_lon)
            & (self.stops_df[self.lon_col] <= max_lon)
        )
        return self.stops_df.loc[bbox_mask].copy()

    def compute_distance_matrix(self, group_df, stops_df_filtered):
        route_array = group_df[[self.lat_col, self.lon_col]].to_numpy()  # shape (N, 2)
        stops_array = stops_df_filtered[
            [self.lat_col, self.lon_col]
        ].to_numpy()  # shape (M, 2)
        return self.haversine_distance_matrix(route_array, stops_array)

    def find_encountered_stops(
        self, dist_matrix, stops_df_filtered, distance_threshold_meters
    ):
        encountered_stop_ids = []
        encountered = set()
        stop_index_to_id = stops_df_filtered[self.stop_id_col].to_numpy()  # shape (M,)
        N = dist_matrix.shape[0]
        for i in range(N):
            close_mask = dist_matrix[i, :] <= distance_threshold_meters
            close_stops_indices = np.where(close_mask)[0]
            for stop_idx in close_stops_indices:
                stop_id = stop_index_to_id[stop_idx]
                if stop_id not in encountered:
                    encountered.add(stop_id)
                    encountered_stop_ids.append(stop_id)
        return encountered_stop_ids

    def stop_sequence_by_shape(
        self, group_df, distance_threshold_meters=50.0, buffer_degrees=0.005
    ):
        """
        Vectorized approach to:
        1) Build bounding box of the route (+ optional buffer).
        2) Filter stops to that bounding box.
        3) Compute the full NxM distance matrix.
        4) "Walk" the route in index order, picking up stops
           that haven't been encountered and are within threshold.

        group_df: Pandas DataFrame with columns ['lat', 'lon'] for each route point in order.
        distance_threshold_meters: float, distance threshold.
        buffer_degrees: float, bounding box buffer in degrees.

        Returns: A list of stop IDs in the order they are first encountered.
        """
        # safety sort
        # group_df = group_df.sort_values(["observation_sequence_id"])
        min_lat, max_lat, min_lon, max_lon = self.build_bounding_box(
            group_df, buffer_degrees
        )
        stops_df_filtered = self.filter_stops_by_bounding_box(
            min_lat, max_lat, min_lon, max_lon
        )
        if stops_df_filtered.empty:
            return []
        dist_matrix = self.compute_distance_matrix(group_df, stops_df_filtered)
        return self.find_encountered_stops(
            dist_matrix, stops_df_filtered, distance_threshold_meters
        )

    def stop_sequence_by_shape_grouped(
        self, group_id: str, distance_threshold_meters=50.0, buffer_degrees=0.005
    ):
        """
        Apply vectorized_filter_stops_by_shape for each group_id in route_df.

        group_id: str, column name for group identifier in route_df.
        distance_threshold_meters: float, distance threshold.
        buffer_degrees: float, bounding box buffer in degrees.

        Returns: A DataFrame with columns ['group_id', 'stop_id'].
        """
        grouped_stops = []
        for gid, group_df in self.route_df.groupby(group_id):
            encountered_stops = self.stop_sequence_by_shape(
                group_df,
                distance_threshold_meters=distance_threshold_meters,
                buffer_degrees=buffer_degrees,
            )
            for stop_id in encountered_stops:
                grouped_stops.append({group_id: gid, "stop_id": stop_id})
        return pd.DataFrame(grouped_stops)
