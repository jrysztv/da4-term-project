# from loguru import logger


# ---------------------------------------------
# Interpolation Helper Class
# ---------------------------------------------
import numpy as np
import pandas as pd
from scipy.spatial import geometric_slerp
from tqdm import tqdm


class ShapeInterpolator:
    """
    A class responsible for all operations related to shape interpolation:
    - Calculating distances between consecutive points in each shape group.
    - Determining how many intermediate points to insert.
    - Performing spherical (great-circle) interpolation using geometric_slerp.
    """

    def __init__(
        self,
        route_df,
        group_id="elvira_id",
        distance_limit=500,
        num_intermediate_points_column=None,
    ):
        """
        :param route_df:      DataFrame containing at least [stop_lat, stop_lon] columns.
        :param group_id:      The column name by which to group shapes (e.g. 'line_id', 'elvira_id', etc.).
        :param distance_limit: The max distance (meters) per segment before inserting intermediate points.
        :param num_intermediate_points_column: Optional column name in route_df specifying the number of intermediate points.
                                               Overrides infering number of intemediate points from distance_limit.
        """
        self.route_df = route_df.copy()
        self.group_id = group_id
        self.distance_limit = distance_limit
        self.num_intermediate_points_column = num_intermediate_points_column

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Return distance in meters between two lat/lon arrays using the Haversine formula.
        """
        R = 6371000  # Earth radius in meters
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    @staticmethod
    def latlon_to_xyz(lat_deg, lon_deg):
        """
        Convert lat, lon (in degrees) on Earth to 3D coordinates (x, y, z) on the unit sphere.
        """
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.array([x, y, z])

    @staticmethod
    def xyz_to_latlon(x, y, z):
        """
        Convert unit-sphere (x, y, z) to latitude, longitude in degrees (rounded to 5 decimals).
        """
        norm = np.sqrt(x * x + y * y + z * z)
        x, y, z = x / norm, y / norm, z / norm
        lat_rad = np.arcsin(z)
        lon_rad = np.arctan2(y, x)
        lat_deg = round(np.degrees(lat_rad), 5)
        lon_deg = round(np.degrees(lon_rad), 5)
        return lat_deg, lon_deg

    def _get_next_lat_lon(self, df, lat_col="stop_lat", lon_col="stop_lon"):
        """
        Return lat/lon arrays shifted by one row.
        """
        return df[lat_col].shift(-1), df[lon_col].shift(-1)

    def _calc_distance_by_group(self, df):
        """
        Return distance in meters between consecutive points for each shape group.
        """

        def group_haversine(group):
            return self.haversine_distance(
                group["stop_lat"],
                group["stop_lon"],
                group["stop_lat_next"],
                group["stop_lon_next"],
            )

        distances = df.groupby(self.group_id, group_keys=False).apply(
            group_haversine, include_groups=False
        )
        return distances

    def interpolate_shapes(self):
        """
        Main pipeline to:
        1. Add next lat/lon columns.
        2. Calculate distance between consecutive shape points.
        3. Determine number of intermediate points per segment.
        4. Generate those intermediate points (using geometric_slerp).
        5. Merge back into the original DataFrame.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with additional lat/lon rows for interpolated segments.
        """
        df = self.route_df.copy()

        # 1. Add columns for next lat/lon
        df[["stop_lat_next", "stop_lon_next"]] = df.groupby(self.group_id)[
            ["stop_lat", "stop_lon"]
        ].shift(-1)

        if (
            self.num_intermediate_points_column
        ):  # workaround. TODO: Needs to be checked out why is this the case.
            df.dropna(subset=["stop_lat_next", "stop_lon_next"], inplace=True)

        # 2. Calculate distance between consecutive shape points
        df["distance"] = self._calc_distance_by_group(df)

        # 3. Determine how many intermediate points to add
        if self.num_intermediate_points_column:
            df["num_intermediate_points"] = (
                df[self.num_intermediate_points_column].fillna(0).astype(int)
            )
        else:
            df["num_intermediate_points"] = (
                (df["distance"] // self.distance_limit).fillna(0).astype(int)
            )

        # 4. Perform spherical interpolation where needed
        #    Only expand rows with num_intermediate_points > 0
        rows_to_expand = df[df["num_intermediate_points"] > 0].copy()

        # Safety check: sort by observation_sequence_id to ensure interpolation order
        rows_to_expand.sort_values(by=["observation_sequence_id"], inplace=True)

        interpolated_rows = []
        for idx, row in tqdm(
            rows_to_expand.iterrows(),
            desc="Interpolating points...",
            total=len(rows_to_expand),
        ):
            lat1, lon1 = row["stop_lat"], row["stop_lon"]
            lat2, lon2 = row["stop_lat_next"], row["stop_lon_next"]
            n_points = row["num_intermediate_points"] + 2  # +2 for start & end

            start_xyz = self.latlon_to_xyz(lat1, lon1)
            end_xyz = self.latlon_to_xyz(lat2, lon2)

            t_vals = np.linspace(0, 1, n_points)
            xyz_points = geometric_slerp(start_xyz, end_xyz, t_vals)

            # Convert each 3D point back to lat/lon
            latlon_list = []
            for x, y, z in xyz_points:
                lat_deg, lon_deg = self.xyz_to_latlon(x, y, z)
                latlon_list.append((lat_deg, lon_deg))

            # Save interpolation results
            interpolated_rows.append(
                {
                    self.group_id: row[self.group_id],
                    "observation_sequence_id": row[
                        "observation_sequence_id"
                    ],  # or however you track it
                    "interpolated_point": latlon_list,
                }
            )

        # 5. Explode the results and merge
        if not interpolated_rows:
            # No interpolation needed; just remove helper columns and return
            df.drop(
                columns=[
                    "stop_lat_next",
                    "stop_lon_next",
                    "distance",
                    "num_intermediate_points",
                ],
                inplace=True,
                errors="ignore",
            )
            return df

        interpolated_points_df = pd.DataFrame(interpolated_rows).explode(
            "interpolated_point"
        )
        interpolated_points_df[["stop_lat", "stop_lon"]] = pd.DataFrame(
            interpolated_points_df["interpolated_point"].tolist(),
            index=interpolated_points_df.index,
        )

        # Merge back on [group_id, sequence_id]
        # (Make sure 'sequence_id' exists in your df; see below example that inserts it)
        merged_df = df.merge(
            interpolated_points_df,
            on=["observation_sequence_id"],
            how="left",
            suffixes=("", "_interp"),
        )

        # If an interpolated value exists, use it
        merged_df["stop_lat"] = np.where(
            merged_df["stop_lat_interp"].notnull(),
            merged_df["stop_lat_interp"],
            merged_df["stop_lat"],
        )
        merged_df["stop_lon"] = np.where(
            merged_df["stop_lon_interp"].notnull(),
            merged_df["stop_lon_interp"],
            merged_df["stop_lon"],
        )

        # Drop helper columns
        merged_df.drop(
            columns=[
                "stop_lat_next",
                "stop_lon_next",
                "distance",
                "num_intermediate_points",
                "stop_lat_interp",
                "stop_lon_interp",
                "interpolated_point",
            ],
            inplace=True,
            errors="ignore",
        )

        return merged_df
