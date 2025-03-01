###############################################################################
# 1) Imports
###############################################################################
import sys
import numpy as np
import pandas as pd

# from pathlib import Path
from meteostat import Daily, Point
from tqdm.auto import tqdm

# Import your custom configuration that provides directory paths
from mav_analysis.config import dir_config
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


###############################################################################
# 2) Documentation Class for Reference Only
###############################################################################
class MeteostatDailyDoc:
    """
    Meteostat Daily DataFrame Column Reference
    ------------------------------------------
    When using `meteostat.Daily(...).fetch()`, Meteostat returns a DataFrame
    with columns (some may be missing if data isn't available):

    - station (String)  : The Meteostat station ID (only if multiple stations are queried)
    - time (Datetime64) : The date
    - tavg (Float64)    : The average air temperature in °C
    - tmin (Float64)    : The minimum air temperature in °C
    - tmax (Float64)    : The maximum air temperature in °C
    - prcp (Float64)    : The daily precipitation total in mm
    - snow (Float64)    : The snow depth in mm
    - wdir (Float64)    : The average wind direction in degrees (°)
    - wspd (Float64)    : The average wind speed in km/h
    - wpgt (Float64)    : The peak wind gust in km/h
    - pres (Float64)    : The average sea-level air pressure in hPa
    - tsun (Float64)    : The daily sunshine total in minutes (m)

    Missing values can be common for certain locations or time periods.
    """


###############################################################################
# 3) WeatherEnricher Class (same as your original, for optional usage)
###############################################################################
class WeatherEnricher:
    """
    Class to fetch Meteostat daily data for each unique lat-lon in the dataset
    over the required date range, then merge that weather onto the main DataFrame.

    This version also supports caching the fetched results in a file
    under dir_config.working.

    Parameters
    ----------
    weather_vars : list of str
        By default, the standard set from Meteostat:
        [tmax, tmin, tavg, prcp, snow, wdir, wspd, wpgt, pres, tsun].
    overwrite : bool
        If True (default), fresh scraping is done and results are saved to the cache file.
        If False, and the cache file already exists, it will skip scraping and load from file.
    """

    def __init__(self, weather_vars=None, overwrite=True):
        # Default Meteostat columns if none provided
        if weather_vars is None:
            self.weather_vars = [
                "tmax",
                "tmin",
                "tavg",
                "prcp",
                "snow",
                "wdir",
                "wspd",
                "wpgt",
                "pres",
                "tsun",
            ]
        else:
            self.weather_vars = weather_vars

        # Overwrite indicates whether to fetch fresh data or load from cache
        self.overwrite = overwrite

        # Path for caching the scraped weather data
        self.scrape_weather_path = (
            dir_config.working_dir / "scraped_weather_data.parquet"
        )

    def fetch_weather_for_latlon(self, lat, lon, start_date, end_date):
        """
        Use meteostat.Point(lat, lon) + Daily to fetch daily data.
        Returns a DataFrame with columns like:
         [time, tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun]
        plus [stop_lat, stop_lon].
        """
        loc = Point(lat, lon)
        try:
            df_weather = Daily(loc, start=start_date, end=end_date).fetch()
            if df_weather.empty:
                return pd.DataFrame()
            # Reset index so 'time' becomes a visible column
            df_weather = df_weather.reset_index()
            # Attach lat/lon to identify which location each row refers to
            df_weather["stop_lat"] = lat
            df_weather["stop_lon"] = lon
            return df_weather
        except Exception as e:
            print(f"Failed to fetch weather for lat={lat}, lon={lon}. Error: {e}")
            return pd.DataFrame()

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method: fetch data for all unique lat-lon pairs over the min..max date in df,
        then merge results on [stop_lat, stop_lon, date].

        If overwrite=False and self.scrape_weather_path exists, we'll load from the cache
        instead of scraping again.

        Returns
        -------
        pd.DataFrame (with new weather columns added)
        """
        # Make sure 'date' is in datetime format
        df["date"] = pd.to_datetime(df["date"])

        # Identify date range we need
        start_date = df["date"].min()
        end_date = df["date"].max()
        print(f"[WeatherEnricher] Date range: {start_date.date()} -> {end_date.date()}")

        # If file exists and overwrite=False, skip fresh scraping
        if (not self.overwrite) and self.scrape_weather_path.exists():
            print(
                f"[WeatherEnricher] Cached weather data file found at "
                f"{self.scrape_weather_path}. Using it."
            )
            df_weather = pd.read_parquet(self.scrape_weather_path)
        else:
            # Collect unique lat-lon pairs from the DataFrame
            unique_coords = df[["stop_lat", "stop_lon"]].drop_duplicates()
            print(f"[WeatherEnricher] Unique lat/lon pairs: {len(unique_coords)}")

            # Accumulate all fetched DataFrames
            all_weather = []
            for _, row_coords in tqdm(
                unique_coords.iterrows(),
                total=len(unique_coords),
                desc="Fetching Meteostat",
            ):
                lat, lon = row_coords["stop_lat"], row_coords["stop_lon"]
                wdf = self.fetch_weather_for_latlon(lat, lon, start_date, end_date)
                if not wdf.empty:
                    all_weather.append(wdf)

            if not all_weather:
                print(
                    "[WeatherEnricher] No weather data fetched. Returning original df."
                )
                return df

            # Combine all weather data
            df_weather = pd.concat(all_weather, ignore_index=True)

            # Save to a Parquet file for future runs
            df_weather.to_parquet(self.scrape_weather_path)
            print(
                f"[WeatherEnricher] Cached weather data saved at {self.scrape_weather_path}"
            )

        # Rename 'time' -> 'date' to merge on the same column
        df_weather = df_weather.rename(columns={"time": "date"})

        # Merge (left join) the weather onto your original DataFrame
        df_merged = df.merge(
            df_weather, on=["stop_lat", "stop_lon", "date"], how="left"
        )

        return df_merged


###############################################################################
# 4) Vectorized-Distance Weather Imputer With Flags
###############################################################################
class WeatherImputerVectorizedWithFlags:
    """
    Imputes missing weather data purely from the nearest stop's non-missing values,
    and adds:
      - <var>_was_imputed (bool)
      - <var>_impute_source_stop (which stop provided the data, or "ORIGINAL")
      - <var>_impute_source_dist (distance in km, or 0 if original, NaN if none found)

    Steps:
      1. Build a vectorized distance matrix (index=stop_id, columns=stop_id).
      2. For each weather variable:
         - Pivot df -> wide [date x stop_id].
         - For each origin stop, reindex columns by ascending distance, forward-fill
           across columns => the rightmost column is the first non-null value from left to right.
         - Identify the *source stop* that actually provided data (or None if all were missing).
         - Merge these results back; fill only where original was NaN.
         - Add flags:
           * <var>_was_imputed=True if we replaced an NaN with a value
           * <var>_impute_source_stop=<some_stop_id> or "ORIGINAL" if no change
           * <var>_impute_source_dist=distance or 0 if original
      3. If all stops are missing for a given date, the final remains NaN, <var>_was_imputed=False, etc.
    """

    def __init__(self, weather_vars=None):
        """
        Parameters
        ----------
        weather_vars: list of str
            The columns/variables to impute, e.g. ["tmax", "tmin", "tavg", "prcp", "snow"].
        """
        if weather_vars is None:
            self.weather_vars = [
                "tavg",
                "tmin",
                "tmax",
                "prcp",
                "snow",
                "wdir",
                "wspd",
                "wpgt",
                "pres",
                "tsun",
            ]
        else:
            self.weather_vars = weather_vars

    @staticmethod
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        """
        Vectorized haversine distance in km.

        lat1, lon1, lat2, lon2 can be arrays of shape (N,) or broadcastable.
        Returns an array of distances (same shape).
        """
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2.0 * np.arcsin(np.sqrt(a))
        return R * c

    def build_distance_matrix(
        self,
        df: pd.DataFrame,
        stop_id_col="stop_id",
        lat_col="stop_lat",
        lon_col="stop_lon",
    ) -> pd.DataFrame:
        """
        Create a distance matrix DataFrame with index=stop_id, columns=stop_id,
        values = haversine distance (km).
        """
        # Drop duplicates so each stop_id is unique
        stops_unique = df.drop_duplicates(subset=[stop_id_col]).copy()
        stops_unique = stops_unique[[stop_id_col, lat_col, lon_col]].reset_index(
            drop=True
        )

        lat_arr = stops_unique[lat_col].values
        lon_arr = stops_unique[lon_col].values
        n = len(stops_unique)

        # Broadcast lat/lon over an (n,n) grid
        lat_grid_1 = np.repeat(lat_arr.reshape(-1, 1), n, axis=1)  # shape (n,n)
        lon_grid_1 = np.repeat(lon_arr.reshape(-1, 1), n, axis=1)
        lat_grid_2 = lat_arr.reshape(1, -1)  # shape (1,n)
        lon_grid_2 = lon_arr.reshape(1, -1)

        dist_matrix = self.haversine_vectorized(
            lat_grid_1, lon_grid_1, lat_grid_2, lon_grid_2
        )

        dist_df = pd.DataFrame(
            dist_matrix,
            index=stops_unique[stop_id_col],
            columns=stops_unique[stop_id_col],
        )
        return dist_df

    def impute_nearest_stop(
        self,
        df: pd.DataFrame,
        dist_df: pd.DataFrame,
        stop_id_col="stop_id",
        date_col="date",
    ) -> pd.DataFrame:
        """
        For each weather var, pivot wide [date x stop_id], reorder columns by ascending
        distance for each 'origin' stop, forward-fill across columns, and merge results
        back. Also track flags for source/distance/imputed.

        Returns a copy of df with additional columns:
          <var>_was_imputed, <var>_impute_source_stop, <var>_impute_source_dist
        """

        df = df.copy()
        df["_orig_index"] = np.arange(len(df))  # to restore ordering after merges

        # Ensure date is datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # For each variable we want to fill
        for var in self.weather_vars:
            # Prepare columns for flags
            was_imputed_col = f"{var}_was_imputed"
            source_stop_col = f"{var}_impute_source_stop"
            source_dist_col = f"{var}_impute_source_dist"

            # Initialize them
            df[was_imputed_col] = False
            df[source_stop_col] = "ORIGINAL"  # if not imputed, remains "ORIGINAL"
            df[source_dist_col] = 0.0  # if not imputed, distance = 0

            # Pivot to wide
            wide = df.pivot(index=date_col, columns=stop_id_col, values=var)

            # We'll store the final values in 'filled_wide' (same shape), plus
            # 'source_stop_wide' and 'source_dist_wide' to track source stops/distances.
            filled_wide = pd.DataFrame(
                index=wide.index, columns=wide.columns, dtype=float
            )
            source_stop_wide = pd.DataFrame(
                index=wide.index, columns=wide.columns, dtype=object
            )
            source_dist_wide = pd.DataFrame(
                index=wide.index, columns=wide.columns, dtype=float
            )

            # For each 'origin_stop' (column) in wide
            for origin_stop in wide.columns:
                # 1) Sort other stops by ascending distance from `origin_stop`
                dist_series = dist_df.loc[origin_stop].sort_values()
                # Reindex wide's columns in that order
                wide_sorted = wide.reindex(columns=dist_series.index)

                # 2) Forward-fill left->right so the rightmost col becomes the first non-null
                wide_filled = wide_sorted.ffill(axis=1)

                # 3) The final values for the origin stop is the last column of wide_filled
                #    i.e. wide_filled.iloc[:, -1]
                final_values_for_origin = wide_filled.iloc[:, -1].copy()

                # 4) We also want to know *which* column originally provided that value:
                #    i.e. the leftmost non-null from wide_sorted. We can find that with:
                mask_notnull = wide_sorted.notna()
                # 'idxmax' returns the *first* True from left->right. If all are False => it picks the first col by default
                # We'll fix that by checking rows with any True at all:
                any_notnull = mask_notnull.any(axis=1)
                # For rows that are all-NaN, idxmax will be the *first* column. We'll set it to NaN for those rows
                source_stops = mask_notnull.idxmax(axis=1)
                source_stops[~any_notnull] = np.nan

                # 5) Distances from origin_stop to whichever source_stop:
                dist_vals = []
                for row_idx, src_stop in source_stops.items():
                    if pd.isna(src_stop):
                        dist_vals.append(np.nan)
                    else:
                        dist_vals.append(dist_df.loc[origin_stop, src_stop])

                dist_vals = pd.Series(dist_vals, index=source_stops.index)

                # Now place these final data back into our wide storage
                filled_wide[origin_stop] = final_values_for_origin
                source_stop_wide[origin_stop] = source_stops
                source_dist_wide[origin_stop] = dist_vals

            # Stack them back to long
            filled_long = filled_wide.stack(dropna=False).rename(var)  # final values
            filled_long_df = filled_long.reset_index()
            # 'filled_long_df' => columns = [date, stop_id, var]

            src_stop_long = source_stop_wide.stack(dropna=False).rename("src_stop")
            src_stop_long_df = src_stop_long.reset_index()

            src_dist_long = source_dist_wide.stack(dropna=False).rename("src_dist")
            src_dist_long_df = src_dist_long.reset_index()

            # Merge them into one
            merged_filled = filled_long_df.merge(
                src_stop_long_df, on=[date_col, stop_id_col], how="left"
            ).merge(src_dist_long_df, on=[date_col, stop_id_col], how="left")
            # merged_filled has columns [date, stop_id, var, src_stop, src_dist]

            # We'll merge that back onto df
            df = df.merge(
                merged_filled,
                on=[date_col, stop_id_col],
                how="left",
                suffixes=("", "_filled"),
            )

            # If original var is NaN but var_filled is not => we impute
            # so <var> = <var_filled>, <var>_was_imputed=True, <var>_impute_source_stop=src_stop, <var>_impute_source_dist=src_dist
            missing_mask = df[var].isna() & df[f"{var}_filled"].notna()

            # Overwrite the missing with the newly filled
            df.loc[missing_mask, var] = df.loc[missing_mask, f"{var}_filled"]
            df.loc[missing_mask, was_imputed_col] = True
            df.loc[missing_mask, source_stop_col] = df.loc[missing_mask, "src_stop"]
            df.loc[missing_mask, source_dist_col] = df.loc[missing_mask, "src_dist"]

            # If the row was all missing (including the final), src_stop=NaN => remains NaN => was_imputed=False
            # We do NOT fill those.

            # Clean up
            df.drop(columns=[f"{var}_filled", "src_stop", "src_dist"], inplace=True)
            df[source_stop_col] = df[source_stop_col].astype(str)

        df = (
            df.sort_values("_orig_index")
            .drop(columns="_orig_index")
            .reset_index(drop=True)
        )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method:
         1) Ensure we have stop_id
         2) Build distance matrix
         3) Impute with flags
        """
        df = df.copy()
        if "stop_id" not in df.columns:
            df["stop_id"] = (
                df["stop_lat"].round(4).astype(str)
                + "_"
                + df["stop_lon"].round(4).astype(str)
            )

        dist_df = self.build_distance_matrix(df)
        df_imputed = self.impute_nearest_stop(df, dist_df)
        return df_imputed


###############################################################################
# 5) Main Execution (Example)
###############################################################################
def run_weather_data_enrichment_pipeline(overwrite=False):
    """
    Demonstrate:
      1) Loading your data
      2) (Optional) Enriching with weather from Meteostat
      3) Imputing missing weather purely by nearest stops (vectorized matrix),
         adding _was_imputed, _impute_source_stop, _impute_source_dist columns.
      4) Final check ensuring no originally non-missing values changed.
      5) Saving results.
      6) (Optional) Comparing imputed values with original to ensure no changes where no missing values.
    """

    # ---------------------------------------------------------------------
    # 1) Load your DataFrame
    # ---------------------------------------------------------------------
    df = pd.read_parquet(
        dir_config.analysis_data_dir / "daily_vertex_attributes.parquet"
    )
    print("[MAIN] Loaded df with shape:", df.shape)

    # We'll track original for final comparison
    df_original = df.copy()

    # Choose which weather variables to fill
    weather_vars = [
        "tavg",
        "tmin",
        "tmax",
        "prcp",
        "snow",
        "wdir",
        "wspd",
        "wpgt",
        "pres",
        "tsun",
    ]

    # ---------------------------------------------------------------------
    # 2) (Optional) WeatherEnricher (if you want fresh data from Meteostat)
    # ---------------------------------------------------------------------
    enricher = WeatherEnricher(weather_vars=weather_vars, overwrite=overwrite)
    df_enriched = enricher.enrich(df)
    print("[MAIN] df_enriched shape:", df_enriched.shape)

    # ---------------------------------------------------------------------
    # 3) Vectorized nearest-stop imputation with flags
    # ---------------------------------------------------------------------
    imputer = WeatherImputerVectorizedWithFlags(weather_vars=weather_vars)
    df_imputed = imputer.fit_transform(df_enriched)
    print("[MAIN] df_imputed shape:", df_imputed.shape)

    # ---------------------------------------------------------------------
    # 4) Final check: confirm no originally non-missing values were changed
    # ---------------------------------------------------------------------
    for var in weather_vars:
        if var not in df_original.columns:
            continue  # e.g. if it wasn't in the original at all
        mask_non_missing = df_original[var].notna()
        # Compare final vs. original only for those rows
        final_vals = df_imputed.loc[mask_non_missing, var]
        orig_vals = df_original.loc[mask_non_missing, var]

        # .ne() is 'not equal', fillna(False) handles potential missing
        changed_mask = final_vals.ne(orig_vals).fillna(False)

        if changed_mask.any():
            print(f"[CHECK ERROR] Some originally non-missing {var} values changed!")
        else:
            print(f"[CHECK OK] Original non-missing {var} remain unchanged.")

    # ---------------------------------------------------------------------
    # 5) Save the final result
    # ---------------------------------------------------------------------
    out_path = dir_config.analysis_data_dir / "daily_vertex_attributes_imputed.parquet"

    df_imputed.to_parquet(out_path)
    print("[MAIN] Saved imputed data to:", out_path)
    print("[MAIN] Done.")

    # ---------------------------------------------------------------------
    # 6) Compare inputed values with original to ensure no changes where no missing values
    # ---------------------------------------------------------------------
    differences = compare_dataframes(df_original, df_imputed, weather_vars)
    if differences:
        for var, diff_df in differences.items():
            print(f"[CHECK ERROR] Some originally non-missing {var} values changed!")
            print(diff_df)
    else:
        print("[CHECK OK] All originally non-missing values remain unchanged.")


###############################################################################
# 6) Compare inputed values with original to ensure no changes where no missing values
###############################################################################
def compare_dataframes(df_original, df_imputed, weather_vars):
    """
    Compare the original and imputed DataFrames to ensure no originally non-missing values were changed.
    """
    differences = {}
    for var in weather_vars:
        if var not in df_original.columns:
            continue  # e.g. if it wasn't in the original at all
        mask_non_missing = df_original[var].notna()
        # Compare final vs. original only for those rows
        final_vals = df_imputed.loc[mask_non_missing, var]
        orig_vals = df_original.loc[mask_non_missing, var]

        # .ne() is 'not equal', fillna(False) handles potential missing
        changed_mask = final_vals.ne(orig_vals).fillna(False)

        if changed_mask.any():
            differences[var] = df_original.loc[
                changed_mask, ["stop_lat", "stop_lon", "date", var]
            ].merge(
                df_imputed.loc[changed_mask, ["stop_lat", "stop_lon", "date", var]],
                on=["stop_lat", "stop_lon", "date"],
                suffixes=("_original", "_imputed"),
            )
    return differences


###############################################################################
# 7) Guarded Entry
###############################################################################
if __name__ == "__main__":
    sys.exit(run_weather_data_enrichment_pipeline())
    # %%
# %%
