# %%
import pickle
import pandas as pd
from mav_analysis.config import dir_config
from tqdm import tqdm


# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------

DELAY_THRESHOLDS = [1, 15, 30, 60, 120]  # you can adjust these as you like
BIDIRECTED = True


# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------


def load_schedules() -> pd.DataFrame:
    """
    Loads the interpolated schedules from parquet.
    """
    return pd.read_parquet(
        dir_config.working_dir / "interpolated_train_schedules.parquet"
    )


def load_graph(bidirected: bool = True):
    """
    Loads the graph from pickle. If bidirected=True,
    loads the undirected (mutual) version.
    """
    if bidirected:
        with open(
            dir_config.full_transit_graphs_dir / "gtfs_graph_undirected.pkl", "rb"
        ) as f:
            g = pickle.load(f)
        # Convert to directed in mutual mode
        g = g.as_directed(mode="mutual")
    else:
        with open(dir_config.full_transit_graphs_dir / "gtfs_graph.pkl", "rb") as f:
            g = pickle.load(f)

    return g


# ------------------------------------------------------------------------
# PREPARATION STEPS
# ------------------------------------------------------------------------


def prepare_edge_df(graph):
    """
    Returns the edge dataframe with an 'edge_id' column, plus a 'consecutive_vertex_ids'
    column that identifies source->target.
    """
    edge_df = (
        graph.get_edge_dataframe().reset_index().rename(columns={"edge ID": "edge_id"})
    )
    edge_df["consecutive_vertex_ids"] = (
        edge_df["source"].astype(float).astype(str)
        + "->"
        + edge_df["target"].astype(float).astype(str)
    )
    return edge_df


def prepare_vertex_df(graph):
    """
    Returns the vertex dataframe with a 'vertex_id' column.
    """
    vertex_df = (
        graph.get_vertex_dataframe()
        .reset_index()
        .rename(columns={"vertex ID": "vertex_id"})
    )
    return vertex_df


def process_schedules_df(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies any pre-processing steps to the schedules data.
    Ensures no negative 'estimated_arrival_diff_minutes'.
    """
    # Clip the arrival diff at 0 (safety measure)
    schedules_df["estimated_arrival_diff_minutes"] = (
        schedules_df["estimated_arrival_diff"].dt.seconds / 60
    ).clip(lower=0)

    return schedules_df


# ------------------------------------------------------------------------
# DELAY THRESHOLD AGGREGATION
# ------------------------------------------------------------------------


def compute_daily_summary(schedules_df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """
    Creates a daily summary DataFrame that includes:
      - The total number of trains per edge per day (num_trains).
      - The average delay (daily_avg_delay_minutes).
      - The count of delays >= each threshold (delays_{threshold}_plus).
    """
    if thresholds is None:
        thresholds = DELAY_THRESHOLDS

    # First, compute the base daily summary with num_trains
    # (unique trains per consecutive_vertex_ids per date)
    df_base = (
        schedules_df[["date", "consecutive_vertex_ids", "prev_vertex_id", "vertex_id"]]
        .drop_duplicates()
        .merge(
            schedules_df.groupby(["date", "consecutive_vertex_ids"])
            .size()
            .reset_index(name="num_trains"),
            on=["date", "consecutive_vertex_ids"],
        )
    )

    # Compute the average delay by date & edge
    temp_df = schedules_df[["date", "consecutive_vertex_ids", "delay_minutes"]]
    avg_delay_df = (
        temp_df.groupby(["date", "consecutive_vertex_ids"])
        .agg(daily_avg_delay_minutes=("delay_minutes", "mean"))
        .reset_index()
    )

    # Merge the average delay back
    df_base = df_base.merge(
        avg_delay_df, on=["date", "consecutive_vertex_ids"], how="left"
    )

    # For each threshold, count how many trains are delayed >= threshold
    for threshold in thresholds:
        above_thr_mask = temp_df["delay_minutes"] >= threshold
        counts_thr_df = (
            temp_df[above_thr_mask]
            .groupby(["date", "consecutive_vertex_ids"])
            .size()
            .reset_index(name=f"delays_{threshold}_plus")
        )
        df_base = df_base.merge(
            counts_thr_df, on=["date", "consecutive_vertex_ids"], how="left"
        )
        df_base[f"delays_{threshold}_plus"] = (
            df_base[f"delays_{threshold}_plus"].fillna(0).astype(int)
        )

    return df_base


def compute_edge_stats(
    schedules_df: pd.DataFrame, edge_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges in average distance and average edge time into edge_df,
    using the schedules_df.
    """
    # Group by edge to get mean distance_diff and mean time
    agg_df = (
        schedules_df.groupby(["consecutive_vertex_ids"])
        .agg({"distance_diff": "mean", "estimated_arrival_diff_minutes": "mean"})
        .reset_index()
        .rename(
            columns={
                "distance_diff": "edge_distance_km",
                "estimated_arrival_diff_minutes": "average_edge_time_minutes",
            }
        )
    )

    # Merge into edge_df
    edge_df = edge_df.merge(agg_df, on="consecutive_vertex_ids", how="left")
    edge_df.sort_values("edge_id", inplace=True)
    return edge_df


# ------------------------------------------------------------------------
# GRAPH-BUILDING PER DAY
# ------------------------------------------------------------------------


def build_daily_graphs(
    graph, daily_summary_df: pd.DataFrame, edge_df: pd.DataFrame, thresholds=None
):
    """
    Builds an igraph object for each date, attaching daily attributes:
      - 'weight' = num_trains
      - 'delay' = daily_avg_delay_minutes
      - 'distance'
      - 'time'
      - 'speed'
      - For each threshold X in thresholds, an edge attribute = 'weight_{X}_plus' (counts of delays >= X)
      - Vertex-level aggregated metrics (strength, betweenness, closeness, etc.)
        based on the various edge-level weights.
    Returns a dict of {date -> igraph.Graph}.
    """

    if thresholds is None:
        thresholds = DELAY_THRESHOLDS

    # Dictionary of graphs, keyed by date
    daily_graphs = {}

    # List of dates in the daily_summary_df
    date_list = daily_summary_df["date"].dropna().unique()

    # Precompute global in/out edge counts on the original graph
    # (used as vertex attributes)
    num_incoming_edges = graph.strength(mode="in")
    num_outgoing_edges = graph.strength(mode="out")

    # Sort edge_df by edge_id once, for consistent alignment
    edge_df_sorted = edge_df.sort_values("edge_id").reset_index(drop=True)

    for date in tqdm(
        date_list,
        desc="Building daily graphs...",
        total=len(date_list),
        unit="day",
    ):
        # Filter daily_summary_df for the current date
        day_df = daily_summary_df.query("date == @date")

        # ----------------------------------------------------------------
        # 1) Merge the per-date DataFrame (day_df) on edge_id
        #    This ensures each edge lines up with its daily values.
        # ----------------------------------------------------------------
        merged_for_the_day = (
            pd.DataFrame(edge_df_sorted["edge_id"].astype(float), columns=["edge_id"])
            .merge(day_df, on="edge_id", how="left")
            .sort_values("edge_id")
        )

        # ----------------------------------------------------------------
        # 2) Edge-level series for the daily graph
        # ----------------------------------------------------------------
        w_num_trains = merged_for_the_day["num_trains"].fillna(0)
        w_delay = merged_for_the_day["daily_avg_delay_minutes"].fillna(0)

        # Distances and times come from the pre-sorted edge_df
        w_distance = edge_df_sorted["edge_distance_km"]
        w_time = edge_df_sorted["average_edge_time_minutes"]

        # Copy the base graph to create a day-specific version
        g_day = graph.copy()

        # Attach the base edge attributes
        g_day.es["weight"] = w_num_trains
        g_day.es["delay"] = w_delay
        g_day.es["distance"] = w_distance
        g_day.es["time"] = w_time

        # Speed = distance / (time in hours); handle zero times
        safe_time = w_time.replace(0, pd.NA)
        g_day.es["speed"] = w_distance / (safe_time / 60)

        # Now handle each threshold for delayed trains
        for thr in thresholds:
            thr_col = f"delays_{thr}_plus"  # column in daily_summary_df
            # If that column is missing, fill with 0
            if thr_col not in merged_for_the_day.columns:
                merged_for_the_day[thr_col] = 0
            # Attach to edges
            g_day.es[f"weight_{thr}_plus"] = merged_for_the_day[thr_col].fillna(0)

        # ----------------------------------------------------------------
        # 3) Vertex-level metrics
        # ----------------------------------------------------------------

        # Store the global in/out edges as vertex attributes
        g_day.vs["num_incoming_edges"] = num_incoming_edges
        g_day.vs["num_outgoing_edges"] = num_outgoing_edges

        # =========== Base "num_trains" related sums ===========
        g_day.vs["incoming"] = g_day.strength(mode="in", weights=g_day.es["weight"])
        g_day.vs["outgoing"] = g_day.strength(mode="out", weights=g_day.es["weight"])
        g_day.vs["sum_incoming_outgoing"] = g_day.strength(
            mode="all", weights=g_day.es["weight"]
        )

        # =========== For each threshold, replicate "strength" logic ===========
        for thr in thresholds:
            w_attr = f"weight_{thr}_plus"
            in_attr = f"incoming_{thr}_plus"
            out_attr = f"outgoing_{thr}_plus"
            sum_attr = f"sum_incoming_outgoing_{thr}_plus"

            g_day.vs[in_attr] = g_day.strength(mode="in", weights=g_day.es[w_attr])
            g_day.vs[out_attr] = g_day.strength(mode="out", weights=g_day.es[w_attr])
            g_day.vs[sum_attr] = g_day.strength(mode="all", weights=g_day.es[w_attr])

        # =========== Average neighboring delay =============
        # Using total 'weight' (num_trains) as the denominator
        denom_series = pd.Series(g_day.vs["sum_incoming_outgoing"]).replace(0, pd.NA)
        g_day.vs["avg_neighboring_delay"] = (
            pd.Series(g_day.strength(weights=g_day.es["delay"], mode="all"))
            / denom_series
        )

        # =========== Average incoming delay =============
        in_series = pd.Series(g_day.vs["incoming"]).replace(0, pd.NA)
        g_day.vs["avg_incoming_delay"] = (
            pd.Series(g_day.strength(weights=g_day.es["delay"], mode="in")) / in_series
        )

        # =========== Weighted betweenness & closeness (base weight) ==========
        g_day.vs["weighted_betweenness"] = g_day.betweenness(
            weights=pd.Series(g_day.es["weight"]) + 1, directed=True
        )
        g_day.vs["betweenness"] = g_day.betweenness(directed=True)

        g_day.vs["weighted_closeness"] = g_day.closeness(
            weights=pd.Series(g_day.es["weight"]) + 1
        )
        g_day.vs["closeness"] = g_day.closeness()

        # =========== Weighted betweenness/closeness for each threshold ======
        for thr in thresholds:
            w_attr = f"weight_{thr}_plus"
            g_day.vs[f"weighted_betweenness_{thr}_plus"] = g_day.betweenness(
                weights=pd.Series(g_day.es[w_attr]) + 1, directed=True
            )
            g_day.vs[f"weighted_closeness_{thr}_plus"] = g_day.closeness(
                weights=pd.Series(g_day.es[w_attr]) + 1
            )

        # =========== day_of_week at vertex and edge level ===========
        day_of_week = pd.to_datetime(date).dayofweek
        g_day.vs["day_of_week"] = day_of_week
        g_day.es["day_of_week"] = day_of_week

        # =========== Summed distance/time/speed strengths ===========
        g_day.vs["incoming_distance_strength"] = g_day.strength(
            weights=g_day.es["distance"], mode="in"
        )
        g_day.vs["outgoing_distance_strength"] = g_day.strength(
            weights=g_day.es["distance"], mode="out"
        )
        g_day.vs["total_distance_strength"] = g_day.strength(
            weights=g_day.es["distance"], mode="all"
        )

        g_day.vs["incoming_time_strength"] = g_day.strength(
            weights=g_day.es["time"], mode="in"
        )
        g_day.vs["outgoing_time_strength"] = g_day.strength(
            weights=g_day.es["time"], mode="out"
        )
        g_day.vs["total_time_strength"] = g_day.strength(
            weights=g_day.es["time"], mode="all"
        )

        g_day.vs["incoming_speed_strength"] = g_day.strength(
            weights=g_day.es["speed"], mode="in"
        )
        g_day.vs["outgoing_speed_strength"] = g_day.strength(
            weights=g_day.es["speed"], mode="out"
        )
        g_day.vs["total_speed_strength"] = g_day.strength(
            weights=g_day.es["speed"], mode="all"
        )

        # ----------------------------------------------------------------
        # 4) Store in dictionary
        # ----------------------------------------------------------------
        daily_graphs[date] = g_day

    return daily_graphs


def merge_edge_ids(
    daily_summary_df: pd.DataFrame, edge_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the edge_id from edge_df into daily_summary_df based on consecutive_vertex_ids.
    """
    daily_summary_df["edge_id"] = daily_summary_df.merge(
        edge_df[["edge_id", "consecutive_vertex_ids"]],
        on="consecutive_vertex_ids",
        how="left",
    )["edge_id"]
    return daily_summary_df


# ------------------------------------------------------------------------
# FINAL DATAFRAME EXPORT
# ------------------------------------------------------------------------


def consolidate_vertex_attributes(daily_graphs: dict) -> pd.DataFrame:
    """
    Gathers all vertex attributes from all daily graphs into a single DataFrame.
    """
    vertex_frames = []
    for date, g_day in daily_graphs.items():
        vdf = g_day.get_vertex_dataframe().assign(date=date)
        vertex_frames.append(vdf)
    return pd.concat(vertex_frames, ignore_index=True)


def consolidate_edge_attributes(daily_graphs: dict) -> pd.DataFrame:
    """
    Gathers all edge attributes from all daily graphs into a single DataFrame.
    """
    edge_frames = []
    for date, g_day in daily_graphs.items():
        edf = g_day.get_edge_dataframe().assign(date=date)
        edge_frames.append(edf)
    return pd.concat(edge_frames, ignore_index=True)


# ------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------


def construct_daily_graph_features_data(bidirected: bool = True):
    # 1. Load data
    schedules_df = load_schedules()
    g = load_graph(bidirected=bidirected)

    # 2. Prepare base dataframes
    vertex_df = prepare_vertex_df(g)
    edge_df = prepare_edge_df(g)

    # 3. Process schedules data (clip negative diffs, etc.)
    schedules_df = process_schedules_df(schedules_df)

    # 4. Compute daily summary (num_trains, daily_avg_delay, delays_{threshold}_plus)
    daily_summary_df = compute_daily_summary(schedules_df, thresholds=DELAY_THRESHOLDS)

    # 5. Merge with vertex_df if needed (the original code did so, but often isn't strictly necessary)
    daily_summary_df = daily_summary_df.merge(vertex_df, on="vertex_id", how="left")

    # 6. Compute average distances/times per edge, merge into edge_df
    edge_df = compute_edge_stats(schedules_df, edge_df)

    # 7. Attach edge_id to daily_summary_df
    daily_summary_df = merge_edge_ids(daily_summary_df, edge_df)

    # 8. Build the daily graphs
    daily_graphs = build_daily_graphs(
        graph=g,
        daily_summary_df=daily_summary_df,
        edge_df=edge_df,
        thresholds=DELAY_THRESHOLDS,
    )

    # 9. Consolidate attributes into final DataFrames
    vertex_out_df = consolidate_vertex_attributes(daily_graphs)
    edge_out_df = consolidate_edge_attributes(daily_graphs)

    # 10. Export
    vertex_out_df.to_parquet(
        dir_config.analysis_data_dir / "daily_vertex_attributes.parquet"
    )
    edge_out_df.to_parquet(
        dir_config.analysis_data_dir / "daily_edge_attributes.parquet"
    )


# %%
if __name__ == "__main__":
    # %%
    # 1. Load data
    schedules_df = load_schedules()
    g = load_graph(bidirected=BIDIRECTED)

    # 2. Prepare base dataframes
    vertex_df = prepare_vertex_df(g)
    edge_df = prepare_edge_df(g)

    # 3. Process schedules data (clip negative diffs, etc.)
    schedules_df = process_schedules_df(schedules_df)

    # 4. Compute daily summary (num_trains, daily_avg_delay, delays_{threshold}_plus)
    daily_summary_df = compute_daily_summary(schedules_df, thresholds=DELAY_THRESHOLDS)
    # %%
    # 5. Merge with vertex_df if needed (the original code did so, but often isn't strictly necessary)
    daily_summary_df = daily_summary_df.merge(vertex_df, on="vertex_id", how="left")

    # 6. Compute average distances/times per edge, merge into edge_df
    edge_df = compute_edge_stats(schedules_df, edge_df)

    # 7. Attach edge_id to daily_summary_df
    daily_summary_df = merge_edge_ids(daily_summary_df, edge_df)

    # 8. Build the daily graphs
    daily_graphs = build_daily_graphs(
        graph=g,
        daily_summary_df=daily_summary_df,
        edge_df=edge_df,
        thresholds=DELAY_THRESHOLDS,
    )

    # 9. Consolidate attributes into final DataFrames
    vertex_out_df = consolidate_vertex_attributes(daily_graphs)
    edge_out_df = consolidate_edge_attributes(daily_graphs)
    # %%
    # 10. Export
    vertex_out_df.to_parquet(
        dir_config.analysis_data_dir / "daily_vertex_attributes.parquet"
    )
    edge_out_df.to_parquet(
        dir_config.analysis_data_dir / "daily_edge_attributes.parquet"
    )

# %%
