# %%
import pandas as pd
from mav_analysis.config import dir_config
import pickle

# %% pandas column display limit off
pd.set_option("display.max_columns", None)


def construct_daily_graph_features_data():
    # %%
    schedules_df = pd.read_parquet(
        dir_config.working_dir / "interpolated_train_schedules.parquet"
    )
    with open(
        dir_config.full_transit_graphs_dir / "gtfs_graph_undirected.pkl", "rb"
    ) as f:
        g = pickle.load(f)

    g = g.as_directed(mode="mutual")

    vertex_df = (
        g.get_vertex_dataframe()
        .reset_index()
        .rename(columns={"vertex ID": "vertex_id"})
    )
    edge_df = (
        g.get_edge_dataframe().reset_index().rename(columns={"edge ID": "edge_id"})
    )
    # %%
    date_list = schedules_df["date"].unique()
    # consequtive_vertex_ids_df = (
    #     edge_df["source"].astype(float).astype(str)
    #     + "->"
    #     + edge_df["target"].astype(float).astype(str)
    # )
    # consequtive_vertex_ids_df

    # %% daily sum of trains on an edge
    daily_summary_df = (
        schedules_df[["date", "consecutive_vertex_ids", "prev_vertex_id", "vertex_id"]]
        .drop_duplicates()
        .merge(
            schedules_df.groupby(["date", "consecutive_vertex_ids"])
            .size()
            .reset_index()
            .rename(columns={0: "num_trains"}),
            on=["date", "consecutive_vertex_ids"],
        )
    )
    # %% average daily delay per edge
    daily_summary_df = daily_summary_df.merge(
        schedules_df.groupby(["date", "consecutive_vertex_ids"])
        .agg({"delay_minutes": "mean"})
        .reset_index()
        .rename(columns={"delay_minutes": "daily_avg_delay_minutes"}),
        on=["date", "consecutive_vertex_ids"],
    )
    # %%
    daily_summary_df = daily_summary_df.merge(
        schedules_df.query("delay_minutes>0")
        .groupby(["date", "consecutive_vertex_ids"])
        .size()
        .reset_index()
        .rename(columns={0: "num_delayed_trains"}),
        on=["date", "consecutive_vertex_ids"],
        how="left",
    )
    daily_summary_df["num_delayed_trains"] = daily_summary_df[
        "num_delayed_trains"
    ].fillna(0)
    # %%
    daily_summary_df = daily_summary_df.merge(vertex_df, on="vertex_id", how="left")
    # %% cutoff estimated arrival diff at 0 - safety measure
    schedules_df["estimated_arrival_diff_minutes"] = (
        schedules_df["estimated_arrival_diff"].dt.seconds / 60
    ).clip(lower=0)
    # %%
    edge_df = (
        g.get_edge_dataframe().reset_index().rename(columns={"edge ID": "edge_id"})
    )
    edge_df["consecutive_vertex_ids"] = (
        edge_df["source"].astype(float).astype(str)
        + "->"
        + edge_df["target"].astype(float).astype(str)
    )
    edge_df = edge_df.merge(
        schedules_df.groupby(["consecutive_vertex_ids"])
        .agg({"distance_diff": "mean", "estimated_arrival_diff_minutes": "mean"})
        .reset_index()
        .rename(
            columns={
                "distance_diff": "edge_distance_km",
                "estimated_arrival_diff_minutes": "average_edge_time_minutes",
            }
        ),
        how="left",
        on="consecutive_vertex_ids",
    )
    edge_df.sort_values("edge_id", inplace=True)
    edge_df
    # %%
    daily_summary_df["edge_id"] = daily_summary_df.merge(
        edge_df[["edge_id", "consecutive_vertex_ids"]],
        on="consecutive_vertex_ids",
        how="left",
    )["edge_id"]
    # %%
    daily_summary_df.dropna(subset=["edge_id"], inplace=True)
    # %% assign daily weights of the number of trains to the edges
    date_list = daily_summary_df["date"].unique()
    daily_graphs = {}

    for date in date_list:
        day_df = daily_summary_df.query("date==@date")
        weights = (
            pd.DataFrame(edge_df["edge_id"].astype(float))
            .merge(day_df, on="edge_id", how="left")
            .sort_values("edge_id")["num_trains"]
            .fillna(0)
        )

        delays = (
            pd.DataFrame(edge_df["edge_id"].astype(float))
            .merge(day_df, on="edge_id", how="left")
            .sort_values("edge_id")["daily_avg_delay_minutes"]
        )

        distances = edge_df["edge_distance_km"]

        times = edge_df["average_edge_time_minutes"]

        num_incoming_edges = g.strength(mode="in")
        num_outgoing_edges = g.strength(mode="out")

        g_day = g.copy()
        g_day.es["weight"] = weights
        g_day.es["delay"] = delays
        g_day.es["distance"] = distances
        g_day.es["time"] = times
        g_day.es["speed"] = distances / (times / 60).replace(0, pd.NA)
        g_day.vs["num_incoming_edges"] = num_incoming_edges
        g_day.vs["num_outgoing_edges"] = num_outgoing_edges
        daily_graphs[date] = g_day

    # Now daily_graphs contains a graph for each day
    # %% calculate incoming and outgoing edges for each vertex, multiply the weights of the incoming and outgoing edges
    # and assign the result to the vertex as incoming or outgoing attributes

    for date, g_day in daily_graphs.items():
        g_day.vs["incoming"] = g_day.strength(mode="in", weights=g_day.es["weight"])
        g_day.vs["outgoing"] = g_day.strength(mode="out", weights=g_day.es["weight"])
        g_day.vs["sum_incoming_outgoing"] = g_day.strength(
            mode="all", weights=g_day.es["weight"]
        )

    # %% calculate the average neighboring delays for each vertex
    for date, g_day in daily_graphs.items():
        g_day.vs["avg_neighboring_delay"] = pd.Series(
            g_day.strength(weights=g_day.es["delay"], mode="all")
        ) / pd.Series(g_day.vs["sum_incoming_outgoing"]).replace(0, pd.NA)

    # %% calculate the average delays for all incoming edges for each vertex
    for date, g_day in daily_graphs.items():
        g_day.vs["avg_incoming_delay"] = pd.Series(
            g_day.strength(weights=g_day.es["delay"], mode="in")
        ) / pd.Series(g_day.vs["incoming"]).replace(0, pd.NA)
    # %% calculate the weighted betweenness centrality for each vertex
    for date, g_day in daily_graphs.items():
        g_day.vs["weighted_betweenness"] = g_day.betweenness(
            weights=pd.Series(g_day.es["weight"]) + 1,
            directed=True,  # adding 1 to avoid division by zero
        )
        g_day.vs["betweenness"] = g_day.betweenness(directed=True)
    # %%
    for date, g_day in daily_graphs.items():
        g_day.vs["weighted_closeness"] = g_day.closeness(
            weights=pd.Series(g_day.es["weight"]) + 1
        )
        g_day.vs["closeness"] = g_day.closeness()
    # %% adding day of week to the vertex attributes and to the edge attributes
    for date, g_day in daily_graphs.items():
        g_day.vs["day_of_week"] = pd.to_datetime(date).dayofweek
        g_day.es["day_of_week"] = pd.to_datetime(date).dayofweek
    # %% adding average incoming, outgoing and total distance to the vertex attributes
    for date, g_day in daily_graphs.items():
        g_day.vs["incoming_distance_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["distance"], mode="in")
        )
        g_day.vs["outgoing_distance_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["distance"], mode="out")
        )
        g_day.vs["total_distance_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["distance"], mode="all")
        )
    # %% adding average incoming, outgoing and total time to the vertex attributes
    for date, g_day in daily_graphs.items():
        g_day.vs["incoming_time_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["time"], mode="in")
        )
        g_day.vs["outgoing_time_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["time"], mode="out")
        )
        g_day.vs["total_time_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["time"], mode="all")
        )
    # %% adding average incoming, outgoing and total speed to the vertex attributes
    for date, g_day in daily_graphs.items():
        g_day.vs["incoming_speed_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["speed"], mode="in")
        )
        g_day.vs["outgoing_speed_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["speed"], mode="out")
        )
        g_day.vs["total_speed_strength"] = pd.Series(
            g_day.strength(weights=g_day.es["speed"], mode="all")
        )

    # %% merging the vertex and edge attributes of all days into a single DataFrame
    vertex_df = pd.concat(
        [
            g_day.get_vertex_dataframe().assign(date=date)
            for date, g_day in daily_graphs.items()
        ]
    )
    edge_df = pd.concat(
        [
            g_day.get_edge_dataframe().assign(date=date)
            for date, g_day in daily_graphs.items()
        ]
    )
    # %%
    vertex_df.to_parquet(
        dir_config.analysis_data_dir / "daily_vertex_attributes.parquet"
    )
    edge_df.to_parquet(dir_config.analysis_data_dir / "daily_edge_attributes.parquet")
    # %%
