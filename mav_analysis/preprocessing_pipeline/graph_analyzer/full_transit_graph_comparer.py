# %%
import pandas as pd
from igraph import Graph

from mav_analysis.config import dir_config
from mav_analysis.preprocessing_pipeline.graph_analyzer.utils.graph_builder import (
    StopSequenceGraphBuilder,
)
from mav_analysis.preprocessing_pipeline.graph_analyzer.utils.graph_visualizer import (
    MapVisualizer,
)


class TransitGraphPipeline:
    """
    Builds a transit graph from stop/line data, then saves a Folium map visualization.
    """

    def __init__(
        self,
        stops_df: pd.DataFrame,
        stop_lines_df: pd.DataFrame,
        group_by: str,
        output_filename: str,
    ):
        """
        :param stops_df: DataFrame containing stops info (stop_id, stop_name, lat/lon, etc.)
        :param stop_lines_df: DataFrame with stop-line relationships (trip_id or elvira_id, sequence, etc.)
        :param group_by: The column used to group stops into lines or trips (e.g., 'trip_id' or 'elvira_id')
        :param output_filename: Filename for the Folium map output (e.g., "gtfs_map.html")
        """
        self.stops_df = stops_df
        self.stop_lines_df = stop_lines_df
        self.group_by = group_by
        self.output_filename = output_filename
        self.graph = None

    def build_transit_graph(self) -> None:
        """
        Uses the StopSequenceGraphBuilder to build an igraph Graph.
        """
        builder = StopSequenceGraphBuilder(
            self.stops_df, self.stop_lines_df, group_by=self.group_by
        )
        builder.run()
        self.graph = builder.graph

    def save_graph_map(self) -> None:
        """
        Saves a Folium map of the built graph using MapVisualizer.
        """
        map_visualizer = MapVisualizer(self.graph)
        map_visualizer.save_folium_map(dir_config.map_graph_dir / self.output_filename)

    def run_pipeline(self) -> None:
        """
        High-level method to build and then save the graph's Folium map.
        """
        self.build_transit_graph()
        self.save_graph_map()


class DirectedToUndirectedTransformer:
    """
    Converts a directed igraph Graph to an undirected version, preserving all vertex attributes.
    """

    def __init__(self, directed_graph: Graph):
        self.directed_graph = directed_graph
        self.undirected_graph = None

    def transform(self) -> Graph:
        """
        Converts the directed graph to an undirected one (with collapsed edges),
        preserving vertex attributes.
        :return: A new undirected Graph instance.
        """
        self.undirected_graph = self.directed_graph.as_undirected(mode="collapse")

        # Ensure vertex attributes are preserved explicitly.
        for attr_name in self.directed_graph.vertex_attributes():
            self.undirected_graph.vs[attr_name] = self.directed_graph.vs[attr_name]

        return self.undirected_graph


class GraphVertexAligner:
    """
    Handles aligning vertices across multiple graphs by reindexing stop_id to a consistent integer index.
    Also handles attaching stop-related attributes such as name, lat, lon.
    """

    @staticmethod
    def align_vertices(
        original_graph: Graph, stopid_to_newidx: dict, all_stop_ids: list
    ) -> Graph:
        """
        Create a new graph with vertex IDs aligned to stop_id -> new vertex index.
        Copy over all relevant vertex and edge attributes.
        """
        new_graph = Graph(n=len(all_stop_ids), directed=original_graph.is_directed())

        # Copy the vertex attributes (like 'stop_id') in a new, consistent order
        new_graph.vs["stop_id"] = all_stop_ids

        # Prepare to copy edge attributes
        edge_attrs = {attr_name: [] for attr_name in original_graph.edge_attributes()}
        edges_to_add = []

        for edge in original_graph.es:
            old_source, old_target = edge.tuple
            stop_id_source = original_graph.vs[old_source]["stop_id"]
            stop_id_target = original_graph.vs[old_target]["stop_id"]

            new_source = stopid_to_newidx[stop_id_source]
            new_target = stopid_to_newidx[stop_id_target]
            edges_to_add.append((new_source, new_target))

            for attr_name in edge_attrs:
                edge_attrs[attr_name].append(edge[attr_name])

        new_graph.add_edges(edges_to_add)

        # Attach edge attributes
        for attr_name, values in edge_attrs.items():
            new_graph.es[attr_name] = values

        return new_graph

    @staticmethod
    def add_stop_attributes(stops_df: pd.DataFrame, graph: Graph) -> Graph:
        """
        Enrich each vertex in the graph with stop attributes (name, lat, lon).
        """
        graph.vs["stop_name"] = [
            stops_df.loc[stops_df["stop_id"] == stop_id, "stop_name"].values[0]
            for stop_id in graph.vs["stop_id"]
        ]
        graph.vs["stop_lat"] = [
            stops_df.loc[stops_df["stop_id"] == stop_id, "stop_lat"].values[0]
            for stop_id in graph.vs["stop_id"]
        ]
        graph.vs["stop_lon"] = [
            stops_df.loc[stops_df["stop_id"] == stop_id, "stop_lon"].values[0]
            for stop_id in graph.vs["stop_id"]
        ]
        return graph


class UndirectedGraphComparator:
    """
    Compares two undirected igraph Graphs, identifies common edges, and creates complementary graphs.
    """

    def __init__(self, graph_a: Graph, graph_b: Graph):
        self.graph_a = graph_a
        self.graph_b = graph_b

    def _compute_edge_set(self, graph: Graph) -> set:
        """
        Returns a set of edges (as sorted tuples) from an undirected graph.
        """
        return set(tuple(sorted(e)) for e in graph.get_edgelist())

    def compute_common_edges(self) -> set:
        """
        Identifies the edges shared by both undirected graphs.
        """
        edges_a = self._compute_edge_set(self.graph_a)
        edges_b = self._compute_edge_set(self.graph_b)
        return edges_a.intersection(edges_b)

    def build_complementary_graphs(self) -> tuple[Graph, Graph]:
        """
        Returns two new graphs (complements). For each graph, remove edges that exist in both.
        """
        common_edges = self.compute_common_edges()

        # Graph A complement
        graph_a_comp = self.graph_a.copy()
        to_remove_a = [
            e.index for e in self.graph_a.es if tuple(sorted(e.tuple)) in common_edges
        ]
        graph_a_comp.delete_edges(to_remove_a)

        # Graph B complement
        graph_b_comp = self.graph_b.copy()
        to_remove_b = [
            e.index for e in self.graph_b.es if tuple(sorted(e.tuple)) in common_edges
        ]
        graph_b_comp.delete_edges(to_remove_b)

        return graph_a_comp, graph_b_comp


def build_and_run_all_transit_pipelines(
    stops_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stop_lines_df: pd.DataFrame,
    stop_lines_df_uninterpolated: pd.DataFrame,
) -> tuple[TransitGraphPipeline, TransitGraphPipeline, TransitGraphPipeline]:
    """
    Creates and runs three pipelines (GTFS, line-shape, uninterpolated line-shape).
    Returns the three TransitGraphPipeline objects, each holding its final built graph.
    """
    # GTFS pipeline
    gtfs_pipeline = TransitGraphPipeline(
        stops_df, stop_times_df, group_by="trip_id", output_filename="gtfs_map.html"
    )
    gtfs_pipeline.run_pipeline()

    # Line-shape pipeline
    line_shape_pipeline = TransitGraphPipeline(
        stops_df,
        stop_lines_df,
        group_by="elvira_id",
        output_filename="line_shape_map.html",
    )
    line_shape_pipeline.run_pipeline()

    # Uninterpolated line-shape pipeline
    line_shape_uninterpolated_pipeline = TransitGraphPipeline(
        stops_df,
        stop_lines_df_uninterpolated,
        group_by="elvira_id",
        output_filename="line_shape_map_uninterpolated.html",
    )
    line_shape_uninterpolated_pipeline.run_pipeline()

    return gtfs_pipeline, line_shape_pipeline, line_shape_uninterpolated_pipeline


def build_and_save_graph_visualizations(
    stops_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stop_lines_df: pd.DataFrame,
    stop_lines_df_uninterpolated: pd.DataFrame,
) -> None:
    """
    Orchestrates building three transit graphs (GTFS, line shape, line shape uninterpolated)
    and saving their Folium maps.
    """
    build_and_run_all_transit_pipelines(
        stops_df, stop_times_df, stop_lines_df, stop_lines_df_uninterpolated
    )
    # Each pipeline saves its own Folium map, so we don't need additional logic here.


def build_compare_save_transit_graphs():
    """
    Main entry point for:
      1) Loading data.
      2) Building pipelines for GTFS and line-shape graphs.
      3) Converting them to undirected, aligning vertices, and adding stop attributes.
      4) Comparing and generating complementary graphs.
      5) Saving all outputs (maps, pickles, etc.).
    Returns a dictionary of all created graphs.
    """
    # 1) Load data
    stops_df = pd.read_csv(dir_config.mav_gtfs_dir / "stops.txt")
    stop_times_df = pd.read_csv(dir_config.mav_gtfs_dir / "stop_times.txt")
    stop_lines_df = pd.read_parquet(
        dir_config.input_graph_data_dir / "stop_lines_df.parquet"
    )
    stop_lines_df_uninterpolated = pd.read_parquet(
        dir_config.input_graph_data_dir / "stop_lines_uninterpolate_df.parquet"
    )

    # 2) Run pipelines
    (
        gtfs_pipeline,
        line_shape_pipeline,
        line_shape_uninterpolated_pipeline,
    ) = build_and_run_all_transit_pipelines(
        stops_df, stop_times_df, stop_lines_df, stop_lines_df_uninterpolated
    )

    # 3) Transform to undirected
    line_shape_transformer = DirectedToUndirectedTransformer(line_shape_pipeline.graph)
    line_shape_undirected = line_shape_transformer.transform()

    gtfs_transformer = DirectedToUndirectedTransformer(gtfs_pipeline.graph)
    gtfs_undirected = gtfs_transformer.transform()

    # 4) Align vertices and add attributes
    stop_ids_line_undirected = set(line_shape_undirected.vs["stop_id"])
    stop_ids_gtfs_undirected = set(gtfs_undirected.vs["stop_id"])
    all_stop_ids = sorted(stop_ids_line_undirected.union(stop_ids_gtfs_undirected))

    stopid_to_newidx = {stop_id: idx for idx, stop_id in enumerate(all_stop_ids)}

    line_shape_undirected = GraphVertexAligner.align_vertices(
        line_shape_undirected, stopid_to_newidx, all_stop_ids
    )
    line_shape_undirected = GraphVertexAligner.add_stop_attributes(
        stops_df, line_shape_undirected
    )

    gtfs_undirected = GraphVertexAligner.align_vertices(
        gtfs_undirected, stopid_to_newidx, all_stop_ids
    )
    gtfs_undirected = GraphVertexAligner.add_stop_attributes(stops_df, gtfs_undirected)

    # 5) Compare and build complementary graphs
    comparator = UndirectedGraphComparator(line_shape_undirected, gtfs_undirected)
    line_shape_complement, gtfs_complement = comparator.build_complementary_graphs()

    # 6) Save complementary maps
    line_shape_complement_visualizer = MapVisualizer(
        line_shape_complement, directed=False
    )
    line_shape_complement_visualizer.save_folium_map(
        dir_config.map_graph_dir / "line_shape_complementary.html"
    )

    gtfs_complement_visualizer = MapVisualizer(gtfs_complement, directed=False)
    gtfs_complement_visualizer.save_folium_map(
        dir_config.map_graph_dir / "gtfs_complementary.html"
    )

    # 7) Save all graph pickles
    line_shape_pipeline.graph.write_pickle(
        dir_config.full_transit_graphs_dir / "line_shape_graph.pkl"
    )
    gtfs_pipeline.graph.write_pickle(
        dir_config.full_transit_graphs_dir / "gtfs_graph.pkl"
    )
    line_shape_uninterpolated_pipeline.graph.write_pickle(
        dir_config.full_transit_graphs_dir / "line_shape_graph_uninterpolated.pkl"
    )

    gtfs_undirected.write_pickle(
        dir_config.full_transit_graphs_dir / "gtfs_graph_undirected.pkl"
    )
    line_shape_undirected.write_pickle(
        dir_config.full_transit_graphs_dir / "line_shape_graph_undirected.pkl"
    )

    gtfs_complement.write_pickle(
        dir_config.full_transit_graphs_dir / "gtfs_graph_complementary.pkl"
    )
    line_shape_complement.write_pickle(
        dir_config.full_transit_graphs_dir / "line_shape_graph_complementary.pkl"
    )

    return {
        "gtfs_graph": gtfs_pipeline.graph,
        "line_shape_graph": line_shape_pipeline.graph,
        "line_shape_graph_uninterpolated": line_shape_uninterpolated_pipeline.graph,
        "gtfs_graph_undirected": gtfs_undirected,
        "line_shape_graph_undirected": line_shape_undirected,
        "gtfs_graph_complementary": gtfs_complement,
        "line_shape_graph_complementary": line_shape_complement,
    }


if __name__ == "__main__":
    graphs = build_compare_save_transit_graphs()
    # Optionally, do something with 'graphs' (e.g., analyze them further)
# %%
