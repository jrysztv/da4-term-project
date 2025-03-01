from igraph import Graph


class StopSequenceGraphBuilder:
    def __init__(self, stops_df, stop_sequence_df, group_by="trip_id"):
        self.stops_df = stops_df
        self.stop_sequence_df = stop_sequence_df
        self.group_by_col = group_by
        self.vertex_attributes = None
        self.edge_df = None
        self.graph = None

    def add_vertice_id(self):
        self.stop_sequence_df["vertice_id"] = (
            self.stop_sequence_df["stop_id"].astype("category").cat.codes
        )

    def add_prev_vertice_id(self):
        self.stop_sequence_df["prev_vertice_id"] = (
            self.stop_sequence_df.groupby(self.group_by_col)["vertice_id"]
            .shift(-1)
            .astype("Int64")
        )

    def create_edge_df(self):
        self.edge_df = (
            self.stop_sequence_df[["prev_vertice_id", "vertice_id"]]
            .dropna()
            .drop_duplicates()
        )

    def create_vertex_attributes(self):
        vertex_attributes = (
            self.stop_sequence_df[["vertice_id", "stop_id"]]
            .drop_duplicates()
            .set_index("stop_id")
        )
        vertex_attributes["stop_name"] = self.stops_df.set_index("stop_id")["stop_name"]
        vertex_attributes["stop_lat"] = self.stops_df.set_index("stop_id")["stop_lat"]
        vertex_attributes["stop_lon"] = self.stops_df.set_index("stop_id")["stop_lon"]
        self.vertex_attributes = (
            vertex_attributes.reset_index().set_index("vertice_id").sort_index()
        )

    def create_graph(self):
        self.graph = Graph(
            vertex_attrs=self.vertex_attributes.to_dict("list"),
            edges=self.edge_df[["vertice_id", "prev_vertice_id"]].values.tolist(),
            directed=True,
        )

    def run(self):
        self.add_vertice_id()
        self.add_prev_vertice_id()
        self.create_edge_df()
        self.create_vertex_attributes()
        self.create_graph()
