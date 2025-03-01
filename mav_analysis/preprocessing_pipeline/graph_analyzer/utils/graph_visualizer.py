import folium
from folium.features import DivIcon
from geopy.distance import geodesic


import math


class MapVisualizer:
    def __init__(self, graph, shapes_block_request=None, directed=True):
        self.graph = graph
        self.shapes_block_request = shapes_block_request
        self.place_arrows = directed

    def get_map_center(self):
        lats = [v["stop_lat"] for v in self.graph.vs]
        lons = [v["stop_lon"] for v in self.graph.vs]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def calculate_edge_length(self, lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).km

    def get_edge_color(self, length):
        if length >= 20:
            return "red"
        elif length >= 10:
            return "yellow"
        else:
            return "blue"

    def plot_shapes(self, m):
        for shape in self.shapes_block_request:
            folium.PolyLine(
                locations=shape["coordinates"],
                color=shape.get("color", "green"),
                weight=shape.get("weight", 2),
                opacity=shape.get("opacity", 0.6),
            ).add_to(m)

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Returns the bearing in degrees between two lat/lon points (start -> end).
        Bearing is measured clockwise from North.
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
            lat2
        ) * math.cos(dlon)

        initial_bearing = math.atan2(x, y)
        bearing = (math.degrees(initial_bearing) + 360) % 360
        return bearing

    def create_folium_map(self):
        center_lat, center_lon = self.get_map_center()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        # 1. Draw stop markers
        for vertex in self.graph.vs:
            lat, lon = vertex["stop_lat"], vertex["stop_lon"]
            tooltip_text = vertex["stop_name"]
            folium.CircleMarker(
                [lat, lon], tooltip=tooltip_text, radius=3, weight=5, color="#006dfc"
            ).add_to(m)

        # 2. Draw edges and place oriented arrows at midpoints
        for edge in self.graph.es:
            source = edge.source
            target = edge.target
            lat1, lon1 = (
                self.graph.vs[source]["stop_lat"],
                self.graph.vs[source]["stop_lon"],
            )
            lat2, lon2 = (
                self.graph.vs[target]["stop_lat"],
                self.graph.vs[target]["stop_lon"],
            )

            # PolyLine for the edge
            length = self.calculate_edge_length(lat1, lon1, lat2, lon2)
            color = self.get_edge_color(length)
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=color,
                weight=3,
                opacity=0.6,
            ).add_to(m)

            if self.place_arrows:
                # Compute bearing and midpoint
                bearing = self.calculate_bearing(lat1, lon1, lat2, lon2)
                # Correct so that 0 degrees means arrow "▶" points North instead of East
                bearing_corrected = (bearing - 90) % 360

                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lat1 + lat2) / 2

                # Unicode arrow styled with rotation + color
                arrow_html = f"""
                <div style="transform: rotate({bearing_corrected}deg);
                            -webkit-transform: rotate({bearing_corrected}deg);
                            -moz-transform: rotate({bearing_corrected}deg);
                            color: {color};
                            font-size: 16px;
                            line-height: 16px;">
                    ▶
                </div>
                """

                # Place a marker at the midpoint
                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=DivIcon(
                        html=arrow_html,
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),  # adjust if arrow is off-center
                    ),
                ).add_to(m)

        # 3. (Optional) Plot any shapes - unused for now
        if self.shapes_block_request is not None:
            self.plot_shapes(m)

        return m

    def save_folium_map(self, file_path):
        m = self.create_folium_map()
        m.save(file_path)
