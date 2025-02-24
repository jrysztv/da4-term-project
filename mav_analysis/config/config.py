from pydantic import BaseModel
from pathlib import Path


class DirConfig(BaseModel):
    # data directories
    data_dir: Path
    input_graph_data_dir: Path
    mav_gtfs_dir: Path
    working_dir: Path
    analysis_data_dir: Path
    full_transit_graphs_dir: Path
    mav_scrape_dir: Path
    # results directories
    results_dir: Path
    map_graph_dir: Path
    graph_dir: Path
    table_dir: Path
