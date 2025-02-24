from pathlib import Path
from .config import DirConfig

# file paths are resolved relative to the init file
# DATA DIRECTORIES
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_GRAPH_DATA_DIR = DATA_DIR / "input_graph_data"
INPUT_GRAPH_DATA_DIR.mkdir(parents=True, exist_ok=True)

MAV_GTFS_DIR = DATA_DIR / "mav_gtfs"
MAV_GTFS_DIR.mkdir(parents=True, exist_ok=True)

WORKING_DIR = DATA_DIR / "working"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# in working directory
FULL_TRANSIT_GRAPHS_DIR = WORKING_DIR / "full_transit_graphs"
FULL_TRANSIT_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

MAV_SCRAPE_DIR = DATA_DIR / "mav_scrape"
MAV_SCRAPE_DIR.mkdir(parents=True, exist_ok=True)

# analysis data directory
ANALYSIS_DATA_DIR = DATA_DIR / "analysis_data"
ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# RESULTS DIRECTORIES
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAP_GRAPH_DIR = RESULTS_DIR / "map_graph_visualizations"
MAP_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_DIR = RESULTS_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = RESULTS_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

dir_config = DirConfig(
    # data directories
    data_dir=DATA_DIR,
    input_graph_data_dir=INPUT_GRAPH_DATA_DIR,
    mav_gtfs_dir=MAV_GTFS_DIR,
    working_dir=WORKING_DIR,
    analysis_data_dir=ANALYSIS_DATA_DIR,
    full_transit_graphs_dir=FULL_TRANSIT_GRAPHS_DIR,
    mav_scrape_dir=MAV_SCRAPE_DIR,
    # results directories
    results_dir=RESULTS_DIR,
    map_graph_dir=MAP_GRAPH_DIR,
    graph_dir=GRAPH_DIR,
    table_dir=TABLE_DIR,
)
