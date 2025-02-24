from mav_analysis.align_schedule_to_graph import align_schedule_to_graph
from mav_analysis.gather_input_graph_data import gather_and_save_graph_building_inputs
from mav_analysis.feature_table_finalization import construct_daily_graph_features_data
from mav_analysis.download_mav_scrape_data import download_and_save_all_scraped_mav_data
from mav_analysis.graph_analyzer.full_transit_graph_comparer import (
    build_and_compare_transit_graphs,
)


def reproduce_preprocessing_steps(
    download_scraped_data=True,
    gather_graph_inputs=True,
    build_graphs=True,
    align_schedule=True,
    create_feature_tables=True,
):
    """
    Execute the preprocessing steps for the MAV analysis project.

    This function performs the following steps:
    1. Downloads and saves all scraped MAV data.
    2. Gathers and saves graph building inputs.
    3. Aligns the schedule to the graph.
    4. Creates daily feature tables.

    After running this function, the preprocessing steps will be completed,
    and the analysis can be run in the analysis.ipynb notebook in the project root.
    """
    print("Starting preprocessing steps...")
    if download_scraped_data:
        print("Downloading and saving scraped data...")
        download_and_save_all_scraped_mav_data()
        print("Scraped data downloaded and saved.")
    if gather_graph_inputs:
        print("Gathering and saving graph building inputs...")
        gather_and_save_graph_building_inputs()
        print("Graph building inputs gathered and saved.")
    if build_graphs:
        print("Building the graph...")
        build_and_compare_transit_graphs()
        print("Graph built.")
    if align_schedule:
        print("Aligning schedule to graph...")
        align_schedule_to_graph()
        print("Schedule aligned to graph.")
    if create_feature_tables:
        print("Creating daily feature tables...")
        construct_daily_graph_features_data()
        print("Daily feature tables created.")
    print("Preprocessing steps completed successfully.")
    print("You can now proceed to run the analysis in analysis.ipynb in project root.")
    print("Have a nice day!")


if __name__ == "__main__":
    reproduce_preprocessing_steps(
        download_scraped_data=False,
        gather_graph_inputs=False,
        build_graphs=True,
        align_schedule=True,
        create_feature_tables=True,
    )
