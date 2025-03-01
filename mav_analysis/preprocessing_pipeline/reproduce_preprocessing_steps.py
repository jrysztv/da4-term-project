from mav_analysis.preprocessing_pipeline.align_schedule_to_graph import (
    align_schedule_to_graph,
)
from mav_analysis.preprocessing_pipeline.gather_input_graph_data import (
    gather_and_save_graph_building_inputs,
)
from mav_analysis.preprocessing_pipeline.feature_table_finalization import (
    construct_daily_graph_features_data,
)
from mav_analysis.preprocessing_pipeline.download_mav_scrape_data import (
    download_and_save_all_scraped_mav_data,
)
from mav_analysis.preprocessing_pipeline.graph_analyzer.full_transit_graph_comparer import (
    build_compare_save_transit_graphs,
)
from mav_analysis.preprocessing_pipeline.weather_data_enrichment import (
    run_weather_data_enrichment_pipeline,
)


def reproduce_preprocessing_steps(
    download_scraped_data=True,
    gather_graph_inputs=True,
    build_graphs=True,
    align_schedule=True,
    create_feature_tables=True,
    enrich_weather_data=True,
):
    """
    Execute the preprocessing steps for the MAV analysis project.
    MAV is the public train service in Hungary.
    This pipeline puts all of the building blocks in place for the preprocessing.
    The end result is balanced panel data with daily graph features based on number of trains and size of delay, and weather data.

    This function performs the following steps:
    1. Downloads and saves all scraped MAV data.
    2. Gathers and saves graph building inputs.
    3. Builds the transit graph.
    4. Aligns the schedule to the graph.
    5. Creates daily feature tables.
    6. Enriches weather data.

    After running this function, the preprocessing steps will be completed,
    and the analysis can be run in the analysis.ipynb notebook in the project root.
    """
    bidirected_fullgraph = False
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
        build_compare_save_transit_graphs()
        print("Graph built.")
    if align_schedule:
        print("Aligning schedule to graph...")
        align_schedule_to_graph(bidirected=bidirected_fullgraph)
        print("Schedule aligned to graph.")
    if create_feature_tables:
        print("Creating daily feature tables...")
        construct_daily_graph_features_data(bidirected=bidirected_fullgraph)
        print("Daily feature tables created.")
    if enrich_weather_data:
        print("Enriching weather data...")
        run_weather_data_enrichment_pipeline(overwrite=True)
        print("Weather data enriched.")
    print("Preprocessing steps completed successfully.")
    print("You can now proceed to run the analysis in analysis.ipynb in project root.")
    print("Have a nice day!")


if __name__ == "__main__":
    reproduce_preprocessing_steps(
        download_scraped_data=False,
        gather_graph_inputs=False,
        build_graphs=False,
        align_schedule=False,
        create_feature_tables=True,
        enrich_weather_data=True,
    )
