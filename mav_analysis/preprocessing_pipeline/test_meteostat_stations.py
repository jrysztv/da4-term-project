from datetime import datetime
import pandas as pd
from meteostat import Stations, Daily


def find_nearest_stations(lat: float, lon: float) -> pd.DataFrame:
    """
    Find nearest weather stations to given coordinates.
    Returns stations sorted by distance.
    """
    # Get nearby stations
    stations = Stations()
    # Fetch all stations and let Meteostat sort by distance
    stations = stations.nearby(lat, lon).fetch()

    if stations.empty:
        print(f"No stations found near ({lat}, {lon})")
        return pd.DataFrame()

    # Debug: Print column names
    print("\nAvailable columns:", stations.columns.tolist())

    # Print details for top 5 stations
    print("\nStation details:")
    for _, station in stations.head().iterrows():
        print(f"- {station.get('name', 'Unknown')}")
        print(f"  Country: {station.get('country', 'Unknown')}")
        print(f"  Distance: {station.get('distance', 0):.1f}km")
        print(f"  Elevation: {station.get('elevation', 0)}m")
        print(f"  Station ID: {station.name}")  # Using index as station ID
        print("  Raw data:", station.to_dict())
        print()

    return stations


def test_station_data(station_id: str) -> None:
    """Test data availability for a station."""
    print(f"\nTesting data for station ID: {station_id}")

    # Get data for 2023
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)

    try:
        data = Daily(station_id, start, end)
        data = data.fetch()

        # Print coverage stats
        total_days = (end - start).days + 1
        available_days = len(data)
        coverage = (available_days / total_days) * 100

        print(f"Days with data: {available_days}/{total_days} ({coverage:.1f}%)")
        if not data.empty:
            print("\nAvailable columns:", data.columns.tolist())
            print("\nSample of available data:")
            print(data.head())
        else:
            print("No data available for this station in 2023")

    except Exception as e:
        print(f"Error fetching data: {str(e)}")


def main():
    # Test with Budapest coordinates
    lat, lon = 47.4979, 19.0402
    print(f"Searching for stations near Budapest ({lat}, {lon})...")

    stations = find_nearest_stations(lat, lon)
    if not stations.empty:
        # Test data availability for closest station
        closest_station = stations.iloc[0]
        station_id = closest_station.name  # Using index as station ID
        test_station_data(station_id)


if __name__ == "__main__":
    main()
