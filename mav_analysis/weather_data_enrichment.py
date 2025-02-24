from __future__ import annotations

import asyncio
import datetime
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import httpx
import pandas as pd
import structlog
from inputimeout import inputimeout, TimeoutOccurred
from pydantic import BaseModel, Field, validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import dir_config

logger = structlog.get_logger()


class WeatherParams(BaseModel):
    """Pydantic model for weather API parameters validation."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    start_date: str
    end_date: str
    daily: List[str]
    timezone: str = "Europe/Budapest"

    @validator("start_date", "end_date")
    def validate_date(cls, v):
        try:
            datetime.datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")


@dataclass
class WeatherData:
    """Data class for weather information."""

    max_temp: Optional[float] = None
    min_temp: Optional[float] = None
    precipitation_sum: Optional[float] = None
    rain_sum: Optional[float] = None
    snowfall_sum: Optional[float] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> Dict[str, WeatherData]:
        """Create WeatherData instances from API response, one per date."""
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        result = {}

        for i, date in enumerate(dates):
            result[date] = cls(
                max_temp=daily.get("temperature_2m_max", [None] * len(dates))[i],
                min_temp=daily.get("temperature_2m_min", [None] * len(dates))[i],
                precipitation_sum=daily.get("precipitation_sum", [None] * len(dates))[
                    i
                ],
                rain_sum=daily.get("rain_sum", [None] * len(dates))[i],
                snowfall_sum=daily.get("snowfall_sum", [None] * len(dates))[i],
            )

        return result


class WeatherEnricher:
    """Class to handle weather data enrichment for location data."""

    def __init__(self, max_concurrent_requests: int = 10, auto_confirm: bool = False):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.auto_confirm = auto_confirm
        self.daily_params = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
        ]

    def estimate_requests(self, df: pd.DataFrame) -> dict:
        """Calculate the number of API requests needed and potential reduction."""
        df_valid = df.dropna(subset=["stop_lat", "stop_lon"])
        unique_locations = df_valid.drop_duplicates(subset=["stop_lat", "stop_lon"])
        location_date_pairs = df_valid.drop_duplicates(
            subset=["stop_lat", "stop_lon", "date"]
        )

        return {
            "unique_locations": len(unique_locations),
            "location_date_pairs": len(location_date_pairs),
            "reduction_percent": round(
                (1 - len(unique_locations) / len(location_date_pairs)) * 100, 2
            ),
        }

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
    )
    async def _fetch_weather(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> Optional[Dict[str, WeatherData]]:
        """
        Fetch weather data for a specific location and date range with retry logic.
        """
        try:
            params = WeatherParams(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
                daily=self.daily_params,
            )

            async with self.semaphore:
                response = await client.get(self.base_url, params=params.model_dump())
                response.raise_for_status()
                return WeatherData.from_api_response(response.json())

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.error(
                "Error fetching weather data",
                error=str(e),
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error",
                error=str(e),
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
            )
            return None

    async def process_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all locations asynchronously and enrich with weather data.
        """
        # Filter valid coordinates
        df_valid = df.dropna(subset=["stop_lat", "stop_lon"])

        # Get date range
        min_date = df_valid["date"].min().strftime("%Y-%m-%d")
        max_date = df_valid["date"].max().strftime("%Y-%m-%d")

        # Get unique locations
        unique_locations = df_valid.drop_duplicates(subset=["stop_lat", "stop_lon"])

        # Calculate and show request estimation
        est = self.estimate_requests(df)
        logger.info(
            "Weather data enrichment estimation",
            unique_locations=est["unique_locations"],
            location_date_pairs=est["location_date_pairs"],
            reduction_percent=est["reduction_percent"],
            date_range=f"{min_date} to {max_date}",
        )

        # Get user confirmation in interactive mode
        if not self.auto_confirm and sys.stdin.isatty():
            try:
                resp = inputimeout(
                    f"\nWill make {est['unique_locations']} API requests "
                    f"(reduced from {est['location_date_pairs']}, "
                    f"{est['reduction_percent']}% reduction)\n"
                    "Continue? [y/N] (15s timeout): ",
                    timeout=30,
                )
                if not resp.lower().startswith("y"):
                    raise ValueError("Aborted by user")
            except TimeoutOccurred:
                logger.info("Input timeout reached, continuing automatically")
            except Exception as e:
                logger.error("User confirmation failed", error=str(e))
                raise

        weather_data_dict = {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                self._fetch_weather(
                    client,
                    row["stop_lat"],
                    row["stop_lon"],
                    min_date,
                    max_date,
                )
                for _, row in unique_locations.iterrows()
            ]

            weather_data_list = await asyncio.gather(*tasks)

            # Create a mapping of (lat, lon, date) -> WeatherData
            for (_, loc), weather_dict in zip(
                unique_locations.iterrows(), weather_data_list
            ):
                if weather_dict:
                    for date, weather in weather_dict.items():
                        weather_data_dict[(loc["stop_lat"], loc["stop_lon"], date)] = (
                            weather
                        )

        # Create a list of weather data records
        weather_records = []
        for _, row in df_valid.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            key = (row["stop_lat"], row["stop_lon"], date_str)
            weather = weather_data_dict.get(key, WeatherData())
            weather_records.append(
                {
                    "stop_lat": row["stop_lat"],
                    "stop_lon": row["stop_lon"],
                    "date": row["date"],
                    "max_temp": weather.max_temp,
                    "min_temp": weather.min_temp,
                    "precipitation_sum": weather.precipitation_sum,
                    "rain_sum": weather.rain_sum,
                    "snowfall_sum": weather.snowfall_sum,
                }
            )

        # Create weather DataFrame and merge with original
        weather_df = pd.DataFrame(weather_records)
        result_df = df.merge(
            weather_df,
            on=["stop_lat", "stop_lon", "date"],
            how="left",
        )

        logger.info("Weather data enrichment completed")
        return result_df


async def main():
    """Main function to run the weather enrichment process."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich vertex data with weather information"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically confirm and proceed with API requests",
    )
    args = parser.parse_args()

    # Read the data
    df = (
        pd.read_parquet(
            dir_config.analysis_data_dir / "daily_vertex_attributes.parquet"
        )
        .reset_index()
        .rename(
            columns={
                "incoming": "num_incoming_trains",
                "outgoing": "num_outgoing_trains",
                "sum_incoming_outgoing": "num_total_trains",
                "vertex ID": "vertex_id",
            }
        )
    )

    # Initialize and run weather enrichment
    enricher = WeatherEnricher(max_concurrent_requests=10, auto_confirm=args.yes)
    enriched_df = await enricher.process_locations(df)

    # Save enriched data
    enriched_df.to_parquet(
        dir_config.analysis_data_dir / "daily_vertex_attributes_with_weather.parquet"
    )


if __name__ == "__main__":
    asyncio.run(main())
