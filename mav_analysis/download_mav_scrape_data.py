# %%
from config import dir_config
import pandas as pd
import datetime
import pytz

# declaring
FILENAMES = [
    "parsed_train_shapes",
    "parsed_train_schedules",
    "spot_train_locations",
]

DATETIME_NOW = datetime.datetime.now(pytz.utc)


# %%
def download_and_save_scraped_data(filename):
    df = pd.read_parquet(
        f"https://raw.githubusercontent.com/jrysztv/MAV-scraper/main/mav_scraper/data/parquet_store/{filename}.parquet"
    )
    df["download_date"] = DATETIME_NOW
    df.to_parquet(dir_config.mav_scrape_dir / f"{filename}.parquet")


def download_and_save_all_scraped_mav_data():
    for filename in FILENAMES:
        download_and_save_scraped_data(filename)


# %%
if __name__ == "__main__":
    download_and_save_all_scraped_mav_data()
    # %%
