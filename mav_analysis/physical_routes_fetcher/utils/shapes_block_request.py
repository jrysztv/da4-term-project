# %%
"""
This module provides functionality to fetch and process shape data from the MAV Start API.
Classes:
    ShapesRequest: Handles the request and processing of shape data.
Usage:
Classes:
    ShapesRequest:
        __init__(self, padding=10):
            Initializes the ShapesRequest object with default headers and JSON data.
        request_shape_repsonse(self):
            Sends a POST request to the MAV Start API and returns the shape data.
        extract_shape_dataframe(line):
            Converts a line's polyline data into a pandas DataFrame.
        request_and_extract(self):
            Requests shape data and processes it into a pandas DataFrame.
Note:
    This usage is deprecated. The shapes data is now fetched from the MAV Scraper repository.
    Reason: the MAV Start API does not provide quality shape data.
"""

import httpx
import pandas as pd
import polyline


class ShapesRequest:
    def __init__(self, padding=10):
        self.headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Content-Type": "application/json; charset=UTF-8",
            "DNT": "1",
            "Origin": "https://vonatinfo.mav-start.hu",
            "Referer": "https://vonatinfo.mav-start.hu/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }
        self.padding = padding
        self.json_data = {
            "a": "LINE",
            "jo": {
                "sw": [
                    45.7 - self.padding,
                    16.1 - self.padding,
                ],
                "ne": [
                    48.6 + self.padding,
                    22.9 + self.padding,
                ],
                "id": "bg",
                "hidden": True,
                "history": True,
                "zoom": 1,
            },
        }

    def request_shape_repsonse(self):
        response = httpx.post(
            "https://vonatinfo.mav-start.hu/map.aspx/getData",
            headers=self.headers,
            json=self.json_data,
        )
        return response.json()["d"]["result"]["lines"]

    @staticmethod
    def extract_shape_dataframe(line):
        shape_df = pd.DataFrame(polyline.decode(line["points"]))
        shape_df.columns = ["lat", "lon"]
        shape_df.insert(0, "line_id", line["linenum"])
        return shape_df

    def request_and_extract(self):
        lines = self.request_shape_repsonse()
        shape_df = pd.concat(list(map(self.extract_shape_dataframe, lines)))
        return shape_df


# %%
if __name__ == "__main__":
    shapes_request = ShapesRequest()
    shape_df = shapes_request.request_and_extract()
    print(shape_df.head())

# %%
