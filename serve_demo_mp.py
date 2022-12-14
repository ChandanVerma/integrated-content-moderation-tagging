import os
import sys
import requests
import logging
import json
import pickle
import traceback
import datetime
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

# loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray")
logger.setLevel("INFO")

df = pd.read_csv(
    "./test/test_data/v3_pytest_data.csv",
    usecols=[
        "ID",
        "CREATED",
        "VIDEO",
        "COUNTRY",
        "DATA",
        "PREDICTED_PRIMARY_CATEGORY",
        "PREDICTED_SECONDARY_CATEGORY",
    ],
)
df = df.dropna(subset=["PREDICTED_PRIMARY_CATEGORY", "PREDICTED_SECONDARY_CATEGORY"])
df = df.reset_index(drop=True)
df.columns = df.columns.str.lower()
records = df.to_dict("records")


from multiprocessing import Pool


def mp_send_req(i):
    kinesis_event = {}
    kinesis_event["lomotif"] = records[i]
    kinesis_event["lomotif"]["data"] = json.loads(kinesis_event["lomotif"]["data"])

    video = kinesis_event["lomotif"]["video"]
    resp = requests.get(
        "http://0.0.0.0:8000/composed", json=kinesis_event, timeout=60 * 10
    )
    if resp.status_code == 200:
        output = resp.json()
        print(i, "Success")
        print(output)
    else:
        print(i, "Failed")


if __name__ == "__main__":
    start = datetime.datetime.now()
    # mp_send_req(0)
    num = 40
    total = 40
    with Pool(processes=num) as pool:
        pool.starmap(mp_send_req, [(x,) for x in range(total)])
    end = datetime.datetime.now()
    print("time taken: ", ((end - start).total_seconds() / total))
