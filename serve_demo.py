import os
import sys
import requests
import logging
import json
import pickle
import traceback
import tracemalloc

tracemalloc.start()

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

# loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray")
logger.setLevel("INFO")

snapshot1 = tracemalloc.take_snapshot()

if __name__ == "__main__":
    # for _ in range(1):
    i = 0
    kinesis_event = pickle.load(
        open("./test/test_data/sample_event_{}.pkl".format(i), "rb")
    )
    kinesis_event["lomotif"]["data"] = json.loads(kinesis_event["lomotif"]["data"])
    video = kinesis_event["lomotif"]["video"]
    video_tup = os.path.splitext(video)
    new_video = "".join([video_tup[0], "-vs", video_tup[-1]])
    kinesis_event["lomotif"]["video"] = new_video

    logger.info("Reading sample data.")
    try:

        resp = requests.get(
            "http://0.0.0.0:8000/composed", json=kinesis_event, timeout=60 * 10
        )
        if resp.status_code == 200:
            output = resp.json()
            # logger.info(
            #     "Results to save to snowflake table: \n {}".format(
            #         output["snowflake_outputs"]
            #     )
            # )
            # logger.info(
            #     "Results to save toS3 bucket (no snowflake needed): \n {}".format(
            #         output["raw_outputs"]
            #     )
            # )
            logger.info("Rayserve tasks successful. Output: {}".format(output))
            snapshot2 = tracemalloc.take_snapshot()

            top_stats = snapshot2.compare_to(snapshot1, "lineno")

            print("[ Top 10 differences ]")
            for stat in top_stats[:10]:
                print(stat)

            top_stats = snapshot2.statistics("traceback")

            # pick the biggest memory block
            stat = top_stats[0]
            print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
            for line in stat.traceback.format():
                print(line)

        else:
            logger.error(
                "Error in rayserve tasks. Status code: {} \nTraceback: {}".format(
                    resp.status_code, resp.text
                )
            )
    except:
        assert False, logger.error(
            "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                kinesis_event["lomotif"]["id"], traceback.format_exc()
            )
        )
