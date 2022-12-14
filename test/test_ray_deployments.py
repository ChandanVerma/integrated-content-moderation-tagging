import pandas as pd
import pickle
import json
import requests

df = pd.read_csv(
    "./test_data/v3_pytest_data.csv",
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

# def test_rayserve_samples_0():
#     kinesis_event = pickle.load(open("./test_data/sample_event_0.pkl", "rb"))
#     resp = requests.get(
#         "http://127.0.0.1:8000/composed", json=kinesis_event, timeout=60 * 30
#     )
#     assert resp.status_code == 200, print(resp.text)
#     outputs = resp.json()
#     test_outputs = pickle.load(open("./test_data/test_sample_event_0.pkl", "rb"))
#     for k, v in outputs.items():
#         if k.find("TIME") == -1:
#             if k == "CLIP_PREDICTED_PRIMARY_CATEGORY":
#                 assert v == test_outputs["PREDICTED_PRIMARY_CATEGORY"]
#             elif k == "CLIP_PREDICTED_SECONDARY_CATEGORY":
#                 assert v == test_outputs["PREDICTED_SECONDARY_CATEGORY"]
#             else:
#                 assert v == test_outputs[k]


def test_rayserve_from_df():
    # for i in range(len(records)):
    for i in range(0, 5):
        kinesis_event = {}
        kinesis_event["lomotif"] = records[i]
        kinesis_event["lomotif"]["data"] = json.loads(kinesis_event["lomotif"]["data"])
        resp = requests.get(
            "http://0.0.0.0:8000/composed", json=kinesis_event, timeout=30
        )
        assert resp.status_code == 200


# def test_rayserve_samples_2():
#     kinesis_event = pickle.load(open("./test_data/sample_event_2.pkl", 'rb'))
#     resp = requests.get(
#         "http://127.0.0.1:8000/composed", json=kinesis_event, timeout=60 * 30
#     )
#     assert resp.status_code == 200
#     outputs = resp.json()
#     test_outputs = pickle.load(open("./test_data/test_sample_event_2.pkl", 'rb'))
#     for k, v in outputs.items():
#         if k.find('TIME') == -1:
#             assert v == test_outputs[k]

# def test_rayserve_samples_3():
#     kinesis_event = pickle.load(open("./test_data/sample_event_3.pkl", 'rb'))
#     resp = requests.get(
#         "http://127.0.0.1:8000/composed", json=kinesis_event, timeout=60 * 30
#     )
#     assert resp.status_code == 200
#     outputs = resp.json()
#     test_outputs = pickle.load(open("./test_data/test_sample_event_3.pkl", 'rb'))
#     for k, v in outputs.items():
#         if k.find('TIME') == -1:
#             assert v == test_outputs[k]

# def test_rayserve_samples_4():
#     kinesis_event = pickle.load(open("./test_data/sample_event_4.pkl", 'rb'))
#     resp = requests.get(
#         "http://127.0.0.1:8000/composed", json=kinesis_event, timeout=60 * 30
#     )
#     assert resp.status_code == 200
#     outputs = resp.json()
#     test_outputs = pickle.load(open("./test_data/test_sample_event_4.pkl", 'rb'))
#     for k, v in outputs.items():
#         if k.find('TIME') == -1:
#             assert v == test_outputs[k]

if __name__ == "__main__":
    # test_rayserve_samples_0()
    test_rayserve_from_df()
    # test_rayserve_samples_1()
    # test_rayserve_samples_3()
    # test_rayserve_samples_4()
