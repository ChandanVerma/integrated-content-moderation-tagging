"""bash
pip install pytest
python -m pytest test_integrated_model.py
"""
import os
import sys

sys.path.append("../")
import pickle
import shutil
from src.integrate_nn_clip import Moderation


def test_clip():
    kinesis_event = pickle.load(open("./test_data/sample_kinesis_event_v2.pkl", "rb"))
    test_outputs_clip = pickle.load(open("./test_data/test_outputs_clip_v2.pkl", "rb"))
    moderation_service = Moderation(
        nn_model_path="../models/classifier_lite.onnx", nn_use_gpu=False
    )
    video_path = "./test_data/f99e6e43-2c98-471f-8ff5-4df75f2718d2/f99e6e432c98471f8ff54df75f2718d2-20211101-2150-video-vs-compressed.mp4"

    clip_results = moderation_service.get_clip_predictions(
        clip_path=video_path, kinesis_event=kinesis_event
    )
    for k, v in clip_results.items():
        assert test_outputs_clip[k] == v, print(
            "Mismatch at: {}. Current result: {}, but should be {}.".format(
                k, v, test_outputs_clip[k]
            )
        )


def test_nudenet():
    kinesis_event = pickle.load(open("./test_data/sample_kinesis_event_v2.pkl", "rb"))
    test_outputs_nn = pickle.load(open("./test_data/test_outputs_nudenet_v2.pkl", "rb"))
    moderation_service = Moderation(
        nn_model_path="../models/classifier_lite.onnx", nn_use_gpu=False
    )
    video_path = "./test_data/f99e6e43-2c98-471f-8ff5-4df75f2718d2/f99e6e432c98471f8ff54df75f2718d2-20211101-2150-video-vs-compressed.mp4"

    nn_results = moderation_service.get_nn_predictions(
        clip_path=video_path, kinesis_event=kinesis_event
    )
    for k, v in nn_results.items():
        if k.find("TIME") == -1:
            assert test_outputs_nn[k] == v, print(
                "Mismatch at: {}. Current result: {}, but should be {}.".format(
                    k, v, test_outputs_nn[k]
                )
            )


if __name__ == "__main__":
    test_clip()
    test_nudenet()
