"""
Ref: https://github.com/notAI-tech/NudeNet/blob/v2/nudenet/lite_classifier.py
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
import traceback
import logging

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim
from src.utils.data_processing import (
    process_clip,
    get_interest_frames_from_video,
    get_mime,
)


class NsfwDetect:
    def __init__(self, model_path="./models/classifier_lite.onnx", use_gpu=False):
        """Runs nudity detect on key frames.

        Args:
            model_path (str, optional): Nudenet model checkpoint. Defaults to "./models/classifier_lite.onnx".
            use_gpu (bool, optional): To use GPU or not. Defaults to False.
        """
        self.logger = logging.getLogger("ray")
        try:
            self.logger.setLevel("INFO")
            self.logger.info("NSFW engine starting.")
            try:
                assert os.path.exists(model_path)
                self.model = cv2.dnn.readNet(model_path)
                if use_gpu:
                    # this is only available when opencv is built from source
                    self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

                self.size = (256, 256)
                self.sim_thresh = 0.3  # 0.6 if metric is nrmse
                self.unsafe_threshold = 0.85
                self.skip_seconds = 1  # reads 1 image frame per skip_seconds seconds, if clip is a video
                self.logger.info("Nudenet model loaded")
            except AssertionError:
                self.logger.error("model_path does not exist.")
                assert False  # hard quit the app
        except Exception as e:
            self.logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def reset(self):
        self.mime = None
        self.frames_rgb = None
        self.selected_frame_indices = None
        self.selected_read_indices = None
        self.num_frames = None
        self.fps = None

    def classify_image(self, img_rgb_processed):
        """Nudenet classification for 1 image.

        Args:
            img_rgb_processed ([np.ndarray]): image array, channel-last.

        Returns:
            [tuple]: (dict of unsafe and safe probabilities, and whether or not \
            it should be moderated (bool))
        """
        loaded_images = np.rollaxis(img_rgb_processed, 3, 1)
        self.model.setInput(loaded_images)
        pred = self.model.forward()
        clip_to_be_moderated = pred[0][0] > self.unsafe_threshold
        return {"unsafe": pred[0][0], "safe": pred[0][1]}, clip_to_be_moderated

    def classify_clip_with_key_frames(self, key_frames, clip_path, kinesis_event):
        """Runs nudity detect on key frames

        Args:
            key_frames (list): list of np.ndarray which are key frames
            clip_path (string): file path to downloaded lomotif
            kinesis_event (dict): event payload

        Returns:
            dict: output from nudity detection model
        """
        lomotif_id = kinesis_event["lomotif"]["id"]

        output_dict = {}
        output_dict["LOMOTIF_ID"] = lomotif_id
        start_time = str(pd.Timestamp.utcnow())
        output_dict["NN_PROCESS_START_TIME"] = start_time

        try:
            self.reset()
            self.logger.debug("[{}] Reading clip: {}".format(lomotif_id, clip_path))

            to_be_moderated = False
            self.logger.debug("[{}] Preparing clip for model.".format(lomotif_id))
            images, img_paths = process_clip(
                frames_rgb=key_frames,
                clip_path=clip_path,
                size=self.size,
                frame_indices=np.arange(len(key_frames)),
            )

            self.logger.debug("[{}] Moderating.".format(lomotif_id))

            safe_scores = []
            unsafe_scores = []
            for i in range(len(img_paths)):
                clip_dict, clip_to_be_moderated = self.classify_image(
                    img_rgb_processed=images[[i]]
                )
                safe_scores.append(round(clip_dict["safe"], 5))
                unsafe_scores.append(round(clip_dict["unsafe"], 5))
                if clip_to_be_moderated:
                    to_be_moderated = True

            if len(unsafe_scores) == 0:
                # handle when no key frames are selected
                # due to unforeseen errors
                to_be_moderated = True

            pred_time = str(pd.Timestamp.utcnow())
            output_dict["NN_PREDICTION_TIME"] = pred_time
            output_dict["NN_SAFE_SCORES"] = ", ".join([str(x) for x in safe_scores])
            output_dict["NN_UNSAFE_SCORES"] = ", ".join([str(x) for x in unsafe_scores])
            output_dict["NN_TO_BE_MODERATED"] = to_be_moderated
            output_dict["NN_PREDICTION_SUCCESS"] = True
            output_dict["NN_STATUS"] = 0

        except Exception as e:
            output_dict["NN_PREDICTION_TIME"] = str(pd.Timestamp.utcnow())
            output_dict["NN_SAFE_SCORES"] = ""
            output_dict["NN_UNSAFE_SCORES"] = ""
            output_dict["NN_TO_BE_MODERATED"] = True
            output_dict["NN_PREDICTION_SUCCESS"] = False
            output_dict["NN_STATUS"] = 4
            self.logger.error(
                "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                    lomotif_id, str(e), traceback.format_exc()
                )
            )

        return output_dict
