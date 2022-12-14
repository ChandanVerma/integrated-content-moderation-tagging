import os
import sys
import torch
import pandas as pd
import numpy as np
import clip
import traceback
import logging

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))
from PIL import Image
from scipy import stats
from src.utils.coop_helper_functions import load_coop_model
from src.utils.categories_sub_cat_desc import (
    coop_classnames,
    model_class_index_to_eval_category,
    eval_category_to_lomotif_pri_category,
    primary_category_dict,
    primary_category_dict_inv,
    sec_cat_descriptions_dict,
    sec_to_pri_cat,
)


def aggregate_frame_predictions(
    probs,
    framethresh,
    eval_category_to_lomotif_pri_category,
    model_class_index_to_eval_category,
    lomotiftopk,
    frametopk,
):
    """_summary_

    Args:
        probs (np.ndarray): array of softmax probabilities for each key frame
        framethresh (float): minimum probability threshold to accept the prediction
        eval_category_to_lomotif_pri_category (dict): mapping from model category names \
         to primary category names
        model_class_index_to_eval_category (dict): mapping from model class index \ 
        to model category names
        lomotiftopk (int): top k predictions to retain on a lomotif level
        frametopk (int): top k predictions to retain on a frame level

    Returns:
        list: predicted primary categories
    """

    top_label_idxs = np.argsort(probs, axis=-1)[:, -frametopk:].reshape([-1])
    top_label_probs = np.sort(probs, axis=-1)[:, -frametopk:].reshape([-1])
    subset = top_label_idxs[top_label_probs > framethresh]
    subset_probs = top_label_probs[top_label_probs > framethresh]
    subset_labels = np.array([model_class_index_to_eval_category[x] for x in subset])

    subset_dict = {
        u: np.sum(subset_probs[subset_labels == u]) for u in list(set(subset_labels))
    }
    subset_dict = dict(sorted(subset_dict.items(), key=lambda item: item[1]))
    possible_labels = list(subset_dict.keys())[::-1]

    possible_pris = []
    top_labels = []
    for i in range(len(possible_labels)):
        if (
            eval_category_to_lomotif_pri_category[possible_labels[i]]
            not in possible_pris
        ):
            possible_pris.append(
                eval_category_to_lomotif_pri_category[possible_labels[i]]
            )
        top_labels.append(possible_labels[i])

        if len(possible_pris) >= 3:
            break

    if len(top_labels) == 0:
        # fall back strategy for primary category prediction
        max_prob_index = np.argmax(probs, axis=-1)
        max_prob = np.max(probs, axis=-1)
        preds = np.array(
            [model_class_index_to_eval_category[x] for x in max_prob_index]
        )
        uniq, count = np.unique(preds, return_counts=True)
        count_dict = {uniq[i]: count[i] for i in range(len(uniq))}
        count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]))
        mode_count = list(count_dict.values())[::-1][0:lomotiftopk]
        mode_labels = {k: v for k, v in count_dict.items() if v in mode_count}
        mode_labels = dict(sorted(mode_labels.items(), key=lambda item: item[1]))
        mode_labels = list(mode_labels.keys())[::-1]

        return mode_labels

    else:
        return top_labels


class ContentTagAndModerate:
    def __init__(self, use_gpu=True, model_directory="./models"):
        """Classifies lomotif into top-2 primary and secondary categories.

        Args:
            use_gpu (bool, optional): to use GPU or not. Defaults to True.
            model_directory (str, optional): _description_. Defaults to "./models".
        """

        self.logger = logging.getLogger("ray")
        try:
            self.logger.setLevel("INFO")

            self.coop_classnames = coop_classnames
            self.model_class_index_to_eval_category = model_class_index_to_eval_category
            self.eval_category_to_lomotif_pri_category = (
                eval_category_to_lomotif_pri_category
            )
            self.primary_category_dict = primary_category_dict
            self.primary_category_dict_inv = primary_category_dict_inv
            self.sec_cat_descriptions_dict = sec_cat_descriptions_dict
            self.sec_to_pri_cat = sec_to_pri_cat
            self.framethresh = 0.05
            self.frametopk = 3
            self.topkpri = 2
            self.topksec = 2
            self.model_filename = "model.pth.tar-50"

            if use_gpu and not torch.cuda.is_available():
                self.logger.error("CUDA device not available.")
                assert False

            if use_gpu:
                self.device = "cuda"
            else:
                self.device = "cpu"

            try:
                self.logger.info(f"Loading CoOp model...")
                self.coop_model = load_coop_model(
                    classnames=self.coop_classnames,
                    device=self.device,
                    model_path=os.path.join(
                        model_directory, "prompt_learner", self.model_filename
                    ),
                )
                self.logger.info(f"Loaded CoOp model on {self.device}")
            except:
                self.logger.error(
                    "Error loading CoOp model. \n Traceback: \n{}".format(
                        traceback.format_exc()
                    )
                )
                assert False

            try:
                self.logger.info(f"Loading CLIP model...")
                self.clip_model, self.preprocess = clip.load(
                    "ViT-B/32",
                    device=self.device,
                    download_root=os.path.join(model_directory, "clip"),
                )
                self.logger.info(f"Loaded CLIP model on {self.device}")
            except:
                self.logger.error(
                    "Error loading CLIP model. \n Traceback: \n{}".format(
                        traceback.format_exc()
                    )
                )
                assert False

        except Exception as e:
            self.logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def reset(self):
        pass

    def generate_coop_predictions(self, image_list, lomotif_id):
        """Get primary category predictions from CoOp

        Args:
            image_list (list): list of key frames
            lomotif_id (string):lomotif unique ID

        Returns:
            tuple: (list of probability scores, list of predicted primary categories)
        """
        try:
            self.logger.info("[{}] Generating CoOp predictions.".format(lomotif_id))
            probs_list = []
            torch.cuda.empty_cache()
            for image in image_list:
                image_torch = (
                    self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    result = self.coop_model(image_torch)
                    image_probs = result.softmax(dim=-1).detach().cpu().numpy()
                probs_list.append(image_probs.tolist())

            self.logger.info("[{}] CoOp predictions generated.".format(lomotif_id))
            probs = np.concatenate(probs_list, axis=0)

            # primary category prediction
            pred_pri_cat = aggregate_frame_predictions(
                probs,
                framethresh=self.framethresh,
                eval_category_to_lomotif_pri_category=self.eval_category_to_lomotif_pri_category,
                model_class_index_to_eval_category=self.model_class_index_to_eval_category,
                lomotiftopk=self.topkpri,
                frametopk=self.frametopk,
            )

            return probs_list, pred_pri_cat

        except Exception as e:
            self.logger.error(
                "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                    lomotif_id, str(e), traceback.format_exc()
                )
            )

    def generate_clip_predictions(self, image_list, pred_pri_cat, title, lomotif_id):
        """Get final secondary and primary category predictions.

        Args:
            image_list (list): list of key frames
            pred_pri_cat (list): list of predicted primary categories
            title (string): title of music used in lomotif
            lomotif_id (string):lomotif unique ID

        Returns:
            tuple: (top 2 primary category predictions, \
                    top 2 secondary category predictions, \
                    3rd primary category prediction to consider, \
                    3rd seconday category prediction to consider)
        """
        try:
            # get descriptions for secondary category prediction
            possib_sec_cat = [
                k
                for k, v in self.primary_category_dict_inv.items()
                if v in pred_pri_cat
            ]
            possib_pri_cat = [
                v
                for k, v in self.primary_category_dict_inv.items()
                if v in pred_pri_cat
            ]

            if "selfies" in pred_pri_cat:
                if "make-up" not in possib_sec_cat:
                    possib_sec_cat.append("make-up")
                    possib_pri_cat.append("beauty-and-grooming")
                if "fashion" not in possib_sec_cat:
                    possib_sec_cat.append("fashion")
                    possib_pri_cat.append("life-style")

            if "life-style" in pred_pri_cat and "party" not in possib_pri_cat:
                possib_pri_cat.append("entertainment")
                possib_sec_cat.append("party")

            possib_sec_cat_desc = [
                self.sec_cat_descriptions_dict[x] for x in possib_sec_cat
            ]

            # clip predictions
            self.logger.info("[{}] Generating CLIP predictions.".format(lomotif_id))
            torch.cuda.empty_cache()
            sec_probs_list = []
            text = clip.tokenize(possib_sec_cat_desc).to(self.device)
            for image in image_list:
                image_torch = (
                    self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    logits_per_image, _ = self.clip_model(image_torch, text)
                    image_sec_probs = (
                        logits_per_image.softmax(dim=-1).detach().cpu().numpy()
                    )

                sec_probs_list.append(image_sec_probs)
            self.logger.info("[{}] CLIP predictions generated.".format(lomotif_id))

            sec_probs = np.concatenate(sec_probs_list, axis=0)
            sec_probs = np.sum(sec_probs, axis=0)
            # self.logger.info({possib_sec_cat[i]:sec_probs[i] for i in range(len(sec_probs))})
            indices = np.argsort(sec_probs)[::-1]
            pred_sec_cat_2 = list(np.array(possib_sec_cat)[indices])
            pred_pri_cat_2 = list(np.array(possib_pri_cat)[indices])

            # hierarchical ensembling
            pred_pri_cat = pred_pri_cat[0 : self.topkpri]
            pred_pri_cat_1 = []
            pred_sec_cat_1 = []
            for cat in pred_pri_cat:
                if len([x for x in pred_pri_cat_2 if x == cat]) != 0:
                    indices = np.where(np.array(pred_pri_cat_2) == cat)[0]
                    pred_sec_cat_1.append(pred_sec_cat_2[indices[0]])
                    pred_pri_cat_1.append(cat)

                    if len(pred_sec_cat_1) >= self.topksec:
                        break
                    pred_sec_cat_2.pop(indices[0])
                    pred_pri_cat_2.pop(indices[0])

            # fall back strategy
            if len(pred_sec_cat_1) == 0:
                pred_pri_cat_1 = pred_pri_cat_2[0 : self.topkpri]
                pred_sec_cat_1 = pred_sec_cat_2[0 : self.topksec]

            # final mapping
            pred_pri_cat_1 = [
                self.sec_to_pri_cat[x] if x in self.sec_to_pri_cat.keys() else x
                for x in pred_pri_cat_1
            ]

            # further business logic
            if title is not None:
                if title.find("birthday") != -1 and "birthday" not in pred_sec_cat_1:
                    pred_sec_cat_1 = pred_sec_cat_1[0:1]
                    pred_pri_cat_1 = pred_pri_cat_1[0:1]
                    pred_sec_cat_1.append("birthday")
                    pred_sec_cat_1 = pred_sec_cat_1[::-1]
                    pred_pri_cat_1.append("celebration-and-wishes")
                    pred_pri_cat_1 = pred_pri_cat_1[::-1]
            if (
                "music" in pred_pri_cat_1
                and "performance" in pred_sec_cat_1
                and "music-performance" not in pred_sec_cat_1
            ):
                pred_sec_cat_1 = [
                    "music-performance" if x == "performance" else x
                    for x in pred_sec_cat_1
                ]
            if "patriotic" in pred_sec_cat_1 and "national-asset" not in pred_sec_cat_1:
                pred_pri_cat_1.append("national-asset")

            # to output the additional secondary category and \
            # its corresponding primary categpory
            other_sec_cats = [x for x in pred_sec_cat_2 if x not in pred_sec_cat_1]
            if len(other_sec_cats) == 0:
                top_nplusone_sec = []
                top_nplusone_pri = []
            else:
                top_nplusone_sec = other_sec_cats[0]
                top_nplusone_pri = [self.sec_to_pri_cat[top_nplusone_sec]]
                top_nplusone_sec = [top_nplusone_sec]

            # ensure pri has no duplicates
            unique_pred_pri_cat_1 = set(pred_pri_cat_1)
            if len(pred_pri_cat_1) > len(unique_pred_pri_cat_1):
                pred_pri_cat_1 = np.array(pred_pri_cat_1)
                indices = np.sort(
                    [np.where(pred_pri_cat_1 == x)[0][0] for x in unique_pred_pri_cat_1]
                )
                pred_pri_cat_1 = list(pred_pri_cat_1[indices])

            return pred_pri_cat_1, pred_sec_cat_1, top_nplusone_pri, top_nplusone_sec

        except Exception as e:
            self.logger.error(
                "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                    lomotif_id, str(e), traceback.format_exc()
                )
            )

    def run_service_with_key_frames(self, key_frames_list, kinesis_event):
        """Determine primary and secondary categories using key frames.

        Args:
            key_frames_list (list): list of np.ndarray which are key frames
            kinesis_event (dict): event payload

        Returns:
            tuple: (dict of model outputs, dict of logits)
        """
        output_dict = {}
        lomotif_id = kinesis_event["lomotif"]["id"]
        output_dict["LOMOTIF_ID"] = lomotif_id
        start_time = str(pd.Timestamp.utcnow())
        logits_dict = {}
        logits_dict["lomotif_id"] = lomotif_id

        try:
            self.reset()

            # retrieve audio information
            try:
                if "data" in kinesis_event["lomotif"].keys():
                    if "audio" in kinesis_event["lomotif"]["data"].keys():
                        audio_info = kinesis_event["lomotif"]["data"]["audio"]
                        if audio_info is not None:
                            if len(audio_info) != 0:
                                if "title" in audio_info[0].keys():
                                    title = audio_info[0]["title"]
                                    title = title.lower()
                                else:
                                    title = None
                            else:
                                 title = None
                        else:
                            title = None
                    else:
                        title = None
                else:
                    title = None
            except Exception as e:
                self.logger.warning(
                    "[{}] Lomotif audio title not handled by logic and default to None: {}. \nTraceback: {}".format(
                        lomotif_id, str(e), traceback.format_exc()
                    )
                )
                title = None

            start_time = str(pd.Timestamp.utcnow())
            output_dict["COOP_PROCESS_START_TIME"] = start_time
            logits, pred_pri_cat = self.generate_coop_predictions(
                key_frames_list, lomotif_id
            )
            pred_time = str(pd.Timestamp.utcnow())
            output_dict["COOP_PREDICTION_TIME"] = pred_time
            output_dict["COOP_PREDICTION_SUCCESS"] = True
            output_dict["COOP_STATUS"] = 0

            start_time = str(pd.Timestamp.utcnow())
            output_dict["CLIP_PROCESS_START_TIME"] = start_time
            (
                pred_pri_cat_1,
                pred_sec_cat_1,
                top_nplusone_pri,
                top_nplusone_sec,
            ) = self.generate_clip_predictions(
                key_frames_list, pred_pri_cat, title, lomotif_id
            )
            pred_time = str(pd.Timestamp.utcnow())

            # pred_pri_cat_1 = [
            #     "violence" if x in special_cases else x for x in pred_pri_cat_1
            # ]

            output_dict["CLIP_PREDICTION_TIME"] = pred_time
            output_dict["CLIP_PREDICTION_SUCCESS"] = True
            output_dict["CLIP_STATUS"] = 0

            output_dict["PREDICTED_PRIMARY_CATEGORY"] = ", ".join(pred_pri_cat_1)
            output_dict["PREDICTED_SECONDARY_CATEGORY"] = ", ".join(pred_sec_cat_1)

            output_dict["PREDICTED_TOP3_PRIMARY_CATEGORY"] = ", ".join(
                pred_pri_cat_1
                + [x for x in top_nplusone_pri if x not in pred_pri_cat_1]
            )
            output_dict["PREDICTED_TOP3_SECONDARY_CATEGORY"] = ", ".join(
                pred_sec_cat_1
                + [x for x in top_nplusone_sec if x not in pred_sec_cat_1]
            )

            if_mod_cat_present = len(
                [
                    x
                    for x in pred_pri_cat_1
                    if x
                    in [
                        "violence",
                        "national-asset",
                        "problematic-content",
                        "harmful-acts",
                        "politics",
                        "branding"
                    ]
                ]
            )

            if if_mod_cat_present:
                output_dict["CLIP_TO_BE_MODERATED"] = True
            else:
                output_dict["CLIP_TO_BE_MODERATED"] = False

            logits_dict["logits"] = logits

        except Exception as e:
            logits_dict["logits"] = []
            output_dict["CLIP_PREDICTION_TIME"] = str(pd.Timestamp.utcnow())
            output_dict["COOP_PREDICTION_TIME"] = str(pd.Timestamp.utcnow())
            output_dict["CLIP_TO_BE_MODERATED"] = True
            output_dict["CLIP_PREDICTION_SUCCESS"] = False
            output_dict["CLIP_STATUS"] = 4
            output_dict["COOP_PREDICTION_SUCCESS"] = False
            output_dict["COOP_STATUS"] = 4
            output_dict["PREDICTED_PRIMARY_CATEGORY"] = ""
            output_dict["PREDICTED_SECONDARY_CATEGORY"] = ""
            output_dict["PREDICTED_TOP3_PRIMARY_CATEGORY"] = ""
            output_dict["PREDICTED_TOP3_SECONDARY_CATEGORY"] = ""

            self.logger.error(
                "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                    lomotif_id, str(e), traceback.format_exc()
                )
            )

        return output_dict, logits_dict
