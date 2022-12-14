import os
import sys
import pandas as pd
import numpy as np
import logging
import traceback
import requests
import shutil
import time
import ray
import json
import boto3
import gc
import psutil

from ray import serve
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

from src.utils.download import LomotifDownloader
from src.utils.data_processing import get_mime, get_interest_frames_from_video
from src.utils.download_models import download_models_helper
from src.tag_and_moderate import ContentTagAndModerate
from src.tag_nudity import NsfwDetect
from src.tag_middle_finger import MiddleFingerDetect
from src.utils.generate_outputs import output_template

# from dotenv import load_dotenv

# load_dotenv("./.env")

# loggers
logger = logging.getLogger("ray")
logger.setLevel("INFO")

nn_use_gpu = False
if float(os.environ["ClipNumGPUPerReplica"]) != 0:
    clip_use_gpu = True
else:
    clip_use_gpu = False

# Set up env variables
AWS_ROLE_ARN = os.environ.get("AWS_ROLE_ARN")
AWS_WEB_IDENTITY_TOKEN_FILE = os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AiModelBucket = os.environ.get("AiModelBucket")

SnowflakeResultsQueue = os.environ["SnowflakeResultsQueue"]
RawResultsQueue = os.environ["RawResultsQueue"]


@serve.deployment(
    route_prefix="/download_lomotif",
    max_concurrent_queries=os.environ["DownloadMaxCon"],
    num_replicas=os.environ["DownloadNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["DownloadNumCPUPerReplica"]),
    },
)
class LomotifDownloaderServe:
    def __init__(
        self,
    ):
        """Rayserve module for downloading lomotifs with at most 5 retries \
        and backoff of 30 seconds in between tries.
        """
        try:
            self.downloader = LomotifDownloader(
                save_folder_directory="./downloaded_lomotifs"
            )
            self.num_retries = 5
            self.delay_between_retries = 30  # in seconds
            # logging.basicConfig(level=logging.INFO)
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, video_url, lomotif_id, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            video_url (string): URL to the lomotif
            lomotif_id (string): lomotif content ID
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            tuple: a 3-element tuple where the first element is a boolean variable that \
            is True if a lomotif has been successfully downloaded else False, second variable \
            is the filepath to the downloaded lomotif, and the RayObjectRef of the video bytes \ 
            is the third.
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        for retry_number in range(self.num_retries):
            logger.info(
                "[{}] Download retry {}/{}...".format(
                    lomotif_id, retry_number, self.num_retries
                )
            )
            result, save_file_name, save_file_name_ref = self.downloader.download(
                video_url=video_url, lomotif_id=lomotif_id
            )
            if result:
                end = time.time()
                logger.info(
                    "[{}] Download complete, filename: {}, duration: {}".format(
                        lomotif_id, save_file_name, end - start
                    )
                )
                break
            else:
                time.sleep(self.delay_between_retries)

        return result, save_file_name, save_file_name_ref


@serve.deployment(
    route_prefix="/process_lomotif",
    max_concurrent_queries=os.environ["PreprocessMaxCon"],
    num_replicas=os.environ["PreprocessNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["PreprocessNumCPUPerReplica"]),
    },
)
class PreprocessLomotifServe:
    def __init__(
        self,
    ):
        """Select key frames of a lomotif and collect some metadata like \
        FPS, number of total frames, and selected frame indices.
        """
        try:
            if not os.path.exists("./downloaded_lomotifs"):
                os.makedirs("./downloaded_lomotifs")
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, save_file_name, save_file_name_ref, lomotif_id, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            save_file_name (string): filepath to the downloaded lomotif
            save_file_name_ref (ObjectRef): ObjectRef of video bytes
            lomotif_id (string): lomotif content ID
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            tuple: Returns key frames and other metadata gathered.
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()

        with open(save_file_name, "wb") as f:
            f.write(save_file_name_ref)

        mime = get_mime(save_file_name)

        if mime in ["video", "image"]:
            (
                key_frames,
                # low_key_frames,
                fps,
                num_frames,
                selected_frame_indices,
            ) = get_interest_frames_from_video(save_file_name)

            logger.info("[{}] Key frames generated.".format(lomotif_id))
            end = time.time()
            logger.info(
                "[{}] Preprocess complete, save_file_name: {}, duration: {}".format(
                    lomotif_id, save_file_name, end - start
                )
            )
            len_key_frames = len(key_frames)
        else:
            mime = None
            len_key_frames = 0
            key_frames = []
            fps = -1
            num_frames = -1
            selected_frame_indices = []
            logger.warning(
                "[{}] File is not video or image. File not processed and defaults to to-be-moderated.".format(
                    lomotif_id
                )
            )

        os.remove(save_file_name)
        save_file_name_rewrite = (
            os.path.splitext(save_file_name)[0]
            + "-rewrite"
            + os.path.splitext(save_file_name)[1]
        )
        if os.path.exists(save_file_name_rewrite):
            os.remove(save_file_name_rewrite)

        key_frames = ray.put(key_frames)
        # low_key_frames = ray.put(low_key_frames)

        return (
            mime,
            key_frames,
            # low_key_frames,
            fps,
            num_frames,
            selected_frame_indices,
            len_key_frames,
        )


@serve.deployment(
    route_prefix="/tagging_with_coopclip",
    max_concurrent_queries=os.environ["ClipMaxCon"],
    num_replicas=os.environ["ClipNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["ClipNumCPUPerReplica"]),
        "num_gpus": float(os.environ["ClipNumGPUPerReplica"]),
    },
)
class ContentModTagCoopClipServe:
    def __init__(
        self,
    ):
        """Predicts primary and secondary categories of lomotif.
        """
        try:
            logger.info("Downloading coopclip models from S3...")
            download_models_helper(model_name='coopclip', root="./models")
            logger.info("All coopclip model files downloaded.")
            self.coopclip_model = ContentTagAndModerate(use_gpu=clip_use_gpu)
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, key_frames, save_file_name, kinesis_event, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            key_frames (ObjectRef): ObjectRef of list of arrays that are key frames
            save_file_name (string): filepath to the downloaded lomotif
            kinesis_event (dict): kinesis event payload
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            _type_: _description_
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        self.coopclip_model.reset()
        clip_results, logits_results = self.coopclip_model.run_service_with_key_frames(
            key_frames_list=key_frames,
            kinesis_event=kinesis_event,
        )
        end = time.time()
        logger.info(
            "[{}] ContentModTagClipServe complete, save_file_name: {}, duration: {}".format(
                kinesis_event["lomotif"]["id"], save_file_name, end - start
            )
        )
        return clip_results, logits_results


@serve.deployment(
    route_prefix="/moderation_with_nudenet",
    max_concurrent_queries=os.environ["NudenetMaxCon"],
    num_replicas=os.environ["NudenetNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["NudenetNumCPUPerReplica"]),
    },
)
class ContentModNudenetServe:
    def __init__(
        self,
    ):
        """Run nudity detection on key frames.
        """
        try:
            logger.info("Downloading nudenet models from S3...")
            download_models_helper(model_name='nudenet', root="./models")
            logger.info("All nudenet model files downloaded.")
            self.nudenet_model = NsfwDetect(
                model_path="./models/classifier_lite.onnx", use_gpu=nn_use_gpu
            )
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, key_frames, save_file_name, kinesis_event, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            key_frames (ObjectRef): ObjectRef of list of arrays that are key frames
            save_file_name (string): filepath to the downloaded lomotif
            kinesis_event (dict): kinesis event payload
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            dict: results of the model
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        self.nudenet_model.reset()
        nn_results = self.nudenet_model.classify_clip_with_key_frames(
            key_frames=key_frames,
            clip_path=save_file_name,
            kinesis_event=kinesis_event,
        )
        end = time.time()
        logger.info(
            "[{}] ContentModNudenetServe complete, save_file_name: {}, duration: {}".format(
                kinesis_event["lomotif"]["id"], save_file_name, end - start
            )
        )
        return nn_results


@serve.deployment(
    route_prefix="/moderation_with_middle_finger_detect",
    max_concurrent_queries=os.environ["MFDMaxCon"],
    num_replicas=os.environ["MFDNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["MFDNumCPUPerReplica"]),
    },
)
class ContentModMiddleFingerDetectServe:
    def __init__(
        self,
    ):
        """Run middle finger detection on key frames.
        """
        try:
            self.mfd_model = MiddleFingerDetect()
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, key_frames, save_file_name, kinesis_event, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            key_frames (ObjectRef): ObjectRef of list of arrays that are key frames
            save_file_name (string): filepath to the downloaded lomotif
            kinesis_event (dict): kinesis event payload
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            dict: results of the model
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        self.mfd_model.reset()
        mfd_results = self.mfd_model.classify_clip_with_key_frames(
            key_frames=key_frames,
            clip_path=save_file_name,
            kinesis_event=kinesis_event,
        )
        end = time.time()
        logger.info(
            "[{}] ContentModMiddleFingerDetectServe complete, save_file_name: {}, duration: {}".format(
                kinesis_event["lomotif"]["id"], save_file_name, end - start
            )
        )
        return mfd_results


@serve.deployment(
    route_prefix="/composed",
    max_concurrent_queries=os.environ["ComposedMaxCon"],
    num_replicas=os.environ["ComposedNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["ComposedNumCPUPerReplica"]),
    },
)
class ComposedModel:
    def __init__(self):
        """Composition of the whole pipeline starting from downloading lomotif to \
        generating all outputs and putting them to SQS.
        """
        try:
            self.download_engine = LomotifDownloaderServe.get_handle(sync=False)
            self.preprocess_engine = PreprocessLomotifServe.get_handle(sync=False)
            self.model_nudenet = ContentModNudenetServe.get_handle(sync=False)
            self.model_coopclip = ContentModTagCoopClipServe.get_handle(sync=False)
            self.model_mfd = ContentModMiddleFingerDetectServe.get_handle(sync=False)
            self.sqs_client = boto3.client("sqs")
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    async def __call__(self, starlette_request):
        """Rayserve __call__ definition.

        Args:
            starlette_request (bytes): incoming request payload

        Returns:
            dict: outputs that will be written to a snowflake table
        """
        vmem = psutil.virtual_memory().percent
        logger.info(
            "% Virtual memory used: {}, gc items: {}".format(vmem, gc.get_count())
        )
        vmem_clear = vmem > 50.0
        vmem_clear_ref = ray.put(vmem_clear)

        if vmem_clear:
            gc.collect()

        start = time.time()
        kinesis_event = await starlette_request.json()
        kinesis_event_ref = ray.put(kinesis_event)
        message_receive_time = str(pd.Timestamp.utcnow())
        logger.info("Message received: {}".format(kinesis_event["lomotif"]["id"]))
        video = kinesis_event["lomotif"]["video"]
        lomotif_id = kinesis_event["lomotif"]["id"]

        output_dict = output_template(
            kinesis_event,
            video,
            message_receive_time
        )

        download_result, save_file_name, save_file_name_ref = await (
            await self.download_engine.remote(video, lomotif_id, vmem_clear_ref)
        )

        if download_result:
            (
                mime,
                key_frames_ref,
                # low_key_frames_ref,
                fps,
                num_frames,
                selected_frame_indices,
                len_key_frames,
            ) = await (
                await self.preprocess_engine.remote(
                    save_file_name, save_file_name_ref, lomotif_id, vmem_clear_ref
                )
            )
            del save_file_name_ref

            if mime is not None:
                if len_key_frames == 0:
                    output_dict["NN_STATUS"] = 5
                    output_dict["CLIP_STATUS"] = 5
                    output_dict["COOP_STATUS"] = 5
                    output_dict["MFD_STATUS"] = 5
                    logits_dict = {}
                    logits_dict["lomotif_id"] = lomotif_id
                    logits_dict["logits"] = []
                    logger.info("[{}] No key frames generated.".format(lomotif_id))
                else:
                    output_dict["KEY_FRAMES"] = ", ".join(
                        [str(x) for x in selected_frame_indices]
                    )
                    output_dict["NUM_FRAMES"] = str(int(num_frames))
                    output_dict["FPS"] = str(fps)
                    logger.info(
                        "[{}] Sending requests to all models.".format(lomotif_id)
                    )
                    mfd_results_ref = await self.model_mfd.remote(
                        key_frames_ref,
                        save_file_name,
                        kinesis_event_ref,
                        vmem_clear_ref,
                    )
                    nn_results_ref = await self.model_nudenet.remote(
                        key_frames_ref,
                        save_file_name,
                        kinesis_event_ref,
                        vmem_clear_ref,
                    )
                    clip_results_ref = await self.model_coopclip.remote(
                        key_frames_ref,
                        save_file_name,
                        kinesis_event_ref,
                        vmem_clear_ref,
                    )
                    mfd_results = await (mfd_results_ref)
                    nn_results = await (nn_results_ref)
                    clip_results, logits_dict = await (clip_results_ref)
                    logger.info("[{}] Getting MFD results.".format(lomotif_id))
                    logger.info("[{}] Getting Nudenet results.".format(lomotif_id))
                    logger.info("[{}] Getting CLIP results.".format(lomotif_id))

                    logger.info("[{}] Aggregating results...".format(lomotif_id))
                    for k, v in mfd_results.items():
                        output_dict[k] = v
                    for k, v in nn_results.items():
                        output_dict[k] = v
                    for k, v in clip_results.items():
                        output_dict[k] = v
                    logger.info("[{}] Results aggregated.".format(lomotif_id))
            else:
                output_dict["NN_STATUS"] = 1
                output_dict["CLIP_STATUS"] = 1
                output_dict["COOP_STATUS"] = 1
                output_dict["MFD_STATUS"] = 1
                logits_dict = {}
                logits_dict["lomotif_id"] = lomotif_id
                logits_dict["logits"] = []
                logger.info("[{}] Mime is None.".format(lomotif_id))

            del key_frames_ref
            # del low_key_frames_ref

        else:
            output_dict["NN_STATUS"] = 403
            output_dict["CLIP_STATUS"] = 403
            output_dict["COOP_STATUS"] = 403
            output_dict["MFD_STATUS"] = 403
            logits_dict = {}
            logits_dict["lomotif_id"] = lomotif_id
            logits_dict["logits"] = []
            logger.info(
                "[{}] Lomotif file does not exist or download has failed.".format(
                    lomotif_id
                )
            )

        del vmem_clear_ref
        del kinesis_event_ref

        if (
            output_dict["NN_TO_BE_MODERATED"]
            or output_dict["CLIP_TO_BE_MODERATED"]
            or output_dict["MFD_TO_BE_MODERATED"]
        ):
            output_dict["TO_BE_MODERATED"] = True
        else:
            output_dict["TO_BE_MODERATED"] = False

        logger.info("[{}] {}".format(lomotif_id, output_dict))
        logits_dict["creation_time"] = kinesis_event["lomotif"]["created"]

        # restructure outputs, and considering legacy
        output_dict_v2 = {}
        for k in [
            "LOMOTIF_ID",
            "VIDEO",
            "COUNTRY",
            "CREATION_TIME",
            "MESSAGE_RECEIVE_TIME",
            "KEY_FRAMES",
            "NUM_FRAMES",
            "FPS",
            "NN_PROCESS_START_TIME",
            "NN_PREDICTION_TIME",
            "NN_SAFE_SCORES",
            "NN_UNSAFE_SCORES",
            "NN_TO_BE_MODERATED",
            "NN_PREDICTION_SUCCESS",
            "NN_STATUS",
            "CLIP_PROCESS_START_TIME",
            "CLIP_PREDICTION_TIME",
            "CLIP_PREDICTION_SUCCESS",
            "CLIP_TO_BE_MODERATED",
            "CLIP_STATUS",
            "PREDICTED_PRIMARY_CATEGORY",
            "PREDICTED_SECONDARY_CATEGORY",
            "TO_BE_MODERATED",
        ]:
            output_dict_v2[k] = output_dict[k]
        output_dict_v2["MODEL_ATTRIBUTES"] = {}
        for k in [
            "MODEL_VERSION",
            "NN_PROCESS_START_TIME",
            "NN_PREDICTION_TIME",
            "NN_SAFE_SCORES",
            "NN_UNSAFE_SCORES",
            "NN_TO_BE_MODERATED",
            "NN_PREDICTION_SUCCESS",
            "NN_STATUS",
            "CLIP_PROCESS_START_TIME",
            "CLIP_PREDICTION_TIME",
            "CLIP_PREDICTION_SUCCESS",
            "CLIP_TO_BE_MODERATED",
            "CLIP_STATUS",
            "COOP_PROCESS_START_TIME",
            "COOP_PREDICTION_TIME",
            "COOP_PREDICTION_SUCCESS",
            "COOP_STATUS",
            "PREDICTED_TOP3_PRIMARY_CATEGORY",
            "PREDICTED_TOP3_SECONDARY_CATEGORY",
            "MFD_PROCESS_START_TIME",
            "MFD_PREDICTION_TIME",
            "MFD_TO_BE_MODERATED",
            "MFD_PREDICTION_SUCCESS",
            "MFD_STATUS",
        ]:
            output_dict_v2["MODEL_ATTRIBUTES"][k] = output_dict[k]

        final_output = {}
        final_output["snowflake_outputs"] = output_dict_v2
        final_output["raw_outputs"] = logits_dict

        try:
            # Send message to SQS queue
            logger.info(
                "[{}] Attempting to send output to SQS: {}.".format(
                    lomotif_id,
                    os.environ["SnowflakeResultsQueue"]
                )
            )
            msg = json.dumps(final_output["snowflake_outputs"])
            response = self.sqs_client.send_message(
                QueueUrl=os.environ["SnowflakeResultsQueue"],
                DelaySeconds=0,
                MessageBody=msg,
            )
            logger.info(
                "[{}] Sent outputs to SQS: {}.".format(lomotif_id, os.environ["SnowflakeResultsQueue"])
            )

            logger.info(
                "[{}] Attempting to send output to SQS: {}.".format(
                    lomotif_id, os.environ["RawResultsQueue"]
                )
            )
            msg = json.dumps(final_output["raw_outputs"])
            response = self.sqs_client.send_message(
                QueueUrl=os.environ["RawResultsQueue"], DelaySeconds=0, MessageBody=msg
            )
            end = time.time()
            logger.info(
                "[{}] Sent raw outputs to SQS: {}.".format(lomotif_id, os.environ["RawResultsQueue"])
            )
            logger.info(
                "[{}] ComposedModel complete, save_file_name: {}, duration: {}".format(
                    lomotif_id, save_file_name, end - start
                )
            )

            # return {}
            return final_output["snowflake_outputs"]

        except Exception as e:
            key_frames_ref = None
            vmem_clear_ref = None
            kinesis_event_ref = None
            save_file_name_ref = None
            # low_key_frames_ref = None
            del save_file_name_ref
            del key_frames_ref
            # del low_key_frames_ref
            del vmem_clear_ref
            del kinesis_event_ref
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script


if __name__ == "__main__":
    env_vars = {
        "AWS_ROLE_ARN": AWS_ROLE_ARN,
        "AWS_WEB_IDENTITY_TOKEN_FILE": AWS_WEB_IDENTITY_TOKEN_FILE,
        "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AiModelBucket": AiModelBucket,
        "RawResultsQueue": RawResultsQueue,
        "SnowflakeResultsQueue": SnowflakeResultsQueue,
    }
    runtime_env = {"env_vars": {}}

    for key, value in env_vars.items():
        if value is not None:
            runtime_env["env_vars"][key] = value

    ray.init(address="auto", namespace="serve", runtime_env=runtime_env)
    serve.start(detached=True, http_options={"host": "0.0.0.0"})

    logger.info("The environment variables in rayserve are: {}".format(runtime_env))
    logger.info("All variables are: {}".format(env_vars))

    logger.info("Starting rayserve server.")
    logger.info("Deploying modules.")

    LomotifDownloaderServe.deploy()
    PreprocessLomotifServe.deploy()
    ContentModNudenetServe.deploy()
    ContentModMiddleFingerDetectServe.deploy()
    ContentModTagCoopClipServe.deploy()
    ComposedModel.deploy()

    logger.info("Deployment completed.")
    logger.info("Waiting for requests...")
