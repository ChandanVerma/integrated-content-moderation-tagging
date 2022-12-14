import os
import numpy as np
import cv2
import mimetypes
import logging
import subprocess

mimetypes.init()
from skimage import metrics as skimage_metrics
from PIL import Image

# loggers
logger = logging.getLogger("ray")
logger.setLevel("INFO")


def rewrite_video(video_path):
    """Rewriting a video with ffmpeg in the event it cannot be read by OpenCV.

    Args:
        video_path (string): full filepath of the video

    Returns:
        string: filepath of the re-written video
    """

    video = cv2.VideoCapture(video_path)

    for frame_i in range(1):
        read_flag, current_frame = video.read()

        if not read_flag:
            break

    if frame_i == 0 and not read_flag:
        output_path = (
            os.path.splitext(video_path)[0]
            + "-rewrite"
            + os.path.splitext(video_path)[1]
        )
        cmd_str = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-c",
            "copy",
            "-an",
            output_path,
        ]
        subprocess.run(cmd_str, capture_output=False)
        logger.info("Rewriting {} with ffmpeg.".format(video_path))
        video_path = output_path

    video.release()

    return video_path


def get_mime(clip_path):
    """Identify data type (image or video) from a filepath.

    Args:
        clip_path (string): filepath of the asset

    Returns:
        string: mime type of the asset
    """
    mimestart = mimetypes.guess_type(clip_path)[0].split("/")[0]
    if mimestart is None:
        if clip_path.split(".")[-1] == "mp4":
            mimestart = "video"
    return mimestart


def resize_image2(im, size):
    """Resize an image to a target size without changing aspect ratio \
    using zero padding.

    Args:
        im (PIL.Image): image object of key frame
        size (tuple): size of image to resize to

    Returns:
        PIL.Image: new resized image
    """
    desired_size = size[0]
    orig_size = (im.height, im.width)
    ratio = float(desired_size) / max(orig_size)
    new_size = tuple([int(x * ratio) for x in orig_size])[::-1]
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(
        im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    )

    return new_im


def process_image(img_rgb, size):
    """Nudenet's image processing before passing into the model

    Args:
        img_rgb ([np.ndarray]): image array, channel-last
        size ([tuple]): size to resize into

    Returns:
        [np.ndarray]: image array after processing.
    """
    img = Image.fromarray(img_rgb).convert("RGB")
    img = resize_image2(img, size)
    img = np.asarray(img, dtype=np.float32)
    img /= 255
    return img


def process_clip(frames_rgb, clip_path, size, frame_indices):
    """Iterate nudenet's processing for each clip.

    Args:
        frames_rgb ([np.ndarray]): image array, channel-last
        clip_path ([str]): file path to clip
        size ([tuple]): size to resize into
        frame_indices ([list]): list of 0-indexed frame indices that is read from clip

    Returns:
        [(np.ndarray, list)]: return processed image arrays \
            and corresponding text to write into outputs
    """
    filename = os.path.basename(clip_path).split(".")[0]
    img_list = [process_image(img_rgb=x, size=size) for x in frames_rgb]
    img_paths = ["{}_frame_{}".format(filename, i) for i in frame_indices]
    return np.asarray(img_list), img_paths


def is_similar_frame(f1, f2, resize_to=(64, 64), thresh=0.3, return_score=False):
    """Adapted from https://github.com/notAI-tech/NudeNet/blob/v2/nudenet/video_utils.py
    """

    if f1 is None or f2 is None:
        return False

    if isinstance(f1, str) and os.path.exists(f1):
        try:
            f1 = cv2.imread(f1)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if isinstance(f2, str) and os.path.exists(f2):
        try:
            f2 = cv2.imread(f2)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if resize_to:
        f1 = cv2.resize(f1, resize_to)
        f2 = cv2.resize(f2, resize_to)

    if len(f1.shape) == 3:
        f1 = f1[:, :, 0]

    if len(f2.shape) == 3:
        f2 = f2[:, :, 0]

    score = skimage_metrics.structural_similarity(f1, f2, channel_axis=None)

    if return_score:
        return score

    if score >= thresh:
        return True

    return False


def get_interest_frames_from_video(
    video_path,
    frame_similarity_threshold=0.5,
    similarity_context_n_frames=3,
):
    """Adapted from https://github.com/notAI-tech/NudeNet/blob/v2/nudenet/video_utils.py
    """
    important_frames = []
    all_frames = []
    selected_frame_indices = []
    fps = 0
    video_length = 0

    try:
        video_path = rewrite_video(video_path)

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_seconds = int(fps)
        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_i in range(length + 1):
            read_flag, current_frame = video.read()

            if not read_flag:
                break

            if skip_seconds > 0 and frame_i % skip_seconds != 0:
                pass
            else:
                frame_i += 1

                found_similar = False
                for context_frame_i, context_frame in reversed(
                    important_frames[-1 * similarity_context_n_frames :]
                ):
                    if is_similar_frame(
                        context_frame, current_frame, thresh=frame_similarity_threshold
                    ):
                        found_similar = True
                        break

                if not found_similar:
                    important_frames.append((frame_i, current_frame))
                    selected_frame_indices.append(frame_i)

                all_frames.append(current_frame[:, :, [2, 1, 0]])


        video.release()
        cv2.destroyAllWindows()

    except Exception as ex:
        logger.error(str(ex))

    if len(important_frames) >= 3:
        important_frames = [i[1][:, :, [2, 1, 0]] for i in important_frames]
    else:
        important_frames = all_frames
        selected_frame_indices = np.arange(len(all_frames))

    return (
        important_frames,
        # low_important_frames,
        fps,
        video_length,
        selected_frame_indices,
    )
