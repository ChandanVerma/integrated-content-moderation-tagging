import os
import boto3
import logging

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray")
logger.setLevel("INFO")


def download_models_helper(model_name, root="./models"):
    """Download models from respective S3 buckets.

    Args:
        model_name (string): model identifier
        root (str, optional): downloaded models will be saved to this directory.\
        Defaults to "./models".
    """
    try:
        assert model_name in ['nudenet', 'coopclip']
    except:
        assert False
        
    logger.info(
        "Env vars within download models helper are: {}, {}, {}".format(
            os.environ.get("AWS_ROLE_ARN"),
            os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"),
            os.environ.get("AWS_DEFAULT_REGION"),
        )
    )
    logger.info(
        "Model files will be saved in: {}".format(os.path.join(os.getcwd(), "models"))
    )
    s3 = boto3.client("s3")

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root, "clip")):
        os.makedirs(os.path.join(root, "clip"))
    if not os.path.exists(os.path.join(root, "prompt_learner")):
        os.makedirs(os.path.join(root, "prompt_learner"))

    if model_name == 'nudenet':
        files = [
            "classifier_lite.onnx",
        ]
    if model_name == 'coopclip':
        files = [
        "clip/ViT-B-32.pt",
        "clip/ViT-B-16.pt",
        "prompt_learner/checkpoint",
        "prompt_learner/model.pth.tar-50",
    ]
    for f in files:
        if not os.path.exists(os.path.join(root, "{}".format(f))):
            s3.download_file(
                os.environ.get("AiModelBucket"),
                "data_science/ai_model_files/content-moderation-tagging/version_4/{}".format(
                    f
                ),
                os.path.join(root, "{}".format(f)),
            )
