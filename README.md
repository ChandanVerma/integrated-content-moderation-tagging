# Content-Moderation-Tagging
This repo does content moderation and content tagging.

# Feature list
This list will be continually updated
1) Moderating lomotifs using CLIP and Nudenet models to identify violence and sexual content
2) Content tagging lomotifs using CLIP and Nudenet models to identify their primary and secondary categories.
3) Wrapped the code base with ML deployment framework, Rayserve.


# Set-up .env file for testing in local
There needs to be a `.env` file with following parameters.
```
DownloadNumCPUPerReplica=0.2
DownloadNumReplicas=1
DownloadMaxCon=100


PreprocessNumCPUPerReplica=1
PreprocessNumReplicas=1
PreprocessMaxCon=100


NudenetNumCPUPerReplica=0.8
NudenetNumReplicas=1
NudenetMaxCon=100

MFDNumCPUPerReplica=0.8
MFDNumReplicas=1
MFDMaxCon=100

ClipNumCPUPerReplica=0.1
ClipNumGPUPerReplica=0.14
ClipNumReplicas=1
ClipMaxCon=100


ComposedNumCPUPerReplica=0.1
ComposedNumReplicas=1
ComposedMaxCon=100

SnowflakeResultsQueue=content_moderation_tagging-results_dev
RawResultsQueue=content_moderation_tagging-raw-results_dev
AiModelBucket=datalake-dev
```

# Additional variables for internal testing
For DS Team internal testing, we also need to add the following env vars to the `.env` file:
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-2
```
and uncomment these lines in `tasks.py`:
```
# from dotenv import load_dotenv

# load_dotenv("./.env")
```
To prepare the conda environment to test the script:
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install -U "ray[default]==1.11.1"
pip install "ray[serve]"
pip install pytest
```

# For use in `g4dn.2xlarge` instance, use the following variables instead
```
DownloadNumCPUPerReplica=0.2
DownloadNumReplicas=1
DownloadMaxCon=100


PreprocessNumCPUPerReplica=1
PreprocessNumReplicas=1
PreprocessMaxCon=100


NudenetNumCPUPerReplica=0.8
NudenetNumReplicas=2
NudenetMaxCon=100

MFDNumCPUPerReplica=0.8
MFDNumReplicas=2
MFDMaxCon=100

ClipNumCPUPerReplica=0.1
ClipNumGPUPerReplica=0.14
ClipNumReplicas=4
ClipMaxCon=100


ComposedNumCPUPerReplica=0.1
ComposedNumReplicas=1
ComposedMaxCon=100

SnowflakeResultsQueue=content_moderation_tagging-results_dev
RawResultsQueue=content_moderation_tagging-raw-results_dev
AiModelBucket=lomotif-datalake-dev
```

# Instructions (Docker)
1) Ensure there are environment variables or `.env` file, see section above for environment variables.
2) Ensure GPU for docker is enabled. See section below.
3) Once the container is able to detect the GPU, we can follow the normal process of

```
docker-compose build
docker-compose up
```

# Enabling GPU for Docker
To enable the GPU for Docker, make sure Nvidia drivers for the system are installed. [Refer link for details](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)

Commands which can help install Nvidia drivers are:
```
unbuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

Then nvidia-docker2 tools needs to be installed.
To install follow the below instructions.
[Refer link for details](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```
curl https://get.docker.com | sh   && sudo systemctl --now enable docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

# Pytest
1) Test if the code is working as expected. Firstly on terminal, do:
```bash
ray start --head --port=6300
```
2) Then, deploy the ray services:
```bash
python serve_tasks/tasks.py
```
3) Finally you can conduct the pytest and shutdown the ray cluster.
```bash
cd ./test
python -m pytest test_ray_deployments.py 
ray stop --force
```

<!-- # Instructions (non-Docker)
- Python 3.8 required.
```bash
conda create -n [YOUR_ENV_NAME] python=3.8
conda activate [YOUR_ENV_NAME]
```
- Prerequisites are in `requirements.txt`. You may install via pip with the following.
```bash
pip install -r requirements.txt
```
- Spin up celery worker server
```bash
-- celery -A tasks.tasks worker --loglevel=INFO -P solo -l info
```
- Run kinesis stream
```bash
python kinesis_deploy.py
``` -->

# Code structure
To be updated.
<!-- - `./kinesis_deploy.py`: script to run predictions on a kinesis stream.
- `./tasks/`: celery tasks
- `./celery_app.py`: celery app file


- `./src/`: main scripts are stored here.
    - `src/utils/data_process_clip.py`: data reading and processing utils
    - `src/utils/write_tables_clip.py`: authenticate into Snowflake and functions for writing tables to store model outputs
    - `src/content_tag_predictor.py`: main predictor script -->
