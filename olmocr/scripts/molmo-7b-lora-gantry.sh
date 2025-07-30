#!/usr/bin/env bash

set -ex

# check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install it."
    exit
fi


EXTRA_ARGS="-c olmocr/train/config/molmo-o-lora-8192.yaml --num_proc 64 --save.path \"s3://ai2-oe-data/jakep/experiments/molmo-pdf/v1/models/\${BEAKER_USER_ID}\""

run_name=$(basename "$0" .sh)

# --cluster 'ai2/jupiter*' \
# --cluster 'ai2/pluto*' \
# --cluster 'ai2/allennlp-cirrascale' \
# --priority high \

CLUSTER='jupiter'

gantry run \
    --description "${run_name}-8192"\
    --task-name "${run_name}-8192"\
    --allow-dirty \
    --host-networking \
    --workspace ai2/oe-data-model-based-cleanup \
    --beaker-image 'jakep/jakep-pdf-finetunev1.2' \
    --venv 'base' \
    --pip gantry-requirements.txt \
    --priority high \
    --gpus 8 \
    --cluster "ai2/${CLUSTER}*" \
    --budget ai2/oe-data \
    --weka "oe-data-default:/data" \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=S2_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=S2_AWS_SECRET_ACCESS_KEY \
    --env-secret DS_AWS_ACCESS_KEY_ID=S2_AWS_ACCESS_KEY_ID \
    --env-secret DS_AWS_SECRET_ACCESS_KEY=S2_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=JAKE_WANDB_API_KEY \
    --shared-memory 10GiB \
    --yes \
    -- /bin/bash -c "source scripts/beaker/${CLUSTER}-ib.sh && python -m olmocr.train.loaddataset ${EXTRA_ARGS} && accelerate launch --multi_gpu --num_processes \${BEAKER_ASSIGNED_GPU_COUNT} --mixed_precision bf16 -m olmocr.train.train ${EXTRA_ARGS}"