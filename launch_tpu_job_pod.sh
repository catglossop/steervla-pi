#!/bin/bash
set -euo pipefail

VM_NAME="${1:?Usage: $0 <vm-name> <num-workers> <config-name>}"
NUM_WORKERS="${2:?Usage: $0 <vm-name> <num-workers> <config-name>}"
CONFIG_NAME="${3:?Usage: $0 <vm-name> <num-workers> <config-name>}"
ZONE=us-central2-b

if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [ "$NUM_WORKERS" -lt 2 ] || [ "$NUM_WORKERS" -gt 8 ]; then
    echo "ERROR: num-workers must be an integer from 2 to 8 (got: $NUM_WORKERS)"
    exit 1
fi

echo "Launching TPU pod job on $VM_NAME: syncing to workers 0..$((NUM_WORKERS - 1)), config=$CONFIG_NAME"

# Ensure SSH keys are provisioned on the VM (needed after VM recreation).
echo "Ensuring SSH keys are provisioned (worker 0)..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --worker=0 --command="echo ok" || {
    echo "ERROR: Cannot SSH into $VM_NAME worker 0. Is the TPU VM running?"
    exit 1
}

sync_to_worker() {
    local w="$1"
    echo "--- rsync to worker $w ---"
    local SSH_CMD
    SSH_CMD=$(gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --worker="$w" --dry-run 2>&1)
    local SSH_ARGS
    SSH_ARGS=$(echo "$SSH_CMD" | grep -oP '(?<=/usr/bin/ssh ).*(?= noam@)') || {
        echo "ERROR: Could not parse SSH args from dry-run output (worker $w):"
        echo "$SSH_CMD"
        exit 1
    }
    local TARGET_HOST
    TARGET_HOST=$(echo "$SSH_CMD" | grep -oP 'noam@[\d.]+') || {
        echo "ERROR: Could not parse target host from dry-run output (worker $w):"
        echo "$SSH_CMD"
        exit 1
    }

    rsync -avz --exclude='.venv' -e "/usr/bin/ssh ${SSH_ARGS//-t /}" \
        /home/noam/steervla-pi "${TARGET_HOST}":/home/noam
}

for ((w = 0; w < NUM_WORKERS; w++)); do
    sync_to_worker "$w"
done

# JAX multi-host: coordinator on worker 0 (internal VPC IP + fixed port).
JAX_COORD_PORT="${JAX_COORD_PORT:-12345}"
echo "Resolving worker 0 internal IP for JAX_COORDINATOR_ADDRESS (port ${JAX_COORD_PORT})..."
COORD_IP="$(
    gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --worker=0 \
        --command="hostname -I" 2>/dev/null | tail -n1 | awk '{print $1}' | tr -d '[:space:]'
)"
if [[ -z "$COORD_IP" || ! "$COORD_IP" =~ ^[0-9.]+$ ]]; then
    echo "ERROR: Could not read worker 0 internal IP (got: '${COORD_IP}')."
    exit 1
fi
JAX_COORDINATOR_ADDRESS="${COORD_IP}:${JAX_COORD_PORT}"
echo "Using JAX_COORDINATOR_ADDRESS=${JAX_COORDINATOR_ADDRESS} JAX_NUM_PROCESSES=${NUM_WORKERS}"

echo "Starting training on each worker (start_tpu_job.sh per worker, with JAX env)..."
for ((w = 0; w < NUM_WORKERS; w++)); do
    echo "--- start_tpu_job.sh on worker $w ---"
    gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --worker="$w" \
        --command="export JAX_COORDINATOR_ADDRESS='${JAX_COORDINATOR_ADDRESS}' JAX_NUM_PROCESSES=${NUM_WORKERS} JAX_PROCESS_INDEX=${w}; bash steervla-pi/start_tpu_job.sh ${WANDB_API_KEY} ${HF_TOKEN} ${CONFIG_NAME}"
done
