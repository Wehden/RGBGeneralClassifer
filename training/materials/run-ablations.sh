#!/usr/bin/env bash
# Run bounded, offline experiments on a host with the training image loaded.
set -euo pipefail

: "${MATERIAL_DATA_DIR:?Set the immutable dataset directory}"
: "${MATERIAL_CACHE_DIR:?Set the preloaded Hugging Face cache directory}"
: "${MATERIAL_FEATURE_DIR:?Set the original frozen-feature output directory}"
: "${MATERIAL_OUTPUT_DIR:?Set a NEW private experiment directory}"
MATERIAL_TRAINING_IMAGE="${MATERIAL_TRAINING_IMAGE:-bolt-rgb-material-training:2026-10-28}"
MATERIAL_STAGE="${MATERIAL_STAGE:-head}"
MATERIAL_DEVICE="${MATERIAL_DEVICE:-cpu}"
material_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "$MATERIAL_STAGE" in head|train) ;; *) echo "Stage must be head or train" >&2; exit 2;; esac
case "$MATERIAL_DEVICE" in cpu|gpu) ;; *) echo "Device must be cpu or gpu" >&2; exit 2;; esac
for material_directory in "$MATERIAL_DATA_DIR" "$MATERIAL_CACHE_DIR" "$MATERIAL_FEATURE_DIR"; do
  if [[ "$material_directory" != /* || ! -d "$material_directory" ]]; then
    echo "Input directories must exist and use absolute paths" >&2
    exit 2
  fi
done
if [[ "$MATERIAL_OUTPUT_DIR" != /* ]]; then
  echo "Output directory must use an absolute path" >&2
  exit 2
fi
if [[ -e "$MATERIAL_OUTPUT_DIR" ]]; then
  echo "Experiment directory already exists; preserve it and use a new path" >&2
  exit 2
fi
mkdir -m 700 -p "$MATERIAL_OUTPUT_DIR"
docker image inspect "$MATERIAL_TRAINING_IMAGE" > "$MATERIAL_OUTPUT_DIR/image-receipt.json"
sha256sum "$material_script_dir/train.py" > "$MATERIAL_OUTPUT_DIR/trainer.sha256"
material_device_args=()
if [[ "$MATERIAL_DEVICE" == gpu ]]; then
  if [[ -n "${MATERIAL_DRIVER_DIR:-}" ]]; then
    # Optional process-local workaround for a loaded/installed driver mismatch.
    # This mounts already extracted matching libraries; it changes no host driver.
    material_device_args=(
      --runtime runc --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm
      -e LD_LIBRARY_PATH=/isolated-driver:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
      -v "$MATERIAL_DRIVER_DIR:/isolated-driver:ro"
    )
  else
    material_device_args=(--gpus all)
  fi
fi
material_runs=(balanced20 balanced40 public-only)
if [[ "$MATERIAL_STAGE" == head ]]; then material_runs=(legacy "${material_runs[@]}"); fi
if [[ -n "${MATERIAL_REVIEW_MANIFEST:-}" ]]; then material_runs+=(reviewed20); fi

for material_run in "${material_runs[@]}"; do
  material_run_output="$MATERIAL_OUTPUT_DIR/$material_run"
  mkdir "$material_run_output"
  cp "$MATERIAL_FEATURE_DIR/features.npz" "$MATERIAL_FEATURE_DIR/feature-manifest.json" \
     "$MATERIAL_FEATURE_DIR/text_embeddings.npy" "$material_run_output/"
  material_training_args=(--weighting domain-balanced --camera-material-mass .2 --camera-rejection-mass .1)
  material_review_args=()
  case "$material_run" in
    legacy) material_training_args=(--weighting legacy);;
    balanced40) material_training_args=(--weighting domain-balanced --camera-material-mass .4 --camera-rejection-mass .1);;
    public-only) material_training_args=(--weighting public-only);;
    reviewed20)
      material_review_args=(-v "$MATERIAL_REVIEW_MANIFEST:/review.json:ro")
      material_training_args+=(--review-manifest /review.json);;
  esac
  docker run --rm --network none --cpus 8 --memory 24g --shm-size 2g \
    --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
    -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "${material_device_args[@]}" "${material_review_args[@]}" \
    -v "$material_script_dir:/source:ro" -v "$MATERIAL_DATA_DIR:/data:ro" \
    -v "$MATERIAL_CACHE_DIR:/cache:ro" -v "$material_run_output:/output" \
    --entrypoint python "$MATERIAL_TRAINING_IMAGE" /source/train.py \
    --stage "$MATERIAL_STAGE" --data /data --output /output --epochs 6 \
    --validation-only --minimum-camera-accepted-references 10 \
    "${material_training_args[@]}" > "$MATERIAL_OUTPUT_DIR/$material_run.log" 2>&1
  echo "$material_run complete; private results in $material_run_output"
done
