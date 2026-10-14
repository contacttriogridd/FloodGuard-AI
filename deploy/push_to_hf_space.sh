#!/usr/bin/env bash
# Upload the trained model to a Hugging Face model repo and the app to a Docker Space.
#
#   export HF_TOKEN=hf_...            # write token: https://huggingface.co/settings/tokens
#   export HF_USER=<your-username>
#   bash deploy/push_to_hf_space.sh                 # uses runs/models
#   ARTIFACTS_DIR=runs/models bash deploy/push_to_hf_space.sh
#
# Creates / updates:
#   model repo  ${FG_MODEL_REPO:-$HF_USER/floodguard-ai-model}   (floodguard_pipeline.joblib, model_card.json)
#   space       ${HF_SPACE:-$HF_USER/floodguard-ai}               (Dockerfile, api/, app/, src/, configs/)
# and sets the Space variable FG_MODEL_REPO so the container downloads the model at start-up.
set -euo pipefail
cd "$(dirname "$0")/.."

# Token: HF_TOKEN env var, or the one saved by `hf auth login` (~/.cache/huggingface/token).
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
: "${HF_TOKEN:?Set HF_TOKEN or run 'hf auth login' first}"
export HF_TOKEN
: "${HF_USER:?Set HF_USER to your Hugging Face username}"
export FG_MODEL_REPO="${FG_MODEL_REPO:-$HF_USER/floodguard-ai-model}"
export HF_SPACE="${HF_SPACE:-$HF_USER/floodguard-ai}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-runs/models}"

if [[ ! -f "$ARTIFACTS_DIR/floodguard_pipeline.joblib" ]]; then
  echo "No trained model in $ARTIFACTS_DIR — run scripts/run_pipeline.py first." >&2
  exit 1
fi
if grep -q '"data_source": "synthetic"' "$ARTIFACTS_DIR/model_card.json" 2>/dev/null; then
  echo "WARNING: $ARTIFACTS_DIR holds a model trained on SYNTHETIC smoke-test data." >&2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -r Dockerfile requirements-serve.txt configs src api app "$STAGE/"
mkdir -p "$STAGE/deploy/model_artifacts" && touch "$STAGE/deploy/model_artifacts/.gitkeep"
cp deploy/hf_space/README.md "$STAGE/README.md"
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +

"${PYTHON:-python}" - "$STAGE" <<'PY'
import os
import sys

from huggingface_hub import HfApi

stage = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
model_repo, space = os.environ["FG_MODEL_REPO"], os.environ["HF_SPACE"]

api.create_repo(model_repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=os.environ["ARTIFACTS_DIR"], repo_id=model_repo, repo_type="model",
                  allow_patterns=["floodguard_pipeline.joblib", "model_card.json"],
                  commit_message="Upload FloodGuard AI model artefacts")
print(f"Model uploaded: https://huggingface.co/{model_repo}")

api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
api.add_space_variable(space, "FG_MODEL_REPO", model_repo)
api.upload_folder(folder_path=stage, repo_id=space, repo_type="space", commit_message="Deploy FloodGuard AI")
print(f"Space deployed: https://huggingface.co/spaces/{space}")
PY
