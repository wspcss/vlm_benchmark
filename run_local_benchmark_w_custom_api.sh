#!/bin/bash
# Run the custom-API VLM benchmark against the server in VLM_API_BASE_URL.
# Set VLM_API_BASE_URL (and optionally VLM_API_KEY) in .env, e.g.:
#   VLM_API_BASE_URL=http://192.168.1.10:8000
#
# --model is optional: if omitted, the script auto-detects the first model
# from GET /v1/models.
cd "$(dirname "$0")"
set -a
source .env
set +a
python3 local_benchmark_w_custom_api.py "$@"
