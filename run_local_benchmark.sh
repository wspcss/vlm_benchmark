#!/bin/bash
cd "$(dirname "$0")"
set -a
source .env
set +a
python3.11 local_benchmark.py --all