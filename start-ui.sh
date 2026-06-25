#!/usr/bin/env bash

# OT actual hybrid cache default patch
export PYTHONPATH="/workspace/OneTrainer:/workspace/OneTrainer/src/mgds/src:${PYTHONPATH:-}"
export OT_ACTUAL_HYBRID_CACHE_VERBOSE="${OT_ACTUAL_HYBRID_CACHE_VERBOSE:-1}"

# OT hybrid cache default patch
export PYTHONPATH="/workspace/OneTrainer:${PYTHONPATH:-}"
export OT_HYBRID_CACHE_VERBOSE="${OT_HYBRID_CACHE_VERBOSE:-1}"

# OT cache-only default patch
export PYTHONPATH="/workspace/OneTrainer:${PYTHONPATH:-}"
export OT_CACHE_ONLY_VERBOSE="${OT_CACHE_ONLY_VERBOSE:-1}"
export OT_CACHE_ONLY_STRICT_ALIGNMENT="${OT_CACHE_ONLY_STRICT_ALIGNMENT:-1}"
export OT_CACHE_ONLY_ALLOW_PARTIAL="${OT_CACHE_ONLY_ALLOW_PARTIAL:-0}"

set -e

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

prepare_runtime_environment

run_python_in_active_env "scripts/train_ui.py" "$@"
