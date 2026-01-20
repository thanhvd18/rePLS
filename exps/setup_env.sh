#!/bin/bash

# Set environment variables for PySurfer
#export SUBJECTS_DIR="/Users/tth/Thanh/Neuroimaging/brain-visualization-tool/data/surface/subjects_dir"
#export ATLASDIR="/Users/tth/Thanh/Neuroimaging/brain-visualization-tool/data/surface/"


set -e

# Resolve repo root as default base_dir (can be overridden by exporting base_dir before sourcing).
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${base_dir:=$(cd "${_script_dir}/.." && pwd)}"
export base_dir

# Matlab command (can be overridden by exporting matlab before sourcing).
: "${matlab:=matlab}"
export matlab

export PYTHONPATH="$PYTHONPATH:$base_dir:$base_dir/figures"
