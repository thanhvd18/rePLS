#!/bin/bash

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for fig in {1..6}; do
    # if fig == 2 continue
    if [ $fig -eq 2 ]; then
        continue
    fi
    echo "Running plot_fig_$fig..."
    bash "$_script_dir/run_figure$fig.sh"

done