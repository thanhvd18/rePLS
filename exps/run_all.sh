#!/bin/bash

 for fig in {1..6}; do
    # if fig == 2 continue
    if [ $fig -eq 2 ]; then
        continue
    fi
    echo "Running plot_fig_$fig..."
    bash run_figure$fig.sh

  done