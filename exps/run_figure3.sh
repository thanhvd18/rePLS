#!/bin/bash

source setup_env.sh
cd "$base_dir/figures"

for random_state in {1..10}; do

    # Run each figure plotting function
    echo "Running plot_fig_3a..."
    python -c "from figures.run_figure3 import plot_fig_3a; plot_fig_3a(random_state=$random_state)"
    echo "Finished plot_fig_3a"

    echo "Running plot_fig_3b..."
    python -c "from figures.run_figure3 import plot_fig_3b; plot_fig_3b(random_state=$random_state)"
    echo "Finished plot_fig_3b"
done