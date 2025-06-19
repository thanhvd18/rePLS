#!/bin/bash

source setup_env.sh
cd "$base_dir/figures"

# Run each figure plotting function
echo "Running plot_fig_5bc..."
python -c "from figures.run_figure5 import prepare_data_fig_5bc; prepare_data_fig_5bc()"
echo "Finished plot_fig_5bc"

echo "Running plot_fig_5b..."
python -c "from figures.run_figure5 import plot_fig_5b; plot_fig_5b()"
echo "Finished plot_fig_5b"

echo "Running plot_fig_5c..."
python -c "from figures.run_figure5 import plot_fig_5c; plot_fig_5c()"
echo "Finished plot_fig_5c"

echo "Running plot_fig_5de..."
python -c "from figures.run_figure5 import plot_fig_5de; plot_fig_5de()"
echo "Finished plot_fig_5de"

$matlab -batch "cd('$base_dir/figures/figure5/matlab'); fig5d;exit();"
