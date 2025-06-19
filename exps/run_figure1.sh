#!/bin/bash

source setup_env.sh
cd "$base_dir/figures"

# Run each figure plotting function
echo "Running plot_fig_1c..."
python -c "from figures.run_figure1 import plot_fig_1c; plot_fig_1c()"
echo "Finished plot_fig_1c"

$matlab -batch "cd('$base_dir/figures/figure1/matlab'); fig1c;exit();"

echo "Running plot_fig_1d..."
python -c "from figures.run_figure1 import plot_fig_1d; plot_fig_1d()"
echo "Finished plot_fig_1d"

$matlab -batch "cd('$base_dir/figures/figure1/matlab'); fig1d;exit();"

echo "Running plot_fig_1e..."
python -c "from figures.run_figure1 import plot_fig_1e; plot_fig_1e()"
echo "Finished plot_fig_1e"

echo "Running plot_fig_1f..."
python -c "from figures.run_figure1 import plot_fig_1f; plot_fig_1f()"
echo "Finished plot_fig_1f"
$matlab -batch "cd('$base_dir/figures/figure1/matlab'); fig1f;exit();"

