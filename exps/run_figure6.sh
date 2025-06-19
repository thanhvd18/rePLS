#!/bin/bash

source setup_env.sh
cd "$base_dir/figures"

echo "Running plot_fig_6b..."
python -c "from figures.run_figure6 import plot_fig_6b; plot_fig_6b()"
echo "Finished plot_fig_6b"

echo "Running plot_fig_6c..."
python -c "from figures.run_figure6 import plot_fig_6c; plot_fig_6c()"
echo "Finished plot_fig_6c"

echo "Running plot_fig_6d..."
python -c "from figures.run_figure6 import plot_fig_6d; plot_fig_6d()"
echo "Finished plot_fig_6d"

echo "Running plot_fig_6e..."
python -c "from figures.run_figure6 import plot_fig_6e; plot_fig_6e()"
echo "Finished plot_fig_6e"

echo "Running plot_fig_6f..."
python -c "from figures.run_figure6 import plot_fig_6f; plot_fig_6f()"
echo "Finished plot_fig_6f"

$matlab -batch "cd('$base_dir/figures/figure6/matlab'); fig6f;exit();"