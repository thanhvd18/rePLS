#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_env.sh"
cd "$base_dir/figures"

run_matlab_batch() {
  local cmd="$1"
  if [ -n "${matlab:-}" ] && { command -v "$matlab" >/dev/null 2>&1 || [ -x "$matlab" ]; }; then
    "$matlab" -batch "$cmd"
  else
    echo "Matlab not found (matlab='$matlab'); skipping: $cmd"
  fi
}

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

run_matlab_batch "cd('$base_dir/figures/figure5/matlab'); fig5d;exit();"
