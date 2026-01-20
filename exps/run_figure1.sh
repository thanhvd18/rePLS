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
echo "Running plot_fig_1c..."
python -c "from figures.run_figure1 import plot_fig_1c; plot_fig_1c()"
echo "Finished plot_fig_1c"

run_matlab_batch "cd('$base_dir/figures/figure1/matlab'); fig1c;exit();"

echo "Running plot_fig_1d..."
python -c "from figures.run_figure1 import plot_fig_1d; plot_fig_1d()"
echo "Finished plot_fig_1d"

run_matlab_batch "cd('$base_dir/figures/figure1/matlab'); fig1d;exit();"

echo "Running plot_fig_1e..."
python -c "from figures.run_figure1 import plot_fig_1e; plot_fig_1e()"
echo "Finished plot_fig_1e"

echo "Running plot_fig_1f..."
python -c "from figures.run_figure1 import plot_fig_1f; plot_fig_1f()"
echo "Finished plot_fig_1f"
run_matlab_batch "cd('$base_dir/figures/figure1/matlab'); fig1f;exit();"

