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

run_matlab_batch "cd('$base_dir/figures/figure6/matlab'); fig6f;exit();"