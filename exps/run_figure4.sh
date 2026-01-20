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

for random_state in {1..2}; do
# Run each figure plotting function
echo "Running plot_fig_4a..."
python -c "from figures.run_figure4 import plot_fig_4a; plot_fig_4a(random_state=$random_state)"
echo "Finished plot_fig_4a"

run_matlab_batch "cd('$base_dir/figures/figure4/matlab'); fig4a;exit();"

echo "Running plot fig 4b..."
python -c "from figures.run_figure4 import figure_4b; figure_4b(random_state=$random_state)"
echo "Finished plot fig 4b"

echo "Running plot fig 4d..."
python -c "from figures.run_figure4 import plot_fig_4d; plot_fig_4d(random_state=$random_state)"
echo "Finished plot fig 4d"

echo "Running plot fig 4e..."
python -c "from figures.run_figure4 import plot_fig_4e; plot_fig_4e(random_state=$random_state)"
echo "Finished plot fig 4e"

echo "Running plot fig 4f..."
python -c "from figures.run_figure4 import plot_fig_4f; plot_fig_4f(random_state=$random_state)"
echo "Finished plot fig 4f"

done
