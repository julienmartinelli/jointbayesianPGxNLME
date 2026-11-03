#!/bin/bash
# Run the fits listed in jobs.txt as a Slurm array, one fit per task:
#   python scripts/run_simulations.py --list > jobs.txt
#   sbatch --array=1-$(wc -l < jobs.txt) scripts/slurm_array.sh
# Each fit runs 5 chains in parallel (one CPU per chain). Most fits take
# minutes to an hour; R2-D2 at p=800 takes about 9 hours (up to ~13 hours) and
# needs about 1.3 GB of memory.
#SBATCH --job-name=pgxnlme
#SBATCH --cpus-per-task=5
#SBATCH --mem=4G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.log

set -euo pipefail
eval "$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.txt)"
