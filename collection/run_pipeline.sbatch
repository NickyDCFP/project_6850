#!/bin/bash
#SBATCH --job-name=corpus_update
#SBATCH --partition=awang,elor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=data/updated/cache/slurm_%j.out
#SBATCH --error=data/updated/cache/slurm_%j.err
#SBATCH --mail-type=FAIL,END

# Filter (resuming after each daily Gemini quota reset) + citations, unattended; see collection/README.md.
#   sbatch --mail-user=<you@example.edu> collection/run_pipeline.sbatch [first-run request budget, default 450]
# Submit from the repository root (log paths above and the cd below are relative to it).
# Resumable: resubmit after a failure; finished decisions and citation lookups are skipped.

cd "${SLURM_SUBMIT_DIR:-.}"
collection/run_pipeline.sh "${1:-450}"
