#!/usr/bin/env bash
# Unattended run of the filter (resuming after each daily Gemini quota reset) and the citation step.
# Normally submitted with collection/run_pipeline.sbatch, whose Slurm mail reports the end or a failure.
# Set NOTIFY_EMAIL to also get start/finish emails (with the filter and citation summaries) via mailx.
#
# Progress: data/updated/cache/pipeline.log (one line per step) and the per-run logs next to it.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGDIR=data/updated/cache
LOG=$LOGDIR/pipeline.log
BUDGET=${1:-450}        # requests for the first filter run (lower it if part of today's quota is used)
DAILY=450               # requests per later run (free tier: 500/day per model)

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
notify() {  # subject, body file or text
    if [ -n "${NOTIFY_EMAIL:-}" ]; then
        { echo "$2"; echo; echo "--- last lines of $LOG ---"; tail -20 "$LOG"; } | mail -s "[project_6850] $1" "$NOTIFY_EMAIL"
    fi
    log "notified: $1"
}
fail() { notify "pipeline FAILED: $1" "Step failed: $1. See $PWD/$LOGDIR for logs. Rerunning the script resumes where it stopped."; exit 1; }

log "pipeline started (pid $$)"
notify "pipeline started" "The filter + citation pipeline started on $(hostname). You'll get another email when it finishes or fails."

# Citations for the shipped corpus can run while the filter waits for quota resets.
python -m collection.fetch_citations > "$LOGDIR/citations_run_$(date +%F_%H%M).log" 2>&1 &
CIT_PID=$!
log "citations (shipped corpus) started, pid $CIT_PID"

stalled=0
while true; do
    run_log="$LOGDIR/filter_run_$(date +%F_%H%M).log"
    log "filter run, budget $BUDGET requests -> $run_log"
    python -m collection.filter_papers --max-calls "$BUDGET" > "$run_log" 2>&1 || fail "filter_papers (see $run_log)"
    grep -E "^this run|retried because" "$run_log" | while read -r line; do log "  $line"; done
    if grep -q "all new candidates judged" "$run_log"; then
        log "filter finished"
        break
    fi
    left=$(grep -oE "^[0-9]+ candidates still to judge" "$run_log" | cut -d' ' -f1)
    if grep -qE "^this run: \{\}|^this run: \{'failed batches'" "$run_log" || ! grep -q "^this run" "$run_log"; then
        stalled=$((stalled + 1))
    else
        stalled=0
    fi
    [ "$stalled" -ge 2 ] && fail "filter made no progress in two runs ($left candidates left)"
    # Gemini daily quotas reset at midnight Pacific; wait until 00:15 PT.
    secs=$(( $(TZ=America/Los_Angeles date -d 'tomorrow 00:15' +%s) - $(date +%s) ))
    log "$left candidates left; sleeping ${secs}s until the quota resets"
    sleep "$secs"
    BUDGET=$DAILY
done

wait "$CIT_PID" || fail "fetch_citations for the shipped corpus"
log "citations (shipped corpus) done"
# New papers, then retry anything Semantic Scholar throttled (each pass only fetches unresolved papers).
for pass in 1 2 3 4; do
    run_log="$LOGDIR/citations_run_$(date +%F_%H%M).log"
    python -m collection.fetch_citations > "$run_log" 2>&1 || fail "fetch_citations (see $run_log)"
    left=$(grep -oE "\(([0-9]+) left for the next run" "$run_log" | grep -oE "[0-9]+" | tail -1)
    log "citations pass $pass done: ${left:-0} papers still throttled; $(tail -1 "$run_log")"
    [ "${left:-0}" -eq 0 ] && break
    sleep 600
done

python -m collection.build_corpus >> "$LOGDIR/pipeline.log" 2>&1 || fail "build_corpus"
log "merged corpus written to data/updated/corpus.jsonl"

notify "pipeline finished" "All new papers are filtered and citation data is collected.

Filter summary (data/updated/filtered/):
$(sed -n '/^   venue year/,$p' "$(ls -t $LOGDIR/filter_run_*.log | head -1)")

Citation coverage (data/updated/citations/summary.txt):
$(cat data/updated/citations/summary.txt)"
