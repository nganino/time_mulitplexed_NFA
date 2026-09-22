#!/usr/bin/env bash
# Tuning chain: N workers (one per GPU, time-boxed) -> report -> best trial + baseline on the full
# grid at the full budget -> report.
#   setsid nohup bash run_tune.sh > logs/tune.log 2>&1 &
#   SMOKE=1 bash run_tune.sh                       # end-to-end check with the tiny study
# Re-running resumes: the study lives in a journal file and finished cards are skipped.
set -u
cd "$(dirname "$0")"
GPUS=${GPUS:-0,1}          # comma-separated GPU ids, one worker each
NT=${NT:-150}              # trials per worker INCLUDING pruned ones
LIMIT_H=${LIMIT_H:-11}     # wall-clock cap of the search phase per worker
SMOKE=${SMOKE:-0}
EXTRA=""; [ "$SMOKE" = 1 ] && EXTRA="--smoke" && NT=6 && LIMIT_H=0.2
mkdir -p logs/tune
echo "tune chain start $(date)  GPUS=$GPUS NT=$NT LIMIT_H=$LIMIT_H SMOKE=$SMOKE"
pids=(); i=0
for gpu in ${GPUS//,/ }; do
  name=$(printf "\\x$(printf %x $((65 + i)))")        # A, B, C ...
  enq=""; [ "$i" = 0 ] && enq="--enqueue"
  python tune_worker.py --worker "$name" --gpu "$gpu" --n-trials "$NT" --time-limit-h "$LIMIT_H" $enq $EXTRA \
      >> "logs/tune/worker_${name}.log" 2>&1 &
  pids+=($!); i=$((i + 1)); sleep 20
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "search phase done $(date), $fail worker(s) exited non-zero"
python tune_report.py $EXTRA >> logs/tune/report.log 2>&1
if [ "$fail" -ge "${#pids[@]}" ]; then
  echo "all workers failed: skipping the final phase"; exit 1
fi
python tune_final.py --gpus "$GPUS" $EXTRA >> logs/tune/final.log 2>&1
echo "final phase done $(date)"
python tune_report.py $EXTRA >> logs/tune/report.log 2>&1
echo "TUNE CHAIN DONE $(date)"
