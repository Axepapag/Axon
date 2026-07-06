#!/bin/bash
# run_forever.sh — crash-proof charslot trainer loop for the GPU box.
#   usage: bash ops/run_forever.sh <TAG> <CORECFG> [extra trainer args...]
# Resumes from the latest rolling checkpoint (pointer.json) after any crash.
# Stops when the 200k target is reached or after MAX_RESTARTS (guards against
# a persistent crash/tripwire loop burning the box).
cd /workspace/axon || exit 1
TAG=$1; CFG=$2; shift 2
mkdir -p "runs/$TAG"
MAX_RESTARTS=50
n=0
while [ $n -lt $MAX_RESTARTS ]; do
  RESUME=""
  if [ -f "runs/$TAG/pointer.json" ]; then
    ACTIVE=$(python3 -c "import json;print(json.load(open('runs/$TAG/pointer.json'))['active'])" 2>/dev/null)
    [ -n "$ACTIVE" ] && [ -f "runs/$TAG/$ACTIVE" ] && RESUME="--resume runs/$TAG/$ACTIVE"
  fi
  echo "[run_forever] start #$n resume='$RESUME' $(date)" >> "runs/$TAG/train.log"
  python3 training/trainer_slot.py --threshold charslot --device cuda \
    --core-cfg "$CFG" --mode phase0 --steps 200000 \
    --history-chars 256 --user-chars 64 --eval-samples 2 \
    --eval-every 500 --checkpoint-every 1000 --log-every 50 \
    --containers-path "" \
    --run-dir "runs/$TAG" $RESUME "$@" >> "runs/$TAG/train.log" 2>&1
  echo "[run_forever] trainer exited code=$? $(date)" >> "runs/$TAG/train.log"
  if grep -q "charslot training finished at step=200000" "runs/$TAG/train.log"; then
    echo "[run_forever] target reached; stopping $(date)" >> "runs/$TAG/train.log"
    break
  fi
  n=$((n+1))
  sleep 15
done
