#!/bin/bash
# run_forever.sh — crash-proof charslot trainer loop for the GPU box.
#   usage: bash scripts/run_forever.sh <TAG> <CORECFG> <TARGET_STEPS> [extra trainer args...]
# Resumes from the latest rolling checkpoint (pointer.json) after any crash.
# To fine-tune from a prior run: copy its ckpt + pointer.json into runs/<TAG>/
# before launching — the resume logic picks it up as its own.
# Stops when TARGET_STEPS is reached or after MAX_RESTARTS (guards against a
# persistent crash/tripwire loop burning the box).
cd /workspace/axon || exit 1
TAG=$1; CFG=$2; TARGET=$3; shift 3
mkdir -p "runs/$TAG"
MAX_RESTARTS=50
n=0
while [ $n -lt $MAX_RESTARTS ]; do
  RESUME=""
  if [ -f "runs/$TAG/pointer.json" ]; then
    ACTIVE=$(python3 -c "import json;print(json.load(open('runs/$TAG/pointer.json'))['active'])" 2>/dev/null)
    [ -n "$ACTIVE" ] && [ -f "runs/$TAG/$ACTIVE" ] && RESUME="--resume runs/$TAG/$ACTIVE"
  fi
  echo "[run_forever] start #$n target=$TARGET resume='$RESUME' $(date)" >> "runs/$TAG/train.log"
  python3 training/trainer_slot.py --threshold charslot --device cuda \
    --core-cfg "$CFG" --mode phase0 --steps "$TARGET" \
    --history-chars 256 --user-chars 64 --eval-samples 2 \
    --eval-every 500 --checkpoint-every 1000 --log-every 50 \
    --containers-path "" \
    --run-dir "runs/$TAG" $RESUME "$@" >> "runs/$TAG/train.log" 2>&1
  echo "[run_forever] trainer exited code=$? $(date)" >> "runs/$TAG/train.log"
  if grep -qE "charslot training finished at step=$TARGET|already reached target" "runs/$TAG/train.log"; then
    echo "[run_forever] target reached; stopping $(date)" >> "runs/$TAG/train.log"
    break
  fi
  n=$((n+1))
  sleep 15
done
