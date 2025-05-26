#!/usr/bin/env bash

EXEC=./exponentialIntegral.out
TIMING="-t"
CSV="gpu_only.csv"

BLOCKS=(16 24 32 48 64 96 128 160 192 256 320 384 448 512 768 1024)

PROBLEMS=(
  "5000 5000"
  "8192 8192"
  "16384 16384"
  "20000 20000"
)

echo "n,m,block,GPUf_ms,GPUd_ms" > $CSV

for size in "${PROBLEMS[@]}"; do
  read n m <<< "$size"
  for blk in "${BLOCKS[@]}"; do
    echo "▶ Testing n=$n m=$m blk=$blk"
    out=$($EXEC -n $n -m $m $TIMING -c -B $blk)

    GPUf=$(echo "$out" | awk '/GPU \(float\)/{print $4}')
    GPUd=$(echo "$out" | awk '/GPU \(double\)/{print $3}')

    echo "$n,$m,$blk,$GPUf,$GPUd" >> $CSV
  done
done

echo "GPU-only done. Results saved in $CSV"
