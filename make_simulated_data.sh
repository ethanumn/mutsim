#!/bin/bash
set -euo pipefail

PROTDIR=$LIB_DIR/mutsim
INDIR=$DATA_DIR 

PARALLEL=40

function make_simulated_data {
  mkdir -p $INDIR

  for K in 10 20 30 50 100 150 200 250 300; do
  for S in 1 3 10 30 100; do
  for T in 50 200 1000; do
  for M_per_cluster in 1; do
  for G_frac in 0; do
  for run in $(seq 1); do
    M=$(echo "$K * $M_per_cluster" | bc)
    G=$(echo "(($G_frac * $M) + 0.5) / 1" | bc)
        
    jobname="sim_K${K}_S${S}_T${T}_M${M}_G${G}_run${run}"
    (
    mkdir $INDIR/$jobname
    python3 $PROTDIR/make_simulated_data.py \
      --write-clusters \
      --write-numpy $INDIR/$jobname/$jobname.truth.npz \
      -K $K \
      -S $S \
      -T $T \
      -M $M \
      -G $G \
      $INDIR/$jobname/$jobname.truth.pickle \
      $INDIR/$jobname/$jobname.params.json \
      $INDIR/$jobname/$jobname.ssm \
      > $INDIR/$jobname/$jobname.stdout \
      2>$INDIR/$jobname/$jobname.stderr \
    )
  done
  done
  done
  done
  done
  done #| parallel -j$PARALLEL --halt 1
}

function main {
  make_simulated_data
}

main
