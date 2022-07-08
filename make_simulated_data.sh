#!/bin/bash
set -euo pipefail

PROTDIR=$LIB_DIR/mutsim
INDIR=$DATA_DIR 

PARALLEL=40

function make_simulated_data {
  mkdir -p $INDIR

  for K in 10 30 50 100 200; do
  for S in 10 20 30; do
  for T in 50 200 1000; do
  for M_per_cluster in 1; do
  for G_frac in 0; do
  for run in $(seq 1); do
    M=$(echo "$K * $M_per_cluster" | bc)
    G=$(echo "(($G_frac * $M) + 0.5) / 1" | bc)
        
    jobname="sim_K${K}_S${S}_T${T}_M${M}_G${G}_run${run}"
    
    if [ -d "$INDIR/$jobname" ]; 
    then
      continue 
    else
      mkdir $INDIR/$jobname
      jobdir=$INDIR/$jobname/truth
      mkdir $jobdir
      python3 $PROTDIR/make_simulated_data.py \
        --write-clusters \
        --write-numpy $jobdir/$jobname.truth.npz \
        -K $K \
        -S $S \
        -T $T \
        -M $M \
        -G $G \
        $jobdir/$jobname.truth.pickle \
        $jobdir/$jobname.params.json \
        $jobdir/$jobname.ssm \
        > $jobdir/$jobname.stdout \
        2>$jobdir/$jobname.stderr 
    fi
    
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
