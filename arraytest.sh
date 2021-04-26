#!/usr/bin/env bash
#SBATCH --job-name MySlurmJob # CHANGE this to a name of your choice
#SBATCH --partition batch # equivalent to PBS batch
##SBATCH --time 24:00:00 # Run 24 hours
#SBATCH --qos=1gpulong # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
##SBATCH --nodelist=nv-ai-01.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
#SBATCH --array=0-6
#srun singularity build --fakeroot tensorflow_custom.sif <build-file-path>

declare -a path=("TF-blocks" "TF-cells" "TF-conditioning" "TF-dropout" "TF-hiddensize" "TF-layers" "TF-numhidden")
declare -a param=("blocks" "cells" "conditioning" "dropout_rate" "hidden_size" "layers" "hum_hidden")
declare -a step=(1 5 10 5 10 1 1)
declare -a div=(0 0 0 100 0 0 0)
echo "this is job $SLURM_ARRAY_JOB_ID with task $SLURM_ARRAY_TASK_ID"
srun singularity exec --nv -B src:/src -B results:/results -B data/"$SLURM_ARRAY_TASK_ID":/data  mxnet_21.02-py3.sif python /src/main.py results/Pems/"${path[$SLURM_ARRAY_TASK_ID]}"/metadata.json "${param[$SLURM_ARRAY_TASK_ID]}" "${step[$SLURM_ARRAY_TASK_ID]}" "${div[$SLURM_ARRAY_TASK_ID]}"
echo "finished successful"