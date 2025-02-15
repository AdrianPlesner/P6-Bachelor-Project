#!/usr/bin/env bash
#SBATCH --job-name CNFlows # CHANGE this to a name of your choice
#SBATCH --partition batch # equivalent to PBS batch
##SBATCH --time 24:00:00 # Run 24 hours
#SBATCH --qos=allgpus # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
##SBATCH --nodelist=nv-ai-01.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
#SBATCH --array=0-4
#srun singularity build --fakeroot mxnet_21.02-py3.sif Singularity

declare -a path=("GP-default/metadata.json" "GP-periodic/metadata.json" "DF-combine/metadata.json" "DAR-combine/metadata.json" "TF-combine/m3.json")
echo "this is job $SLURM_ARRAY_JOB_ID with task $SLURM_ARRAY_TASK_ID"
SLEEP 5 * "$SLURM_ARRAY_TASK_ID"
srun singularity exec --nv -B src:/src -B results:/results -B data/"$SLURM_ARRAY_TASK_ID":/data  mxnet_21.02-py3.sif python /src/validate.py results/Pems/"${path[$SLURM_ARRAY_TASK_ID]}"
echo "finished successful"