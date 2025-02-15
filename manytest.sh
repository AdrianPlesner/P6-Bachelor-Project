#!/usr/bin/env bash
#SBATCH --job-name MySlurmJob # CHANGE this to a name of your choice
#SBATCH --partition batch # equivalent to PBS batch
##SBATCH --time 24:00:00 # Run 24 hours
#SBATCH --qos=allgpus # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
##SBATCH --nodelist=nv-ai-01.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
#SBATCH --array=0-5
#srun singularity build --fakeroot mxnet_21.02-py3.sif Singularity

#declare -a path=("DF-combine" "DAR-combine" "TF-combine")
echo "this is job $SLURM_ARRAY_JOB_ID with task $SLURM_ARRAY_TASK_ID"
srun singularity exec --nv mxnet_21.02-py3.sif python test.py
echo "finished successful"