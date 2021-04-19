#!/usr/bin/env bash
#SBATCH --job-name MySlurmJob # CHANGE this to a name of your choice
#SBATCH --partition batch # equivalent to PBS batch
##SBATCH --time 24:00:00 # Run 24 hours
#SBATCH --qos=1gpulong # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
##SBATCH --nodelist=nv-ai-01.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk

#srun singularity build --fakeroot tensorflow_custom.sif <build-file-path>

srun singularity exec --nv -B src:/src -B results:/results -B data:/data  mxnet_21.02-py3.sif python /src/main.py results/Pems/DAR-cells/metadata.json cells 5
srun echo "finished successful"