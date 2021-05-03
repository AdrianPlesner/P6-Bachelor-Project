Bootstrap: docker
From: nvcr.io/nvidia/mxnet:21.02-py3

%post
pip install tables
pip install numpy
pip install pandas
pip install mxnet
pip install sklearn
pip install git+https://github.com/awslabs/gluon-ts.git@master#egg=gluonts
pip install --no-deps pytorchts==0.3.1
pip install wandb
pip install torch
pip install scipy

