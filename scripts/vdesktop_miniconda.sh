#!/bin/bash

### Move to your working directory ###
#cd $HOME/Desktop/GORDE
cd /media/scppebai/Sync2TB/repo_unison/IRAKASKUNTZA/BDA/Flower

### conda ###
# $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# $ bash ./Miniconda3-latest-Linux-x86_64.sh

### accept terms ###
$HOME/miniconda3/bin/conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/main
$HOME/miniconda3/bin/conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/r
$HOME/miniconda3/bin/conda init

### virtual environment ###
$HOME/miniconda3/bin/conda create --name venv_flwr python=3.12.3 -y

### packages ###
$HOME/miniconda3/envs/venv_flwr/bin/pip3 install flwr==1.28.0
$HOME/miniconda3/envs/venv_flwr/bin/pip3 install torch==2.11.0
$HOME/miniconda3/envs/venv_flwr/bin/pip3 install matplotlib
$HOME/miniconda3/envs/venv_flwr/bin/pip3 install scikit-learn
$HOME/miniconda3/envs/venv_flwr/bin/pip3 install pandas

### jupyter-notebook ###
$HOME/miniconda3/envs/venv_flwr/bin/pip3 install notebook
$HOME/miniconda3/envs/venv_flwr/bin/pip3 install ipykernel
$HOME/miniconda3/envs/venv_flwr/bin/python3 -m ipykernel install \
  --user --name=venv_flwr
$HOME/miniconda3/envs/venv_flwr/bin/jupyter-notebook

### activate environment in other terminal window ###
### Activate the venv
# $ conda env list
# $ conda activate venv_flwr
### HTTP server
# $ cd $HOME/Desktop/GORDE/breastcancer/http_data
# $ python3 -m http.server -b 127.0.0.1 8000
### FLOWER
# $ cd $HOME/Desktop/GORDE/breastcancer/
# $ pip3 install -e .
# $ flwr run . --stream
# $ flwr run . --stream --federation-config="num-supernodes=4 init-args-num-cpus=2 init-args-num-gpus=0"
### Deactivate
# $ conda deactivate
