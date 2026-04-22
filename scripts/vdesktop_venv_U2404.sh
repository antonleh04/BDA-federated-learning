#!/bin/bash

### virtual environment ###
cd $HOME
python3 -m venv venv_flwr
cd $HOME/Desktop/GORDE

### packages ###
$HOME/venv_flwr/bin/pip3 install flwr==1.28.0
$HOME/venv_flwr/bin/pip3 install torch==2.11.0
$HOME/venv_flwr/bin/pip3 install matplotlib
$HOME/venv_flwr/bin/pip3 install scikit-learn
$HOME/venv_flwr/bin/pip3 install pandas

### jupyter-notebook ###
#$HOME/venv_flwr/bin/pip3 install notebook
$HOME/venv_flwr/bin/pip3 install ipykernel
$HOME/venv_flwr/bin/python3 -m ipykernel install --user --name=venv_flwr
/usr/bin/jupyter-notebook

### test breastcancer example ###
### HTTP server
# $ cd $HOME/Desktop/GORDE/breastcancer/http_data
# $ $HOME/venv_flwr/bin/python3 -m http.server -b 127.0.0.1 8000
### Activate the venv
# $ cd $HOME/Desktop/GORDE/breastcancer/
# $ source $HOME/venv_flwr/bin/activate
### FLOWER
# $ cd $HOME/Desktop/GORDE/breastcancer/
# $ pip3 install -e .
# $ flwr run . --stream
# $ flwr run . --stream --federation-config="num-supernodes=4 init-args-num-cpus=2 init-args-num-gpus=0"
### Deactivate
# $ deactivate
