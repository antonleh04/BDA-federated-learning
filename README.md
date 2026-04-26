BDA federated learning

Team: 
- Maddi Olaetxea Usabiaga
- Johanka Jakubove
- Anton Lehnerer



setup:

```bash
pip install -e .
```

running the federated learning

- first serve the data:
```bash
cd data
python3 -m http.server -b 127.0.0.1 8000
```

- run the federated strategies:
```bash
flwr run . --stream --federation-config="num-supernodes=12"
flwr run . --stream --federation-config="num-supernodes=12" --run-config="strategy-name='fedavg-unweighted'"
flwr run . --stream --federation-config="num-supernodes=12" --run-config="strategy-name='fedprox-weighted'"
flwr run . --stream --federation-config="num-supernodes=12" --run-config="strategy-name='fedprox-unweighted'"
```


