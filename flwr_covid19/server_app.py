"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from dataclasses import replace as dc_replace

from flwr_covid19.centralized import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


class UnweightedFedAvg(FedAvg):
    """FedAvg with equal client weight regardless of local dataset size."""

    def aggregate_fit(self, server_round, results, failures):
        equal_results = [
            (proxy, dc_replace(fit_res, num_examples=1))
            for proxy, fit_res in results
        ]
        return super().aggregate_fit(server_round, equal_results, failures)


class UnweightedFedProx(FedProx):
    """FedProx with equal client weight regardless of local dataset size."""

    def aggregate_fit(self, server_round, results, failures):
        equal_results = [
            (proxy, dc_replace(fit_res, num_examples=1))
            for proxy, fit_res in results
        ]
        return super().aggregate_fit(server_round, equal_results, failures)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    strategy_name: str = context.run_config["strategy-name"]
    proximal_mu: float = context.run_config["proximal-mu"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize strategy
    strategy_kwargs = dict(fraction_evaluate=fraction_evaluate)

    if strategy_name == "fedavg-weighted":
        strategy = FedAvg(**strategy_kwargs)
    elif strategy_name == "fedavg-unweighted":
        strategy = UnweightedFedAvg(**strategy_kwargs)
    elif strategy_name == "fedprox-weighted":
        strategy = FedProx(proximal_mu=proximal_mu, **strategy_kwargs)
    elif strategy_name == "fedprox-unweighted":
        strategy = UnweightedFedProx(proximal_mu=proximal_mu, **strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy-name: {strategy_name!r}. "
                         "Choose from: fedavg-weighted, fedavg-unweighted, "
                         "fedprox-weighted, fedprox-unweighted")

    # Start strategy, run for num_rounds.
    # proximal_mu is forwarded to clients so FedProx variants apply the proximal
    # penalty locally; FedAvg variants ignore it (client uses 0.0 by default).
    train_cfg = {"lr": lr}
    if strategy_name in ("fedprox-weighted", "fedprox-unweighted"):
        train_cfg["proximal_mu"] = proximal_mu

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_cfg),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"models/final_model_{strategy_name}.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global model on the full held-out test set."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set (train loader discarded — only test set used here)
    _, test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    loss, accuracy, precision, recall, auc = test(model, test_dataloader, device)

    print(f"[Round {server_round}] loss={loss:.4f}  acc={accuracy:.4f}  "
          f"prec={precision:.4f}  rec={recall:.4f}  auc={auc:.4f}")

    # Return the evaluation metrics
    return MetricRecord({
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    })
