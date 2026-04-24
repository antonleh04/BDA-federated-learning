"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flwr_covid19.centralized import StepByStep, Net, load_data
from flwr_covid19.centralized import test as test_fn
from flwr_covid19.centralized import train as train_fn

# Flower ClientApp
app = ClientApp()

# MEDICAL_UNIT values used as federated clients.
# Unit 0 is excluded (only 12 records, 0 deaths — stratified split would crash).
# Unit 2 is absent from the dataset entirely.
MEDICAL_UNITS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]



@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    sbs = StepByStep(model)
    sbs.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    medical_unit = MEDICAL_UNITS[partition_id]
    trainloader, valloader = load_data(medical_unit, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        valloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        sbs.device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord({
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    })
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local test data."""

    # Load the model and initialize it with the received weights
    model = Net()
    sbs = StepByStep(model)
    sbs.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    medical_unit = MEDICAL_UNITS[partition_id]
    _, valloader = load_data(medical_unit, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc, eval_prec, eval_rec, eval_auc = test_fn(
        model,
        valloader,
        sbs.device,
    )

    # Construct and return reply Message
    metric_record = MetricRecord({
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "eval_precision": eval_prec,
        "eval_recall": eval_rec,
        "eval_auc": eval_auc,
        "num-examples": len(valloader.dataset),
    })
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
