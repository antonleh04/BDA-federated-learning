import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, recall_score, roc_auc_score

DATA_PATH = "http://127.0.0.1:8000/Covid19.csv"

FEATURE_COLS = [
    "USMER", "SEX", "PATIENT_TYPE", "PNEUMONIA", "AGE", "PREGNANT",
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION", "OTHER_DISEASE",
    "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO", "CLASIFFICATION_FINAL"
]


def Net():
    model = nn.Sequential()
    model.add_module('hidden0', nn.Linear(17, 60))
    model.add_module('activation0', nn.ReLU())
    model.add_module('dr0', nn.Dropout(0.3))
    model.add_module('output', nn.Linear(60, 1))
    model.add_module('sigmoid', nn.Sigmoid())
    return model



def preprocess_covid(df):
    df = df.copy()
    # SEX: Female -> 0, Male -> 1
    df["SEX"] = (df["SEX"] == "Male").astype(int)
    # CLASIFFICATION_FINAL: ordinal 1,2,3 -> 0, 0.5, 1
    df["CLASIFFICATION_FINAL"] = (df["CLASIFFICATION_FINAL"] - 1) / 2.0
    # AGE is left as-is; StandardScaler in load_data/load_centralized_dataset handles it
    return df





def load_data(partition_id: int, batch_size: int):
    df = pd.read_csv(DATA_PATH, index_col=0)

    # partition by medical unit
    df = df[df["MEDICAL_UNIT"] == partition_id]

    df = preprocess_covid(df)

    x = df[FEATURE_COLS].values
    y = df["DEATH"].values

    # apply stratified train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # oversample DEATH==1 in train set
    ros = RandomOverSampler(random_state=42)
    x_train, y_train = ros.fit_resample(x_train, y_train)

    # standardize, fit on train set
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_t = torch.as_tensor(x_train).float()
    y_train_t = torch.as_tensor(y_train.reshape(-1, 1)).float()
    x_test_t = torch.as_tensor(x_test).float()
    y_test_t = torch.as_tensor(y_test.reshape(-1, 1)).float()

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size)

    return train_loader, test_loader


def load_centralized_dataset(batch_size: int = 16):
    df = pd.read_csv(DATA_PATH, index_col=0)

    df = preprocess_covid(df)

    x = df[FEATURE_COLS].values
    y = df["DEATH"].values

    # holdout split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # oversample DEATH==1 in train set
    ros = RandomOverSampler(random_state=42)
    x_train, y_train = ros.fit_resample(x_train, y_train)

    # standardize
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_t = torch.as_tensor(x_train).float()
    y_train_t = torch.as_tensor(y_train.reshape(-1, 1)).float()
    x_test_t = torch.as_tensor(x_test).float()
    y_test_t = torch.as_tensor(y_test.reshape(-1, 1)).float()

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size)

    return train_loader, test_loader
 
 
class StepByStep(object):
    def __init__(self, model, loss_fn=None, optimizer=None, proximal_mu=0.0, global_params=None):
        # Here we define the attributes of our class
        # We start by storing the arguments as attributes
        # to use later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # FedProx proximal term: snapshot of the global weights at round start
        # and the strength mu. Set mu=0 to disable.
        self.proximal_mu = proximal_mu
        self.global_params = global_params
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not available at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.total_epochs = 0

        # Creates the train_step function for our model,
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader
        # (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_valloader(self, val_loader):
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can use directly
        # the attributes: self.model, self.loss_fn and self.optimizer
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - Computes model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # FedProx: add (mu/2) * ||w - w_global||^2 proximal term.
            # Skipped when mu=0 or no global snapshot is set (plain FedAvg / centralized).
            if self.proximal_mu > 0.0 and self.global_params is not None:
                proximal_term = 0.0
                for w, w_global in zip(self.model.parameters(), self.global_params):
                    proximal_term = proximal_term + (w - w_global).pow(2).sum()
                loss = loss + (self.proximal_mu / 2.0) * proximal_term
            # Step 3 - Computes gradients for "b" and "w" parameters
            loss.backward()
            # Step 4 - Updates parameters using gradients and the
            # learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn

    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()
            
            # Step 1 - Computes model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # There is no need to compute Steps 3 and 4,
            # since we don't update parameters during evaluation
            return loss.item()
        
        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation` defines which loader and
        # corresponding step function are going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
        
        if data_loader is None:
            return None

        # Once the data loader and step function are set, this is the
        # same mini-batch loop we had before
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        
        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=None):
        if seed is not None:
            self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the number of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1
            
            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            
            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

    def load_state_dict(self, server_state_dict):
        # Restore state for model and optimizer
        self.model.load_state_dict(server_state_dict)
        self.model.train() # always use TRAIN for resuming training

    def predict(self, x):
        # Set it to evaluation mode for predictions
        self.model.eval()
        # Take a Numpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and use model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detach it, bring it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()


def train(model, trainloader, valoader, epochs, lr, device, proximal_mu=0.0):
    # Move model to GPU if available
    model.to(device)

    # Defines a binary cross-entropy loss function
    loss_fn = nn.BCELoss()

    # Defines an SGD optimizer to update the parameters
    # (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train mode
    model.to(device)

    # FedProx: snapshot the global weights received from the server before
    # any local SGD steps. The proximal penalty pulls local updates back
    # toward this anchor. mu=0 disables the term (== FedAvg / centralized).
    global_params = None
    if proximal_mu > 0.0:
        global_params = [p.detach().clone() for p in model.parameters()]

    # Train the model
    sbs = StepByStep(model, loss_fn, optimizer,
                     proximal_mu=proximal_mu, global_params=global_params)
    sbs.set_loaders(trainloader, valoader)
    sbs.train(epochs)
    
    # Get average of all losses
    avg_train_loss = sum(sbs.losses) / len(sbs.losses)
    return avg_train_loss

def test(model, testloader, device):
    model.to(device)
    sbs = StepByStep(model)
    loss_fn = nn.BCELoss()

    loss_sum = 0
    all_probs = []
    all_preds = []
    all_labels = []

    for x_data, y_data in testloader:
        probs = torch.as_tensor(sbs.predict(x_data)).float()
        loss_sum += loss_fn(probs, y_data).item()
        all_probs.append(probs.detach().cpu())
        all_preds.append(torch.round(probs).detach().cpu())
        all_labels.append(y_data.detach().cpu())

    all_probs = torch.cat(all_probs).numpy().flatten()
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()

    loss = loss_sum / len(testloader)
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    return loss, accuracy, precision, recall, auc

if __name__ == "__main__":
	lr = 0.1
	torch.manual_seed(42)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = Net()

	# Match federated setup: batch_size=32, ~60 effective epochs
	# (20 server rounds x 3 local epochs) for a fair baseline.
	train_loader, test_loader = load_centralized_dataset(batch_size=32)

	epochs = 60

	trloss = train(model, train_loader, test_loader, epochs, lr, device)
	print(trloss)

	loss, accuracy, precision, recall, auc = test(model, test_loader, device)
	print(f"loss={loss:.4f}  acc={accuracy:.4f}  prec={precision:.4f}  rec={recall:.4f}  auc={auc:.4f}")

	torch.save(model.state_dict(), "models/final_model_centralized.pt")
