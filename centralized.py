import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

def Net():
    model = nn.Sequential()
    model.add_module('hidden0', nn.Linear(30, 60))
    model.add_module('activation0', nn.ReLU())
    model.add_module('dr0', nn.Dropout(0.3))
    model.add_module('output', nn.Linear(60, 1))
    model.add_module('sigmoid', nn.Sigmoid())
    return model


#TODO: preprocess features
#TODO: adapt the load_data function to the covid19.csv dataset. we need to divide the dataset by medical unit
#TODO: add oversampling or undersampling and stratified data splitting
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    #partition_id = 3
    #num_partitions = 5
    #batch_size = 16

    # Load data from file
    # df_breast = pd.read_csv("./wdbc.csv", header=None)
    # python3 -m http.server -b 127.0.0.1
    # http://127.0.0.1:8000/wdbc.csv
    df_breast = pd.read_csv("http://127.0.0.1:8000/wdbc.csv")
    
    # Rename columns
    df_breast.columns = ["id", "diagnosis", 
        "a_radius", "a_texture", "a_perimeter", "a_area", "a_smoothness",
        "a_compactness", "a_concavity", "a_concave", "a_symmetry", "a_fractal",
        "b_radius", "b_texture", "b_perimeter", "b_area", "b_smoothness", 
        "b_compactness", "b_concavity", "b_concave", "b_symmetry", "b_fractal",
        "c_radius", "c_texture", "c_perimeter", "c_area", "c_smoothness", 
        "c_compactness", "c_concavity", "c_concave", "c_symmetry", "c_fractal"]

    # Attributes and class
    x = df_breast.iloc[:,2:]
    y = df_breast.iloc[:,1]
    y = y.replace('B', 0).replace('M',1)

    # Make num_partitions splits and select the partition_id
    kf = KFold(n_splits=num_partitions)
    for i, (ktr_index, kts_index) in enumerate(kf.split(x)):
        if i == partition_id:
            break

    # Generate train and validation sets
    x_tot = x.iloc[kts_index.tolist(),]
    y_tot = y.iloc[kts_index.tolist(),]

    # Train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_tot, y_tot, test_size=0.33, shuffle=False)

    # Standardize
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(x_train)
    scaled_x_train = scaler.transform(x_train)
    scaled_x_val = scaler.transform(x_val)

    # Transform into PyTorch's Tensors
    x_train_tensor = torch.as_tensor(scaled_x_train).float()
    y_train_tensor = torch.as_tensor(np.array(y_train, dtype=float).reshape((-1,1))).float()
    x_val_tensor = torch.as_tensor(scaled_x_val).float()
    y_val_tensor = torch.as_tensor(np.array(y_val, dtype=float).reshape((-1,1))).float()

    # Builds Dataset
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    val_data = TensorDataset(x_val_tensor, y_val_tensor)

    # Builds DataLoader
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_data, 
        batch_size=batch_size)

    return train_loader, val_loader


#TODO: load covid19 dataset
#TODO: also return dataloader for test split
def load_centralized_dataset():
    # Load data from file
    # df_breast = pd.read_csv("./wdbc.csv", header=None)
    # python3 -m http.server -b 127.0.0.1
    # http://127.0.0.1:8000/wdbc.csv
    df_breast = pd.read_csv("http://127.0.0.1:8000/wdbc.csv", header=None)

    # Rename columns
    df_breast.columns = ["id", "diagnosis", 
        "a_radius", "a_texture", "a_perimeter", "a_area", "a_smoothness",
        "a_compactness", "a_concavity", "a_concave", "a_symmetry", "a_fractal",
        "b_radius", "b_texture", "b_perimeter", "b_area", "b_smoothness", 
        "b_compactness", "b_concavity", "b_concave", "b_symmetry", "b_fractal",
        "c_radius", "c_texture", "c_perimeter", "c_area", "c_smoothness", 
        "c_compactness", "c_concavity", "c_concave", "c_symmetry", "c_fractal"]

    # Attributes and class
    x = df_breast.iloc[:,2:]
    y = df_breast.iloc[:,1]
    y = y.replace('B', 0).replace('M',1)

    # Train and validation sets
    x_train, y_train = x, y
    
    # Standardize
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(x_train)
    scaled_x_train = scaler.transform(x_train)

    # Transform into PyTorch's Tensors
    x_train_tensor = torch.as_tensor(scaled_x_train).float()
    y_train_tensor = torch.as_tensor(np.array(y_train, dtype=float).reshape((-1,1))).float()

    # Builds Dataset
    train_data = TensorDataset(x_train_tensor, y_train_tensor)

    # Builds DataLoader
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=16,
        shuffle=True,
    )

    return train_loader
 
 
class StepByStep(object):
    def __init__(self, model, loss_fn=None, optimizer=None):
        # Here we define the attributes of our class
        # We start by storing the arguments as attributes
        # to use later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
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


def train(model, trainloader, valoader, epochs, lr, device):
    # Move model to GPU if available
    model.to(device)

    # Defines a binary cross-entropy loss function
    loss_fn = nn.BCELoss()
    
    # Defines an SGD optimizer to update the parameters
    # (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train mode
    model.to(device)

    # Train the model
    sbs = StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(trainloader, valoader)
    sbs.train(epochs)
    
    # Get average of all losses
    avg_train_loss = sum(sbs.losses) / len(sbs.losses)
    return avg_train_loss.item()

#TODO: add other metrics: precision + recall or the ROC curve(AUC)
def test(model, testloader, device):
    # Move model to GPU if available
    model.to(device)
    
    # Initialize the class
    sbs = StepByStep(model)

    # Defines a binary cross-entropy loss function
    loss_fn = nn.BCELoss()

    # Test example per example
    loss_sum = 0
    hits_sum = 0
    n = 0
    for batch in testloader:
        x_data = batch[0]
        y_data = batch[1]
        predictions = torch.as_tensor(sbs.predict(x_data)).float()
        # loss
        loss_sum += loss_fn(predictions, y_data).item()
        # accuracy
        hits_sum += (torch.round(predictions) == y_data).sum().item()
        
    accuracy = hits_sum / len(testloader.dataset)
    loss = loss_sum / len(testloader)
    return loss, accuracy

if __name__ == "__main__":
	#def train(net, trainloader, epochs, lr, device):
	# Sets learning rate - this is "eta" ~ the "n"-like Greek letter
	lr = 0.1

	# Reproducibility
	torch.manual_seed(42)

	# Now we can create a model and send it at once to the device
	model = Net()

	# GPU available?
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	tr_loader, val_loader = load_data(0, 5, 16)

	epochs = 200

	trloss = train(model, tr_loader, val_loader, epochs, lr, device)
	print(trloss)

	loss, accuracy = test(model, val_loader, device)
	print(loss)
	print(accuracy)
	
	test_loader = load_centralized_dataset()
	loss, accuracy = test(model, test_loader, device)
	print(loss)
	print(accuracy)
