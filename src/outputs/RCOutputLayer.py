import os
import torch
import numpy as np

class FullyConnected:
    """
    A simple output layer for a reservoir computer. It is a fully connected linear layer followed by a softmax layer.

    Attributes:
    RC_output_layer (torch.nn.Sequential): the output layer
    loss_fn (torch.nn.CrossEntropyLoss): the loss function
    optimizer (torch.optim.Adam): the optimizer

    Methods:
    forward: forward pass
    train_epoch: train the model for one epoch
    train: train the model for multiple epochs
    save: save the model to a file
    
    """

    def __init__(self, n_input, n_output,lr = 0.001, bias=True, name = "RC_output_layer"):
        """
        Parameters:
        n_input (int): number of input nodes
        n_output (int): number of output nodes
        bias (bool): whether to use a bias term in the linear layer

        """
        self.RC_output_layer = torch.nn.Sequential()
        self.RC_output_layer.add_module('output', torch.nn.Linear(n_input, n_output))
        self.RC_output_layer.add_module('softmax', torch.nn.Softmax(dim=1))

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.RC_output_layer.parameters(), lr=lr)

        self.name = name

    def forward(self, x):
        """

        Parameters:
        x (torch.Tensor): input tensor

        Returns:
        torch.Tensor: output tensor

        """
        return self.RC_output_layer(x)
    
    def train_epoch(self, x, y):
        """
        
        Parameters:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor

        Returns:
        torch.Tensor: loss
        """
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train(self, x_train, y_train, x_test = None, y_test = None, epochs=10, verbose = True):
        """
        Train the model for multiple epochs

        Parameters:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        epochs (int): number of epochs
        verbose (bool): whether to print the loss

        Returns:
        history (list): list of losses

        """

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train).float()
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train).long()

        if x_test is not None and not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test).float()
        if y_test is not None and not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test).long()

        history = {"loss":[]}
        if x_test is not None:
            history["test_loss"] = []
        for epoch in range(epochs):
            loss = self.train_epoch(x_train, y_train)

            if x_test is not None:

                

                y_pred = self.forward(x_test)
                test_loss = self.loss_fn(y_pred, y_test)

                history["loss"].append(loss.item())
                history["test_loss"].append(test_loss.item())

                if verbose:
                        print(f'Epoch {epoch} loss: {loss}, test loss: {test_loss}')
            else:
                test_loss = None

                if verbose:
                    print(f'Epoch {epoch} loss: {loss}')
                history["loss"].append(loss.item())

        return history
    
    def save(self, path):
        """
        Save the model to a file

        Parameters:
        path (str): path to the file

        

        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.RC_output_layer.state_dict(), os.path.join(path, self.name + ".pickle"))

    def load(self, path):
        """
        Load the model from a pickle file

        Parameters:
        path (str): path to the file

        """
        self.RC_output_layer.load_state_dict(torch.load(path))




def getVirtualNodeIndexes(N, tau, t):
    theta = tau / N
    indexes = [0]
    for i in range(len(t)):
        if t[i] > theta * (len(indexes)):
            indexes.append(i)

    return indexes

def prepareDataForOutputLayer(sample, N, tau):
    """
    Prepare the data for the output layer by selecting the value at the time of the virtual nodes of each channel, and concatenating them.
    
    If there are K channels, and N virtual nodes, the input tensor will have K*N elements per period tau
    
    Parameters:
    sample (list): list of samples
    N (int): number of nodes in the reservoir
    tau (float): time constant of the reservoir

    Returns:
    X (torch.Tensor): input tensor
    Y (torch.Tensor): target tensor

    """
    X = []
    Y = []

    for item in sample:
        indexes = getVirtualNodeIndexes(N, tau, item['t'])
        x = np.zeros(len(indexes)*len(item['channels']))
                    
        for i in range(len(item['channels'])):
            x[i*len(indexes):(i+1)*len(indexes)] = item['channels'][i][indexes].reshape(-1)
        

        X.append(x)
        Y.append(item['label'])

    X = torch.tensor(X).float()
    Y = torch.tensor(np.array(Y).astype(int))
    return X, Y