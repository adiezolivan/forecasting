import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import random


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Processing available: " + str(device))


def seed_all(seed):
    if not seed:
        seed = 10
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n['sales'].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()
        # Makes predictions
        yhat = self.model(x)
        # Computes loss
        loss = self.loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    def train(self,
              train_loader,
              val_loader,
              batch_size=64,
              n_epochs=50,
              n_features=1,
              model_name='model.pt',
              verbose=1):
        # early stop stuff
        min_avg_loss = np.Inf
        n_epochs_stop = 6
        epochs_no_improve = 0
        early_stop = False
        for epoch in range(1, n_epochs + 1):
            avg_loss = 0.
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                #y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)

                avg_loss += loss
                # If the avg loss is at a minimum
                if avg_loss < min_avg_loss:
                    epochs_no_improve = 0
                    min_avg_loss = avg_loss
                else:
                    epochs_no_improve += 1
                if epoch > 5 and epochs_no_improve == n_epochs_stop:
                    if verbose > 0:
                        print('Early stopping!')
                    early_stop = True
                    break
                else:
                    continue
                break

            if early_stop:
                if verbose > 0:
                    print("Stopped")
                break

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    #y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            if verbose > 0:
                if (epoch <= 10) | (epoch % 50 == 0):
                    print(
                        f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                    )
        # save model to file
        torch.save(self.model.state_dict(), model_name)

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                #y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

