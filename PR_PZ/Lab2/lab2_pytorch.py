#%%
import torch
from numpy.f2py.auxfuncs import throw_error
from sklearn.metrics import r2_score, mean_squared_error
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, ReLU, Sequential, RNN
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
from sklearn.model_selection import train_test_split
#%%
# Defining device for training, x, y and f(x, y)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, y = torch.arange(0, 10, 0.1), torch.arange(0, 10, 0.1)
XX, yy = torch.meshgrid(X, y, indexing='ij')

X_cords = XX.flatten()
y_cords = yy.flatten()

z = 0.5 * X_cords ** 2 - 2 * y_cords ** 2 + X_cords - 1

X_y = torch.cat((X_cords.unsqueeze(1), y_cords.unsqueeze(1)), dim=1)
X_y_train, X_y_test, z_train, z_test = [i.to(device) for i in train_test_split(X_y, z, test_size=0.2, random_state=42)]
print(f'Device: {device}')
len(X_y_train), len(X_y_test), X_y[:5]
#%%
X_y_train.shape, z_train.shape, z_test.shape
#%%
# Feed forward backpropagation model
class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mean_squared_error = None
        self.r2_score = None
        self.modules_seq = Sequential(Linear(2, hidden_dim), ReLU(), Linear(hidden_dim, 1))

    def forward(self, input_X_y):
        return self.modules_seq(input_X_y)

# Cascade forward backpropagation model
class CascadeForward(nn.Module):
    def __init__(self, num_of_dim, hidden_dim):
        super().__init__()
        self.mean_squared_error = None
        self.r2_score = None
        self.num_of_dim = num_of_dim
        self.hidden_layer_1 = nn.Linear(2, hidden_dim)
        if num_of_dim == 1:
            self.output = nn.Linear(2 + hidden_dim, 1)
        elif num_of_dim == 2:
            self.hidden_layer_2 = nn.Linear(2 + hidden_dim, hidden_dim)
            self.output = nn.Linear(2 + hidden_dim + hidden_dim, 1)
        else:
            throw_error('num_of_dim must be 1 or 2')

    def forward(self, input_X_y):
        h1_layer_out = F.relu(self.hidden_layer_1(input_X_y))
        if self.num_of_dim == 1:
            output_in = torch.cat((input_X_y, h1_layer_out), dim=1)
            return self.output(output_in)

        elif self.num_of_dim == 2:
            h2_layer_in = torch.cat((input_X_y, h1_layer_out), dim=1)
            h2_layer_out = F.relu(self.hidden_layer_2(h2_layer_in))

            output_in = torch.cat((input_X_y, h1_layer_out, h2_layer_out), dim=1)
            return self.output(output_in)

        else:
            throw_error('num_of_dim must be 1 or 2')
            return None

class ElmanBackprop(nn.Module):
    def __init__(self, hidden_size, num_layers=1):
        super().__init__()
        self.mean_squared_error = None
        self.r2_score = None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.elman_layer = RNN(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, nonlinearity='tanh')
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input_X_y):
        input_seq = input_X_y.unsqueeze(1)
        out, _ = self.elman_layer(input_seq)

        last_time_step_out = out[:, -1, :]
        final_prediction = self.output_layer(last_time_step_out)

        return final_prediction
#%%
def training_loop(model, input_train, input_test, output_train, output_test, num_epochs, model_name):
    optimizer = Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss().to(device)

    train_loss_lt = []
    test_loss_lt = []
    epoch_count = []

    for epoch in range(num_epochs):
        model.train()
        train_pred = model(input_train)
        loss = criterion(train_pred, output_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            test_pred = model(input_test)
            test_loss = criterion(test_pred, output_test)

            if epoch % 100 == 0:
                epoch_count.append(epoch)
                train_loss_lt.append(loss.item())
                test_loss_lt.append(test_loss.item())
                print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')

            if test_loss.item() < 3:
                break

    model.mean_squared_error = mean_squared_error(output_test.cpu(), test_pred.cpu())
    model.r2_score = r2_score(output_test.cpu(), test_pred.cpu())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epoch_count, train_loss_lt, label="Train loss")
    ax.plot(epoch_count, test_loss_lt, label="Test loss")
    ax.set_title(f"Графік передбачень тренувальних та тестових даних\nвідповідно до епох, модель: {model_name}")
    ax.set_yscale('log')
    formatter = mticker.ScalarFormatter()
    ax.yaxis.set_major_formatter(formatter)
    # ax.yaxis.set_minor_formatter(formatter)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()
    plt.show()
#%%
# Model with 10 neurons in a hidden layer
model_feed_forward_0 = FeedForward(hidden_dim=10).to(device)
training_loop(model_feed_forward_0, X_y_train, X_y_test, z_train.unsqueeze(1), z_test.unsqueeze(1), 3500, 'FeedForward 10 нейронів')
print(f'Mean Squared Error: {model_feed_forward_0.mean_squared_error}, R2 Error: {model_feed_forward_0.r2_score}')
#%%
# Creating a new model, but with more neurons in a hidden layer
model_feed_forward_1 = FeedForward(hidden_dim=20).to(device)
training_loop(model_feed_forward_1, X_y_train, X_y_test, z_train.unsqueeze(1), z_test.unsqueeze(1), 3500, 'FeedForward 20 нейронів')
print(f'Mean Squared Error: {model_feed_forward_1.mean_squared_error}, R2 Error: {model_feed_forward_1.r2_score}')
#%%
model_cascade_forward_0 = CascadeForward(1, 20).to(device)
training_loop(model_cascade_forward_0, X_y_train, X_y_test, z_train.unsqueeze(1), z_test.unsqueeze(1), 3500, 'CascadeForward 20 нейронів, 1 шар')
print(f'Mean Squared Error: {model_cascade_forward_0.mean_squared_error}, R2 Error: {model_cascade_forward_0.r2_score}')
#%%
model_cascade_forward_1 = CascadeForward(2, 10).to(device)
training_loop(model_cascade_forward_1, X_y_train, X_y_test, z_train.unsqueeze(1), z_test.unsqueeze(1), 3500, 'CascadeForward 10 нейронів, 2 шари')
print(f'Mean Squared Error: {model_cascade_forward_1.mean_squared_error}, R2 Error: {model_cascade_forward_1.r2_score}')
#%%
# Elman model with 15 neurons in a hidden layer
model_elman_backprop_0 = ElmanBackprop(15, 1).to(device)
training_loop(model_elman_backprop_0, X_y_train, X_y_test, z_train.unsqueeze(1), z_test.unsqueeze(1), 3500, 'ElmanBackprop 15 нейронів, 1 шар')
print(f'Mean Squared Error: {model_elman_backprop_0.mean_squared_error}, R2 Error: {model_elman_backprop_0.r2_score}')
#%%
# Elman model with 15 neurons in a hidden layer
model_elman_backprop_1 = ElmanBackprop(5, 3).to(device)
training_loop(model_elman_backprop_1, X_y_train, X_y_test, z_train.unsqueeze(1), z_test.unsqueeze(1), 3500, 'ElmanBackprop 5 нейронів, 3 шари')
print(f'Mean Squared Error: {model_elman_backprop_1.mean_squared_error}, R2 Error: {model_elman_backprop_1.r2_score}')
#%%
