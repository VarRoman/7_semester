#%%
import torch
from sklearn.linear_model import SGDRegressor
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import random
#%%
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 1, 0, 0, 0, 1])

x_train = torch.from_numpy(x).float()
y_train = torch.from_numpy(y).float().unsqueeze(1)

model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.05)
#%%
epochs = 150

train_loss = []
test_loss = []
epoch_count = []

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.inference_mode():
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss.append(loss.item())
            print(f'Epoch: {epoch}, Train Loss: {loss.item()}')
#%%
model.eval()

with torch.inference_mode():
    y_pred = model(x_train[0:7:2])
print(f'Got results via model: {np.int32(np.round(y_pred).reshape(4)).tolist()},\nTrue results: {y[0:7:2].tolist()}')

plt.figure(figsize=(6, 4))
plt.plot(epoch_count, train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training process')
plt.show();
#%%
