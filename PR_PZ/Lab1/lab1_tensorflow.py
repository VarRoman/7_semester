#%%
import tensorflow as tf
import numpy as np
#%%
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 1, 0, 0, 0, 1])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_dim=3, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=100)

loss, accuracy = model.evaluate(x, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

prediction = model.predict(x)
for inp, pred in zip(x, prediction):
    print(inp, round(pred[0]))
#%%
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Softmax
from torch.optim import Adam
import numpy as np
#%%
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 1, 0, 0, 0, 1])

