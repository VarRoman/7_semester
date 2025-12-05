import matplotlib.pyplot as plt
import numpy as np


def visualize_signal(signal: np.ndarray):
    num_freq, total_length = signal.shape
    plt.figure(figsize=(12, 8))
    for i in range(num_freq):
        plt.subplot(num_freq, 1, i + 1)
        plt.plot(signal[i, :])
        plt.title(f'Частота {i + 1}')
        plt.xlabel('Часові слоти')
        plt.ylabel('Значення сигнала')
        plt.grid(True)
    plt.tight_layout()
    plt.show()