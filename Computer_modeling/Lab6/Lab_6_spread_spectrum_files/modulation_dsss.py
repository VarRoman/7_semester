import numpy as np
import random
from scipy.stats import mode
from visualize import visualize_signal


class Sender:
    def __init__(self, seed, chip_rate):
        self.seed = seed
        self.chip_rate = chip_rate

    def modulate(self, message):
        """
        Генерує модульований сигнал за методов DSSS. Результат - 2D-масив (частоти x часові слоти)
        """
        # 1. Текст в двійкове представлення (побайтово, ASCII)
        bits = []
        for c in message:
            bin_str = bin(ord(c))[2:].zfill(8)  # Доповнення до розміру 1 байта
            bits.extend([int(b) for b in bin_str])
        bits = np.array(bits)
        len_bits = len(bits)    # Кількість бітів у повідомленні
        total_length = len_bits * self.chip_rate    # Довжина розширеного сигналу
        # 2. Частоти: 6 частот 1-6 умовних Гц (для наочності)
        num_freq = 6
        freqs = np.arange(1, num_freq + 1)
        # 3. Масив часових слотів (дискретизація по часу)
        t = np.arange(total_length)
        # 4. Сигнал-носій: синусоїди для кожної частоти із дискретизацією
        sins = np.sin(2 * np.pi * freqs[:, np.newaxis] * t / total_length if total_length > 0 else 1)
        # 5. PN-код: random зі спільним seed
        random.seed(self.seed)
        pn_code = np.array([random.choice([0, 1]) for _ in range(total_length)])
        # 6. Розширення сигналу у відповідності до chip_rate
        repeated_bits = np.repeat(bits, self.chip_rate)
        # 7. Модуляція DSSS по XOR із PN-кодом
        modulated = np.bitwise_xor(repeated_bits, pn_code)
        # 8. Конвертація в 2-мірний масив для адитивної модуляції на синусоїди
        modulated_2d = modulated[np.newaxis, :]
        # 9. Сигнал = синусоїди + DSSS-модульоване повідомлення
        signal = sins + modulated_2d
        return signal


class Receiver:
    def __init__(self, seed, chip_rate):
        self.seed = seed
        self.chip_rate = chip_rate

    def demodulate(self, signal):
        """
        Відновлює текстове повідомлення із 2D-масиву сигналу 
        """
        # 1. Форма сигналу: кількість частот, довжина розширеного сигналу
        num_freq, total_length = signal.shape
        len_bits = total_length // self.chip_rate
        # 2. Частоти: 1, 2, 3, ... і масив часових слотів (дискретизація по часу)
        freqs = np.arange(1, num_freq + 1)
        t = np.arange(total_length)
        # 3. Синусоїди сигнала-носія: такі ж синусоїди як у Sender
        sins = np.sin(2 * np.pi * freqs[:, np.newaxis] * t / total_length if total_length > 0 else 1)
        # 4. Модульовані значення = сигнал - синусоїди 
        extracted = signal - sins
        extracted = np.round(extracted).astype(int)  # Округлення для очистки від невеликого шуму
        # 5. PN-код за тим самим seed
        random.seed(self.seed)
        pn = np.array([random.choice([0, 1]) for _ in range(total_length)])
        pn_2d = pn[np.newaxis, :]
        # 6. Відновлення: повторний XOR
        recovered_per_freq = np.bitwise_xor(extracted, pn_2d)
        # 7. "Голосування по більшості": значення конкретних часових слотів серед різних частот
        recovered_spread = mode(recovered_per_freq, axis=0, keepdims=False).mode
        # 8. "Голосування по більшості": значення бітів серед всіх значень в слоті (розширення)
        recovered_bits = []
        for i in range(0, total_length, self.chip_rate):
            chip_group = recovered_spread[i:i + self.chip_rate]
            bit = mode(chip_group, keepdims=False).mode  # Majority vote
            recovered_bits.append(bit)
        recovered_bits = np.array(recovered_bits)
        recovered_bits = np.clip(recovered_bits, 0, 1)
        # 9. Відновлення текста із послідовності бітів (байтів)
        binary_str = ''.join(map(str, recovered_bits))
        message = ''
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8]
            if len(byte) == 8:
                message += chr(int(byte, 2))
        return message
    

if __name__ == '__main__':
    message = 'Hello, world!'
    seed = 42
    chip_rate = 4

    sender = Sender(
        seed=seed,
        chip_rate=chip_rate
    )
    dsss_message = sender.modulate(message=message)

    receiver = Receiver(
        seed=seed,
        chip_rate=chip_rate
    )
    decoded_message = receiver.demodulate(signal=dsss_message)

    print(f'Original message: {message}')
    print(f'Decoded message: {decoded_message}')
    visualize_signal(dsss_message)