import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import time

def plot_sample_sequence(data, width, height, fig_title=None):
    fig, axes = plt.subplots(width, height, figsize=(15, 9))
    data_size = len(data)

    index = np.random.choice(data_size, width * height, replace=False)

    for i in range(width):
        for j in range(height):
            axes[i, j].plot(data[index[i * height + j]])

    if fig_title:
        fig.suptitle(fig_title)
    plt.show()

def plot_sequence(data, width, height, fig_title=None):
    fig, axes = plt.subplots(width, height, figsize=(15, 9))

    for i in range(width):
        for j in range(height):
            axes[i, j].plot(data[i * height + j])

    if fig_title:
        fig.suptitle(fig_title)
    plt.show()

def save_sequence_img(data, width, height, path=None):
    fig, axes = plt.subplots(width, height, figsize=(15, 9))

    for i in range(width):
        for j in range(height):
            axes[i, j].plot(data[i * height + j])

    if path:
        fig.suptitle(path)
    plt.savefig(path)

def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1,
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9,
        random_seed=None, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples, dtype=np.float32)
    return samples

def sine_wave_with_linear(seq_length=30, num_samples=28*5*100, num_signals=1, 
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9, ratio_low=-0.04, ratio_high=0.04,
        random_seed=None, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            linear_fun = np.random.uniform(low=ratio_low, high=ratio_high) * ix
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset) + linear_fun)
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples, dtype=np.float32)
    return samples

def sine_wave_with_noise(seq_length=30, num_samples=28*5*100, num_signals=1, 
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9, ratio_low=-0.04, ratio_high=0.04, 
        noise_amplitude=0.1, random_seed=None, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
    
            noise = np.random.normal(size=seq_length) * noise_amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            linear_fun = np.random.uniform(low=ratio_low, high=ratio_high) * ix
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset) + linear_fun + noise)
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples, dtype=np.float32)
    return samples


def sine_wave_plus_sawtooth(seq_length=50, num_samples=28 * 5 * 100, num_signals=1,
                            freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9,
                            random_seed=None, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)  # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)  # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            t = np.linspace(0, 25, 25)
            saw_wave = A * signal.sawtooth(2 * np.pi * 5 * t)
            saw_wave = np.concatenate((np.zeros(25), saw_wave))
            signals.append((A * np.sin(2 * np.pi * f * ix / float(seq_length) + offset) + saw_wave))

            samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples, dtype=np.float32)
    return samples

def sine_wave_plus_noise_sawtooth(seq_length=50, num_samples=28 * 5 * 100, num_signals=1,
                            freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9, noise_amplitude=0.1,
                            random_seed=None, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)  # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)  # amplitude
            noise = np.random.normal(size=seq_length) * noise_amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)

            split_position = int(seq_length / 2)
            t = np.linspace(0, split_position, split_position)
            saw_wave = A * signal.sawtooth(2 * np.pi * 5 * t)
            saw_wave = np.concatenate((np.zeros(split_position), saw_wave))

            signals.append((A * np.sin(2 * np.pi * f * ix / float(seq_length) + offset + noise) + saw_wave))
            samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples, dtype=np.float32)
    return samples


def visualize_training_generator(train_step_num, start_time, data_np, width, height):
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plot_sample_sequence(data_np, width, height)

def plot_sequence_data(x, y, x_label, y_lable):    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_lable)