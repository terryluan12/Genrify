import librosa as lb
import numpy as np

def augment_sample(sample, sr):
    """
    Data augmentation, used to prevent overfitting
    Technique used: pitch shift, time stretch, add noise

    Args:
        sample: output of librosa.load()
        sr: sampling rate, outout of librosa.load()

    Returns:
        list of augmented samples
    """
    augmented_samples = []

    # 1. add original sample
    augmented_samples.append(sample)

    # 2. Change pitch
    for n_steps in [4, 6]:
        augmented_samples.append(lb.effects.pitch_shift(sample, sr, n_steps))

    # 3. Time Stretch
    for rate in [0.5, 2.0]:
        augmented_samples.append(lb.effects.time_stretch(sample, rate))

    # 4. white noise
    wn = np.random.randn(len(sample))
    augmented_samples.append(sample + 0.005*wn)

    return augmented_samples