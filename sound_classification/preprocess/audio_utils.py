from typing import Tuple
import numpy as np
import librosa
import noisereduce as nr

from .config import SampleRate, NoisePropDecrease


def load_audio(file_path: str, sampling_rate: int = SampleRate) -> Tuple[np.ndarray, int]:

    y, sr = librosa.load(file_path, sr=sampling_rate, mono=True)
    return y, sr


def reduce_noise_spectral_gating(audio: np.ndarray, 
                                 sr: int = SampleRate, 
                                 prop_decrease: float = NoisePropDecrease, 
                                 stationary: bool = False
                            ) -> np.ndarray:

    reduced = nr.reduce_noise(y=audio, sr=sr, stationary=stationary, prop_decrease=prop_decrease)
    return reduced

