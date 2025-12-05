
from typing import List, Tuple
import numpy as np
from scipy.signal import find_peaks

from .config import SampleRate, PeakHeightRatio, PeakMinDistanceSec


def detect_impacts_peak_finding(audio: np.ndarray, 
                                sr: int = SampleRate, 
                                height_ratio: float = PeakHeightRatio, 
                                min_distance_sec: float = PeakMinDistanceSec
                            ) -> Tuple[List[float], List[int]]:
    
    audio_abs = np.abs(audio)
    max_amp = audio_abs.max() if len(audio_abs) > 0 else 0.0
    if max_amp == 0:
        return [], []

    height = max_amp * height_ratio
    distance = int(min_distance_sec * sr)

    peaks, _ = find_peaks(audio_abs, height=height, distance=distance)

    impact_indices = peaks.tolist()
    impact_times = [idx / sr for idx in impact_indices]

    return impact_times, impact_indices

