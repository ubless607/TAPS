
from pathlib import Path
from typing import Tuple
import numpy as np
import librosa
import librosa.util

from .config import SampleRate, MelNFFT, MelHopLength, MelNMels, OutputRoot, get_config_name


def compute_mel_spectrogram(audio: np.ndarray,
                            sr: int = SampleRate,
                            n_fft: int = MelNFFT,
                            hop_length: int = MelHopLength,
                            n_mels: int = MelNMels,
                        ) -> np.ndarray:
    
    mel = librosa.feature.melspectrogram(y=audio, 
                                         sr=sr,
                                         n_fft=n_fft, 
                                         hop_length=hop_length, 
                                         n_mels=n_mels, 
                                         power=2.0,
                                    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def resize_spectrogram_to_128(mel_spec: np.ndarray) -> np.ndarray:
    F, T = mel_spec.shape

    if F > 128:
        start_f = (F - 128) // 2
        mel_spec = mel_spec[start_f : start_f + 128, :]
    elif F < 128:
        pad_top = (128 - F) // 2
        pad_bottom = 128 - F - pad_top
        mel_spec = np.pad(mel_spec, ((pad_top, pad_bottom), (0, 0)), mode="constant")

    mel_resized = librosa.util.fix_length(mel_spec, size=128, axis=1)
    return mel_resized


def normalize_spectrogram_zscore(mel_spec: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(mel_spec))
    std = float(np.std(mel_spec))

    if std == 0:
        mel_norm = np.zeros_like(mel_spec, dtype=np.float32)
    else:
        mel_norm = (mel_spec - mean) / std

    return mel_norm.astype(np.float32), mean, std


def save_spectrogram_npy(mel_norm: np.ndarray, 
                         material: str, 
                         split: str, 
                         idx: int, 
                         output_root: Path = OutputRoot,
                         config_name: str = None,
                    ) -> Path:
    """
    정규화된 스펙트로그램을 .npy로 저장.
    저장 경로: processed_data/<config_name>/<material>/<split>/<idx:05d>.npy
    """
    if config_name is None:
        config_name = get_config_name()
    base_path = output_root / config_name / material / split
    base_path.mkdir(parents=True, exist_ok=True)

    filename = f"{idx:05d}.npy"
    save_path = base_path / filename
    np.save(save_path, mel_norm)
    
    return save_path

