"""
추론(Inference)용 전처리 모듈

실제 로봇에서 들어오는 오디오를 모델 입력 형태로 변환하는 함수들.
학습 시와 동일한 전처리 파이프라인을 적용하여 샘플링 레이트 차이를 해결합니다.
"""
from typing import List, Union

import numpy as np

from .config import SampleRate
from .audio_utils import load_audio, reduce_noise_spectral_gating
from .detection import detect_impacts_peak_finding
from .trimming import trim_audio_around_impacts
from .spectrogram import (
    compute_mel_spectrogram,
    resize_spectrogram_to_128,
    normalize_spectrogram_zscore,
)


def preprocess_for_inference(
    audio_input: Union[np.ndarray, str],
    target_sr: int = SampleRate,
    detect_impacts: bool = True,
) -> List[np.ndarray]:
    """
    추론용 전처리 함수: 오디오를 모델 입력 형태(정규화된 스펙트로그램)로 변환.

    Parameters
    ----------
    audio_input : np.ndarray | str
        - np.ndarray: 이미 로드된 오디오 신호 (샘플링 레이트는 target_sr로 가정)
        - str: 오디오 파일 경로 (자동으로 target_sr로 리샘플링됨)
    target_sr : int, default=SampleRate (44100)
        목표 샘플링 레이트. 학습 시 사용한 샘플링 레이트와 동일해야 함.
    detect_impacts : bool, default=True
        True: 충격 시점을 자동으로 검출하여 각각 전처리
        False: 전체 오디오를 하나의 segment로 처리 (1초 단위로 잘라서 처리)

    Returns
    -------
    spectrograms : List[np.ndarray]
        정규화된 스펙트로그램 리스트. 각 원소는 (128, 128) shape.
        모델 입력 시: torch.from_numpy(spec).unsqueeze(0).unsqueeze(0) 형태로 변환 필요.

    Notes
    -----
    - **샘플링 레이트 자동 처리**: 
      - 파일 경로를 입력하면 `librosa.load()`가 자동으로 target_sr로 리샘플링합니다.
      - 따라서 실제 로봇에서 들어오는 오디오의 샘플링 레이트가 달라도 문제없습니다.
    
    - **학습 시와 동일한 파이프라인**:
      1. 오디오 로드 (자동 리샘플링)
      2. 노이즈 제거 (Spectral Gating)
      3. 충격 시점 검출 (Peak Finding)
      4. 1초 Trimming (0.05s 전 ~ 0.95s 후)
      5. Mel-Spectrogram 변환
      6. 128x128 리사이징
      7. Z-score 정규화

    Examples
    --------
    >>> from preprocess.inference import preprocess_for_inference
    >>> import torch
    >>> 
    >>> # 파일 경로로 입력 (자동 리샘플링)
    >>> specs = preprocess_for_inference("path/to/audio.wav")
    >>> 
    >>> # 모델 입력 형태로 변환
    >>> model_input = torch.from_numpy(specs[0]).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)
    >>> 
    >>> # 예측
    >>> with torch.no_grad():
    >>>     output = model(model_input)
    """
    # 1. 오디오 로드 (자동 리샘플링)
    if isinstance(audio_input, str):
        # 파일 경로: librosa가 자동으로 target_sr로 리샘플링
        audio, sr = load_audio(audio_input, sampling_rate=target_sr)
    else:
        # 이미 로드된 오디오: 샘플링 레이트가 다르면 리샘플링 필요
        audio = audio_input
        sr = target_sr  # 가정: 이미 target_sr로 리샘플링되어 있음
        # 만약 다른 샘플링 레이트라면 librosa.resample() 사용 필요

    # 2. 노이즈 제거
    from .config import NoisePropDecrease
    audio_clean = reduce_noise_spectral_gating(audio, sr=sr, prop_decrease=NoisePropDecrease)

    # 3. 충격 검출 및 Trimming
    if detect_impacts:
        from .config import PeakHeightRatio, PeakMinDistanceSec, TrimBeforeSec, TrimAfterSec, TrimOverlapMode
        
        impact_times, _ = detect_impacts_peak_finding(
            audio_clean,
            sr=sr,
            height_ratio=PeakHeightRatio,
            min_distance_sec=PeakMinDistanceSec,
        )

        if not impact_times:
            # 충격이 없으면 전체 오디오를 1초 단위로 처리
            segments = []
            segment_length_samples = int(target_sr * 1.0)  # 1초
            for i in range(0, len(audio_clean), segment_length_samples):
                seg = audio_clean[i:i + segment_length_samples]
                if len(seg) == segment_length_samples:  # 정확히 1초인 경우만
                    segments.append(seg)
        else:
            segments = trim_audio_around_impacts(
                audio_clean,
                sr=sr,
                impact_times=impact_times,
                before_sec=TrimBeforeSec,
                after_sec=TrimAfterSec,
                overlap_handling=TrimOverlapMode,
            )
    else:
        # 충격 검출 없이 전체 오디오를 1초 단위로 처리
        segments = []
        segment_length_samples = int(target_sr * 1.0)  # 1초
        for i in range(0, len(audio_clean), segment_length_samples):
            seg = audio_clean[i:i + segment_length_samples]
            if len(seg) == segment_length_samples:  # 정확히 1초인 경우만
                segments.append(seg)

    if not segments:
        return []

    # 4. Mel-Spectrogram 변환 및 정규화
    spectrograms = []
    for seg in segments:
        mel_db = compute_mel_spectrogram(seg, sr=target_sr)
        mel_128 = resize_spectrogram_to_128(mel_db)
        mel_norm, _, _ = normalize_spectrogram_zscore(mel_128)
        spectrograms.append(mel_norm)

    return spectrograms


def preprocess_single_segment(
    audio_segment: np.ndarray,
    sr: int = SampleRate,
) -> np.ndarray:
    """
    이미 1초로 잘린 오디오 segment를 스펙트로그램으로 변환.
    
    Parameters
    ----------
    audio_segment : np.ndarray
        1초 길이의 오디오 신호 (샘플링 레이트는 sr로 가정)
    sr : int, default=SampleRate
        오디오의 샘플링 레이트
    
    Returns
    -------
    spectrogram : np.ndarray
        정규화된 스펙트로그램 (128, 128)
    
    Notes
    -----
    - 오디오가 이미 1초로 잘려있고, 샘플링 레이트가 target_sr과 다를 경우,
      librosa.resample()을 먼저 적용해야 합니다.
    """
    # 샘플링 레이트 확인 및 리샘플링
    if sr != SampleRate:
        import librosa
        audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=SampleRate)
        sr = SampleRate
    
    # 노이즈 제거
    from .config import NoisePropDecrease
    audio_clean = reduce_noise_spectral_gating(audio_segment, sr=sr, prop_decrease=NoisePropDecrease)
    
    # Mel-Spectrogram 변환 및 정규화
    mel_db = compute_mel_spectrogram(audio_clean, sr=sr)
    mel_128 = resize_spectrogram_to_128(mel_db)
    mel_norm, _, _ = normalize_spectrogram_zscore(mel_128)
    
    return mel_norm

