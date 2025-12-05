"""
YCB-Impact Dataset 전처리 패키지

전처리 파이프라인:
1) 오디오 로드 (44.1kHz, mono)
2) 노이즈 제거 (Spectral Gating, noisereduce)
3) Peak Finding 기반 충격 시점 검출
4) 각 충격 시점 기준 1초 Trimming (0.05s 전 ~ 0.95s 후), 중첩 처리
5) Mel-Spectrogram (FFT size=400)
6) 128 x 128 리사이징 (주파수 x 시간), 채널=1 (그레이스케일)
7) Z-score 정규화
8) .npy 저장: processed_data/<material>/<split>/<index>.npy

데이터 구조:
- Robot_impact_Data/
    - Horizontal_Pokes/<Material>/<train|test>/<ObjectName>/*.ogg
    - Vertical_Pokes/Known_Objects/<Material>/<ObjectName>/*.ogx
    - Vertical_Pokes/Unknown_Objects/<Material>/<ObjectName>/*.ogx

규칙:
- Vertical_Pokes/Known_Objects -> split = "train"
- Vertical_Pokes/Unknown_Objects -> split = "test"
- Plastic의 하위 타입(Hard/Other/Soft 등)은 모두 material = "Plastic"으로 통합
"""

from .pipeline import run_preprocessing
from .dataset import get_all_audio_files
from .inference import preprocess_for_inference, preprocess_single_segment
from .config import (
    SampleRate,
    NoisePropDecrease,
    PeakHeightRatio,
    PeakMinDistanceSec,
    TrimBeforeSec,
    TrimAfterSec,
    TrimOverlapMode,
    MelNFFT,
    MelHopLength,
    MelNMels,
    OutputRoot,
)

__all__ = [
    "run_preprocessing",
    "get_all_audio_files",
    "preprocess_for_inference",
    "preprocess_single_segment",
    "SampleRate",
    "NoisePropDecrease",
    "PeakHeightRatio",
    "PeakMinDistanceSec",
    "TrimBeforeSec",
    "TrimAfterSec",
    "TrimOverlapMode",
    "MelNFFT",
    "MelHopLength",
    "MelNMels",
    "OutputRoot",
]

