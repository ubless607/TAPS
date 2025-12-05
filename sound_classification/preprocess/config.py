from pathlib import Path

SampleRate = 44100

# Denoising Parameter
NoisePropDecrease = 0.8  # Spectral Gating 강도 (peak 보존 위주)
'''
노이즈 제거 강도 (0.0~1.0)
- 0.0: 노이즈 제거 안 함
- 0.4~0.5: 매우 보수적 (peak 최대 보존, 노이즈 제거 약함)
- 0.6~0.7: peak 보존에 유리 (타격음 보존)
- 0.8~0.9: 균형잡힌 제거
- 1.0: 최대 강도 노이즈 제거
'''

# Peak Finding Parameter
PeakHeightRatio = 0.3  # max_amplitude * 0.3 이상인 peak만 사용
PeakMinDistanceSec = 0.5  # 두 peak 사이 최소 간격 (초)
'''
PeakHeightRatio: 최대 진폭 대비 최소 높이 비율 (예: 0.3이면 max_amplitude * 0.3 이상인 peak만 검출)
min_distance_sec: 두 peak 사이의 최소 시간 간격 (초)
'''

# Trimming Parameter
TrimBeforeSec = 0.05  # 충격 0.05초 전
TrimAfterSec = 0.95  # 충격 0.95초 후 (총 1초)
TrimOverlapMode = "exclude"
'''
TrimBeforeSec: 충격 시점 이전 추출 길이 (초)
TrimAfterSec: 충격 시점 이후 추출 길이 (초)
TrimOverlapMode: 
    - "exclude": 중첩 구간 제외
    - "replace": 중첩 구간을 배경 소음 평균값으로 대체
'''

# Mel-Spectrogram Parameter
MelNFFT = 400
MelHopLength = 160
MelNMels = 128

# Output Path
OutputRoot = Path("processed_data")


def get_config_name(
    noise_prop_decrease: float = None,
    peak_height_ratio: float = None,
    peak_min_distance_sec: float = None,
    trim_before_sec: float = None,
    trim_after_sec: float = None,
    trim_overlap_mode: str = None,
) -> str:
    """
    하이퍼파라미터를 기반으로 설정 이름을 생성.
    
    Parameters
    ----------
    noise_prop_decrease, peak_height_ratio, ... : optional
        파라미터가 제공되지 않으면 config.py의 기본값 사용
    
    Returns
    -------
    config_name : str
        하이퍼파라미터를 포함한 설정 이름
        예: "noise0.8_peak0.3_dist0.5_trim0.05-0.95_exclude"
    """
    # 파라미터 기본값 설정
    noise_prop_decrease = noise_prop_decrease if noise_prop_decrease is not None else NoisePropDecrease
    peak_height_ratio = peak_height_ratio if peak_height_ratio is not None else PeakHeightRatio
    peak_min_distance_sec = peak_min_distance_sec if peak_min_distance_sec is not None else PeakMinDistanceSec
    trim_before_sec = trim_before_sec if trim_before_sec is not None else TrimBeforeSec
    trim_after_sec = trim_after_sec if trim_after_sec is not None else TrimAfterSec
    trim_overlap_mode = trim_overlap_mode if trim_overlap_mode is not None else TrimOverlapMode

    # 주요 하이퍼파라미터만 포함 (간결하게)
    config_name = (
        f"noise{noise_prop_decrease}_"
        f"peak{peak_height_ratio}_"
        f"dist{peak_min_distance_sec}_"
        f"trim{trim_before_sec}-{trim_after_sec}_{trim_overlap_mode}"
    )
    return config_name