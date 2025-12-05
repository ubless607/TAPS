
from typing import List
import numpy as np

from .config import SampleRate, TrimBeforeSec, TrimAfterSec, TrimOverlapMode


def trim_audio_around_impacts(audio: np.ndarray, 
                              sr: int, 
                              impact_times: List[float], 
                              before_sec: float = TrimBeforeSec, 
                              after_sec: float = TrimAfterSec, 
                              overlap_handling: str = TrimOverlapMode,
                        ) -> List[np.ndarray]:

    total_length_sec = before_sec + after_sec
    total_length_samples = int(total_length_sec * sr)
    before_samples = int(before_sec * sr)
    after_samples = int(after_sec * sr)

    trimmed_segments: List[np.ndarray] = []

    background_noise_mean = np.mean(audio[: int(0.5 * sr)]) if len(audio) > int(0.5 * sr) else 0.0

    for impact_time in impact_times:
        impact_idx = int(impact_time * sr)
        start_idx = impact_idx - before_samples
        end_idx = impact_idx + after_samples

        if start_idx < 0:
            segment = np.zeros(total_length_samples, dtype=np.float32)
            actual_start = 0
            actual_end = min(end_idx, len(audio))
            offset = -start_idx
            segment[offset : offset + (actual_end - actual_start)] = audio[actual_start:actual_end]
        elif end_idx > len(audio):
            segment = np.zeros(total_length_samples, dtype=np.float32)
            actual_start = max(start_idx, 0)
            actual_end = len(audio)
            segment[: actual_end - actual_start] = audio[actual_start:actual_end]
        else:
            segment = audio[start_idx:end_idx].astype(np.float32).copy()

        segment_start_time = impact_time - before_sec
        segment_end_time = impact_time + after_sec

        has_overlap = False
        for other_time in impact_times:
            if other_time == impact_time:
                continue
            if segment_start_time <= other_time <= segment_end_time:
                has_overlap = True
                break

        if has_overlap:
            if overlap_handling == "exclude":
                # 이 segment는 사용하지 않음
                continue
            elif overlap_handling == "replace":
                # 중첩된 충격 주변 0.1초 구간을 배경 소음 평균값으로 대체
                for other_time in impact_times:
                    if other_time == impact_time:
                        continue
                    if segment_start_time <= other_time <= segment_end_time:
                        center_idx = int((other_time - segment_start_time) * sr)
                        half_win = int(0.05 * sr)
                        overlap_start_idx = max(0, center_idx - half_win)
                        overlap_end_idx = min(len(segment), center_idx + half_win)
                        segment[overlap_start_idx:overlap_end_idx] = background_noise_mean

        trimmed_segments.append(segment)

    return trimmed_segments

