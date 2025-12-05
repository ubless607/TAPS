
from pathlib import Path
from typing import Dict, Tuple, Optional

from .config import (
    SampleRate,
    NoisePropDecrease,
    PeakHeightRatio,
    PeakMinDistanceSec,
    TrimBeforeSec,
    TrimAfterSec,
    TrimOverlapMode,
    OutputRoot,
    MelNFFT,
    MelHopLength,
    MelNMels,
    get_config_name,
)
from .audio_utils import load_audio, reduce_noise_spectral_gating
from .detection import detect_impacts_peak_finding
from .trimming import trim_audio_around_impacts
from .spectrogram import (
    compute_mel_spectrogram,
    resize_spectrogram_to_128,
    normalize_spectrogram_zscore,
    save_spectrogram_npy,
)
from .dataset import get_all_audio_files


def run_preprocessing(
    dataset_root: str = "Robot_impact_Data",
    output_root: Optional[Path] = None,
    noise_prop_decrease: Optional[float] = None,
    peak_height_ratio: Optional[float] = None,
    peak_min_distance_sec: Optional[float] = None,
    trim_before_sec: Optional[float] = None,
    trim_after_sec: Optional[float] = None,
    trim_overlap_mode: Optional[str] = None,
    mel_n_fft: Optional[int] = None,
    mel_hop_length: Optional[int] = None,
    mel_n_mels: Optional[int] = None,
) -> None:
    # 파라미터 기본값 설정
    output_root = output_root if output_root is not None else OutputRoot
    noise_prop_decrease = noise_prop_decrease if noise_prop_decrease is not None else NoisePropDecrease
    peak_height_ratio = peak_height_ratio if peak_height_ratio is not None else PeakHeightRatio
    peak_min_distance_sec = peak_min_distance_sec if peak_min_distance_sec is not None else PeakMinDistanceSec
    trim_before_sec = trim_before_sec if trim_before_sec is not None else TrimBeforeSec
    trim_after_sec = trim_after_sec if trim_after_sec is not None else TrimAfterSec
    trim_overlap_mode = trim_overlap_mode if trim_overlap_mode is not None else TrimOverlapMode
    mel_n_fft = mel_n_fft if mel_n_fft is not None else MelNFFT
    mel_hop_length = mel_hop_length if mel_hop_length is not None else MelHopLength
    mel_n_mels = mel_n_mels if mel_n_mels is not None else MelNMels

    # 설정 이름 생성
    config_name = get_config_name(
        noise_prop_decrease=noise_prop_decrease,
        peak_height_ratio=peak_height_ratio,
        peak_min_distance_sec=peak_min_distance_sec,
        trim_before_sec=trim_before_sec,
        trim_after_sec=trim_after_sec,
        trim_overlap_mode=trim_overlap_mode,
    )

    files = get_all_audio_files(base_path=dataset_root)
    print(f"총 {len(files)}개의 오디오 파일을 찾았습니다.")

    if not files:
        return
    counters: Dict[Tuple[str, str], int] = {}
    total_segments = 0

    for i, (file_path, material, split) in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path} | material={material}, split={split}")
        
        try:
            audio, sr = load_audio(file_path, sampling_rate=SampleRate)
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            continue

        # 노이즈 제거
        audio_proc = reduce_noise_spectral_gating(audio, sr=sr, prop_decrease=noise_prop_decrease)

        # 충격 검출
        impact_times, _ = detect_impacts_peak_finding(
            audio_proc,
            sr=sr,
            height_ratio=peak_height_ratio,
            min_distance_sec=peak_min_distance_sec,
        )

        if not impact_times:
            # 충격이 없는 경우 스킵
            print(f"[INFO] No impacts detected: {file_path}")
            continue

        # Trimming
        segments = trim_audio_around_impacts(
            audio_proc,
            sr=sr,
            impact_times=impact_times,
            before_sec=trim_before_sec,
            after_sec=trim_after_sec,
            overlap_handling=trim_overlap_mode,
        )

        if not segments:
            print(f"[INFO] No valid segments after trimming (all overlapped): {file_path}")
            continue

        # 재질+split별 인덱스 초기화
        key = (material, split)
        if key not in counters:
            counters[key] = 0

        saved_count = 0

        for seg in segments:
            # Mel-Spectrogram
            mel_db = compute_mel_spectrogram(seg, sr=SampleRate, n_fft=mel_n_fft, hop_length=mel_hop_length, n_mels=mel_n_mels)
            mel_128 = resize_spectrogram_to_128(mel_db)
            mel_norm, mean, std = normalize_spectrogram_zscore(mel_128)

            counters[key] += 1
            idx = counters[key]
            save_spectrogram_npy(
                mel_norm, 
                material=material, 
                split=split, 
                idx=idx, 
                output_root=output_root,
                config_name=config_name,
            )

            saved_count += 1

        # 파일 단위 요약 로그
        print(f"  -> Total segments for this file: {saved_count}")
        total_segments += saved_count

    print("\n=== Preprocessing Finished ===")
    print(f"Total segments saved: {total_segments}")
    print(f"저장 경로: {output_root / config_name}")
    print("Per-material/split counts:")
    for (mat, spl), cnt in sorted(counters.items()):
        print(f"  {mat}/{spl}: {cnt} segments")

