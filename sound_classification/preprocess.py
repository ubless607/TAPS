import argparse
from pathlib import Path

from preprocess import run_preprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description="YCB-Impact Dataset 전처리 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 데이터셋 경로
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Robot_impact_Data",
        help="데이터셋 루트 경로 (기본: Robot_impact_Data)"
    )

    # 출력 경로
    parser.add_argument(
        "--output",
        type=str,
        default="preprocessed_data",
        help="전처리 결과 저장 경로 (기본: preprocessed_data)"
    )

    # Denoising Parameter
    parser.add_argument(
        "--noise",
        type=float,
        default=0.4,
        help="노이즈 제거 강도 (0.0~1.0, 기본: 0.4). "
             "0.4~0.5: 보수적, 0.6~0.7: peak 보존, 0.8~0.9: 균형, 1.0: 최대 강도"
    )

    # Peak Finding Parameter
    parser.add_argument(
        "--peak",
        type=float,
        default=0.3,
        help="Peak 높이 비율 (기본: 0.3). max_amplitude * peak 이상인 peak만 검출"
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=0.5,
        help="Peak 최소 거리 (초, 기본: 0.5). 두 peak 사이 최소 간격"
    )

    # Trimming Parameter
    parser.add_argument(
        "--trim_before",
        type=float,
        default=0.05,
        help="충격 시점 이전 추출 길이 (초, 기본: 0.05)"
    )
    parser.add_argument(
        "--trim_after",
        type=float,
        default=0.95,
        help="충격 시점 이후 추출 길이 (초, 기본: 0.95)"
    )
    parser.add_argument(
        "--overlap",
        type=str,
        choices=["exclude", "replace"],
        default="exclude",
        help="중첩 처리 모드 (기본: exclude). exclude: 제외, replace: 배경 소음으로 대체"
    )

    # Mel-Spectrogram Parameter (일반적으로 변경하지 않음)
    parser.add_argument(
        "--n_fft",
        type=int,
        default=400,
        help="FFT size (기본: 400)"
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=160,
        help="Hop length (기본: 160)"
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=128,
        help="Mel 필터 개수 (기본: 128)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 설정 출력
    print("=" * 60)
    print("전처리 파이프라인 설정")
    print("=" * 60)
    print(f"데이터셋 경로: {args.dataset_root}")
    print(f"출력 경로: {args.output}")
    print(f"\n[하이퍼파라미터]")
    print(f"  노이즈 제거 강도: {args.noise}")
    print(f"  Peak 높이 비율: {args.peak}")
    print(f"  Peak 최소 거리: {args.dist}초")
    print(f"  Trimming: {args.trim_before}초 전 ~ {args.trim_after}초 후")
    print(f"  중첩 처리: {args.overlap}")
    print(f"  Mel-Spectrogram: n_fft={args.n_fft}, hop_length={args.hop_length}, n_mels={args.n_mels}")
    print("=" * 60)
    print()

    # 전처리 실행
    run_preprocessing(
        dataset_root=args.dataset_root,
        output_root=Path(args.output),
        noise_prop_decrease=args.noise,
        peak_height_ratio=args.peak,
        peak_min_distance_sec=args.dist,
        trim_before_sec=args.trim_before,
        trim_after_sec=args.trim_after,
        trim_overlap_mode=args.overlap,
        mel_n_fft=args.n_fft,
        mel_hop_length=args.hop_length,
        mel_n_mels=args.n_mels,
    )


if __name__ == "__main__":
    main()
