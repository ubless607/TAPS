# 전처리 실행 방법


```bash
python preprocess.py
```

- `Robot_impact_Data` 폴더의 모든 오디오 파일을 전처리

#### 커맨드라인 옵션


| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset_root` | `Robot_impact_Data` | 데이터셋 루트 경로 |
| `--output` | `preprocessed_data` | 전처리 결과 저장 경로 |
| `--noise` | `0.4` | 노이즈 제거 강도 (0.0~1.0) |
| `--peak` | `0.3` | Peak 높이 비율 |
| `--dist` | `0.5` | Peak 최소 거리 (초) |
| `--trim_before` | `0.05` | 충격 시점 이전 추출 길이 (초) |
| `--trim_after` | `0.95` | 충격 시점 이후 추출 길이 (초) |
| `--overlap` | `exclude` | 중첩 처리 모드 (`exclude` 또는 `replace`) |
| `--n_fft` | `400` | FFT size |
| `--hop_length` | `160` | Hop length |
| `--n_mels` | `128` | Mel 필터 개수 |


- **노이즈 제거 강도 (`--noise`)**
    - `0.0`: 노이즈 제거 안 함
    - `0.4~0.5`: peak 최대 보존, 노이즈 제거 약함
    - `0.6~0.7`
    - `0.8~0.9`
    - `1.0`: 최대 강도 노이즈 제거

- 중첩 처리 모드 (`--overlap`)
    - `exclude`: 중첩 구간 제외
    - `replace`: 중첩 구간을 배경 소음 평균값으로 대체

#### 저장 경로 구조

```
preprocessed_data/
└── noise0.4_peak0.3_dist0.5_trim0.05-0.95_exclude/
    ├── Aluminum/
    │   ├── train/
    │   │   ├── 00001.npy
    │   │   ├── 00002.npy
    │   │   └── ...
    │   └── test/
    │       └── ...
    └── Glass/
        └── ...
```