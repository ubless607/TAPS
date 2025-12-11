from scipy.signal import find_peaks
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
import noisereduce as nr

material_to_label = {'Aluminum': 0, 'Ceramic': 1, 'Plastic': 2, 'Paper': 3, 'Wood': 4}
label_to_material = {v: k for k, v in material_to_label.items()}

import torch.nn as nn
from panns_inference import AudioTagging

def load_finetuned_panns(checkpoint_path, device, num_classes=5):
    """
    Loads the PANNs Cnn14 model and restores the fine-tuned weights.
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # 1. Initialize the base PANNs (Cnn14) model
    # We use checkpoint_path=None to get the default architecture structure
    at = AudioTagging(checkpoint_path=None, device=device)
    panns_model = at.model
    
    # 2. Modify the final layer to match your fine-tuning (5 classes)
    # The original Cnn14 has 527 classes (AudioSet), we need to change it to 5.
    # We must do this BEFORE loading weights, or shapes won't match.
    in_features = panns_model.fc_audioset.in_features
    panns_model.fc_audioset = nn.Linear(in_features, num_classes)
    
    # 3. Load the saved state dictionary
    # map_location ensures it loads to the correct device (cpu or cuda)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Support both full checkpoint dicts and direct state dicts
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    panns_model.load_state_dict(state_dict)
    
    # 4. Move to device and set to evaluation mode
    panns_model.to(device)
    panns_model.eval()
    
    print("Model loaded successfully.")
    return panns_model, at

def detect_impacts_peak_finding(audio, sr, height_ratio=0.3, min_distance_sec=0.5):
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

def load_audio(file_path, sampling_rate=44100):
    y, sampling_rate = librosa.load(file_path, sr=sampling_rate)
    return y, sampling_rate

def reduce_noise_spectral_gating(audio, sr=44100, prop_decrease=0.8, stationary=False):
    return nr.reduce_noise(y=audio, sr=sr, stationary=stationary, prop_decrease=prop_decrease)

def trim_audio_around_impacts(
    audio,
    sr,
    impact_times,
    before_sec=0.05,
    after_sec=0.95,
    overlap_handling="exclude",
):
    total_length_sec = before_sec + after_sec  # 1초
    total_length_samples = int(total_length_sec * sr)
    before_samples = int(before_sec * sr)
    after_samples = int(after_sec * sr)

    trimmed_segments = []

    background_noise_mean = np.mean(audio[: int(0.5 * sr)]) if len(audio) > int(0.5 * sr) else 0.0

    for impact_time in impact_times:
        impact_idx = int(impact_time * sr)
        start_idx = impact_idx - before_samples
        end_idx = impact_idx + after_samples

        # 1초 segment 만들기 (padding 포함)
        if start_idx < 0:
            segment = np.zeros(total_length_samples)
            actual_start = 0
            actual_end = min(end_idx, len(audio))
            segment[-start_idx : -start_idx + (actual_end - actual_start)] = audio[actual_start:actual_end]
        elif end_idx > len(audio):
            segment = np.zeros(total_length_samples)
            actual_start = max(start_idx, 0)
            actual_end = len(audio)
            segment[: actual_end - actual_start] = audio[actual_start:actual_end]
        else:
            segment = audio[start_idx:end_idx].copy()

        # 다른 impact와 겹치는 경우 처리
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
                continue
            elif overlap_handling == "replace":
                for other_time in impact_times:
                    if other_time == impact_time:
                        continue
                    if segment_start_time <= other_time <= segment_end_time:
                        overlap_start_idx = int((other_time - segment_start_time) * sr) - int(0.05 * sr)
                        overlap_end_idx = int((other_time - segment_start_time) * sr) + int(0.05 * sr)
                        overlap_start_idx = max(0, overlap_start_idx)
                        overlap_end_idx = min(len(segment), overlap_end_idx)
                        segment[overlap_start_idx:overlap_end_idx] = background_noise_mean

        trimmed_segments.append(segment)

    return trimmed_segments

def classify_impacts_in_wav_finetuned(
    wav_path,
    panns_model,
    device,
    sr_orig=44100,
    target_sr=32000,
    height_ratio=0.3,
    min_distance_sec=0.5,
    before_sec=0.05,
    after_sec=0.95,
):
    """
    Inference function for Fine-Tuned PANNs model.
    Detects impacts, batches them, and classifies material.
    """
    wav_path = Path(wav_path)

    # Internal helper to fix length to exactly 32000 samples (1 sec at 32k)
    def _fix_length_for_inference(wav: np.ndarray, target_len: int):
        L = len(wav)
        if L == target_len:
            return wav
        elif L < target_len:
            # Pad end
            pad = target_len - L
            return np.pad(wav, (0, pad), mode="constant")
        else:
            # Center crop
            start = (L - target_len) // 2
            return wav[start : start + target_len]

    # 1. Load Audio & Denoise
    # Note: Ensure load_audio and reduce_noise_spectral_gating are defined in your scope
    try:
        audio, sr = load_audio(str(wav_path), sampling_rate=sr_orig)
    except Exception as e:
        print(f"[Error] Could not load {wav_path}: {e}")
        return []
        
    audio_denoised = reduce_noise_spectral_gating(audio, sr, prop_decrease=0.8)

    # 2. Detect Impacts (Peak Finding)
    impact_times, _ = detect_impacts_peak_finding(
        audio_denoised,
        sr,
        height_ratio=height_ratio,
        min_distance_sec=min_distance_sec,
    )

    if len(impact_times) == 0:
        print(f"[WARN] No impacts detected in {wav_path.name}")
        return []

    # 3. Trim Segments (1 second clips, handling overlap)
    segments = trim_audio_around_impacts(
        audio_denoised,
        sr,
        impact_times,
        before_sec=before_sec,
        after_sec=after_sec,
        overlap_handling="exclude",
    )

    # Filter valid times to match segments (in case trim_audio dropped overlaps)
    # This logic replicates the check in your notebook
    valid_times = []
    for t in impact_times:
        start_t = t - before_sec
        end_t = t + after_sec
        # Simple overlap check logic matching your extraction function
        # (Assuming segments returned correspond sequentially to valid non-overlapping times)
        # For strict robustness, trim_audio_around_impacts should arguably return the times too,
        # but we reconstruct valid_times here assuming 'exclude' logic works sequentially.
        has_overlap = False
        for other_t in impact_times:
            if other_t == t: continue
            if start_t <= other_t <= end_t:
                has_overlap = True
                break
        if not has_overlap:
            valid_times.append(t)
            
    # Safety trim to match lengths
    min_len = min(len(segments), len(valid_times))
    segments = segments[:min_len]
    valid_times = valid_times[:min_len]

    if not segments:
        return []

    # 4. Batch Preprocessing
    processed_tensors = []
    target_len_samples = int(target_sr * (before_sec + after_sec)) # Should be 32000

    for seg in segments:
        # Resample to 32k (PANNs requirement)
        seg_32k = librosa.resample(seg, orig_sr=sr, target_sr=target_sr)
        # Fix length
        seg_fixed = _fix_length_for_inference(seg_32k, target_len_samples)
        # Convert to Tensor
        processed_tensors.append(torch.from_numpy(seg_fixed.astype(np.float32)))

    # Stack into a batch: (Batch_Size, Time_Steps) -> (N, 32000)
    batch_input = torch.stack(processed_tensors).to(device)

    # 5. Batch Inference
    panns_model.eval()
    results = []
    
    with torch.no_grad():
        # Forward pass
        output_dict = panns_model(batch_input)
        
        # Extract logits (support dictionary output)
        if isinstance(output_dict, dict) and "clipwise_output" in output_dict:
            logits = output_dict["clipwise_output"]
        else:
            logits = output_dict

        # Softmax probabilities
        probs_batch = F.softmax(logits, dim=1).cpu().numpy() # (N, 5)
        preds_batch = np.argmax(probs_batch, axis=1)         # (N,)

    # 6. Format Results
    for i, (t, pred_label, probs) in enumerate(zip(valid_times, preds_batch, probs_batch)):
        results.append({
            "impact_idx": i,
            "impact_time": float(t),
            "pred_label": int(pred_label),
            "pred_material": label_to_material.get(int(pred_label), "Unknown"),
            "probs": probs, # Array of probabilities for all classes
            "confidence": float(probs[pred_label])
        })

    return results


def pwm_to_degree(pwm_value):
    """
    다이나믹셀 PWM(0~4095)을 각도(Degree)로 변환
    XL430/330 기준: 1 tick = 0.088도
    """
    # 0 ~ 4095 -> 0 ~ 360도
    return pwm_value * 0.088