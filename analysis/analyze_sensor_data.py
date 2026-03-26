"""
HAR Sensor Data Analysis Script
================================
data 폴더 안의 커스텀 형식 CSV 파일들을 파싱하여:
- 메타데이터(header/footer) + 센서 데이터 분리
- accelerometer / gyroscope 그래프 생성
- 파일별 / label별 / 전체 데이터셋 요약 생성
- HTML 종합 리포트 생성

CSV 형식:
  # key=value          ← header metadata
  elapsed_ms,label,... ← CSV header + body
  # key=value          ← footer metadata

Usage:
  python analyze_sensor_data.py
  python analyze_sensor_data.py --data-dir ../data --out-dir ../analysis_output
"""

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# scipy는 Butterworth LPF에 사용 — 없으면 EMA fallback
try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", font_scale=0.9)

logger = logging.getLogger("sensor_analysis")

# ──────────────────────────────────────────────
# 1. 데이터 모델
# ──────────────────────────────────────────────

@dataclass
class ParsedCSV:
    """하나의 CSV 파일 파싱 결과"""
    filepath: str
    filename: str
    header_metadata: Dict[str, str] = field(default_factory=dict)
    footer_metadata: Dict[str, str] = field(default_factory=dict)
    dataframe: Optional[pd.DataFrame] = None
    parse_error: Optional[str] = None


# ──────────────────────────────────────────────
# 2. 커스텀 CSV 파서
# ──────────────────────────────────────────────

def parse_metadata_line(line: str) -> Optional[Tuple[str, str]]:
    """'# key=value' 형태의 줄에서 (key, value)를 추출. 실패하면 None."""
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    content = stripped[1:].strip()  # '#' 제거
    if "=" not in content:
        return None
    key, _, value = content.partition("=")
    return key.strip(), value.strip()


def parse_sensor_csv(filepath: str) -> ParsedCSV:
    """
    커스텀 형식 CSV를 파싱한다.

    구조:
      1) 파일 상단 — '# key=value' 줄들 → header_metadata
      2) CSV 헤더 + 본문 데이터
      3) 파일 하단 — '# key=value' 줄들 → footer_metadata

    Returns:
        ParsedCSV 객체 (파싱 실패 시 parse_error에 메시지 기록)
    """
    result = ParsedCSV(filepath=filepath, filename=os.path.basename(filepath))

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except Exception as e:
        result.parse_error = f"파일 읽기 실패: {e}"
        return result

    if not raw_lines:
        result.parse_error = "빈 파일"
        return result

    # ── Phase 1: header metadata 추출 ──
    data_start_idx = 0
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            kv = parse_metadata_line(stripped)
            if kv:
                result.header_metadata[kv[0]] = kv[1]
            data_start_idx = i + 1
        elif stripped == "":
            data_start_idx = i + 1
        else:
            # 첫 번째 non-comment, non-empty 줄 = CSV 헤더
            data_start_idx = i
            break

    # ── Phase 2: footer metadata 추출 (뒤에서부터 역방향 탐색) ──
    data_end_idx = len(raw_lines)
    for i in range(len(raw_lines) - 1, data_start_idx, -1):
        stripped = raw_lines[i].strip()
        if stripped.startswith("#"):
            kv = parse_metadata_line(stripped)
            if kv:
                result.footer_metadata[kv[0]] = kv[1]
            data_end_idx = i
        elif stripped == "":
            data_end_idx = i
        else:
            break

    # ── Phase 3: 본문 데이터 파싱 ──
    body_lines = raw_lines[data_start_idx:data_end_idx]
    if not body_lines:
        result.parse_error = "본문 데이터 없음"
        return result

    # comment 줄이 섞여 있을 수 있으므로 제거
    clean_lines = [l for l in body_lines if not l.strip().startswith("#") and l.strip()]
    if not clean_lines:
        result.parse_error = "유효한 데이터 행 없음"
        return result

    try:
        from io import StringIO
        csv_text = "".join(clean_lines)
        df = pd.read_csv(StringIO(csv_text))
        # 숫자 열 변환
        for col in ["elapsed_ms", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        result.dataframe = df
    except Exception as e:
        result.parse_error = f"CSV 파싱 실패: {e}"

    return result


# ──────────────────────────────────────────────
# 3. 파일 탐색
# ──────────────────────────────────────────────

def find_csv_files(data_dir: str) -> List[str]:
    """data_dir 아래의 모든 .csv 파일을 재귀적으로 찾는다."""
    csv_files = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return csv_files


# ──────────────────────────────────────────────
# 4. 데이터 품질 검사
# ──────────────────────────────────────────────

def check_quality(parsed: ParsedCSV) -> Dict:
    """파일 하나에 대한 품질 검사 결과를 딕셔너리로 반환."""
    issues = []
    df = parsed.dataframe

    if df is None or df.empty:
        return {"issues": ["데이터 없음"], "has_missing": True, "time_monotonic": False,
                "near_constant": True, "label_mismatch": False, "sample_density_hz": 0}

    # 결측치
    has_missing = df.isnull().any().any()
    if has_missing:
        issues.append("결측치 존재")

    # elapsed_ms 단조 증가 여부
    time_monotonic = True
    if "elapsed_ms" in df.columns:
        diffs = df["elapsed_ms"].diff().dropna()
        if (diffs < 0).any():
            time_monotonic = False
            issues.append("elapsed_ms가 단조 증가하지 않음")

    # sensor magnitude 변화량이 극히 작은 경우
    near_constant = False
    for prefix in ["acc", "gyro"]:
        cols = [c for c in df.columns if c.startswith(prefix + "_") and c != prefix + "_mag"]
        if cols:
            stds = df[cols].std()
            if (stds < 1e-6).all():
                near_constant = True
                issues.append(f"{prefix} 값이 거의 변하지 않음")

    # label mismatch
    label_mismatch = False
    meta_label = parsed.header_metadata.get("label", None)
    if meta_label and "label" in df.columns:
        body_labels = df["label"].unique()
        if len(body_labels) == 1 and body_labels[0] != meta_label:
            label_mismatch = True
            issues.append(f"메타데이터 label({meta_label}) ≠ 본문 label({body_labels[0]})")
        elif len(body_labels) > 1:
            issues.append(f"본문에 여러 label 존재: {list(body_labels)}")

    # 샘플 밀도 추정
    sample_density_hz = 0
    if "elapsed_ms" in df.columns and len(df) > 1:
        duration_s = (df["elapsed_ms"].max() - df["elapsed_ms"].min()) / 1000.0
        if duration_s > 0:
            sample_density_hz = round(len(df) / duration_s, 1)

    return {
        "issues": issues,
        "has_missing": has_missing,
        "time_monotonic": time_monotonic,
        "near_constant": near_constant,
        "label_mismatch": label_mismatch,
        "sample_density_hz": sample_density_hz,
    }


# ──────────────────────────────────────────────
# 5. 통계 계산
# ──────────────────────────────────────────────

def add_magnitude_columns(df: pd.DataFrame) -> pd.DataFrame:
    """acc_mag, gyro_mag 열 추가."""
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
    if all(c in df.columns for c in acc_cols):
        df["acc_mag"] = np.sqrt((df[acc_cols] ** 2).sum(axis=1))
    if all(c in df.columns for c in gyro_cols):
        df["gyro_mag"] = np.sqrt((df[gyro_cols] ** 2).sum(axis=1))
    return df


# ──────────────────────────────────────────────
# 5-b. 파생 신호 (gravity, dynamic accel, V/H 분해)
# ──────────────────────────────────────────────

def estimate_sampling_rate(df: pd.DataFrame) -> float:
    """elapsed_ms 열로부터 평균 샘플링 레이트(Hz)를 추정한다.
    (전체 시간 범위 / 샘플 수 방식 — 중복 timestamp 행이 있어도 안정적)"""
    if "elapsed_ms" not in df.columns or len(df) < 2:
        return 100.0  # 기본값
    total_ms = df["elapsed_ms"].iloc[-1] - df["elapsed_ms"].iloc[0]
    if total_ms <= 0:
        return 100.0
    return (len(df) - 1) * 1000.0 / total_ms


def _lowpass_butter(signal: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """Butterworth low-pass filter (scipy)."""
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        cutoff_hz = nyq * 0.9
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal)


def _lowpass_ema(signal: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Exponential moving average fallback (scipy 없을 때)."""
    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]
    return out


def add_derived_signal_columns(df: pd.DataFrame, lpf_cutoff: float = 0.3) -> pd.DataFrame:
    """
    파생 신호 열을 추가한다:
      grav_x/y/z   – 중력 추정 (LPF)
      dyn_x/y/z    – 동적 가속도 = acc - grav
      dyn_mag      – 동적 가속도 크기
      a_v          – 수직 동적 가속도 (중력 방향 투영)
      a_h          – 수평 동적 가속도
      w_v          – 수직 각속도
      w_h          – 수평 각속도
    """
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
    if not all(c in df.columns for c in acc_cols):
        return df

    fs = estimate_sampling_rate(df)

    # ── 중력 추정 (Low-pass filter) ──
    for col, gcol in zip(acc_cols, ["grav_x", "grav_y", "grav_z"]):
        sig = df[col].values.astype(float)
        if HAS_SCIPY and len(sig) > 12:
            df[gcol] = _lowpass_butter(sig, lpf_cutoff, fs)
        else:
            df[gcol] = _lowpass_ema(sig)

    # ── 동적 가속도 ──
    for ac, gc, dc in zip(acc_cols, ["grav_x", "grav_y", "grav_z"],
                          ["dyn_x", "dyn_y", "dyn_z"]):
        df[dc] = df[ac] - df[gc]

    df["dyn_mag"] = np.sqrt(df["dyn_x"] ** 2 + df["dyn_y"] ** 2 + df["dyn_z"] ** 2)

    # ── 중력 방향 단위벡터 ──
    grav_mag = np.sqrt(df["grav_x"] ** 2 + df["grav_y"] ** 2 + df["grav_z"] ** 2)
    grav_mag = grav_mag.replace(0, np.nan)
    g_hat_x = df["grav_x"] / grav_mag
    g_hat_y = df["grav_y"] / grav_mag
    g_hat_z = df["grav_z"] / grav_mag

    # ── 수직/수평 동적 가속도 분해 ──
    df["a_v"] = df["dyn_x"] * g_hat_x + df["dyn_y"] * g_hat_y + df["dyn_z"] * g_hat_z
    a_h_sq = df["dyn_mag"] ** 2 - df["a_v"] ** 2
    df["a_h"] = np.sqrt(a_h_sq.clip(lower=0))

    # ── 수직/수평 각속도 분해 ──
    if all(c in df.columns for c in gyro_cols):
        df["w_v"] = df["gyro_x"] * g_hat_x + df["gyro_y"] * g_hat_y + df["gyro_z"] * g_hat_z
        w_h_sq = df["gyro_mag"] ** 2 - df["w_v"] ** 2 if "gyro_mag" in df.columns else (
            df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2 - df["w_v"] ** 2)
        df["w_h"] = np.sqrt(w_h_sq.clip(lower=0))

    return df


# ──────────────────────────────────────────────
# 5-c. 윈도우 기반 특징 추출
# ──────────────────────────────────────────────

def compute_window_features(df: pd.DataFrame, window_sec: float = 2.0,
                            stride_sec: float = 0.5) -> pd.DataFrame:
    """
    슬라이딩 윈도우로 특징을 추출한다.
    Returns:
        DataFrame — 윈도우 별 특징 (window_start_ms, window_end_ms, S_v, R_impact, J_v, R_HF, E_w_h, step_freq, ...)
    """
    if df is None or df.empty or "elapsed_ms" not in df.columns or "a_v" not in df.columns:
        return pd.DataFrame()

    fs = estimate_sampling_rate(df)
    window_samples = max(int(window_sec * fs), 10)
    stride_samples = max(int(stride_sec * fs), 1)

    records = []
    n = len(df)
    start = 0
    while start + window_samples <= n:
        end = start + window_samples
        w = df.iloc[start:end]
        t0 = w["elapsed_ms"].iloc[0]
        t1 = w["elapsed_ms"].iloc[-1]
        av = w["a_v"].values.astype(float)
        ah = w["a_h"].values.astype(float) if "a_h" in w.columns else np.zeros(len(w))

        feat: Dict = {"window_start_ms": t0, "window_end_ms": t1}

        # S_v : 수직 가속도 표준편차
        feat["S_v"] = np.nanstd(av)

        # R_impact : 충격 비대칭성 = max(a_v) / |min(a_v)| (min이 0이면 nan)
        av_max = np.nanmax(av)
        av_min = np.nanmin(av)
        feat["R_impact"] = av_max / abs(av_min) if abs(av_min) > 1e-6 else np.nan

        # J_v : 수직 저크 = mean(|d(a_v)/dt|)
        dt_arr = w["elapsed_ms"].diff().iloc[1:].values / 1000.0  # seconds
        dav = np.diff(av)
        valid = dt_arr > 0
        if valid.any():
            feat["J_v"] = np.nanmean(np.abs(dav[valid] / dt_arr[valid]))
        else:
            feat["J_v"] = np.nan

        # step_freq & R_HF : FFT 기반
        if len(av) >= 8:
            fft_vals = np.fft.rfft(av - np.nanmean(av))
            power = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(len(av), d=1.0 / fs)

            # step_freq: 0.5~4 Hz 범위에서 피크 주파수
            walk_mask = (freqs >= 0.5) & (freqs <= 4.0)
            if walk_mask.any():
                peak_idx = np.argmax(power[walk_mask])
                feat["step_freq"] = freqs[walk_mask][peak_idx]
            else:
                feat["step_freq"] = np.nan

            # R_HF: 고주파(>3Hz) 에너지 / 전체 에너지
            total_power = power[1:].sum()  # DC 제외
            hf_mask = freqs[1:] > 3.0
            feat["R_HF"] = power[1:][hf_mask].sum() / total_power if total_power > 0 else np.nan
        else:
            feat["step_freq"] = np.nan
            feat["R_HF"] = np.nan

        # E_w_h : 수평 각속도 에너지
        if "w_h" in w.columns:
            wh = w["w_h"].values.astype(float)
            feat["E_w_h"] = np.nanmean(wh ** 2)
        else:
            feat["E_w_h"] = np.nan

        # 보조 통계
        feat["mean_a_v"] = np.nanmean(av)
        feat["std_a_v"] = np.nanstd(av)
        feat["mean_a_h"] = np.nanmean(ah)
        feat["std_a_h"] = np.nanstd(ah)

        # acc_mag / gyro_mag 윈도우 통계
        if "acc_mag" in w.columns:
            am = w["acc_mag"].values.astype(float)
            feat["acc_mag_mean"] = np.nanmean(am)
            feat["acc_mag_std"] = np.nanstd(am)
        else:
            feat["acc_mag_mean"] = np.nan
            feat["acc_mag_std"] = np.nan

        if "gyro_mag" in w.columns:
            gm = w["gyro_mag"].values.astype(float)
            feat["gyro_mag_mean"] = np.nanmean(gm)
            feat["gyro_mag_std"] = np.nanstd(gm)
        else:
            feat["gyro_mag_mean"] = np.nan
            feat["gyro_mag_std"] = np.nan

        # zero-crossing count (acc_mag 기준, 평균 기준선)
        if "acc_mag" in w.columns:
            am = w["acc_mag"].values.astype(float)
            am_centered = am - np.nanmean(am)
            feat["zero_crossing_count"] = int(np.sum(np.diff(np.sign(am_centered)) != 0))
        else:
            feat["zero_crossing_count"] = 0

        # ────────────────────────────────────────────
        # Lateral Rotation Smoothness Index (F_LRS)
        # ────────────────────────────────────────────
        # 목적: stairs_up(부드러운 좌우 회전) vs stairs_down(spike성 충격) 구분 보조
        #   F_LRS = A_z * (E_L / (E_H + eps)) * (1 / (K_z + eps))
        #   - 값이 클수록 부드럽고 지속적인 좌우 회전이 크며 spike 적음 → stairs_up 가능성
        #   - 값이 작을수록 spike-like 고주파 충격 우세 → stairs_down 가능성
        eps = 1e-10
        if "gyro_z" in w.columns:
            gz_raw = w["gyro_z"].values.astype(float)
            N_gz = len(gz_raw)

            # 1) 평균 제거된 gyro_z
            gz_mean = np.nanmean(gz_raw)
            gz_centered = gz_raw - gz_mean

            # 2) 회전 진폭 RMS (A_z)
            A_z = np.sqrt(np.nanmean(gz_centered ** 2))
            feat["gyro_z_rms"] = A_z

            # 3) 저주파/고주파 대역 에너지 (FFT magnitude^2 기반)
            #    저주파 회전 대역: 0.3~2.0 Hz / 고주파 충격 대역: 3.0~8.0 Hz
            if N_gz >= 8:
                gz_fft = np.fft.rfft(gz_centered)
                gz_power = np.abs(gz_fft) ** 2  # |FFT|^2 (spectral power)
                gz_freqs = np.fft.rfftfreq(N_gz, d=1.0 / fs)

                low_mask = (gz_freqs >= 0.3) & (gz_freqs <= 2.0)
                high_mask = (gz_freqs >= 3.0) & (gz_freqs <= 8.0)

                E_L = gz_power[low_mask].sum() if low_mask.any() else 0.0
                E_H = gz_power[high_mask].sum() if high_mask.any() else 0.0
            else:
                E_L = 0.0
                E_H = 0.0

            feat["gyro_z_low_band_energy"] = E_L
            feat["gyro_z_high_band_energy"] = E_H
            feat["gyro_z_low_high_ratio"] = E_L / (E_H + eps)

            # 4) Kurtosis (직접 계산: m4/m2^2)
            #    K_z 가 크면 spike-like, 작으면 부드러운 진동
            m2 = np.nanmean(gz_centered ** 2)
            m4 = np.nanmean(gz_centered ** 4)
            K_z = m4 / (m2 ** 2 + eps)
            feat["gyro_z_kurtosis"] = K_z

            # 5) 최종 F_LRS
            F_LRS = A_z * (E_L / (E_H + eps)) * (1.0 / (K_z + eps))
            feat["gyro_z_lateral_rotation_smoothness"] = F_LRS
        else:
            feat["gyro_z_rms"] = np.nan
            feat["gyro_z_low_band_energy"] = np.nan
            feat["gyro_z_high_band_energy"] = np.nan
            feat["gyro_z_low_high_ratio"] = np.nan
            feat["gyro_z_kurtosis"] = np.nan
            feat["gyro_z_lateral_rotation_smoothness"] = np.nan

        records.append(feat)
        start += stride_samples

    return pd.DataFrame(records)


def compute_file_stats(parsed: ParsedCSV) -> Dict:
    """파일 하나에 대한 통계 딕셔너리."""
    df = parsed.dataframe
    if df is None or df.empty:
        return {}

    stats = {}
    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "acc_mag", "gyro_mag"]
    for col in sensor_cols:
        if col in df.columns:
            stats[f"{col}_mean"] = round(df[col].mean(), 6)
            stats[f"{col}_std"] = round(df[col].std(), 6)
            stats[f"{col}_min"] = round(df[col].min(), 6)
            stats[f"{col}_max"] = round(df[col].max(), 6)

    return stats


def build_file_summary(parsed: ParsedCSV, quality: Dict, stats: Dict) -> Dict:
    """파일별 요약 행 하나를 딕셔너리로 만든다."""
    df = parsed.dataframe
    row = {
        "filename": parsed.filename,
        "filepath": parsed.filepath,
        "meta_label": parsed.header_metadata.get("label", ""),
        "body_labels": "",
        "phone_hand": parsed.header_metadata.get("phone_hand", ""),
        "screen_direction": parsed.header_metadata.get("screen_direction", ""),
        "num_samples": 0,
        "elapsed_duration_ms": 0,
        "footer_total_recording_time_ms": parsed.footer_metadata.get("total_recording_time_ms", ""),
        "footer_num_samples_after_trim": parsed.footer_metadata.get("num_samples_after_trim", ""),
        "has_missing": quality.get("has_missing", False),
        "time_monotonic": quality.get("time_monotonic", True),
        "near_constant": quality.get("near_constant", False),
        "label_mismatch": quality.get("label_mismatch", False),
        "sample_density_hz": quality.get("sample_density_hz", 0),
        "issues": "; ".join(quality.get("issues", [])),
    }

    if df is not None and not df.empty:
        row["num_samples"] = len(df)
        if "label" in df.columns:
            row["body_labels"] = ", ".join(df["label"].astype(str).unique())
        if "elapsed_ms" in df.columns:
            row["elapsed_duration_ms"] = df["elapsed_ms"].max() - df["elapsed_ms"].min()

    row.update(stats)
    return row


# ──────────────────────────────────────────────
# 6. 그래프 생성
# ──────────────────────────────────────────────

def plot_file_graphs(parsed: ParsedCSV, out_dir: str):
    """파일 하나에 대해 4개 그래프를 저장한다."""
    df = parsed.dataframe
    if df is None or df.empty or "elapsed_ms" not in df.columns:
        return

    time = df["elapsed_ms"]
    base = os.path.splitext(parsed.filename)[0]
    label_str = parsed.header_metadata.get("label", "unknown")

    # ── accelerometer xyz ──
    if all(c in df.columns for c in ["acc_x", "acc_y", "acc_z"]):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, df["acc_x"], label="acc_x", linewidth=0.7)
        ax.plot(time, df["acc_y"], label="acc_y", linewidth=0.7)
        ax.plot(time, df["acc_z"], label="acc_z", linewidth=0.7)
        ax.set_xlabel("elapsed_ms")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title(f"Accelerometer — {label_str} ({parsed.filename})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_acc_xyz.png"), dpi=150)
        plt.close(fig)

    # ── gyroscope xyz ──
    if all(c in df.columns for c in ["gyro_x", "gyro_y", "gyro_z"]):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, df["gyro_x"], label="gyro_x", linewidth=0.7)
        ax.plot(time, df["gyro_y"], label="gyro_y", linewidth=0.7)
        ax.plot(time, df["gyro_z"], label="gyro_z", linewidth=0.7)
        ax.set_xlabel("elapsed_ms")
        ax.set_ylabel("Angular velocity (rad/s)")
        ax.set_title(f"Gyroscope — {label_str} ({parsed.filename})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_gyro_xyz.png"), dpi=150)
        plt.close(fig)

    # ── acc magnitude ──
    if "acc_mag" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, df["acc_mag"], color="tab:red", linewidth=0.7)
        ax.axhline(y=9.81, color="gray", linestyle="--", linewidth=0.5, label="g ≈ 9.81")
        ax.set_xlabel("elapsed_ms")
        ax.set_ylabel("Acceleration magnitude (m/s²)")
        ax.set_title(f"Acc Magnitude — {label_str} ({parsed.filename})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_acc_mag.png"), dpi=150)
        plt.close(fig)

    # ── gyro magnitude ──
    if "gyro_mag" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, df["gyro_mag"], color="tab:purple", linewidth=0.7)
        ax.set_xlabel("elapsed_ms")
        ax.set_ylabel("Gyro magnitude (rad/s)")
        ax.set_title(f"Gyro Magnitude — {label_str} ({parsed.filename})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_gyro_mag.png"), dpi=150)
        plt.close(fig)

    # ── wf_grid 초기화 (윈도우 기반 그래프 공통) ──
    _WINDOWS = [1.0, 1.5, 2.0, 3.0]
    _STRIDES = [0.5, 1.0, 1.5, 2.0]
    wf_grid = getattr(parsed, '_wf_grid', None)
    if wf_grid is None:
        wf_map = getattr(parsed, '_window_features_map', None)
        if wf_map:
            wf_grid = {(2.0, s): v for s, v in wf_map.items()}

    # ── acc_mag window mean & std (stride-dependent) ──
    if wf_grid:
        for win_val in _WINDOWS:
            for stride_val in _STRIDES:
                wf = wf_grid.get((win_val, stride_val))
                if wf is None or wf.empty or "acc_mag_mean" not in wf.columns:
                    continue
                ws_tag = f"_w{win_val}_s{stride_val}"
                mid_t = (wf["window_start_ms"] + wf["window_end_ms"]) / 2
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                axes[0].plot(mid_t, wf["acc_mag_mean"], marker=".", markersize=2, linewidth=0.8, color="tab:red")
                axes[0].set_ylabel("Acc Mag Mean (m/s²)")
                axes[0].set_title(f"Window Acc-Mag Mean (win={win_val}s, stride={stride_val}s) — {label_str} ({parsed.filename})")
                axes[1].plot(mid_t, wf["acc_mag_std"], marker=".", markersize=2, linewidth=0.8, color="tab:orange")
                axes[1].set_ylabel("Acc Mag Std (m/s²)")
                axes[1].set_xlabel("elapsed_ms")
                axes[1].set_title("Window Acc-Mag Std")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{base}_acc_mag_window{ws_tag}.png"), dpi=150)
                plt.close(fig)

    # ── gyro_mag window mean & std (stride-dependent) ──
    if wf_grid:
        for win_val in _WINDOWS:
            for stride_val in _STRIDES:
                wf = wf_grid.get((win_val, stride_val))
                if wf is None or wf.empty or "gyro_mag_mean" not in wf.columns:
                    continue
                ws_tag = f"_w{win_val}_s{stride_val}"
                mid_t = (wf["window_start_ms"] + wf["window_end_ms"]) / 2
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                axes[0].plot(mid_t, wf["gyro_mag_mean"], marker=".", markersize=2, linewidth=0.8, color="tab:purple")
                axes[0].set_ylabel("Gyro Mag Mean (rad/s)")
                axes[0].set_title(f"Window Gyro-Mag Mean (win={win_val}s, stride={stride_val}s) — {label_str} ({parsed.filename})")
                axes[1].plot(mid_t, wf["gyro_mag_std"], marker=".", markersize=2, linewidth=0.8, color="tab:cyan")
                axes[1].set_ylabel("Gyro Mag Std (rad/s)")
                axes[1].set_xlabel("elapsed_ms")
                axes[1].set_title("Window Gyro-Mag Std")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{base}_gyro_mag_window{ws_tag}.png"), dpi=150)
                plt.close(fig)

    # ── gravity xyz ──
    if all(c in df.columns for c in ["grav_x", "grav_y", "grav_z"]):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, df["grav_x"], label="grav_x", linewidth=0.7)
        ax.plot(time, df["grav_y"], label="grav_y", linewidth=0.7)
        ax.plot(time, df["grav_z"], label="grav_z", linewidth=0.7)
        ax.set_xlabel("elapsed_ms")
        ax.set_ylabel("Gravity (m/s²)")
        ax.set_title(f"Gravity Estimation (LPF) — {label_str} ({parsed.filename})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_gravity_xyz.png"), dpi=150)
        plt.close(fig)

    # ── dynamic acceleration xyz ──
    if all(c in df.columns for c in ["dyn_x", "dyn_y", "dyn_z"]):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, df["dyn_x"], label="dyn_x", linewidth=0.7)
        ax.plot(time, df["dyn_y"], label="dyn_y", linewidth=0.7)
        ax.plot(time, df["dyn_z"], label="dyn_z", linewidth=0.7)
        ax.set_xlabel("elapsed_ms")
        ax.set_ylabel("Dynamic Accel (m/s²)")
        ax.set_title(f"Dynamic Acceleration — {label_str} ({parsed.filename})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_dynamic_xyz.png"), dpi=150)
        plt.close(fig)

    # ── dynamic magnitude ──
    if "dyn_mag" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, df["dyn_mag"], color="tab:orange", linewidth=0.7)
        ax.set_xlabel("elapsed_ms")
        ax.set_ylabel("Dynamic Accel Magnitude (m/s²)")
        ax.set_title(f"Dynamic Accel Magnitude — {label_str} ({parsed.filename})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_dyn_mag.png"), dpi=150)
        plt.close(fig)

    # ── vertical / horizontal accel ──
    if "a_v" in df.columns and "a_h" in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(time, df["a_v"], color="tab:blue", linewidth=0.7)
        axes[0].set_ylabel("a_v (m/s²)")
        axes[0].set_title(f"Vertical Dynamic Accel — {label_str} ({parsed.filename})")
        axes[1].plot(time, df["a_h"], color="tab:green", linewidth=0.7)
        axes[1].set_ylabel("a_h (m/s²)")
        axes[1].set_xlabel("elapsed_ms")
        axes[1].set_title("Horizontal Dynamic Accel")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_a_v_a_h.png"), dpi=150)
        plt.close(fig)

    # ── vertical / horizontal gyro ──
    if "w_v" in df.columns and "w_h" in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(time, df["w_v"], color="tab:red", linewidth=0.7)
        axes[0].set_ylabel("w_v (rad/s)")
        axes[0].set_title(f"Vertical Angular Velocity — {label_str} ({parsed.filename})")
        axes[1].plot(time, df["w_h"], color="tab:purple", linewidth=0.7)
        axes[1].set_ylabel("w_h (rad/s)")
        axes[1].set_xlabel("elapsed_ms")
        axes[1].set_title("Horizontal Angular Velocity")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_w_v_w_h.png"), dpi=150)
        plt.close(fig)

    # ── window feature trends (window × stride 별) ──
    if wf_grid:
        for win_val in _WINDOWS:
            for stride_val in _STRIDES:
                wf = wf_grid.get((win_val, stride_val))
                if wf is None or wf.empty:
                    continue
                ws_tag = f"_w{win_val}_s{stride_val}"
                mid_t = (wf["window_start_ms"] + wf["window_end_ms"]) / 2
                fig, axes = plt.subplots(3, 2, figsize=(14, 10))
                feat_pairs = [
                    ("S_v", "Std a_v"), ("R_impact", "Impact Ratio"),
                    ("J_v", "Jerk (vertical)"), ("R_HF", "HF Energy Ratio"),
                    ("E_w_h", "Horiz Gyro Energy"), ("step_freq", "Step Freq (Hz)"),
                ]
                for ax, (col, title) in zip(axes.flat, feat_pairs):
                    if col in wf.columns:
                        ax.plot(mid_t, wf[col], marker=".", markersize=2, linewidth=0.8)
                    ax.set_title(title)
                    ax.set_xlabel("elapsed_ms")
                fig.suptitle(f"Window Features (win={win_val}s, stride={stride_val}s) — {label_str} ({parsed.filename})", fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{base}_window_features{ws_tag}.png"), dpi=150)
                plt.close(fig)

    # ── E_w_h (Horizontal Gyro Energy) trends (stride-dependent) ──
    if wf_grid:
        for win_val in _WINDOWS:
            for stride_val in _STRIDES:
                wf = wf_grid.get((win_val, stride_val))
                if wf is None or wf.empty or "E_w_h" not in wf.columns:
                    continue
                ws_tag = f"_w{win_val}_s{stride_val}"
                mid_t = (wf["window_start_ms"] + wf["window_end_ms"]) / 2
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(mid_t, wf["E_w_h"], marker=".", markersize=2, linewidth=0.8, color="tab:green")
                ax.set_xlabel("elapsed_ms")
                ax.set_ylabel("E_w_h (Horiz Gyro Energy)")
                ax.set_title(f"Horizontal Gyro Energy (win={win_val}s, stride={stride_val}s) — {label_str} ({parsed.filename})")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{base}_ewh_trend{ws_tag}.png"), dpi=150)
                plt.close(fig)

    # ── zero-crossing count trends (stride-dependent) ──
    if wf_grid:
        for win_val in _WINDOWS:
            for stride_val in _STRIDES:
                wf = wf_grid.get((win_val, stride_val))
                if wf is None or wf.empty or "zero_crossing_count" not in wf.columns:
                    continue
                ws_tag = f"_w{win_val}_s{stride_val}"
                mid_t = (wf["window_start_ms"] + wf["window_end_ms"]) / 2
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(mid_t, wf["zero_crossing_count"], marker=".", markersize=2, linewidth=0.8, color="tab:brown")
                ax.set_xlabel("elapsed_ms")
                ax.set_ylabel("Zero-Crossing Count")
                ax.set_title(f"Zero-Crossing Count (win={win_val}s, stride={stride_val}s) — {label_str} ({parsed.filename})")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{base}_zero_crossing{ws_tag}.png"), dpi=150)
                plt.close(fig)

    # ── gyro_z rotation profile ──
    # 부드러운 좌우 회전 vs spike-like 패턴 확인용
    if "gyro_z" in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(time, df["gyro_z"], color="tab:cyan", linewidth=0.5, alpha=0.7, label="raw gyro_z")
        # rolling mean (약 0.5초 윈도우)
        roll_n = max(int(estimate_sampling_rate(df) * 0.5), 3)
        gz_smooth = df["gyro_z"].rolling(roll_n, center=True, min_periods=1).mean()
        axes[0].plot(time, gz_smooth, color="tab:red", linewidth=1.2, label=f"rolling mean ({roll_n} samples)")
        axes[0].set_ylabel("gyro_z (rad/s)")
        axes[0].set_title(f"Gyro-Z: Smooth Lateral Rotation vs Spike Pattern — {label_str} ({parsed.filename})")
        axes[0].legend(fontsize=8)

        gz_centered = df["gyro_z"] - df["gyro_z"].mean()
        axes[1].plot(time, gz_centered, color="tab:olive", linewidth=0.5, label="mean-centered gyro_z")
        axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1].set_ylabel("centered gyro_z (rad/s)")
        axes[1].set_xlabel("elapsed_ms")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{base}_gyro_z_rotation_profile.png"), dpi=150)
        plt.close(fig)

    # ── window feature trends (gyro_z lateral rotation, window × stride 별) ──
    if wf_grid:
        for win_val in _WINDOWS:
            for stride_val in _STRIDES:
                wf = wf_grid.get((win_val, stride_val))
                if wf is None or wf.empty or "gyro_z_lateral_rotation_smoothness" not in wf.columns:
                    continue
                ws_tag = f"_w{win_val}_s{stride_val}"
                mid_t = (wf["window_start_ms"] + wf["window_end_ms"]) / 2
                fig, axes = plt.subplots(2, 2, figsize=(14, 8))
                gz_feat_pairs = [
                    ("gyro_z_rms", "Gyro-Z RMS (A_z)"),
                    ("gyro_z_low_high_ratio", "Low/High Freq Ratio (E_L/E_H)"),
                    ("gyro_z_kurtosis", "Gyro-Z Kurtosis (K_z)"),
                    ("gyro_z_lateral_rotation_smoothness", "F_LRS (Lateral Rotation Smoothness)"),
                ]
                for ax, (col, title) in zip(axes.flat, gz_feat_pairs):
                    if col in wf.columns:
                        ax.plot(mid_t, wf[col], marker=".", markersize=2, linewidth=0.8)
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel("elapsed_ms")
                fig.suptitle(f"Gyro-Z Rotation Features (win={win_val}s, stride={stride_val}s) — {label_str} ({parsed.filename})", fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{base}_window_feature_trends_gyroz{ws_tag}.png"), dpi=150)
                plt.close(fig)


def plot_comparative_graphs(all_parsed: List[ParsedCSV], out_dir: str):
    """label별 비교 그래프를 생성한다."""
    # label별 전체 데이터 결합
    frames = []
    for p in all_parsed:
        if p.dataframe is not None and not p.dataframe.empty:
            df_copy = p.dataframe.copy()
            df_copy["_source_file"] = p.filename
            df_copy["_meta_label"] = p.header_metadata.get("label", "unknown")
            frames.append(df_copy)

    if not frames:
        return

    all_df = pd.concat(frames, ignore_index=True)
    labels = sorted(all_df["_meta_label"].unique())
    comp_dir = os.path.join(out_dir, "comparative")
    os.makedirs(comp_dir, exist_ok=True)

    _lbl_width = max(6, len(labels) * 1.5)

    # ── label별 acc_mag 분포 비교 (boxplot) ──
    if "acc_mag" in all_df.columns and len(labels) >= 1:
        fig, ax = plt.subplots(figsize=(_lbl_width, 5))
        sns.boxplot(data=all_df, x="_meta_label", y="acc_mag", ax=ax)
        ax.set_xlabel("Label")
        ax.set_ylabel("Acceleration magnitude (m/s²)")
        ax.set_title("Acc Magnitude Distribution by Label")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "acc_mag_boxplot_by_label.png"), dpi=150)
        plt.close(fig)

    # ── label별 gyro_mag 분포 비교 (boxplot) ──
    if "gyro_mag" in all_df.columns and len(labels) >= 1:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        sns.boxplot(data=all_df, x="_meta_label", y="gyro_mag", ax=ax)
        ax.set_xlabel("Label")
        ax.set_ylabel("Gyro magnitude (rad/s)")
        ax.set_title("Gyro Magnitude Distribution by Label")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "gyro_mag_boxplot_by_label.png"), dpi=150)
        plt.close(fig)

    # ── label별 acc_mag histogram ──
    if "acc_mag" in all_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label in labels:
            subset = all_df[all_df["_meta_label"] == label]["acc_mag"]
            ax.hist(subset, bins=50, alpha=0.5, label=label, density=True)
        ax.set_xlabel("Acceleration magnitude (m/s²)")
        ax.set_ylabel("Density")
        ax.set_title("Acc Magnitude Histogram by Label")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "acc_mag_hist_by_label.png"), dpi=150)
        plt.close(fig)

    # ── label별 gyro_mag histogram ──
    if "gyro_mag" in all_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label in labels:
            subset = all_df[all_df["_meta_label"] == label]["gyro_mag"]
            ax.hist(subset, bins=50, alpha=0.5, label=label, density=True)
        ax.set_xlabel("Gyro magnitude (rad/s)")
        ax.set_ylabel("Density")
        ax.set_title("Gyro Magnitude Histogram by Label")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "gyro_mag_hist_by_label.png"), dpi=150)
        plt.close(fig)

    # ── label별 violin plot (acc_mag) ──
    if "acc_mag" in all_df.columns and len(labels) >= 1:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        sns.violinplot(data=all_df, x="_meta_label", y="acc_mag", ax=ax, inner="quartile")
        ax.set_xlabel("Label")
        ax.set_ylabel("Acceleration magnitude (m/s²)")
        ax.set_title("Acc Magnitude Violin Plot by Label")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "acc_mag_violin_by_label.png"), dpi=150)
        plt.close(fig)

    # ── label별 violin plot (gyro_mag) ──
    if "gyro_mag" in all_df.columns and len(labels) >= 1:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        sns.violinplot(data=all_df, x="_meta_label", y="gyro_mag", ax=ax, inner="quartile")
        ax.set_xlabel("Label")
        ax.set_ylabel("Gyro magnitude (rad/s)")
        ax.set_title("Gyro Magnitude Violin Plot by Label")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "gyro_mag_violin_by_label.png"), dpi=150)
        plt.close(fig)

    # ── label별 대표 파일 시계열 비교 ──
    if len(labels) >= 1:
        # 각 label에서 첫 번째 파일을 대표로 선택
        fig, axes = plt.subplots(len(labels), 1, figsize=(12, 3.5 * len(labels)), squeeze=False)
        for idx, label in enumerate(labels):
            rep_files = [p for p in all_parsed
                         if p.header_metadata.get("label") == label
                         and p.dataframe is not None and not p.dataframe.empty]
            if not rep_files:
                continue
            rep = rep_files[0]
            ax = axes[idx, 0]
            if "acc_mag" in rep.dataframe.columns:
                ax.plot(rep.dataframe["elapsed_ms"], rep.dataframe["acc_mag"],
                        label="acc_mag", linewidth=0.7)
            if "gyro_mag" in rep.dataframe.columns:
                ax.plot(rep.dataframe["elapsed_ms"], rep.dataframe["gyro_mag"],
                        label="gyro_mag", linewidth=0.7)
            ax.set_title(f"{label} — {rep.filename}")
            ax.set_xlabel("elapsed_ms")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "representative_timeseries_by_label.png"), dpi=150)
        plt.close(fig)

    # ── recording duration / sample count 분포 ──
    durations = []
    sample_counts = []
    file_labels = []
    for p in all_parsed:
        if p.dataframe is not None and not p.dataframe.empty and "elapsed_ms" in p.dataframe.columns:
            dur = p.dataframe["elapsed_ms"].max() - p.dataframe["elapsed_ms"].min()
            durations.append(dur)
            sample_counts.append(len(p.dataframe))
            file_labels.append(p.header_metadata.get("label", "unknown"))

    if durations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = [sns.color_palette()[labels.index(l) % len(sns.color_palette())] for l in file_labels]

        ax1.bar(range(len(durations)), [d / 1000 for d in durations], color=colors)
        ax1.set_xlabel("File index")
        ax1.set_ylabel("Duration (s)")
        ax1.set_title("Recording Duration per File")

        ax2.bar(range(len(sample_counts)), sample_counts, color=colors)
        ax2.set_xlabel("File index")
        ax2.set_ylabel("Sample count")
        ax2.set_title("Sample Count per File")

        # legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=sns.color_palette()[labels.index(l) % len(sns.color_palette())],
                                 label=l) for l in sorted(set(file_labels))]
        ax1.legend(handles=legend_elements, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "duration_and_samplecount.png"), dpi=150)
        plt.close(fig)

    # ── label별 파생 신호 비교 (boxplot) ──
    derived_cols = [
        ("a_v", "Vertical Accel (a_v)", "m/s²"),
        ("a_h", "Horizontal Accel (a_h)", "m/s²"),
        ("dyn_mag", "Dynamic Accel Magnitude", "m/s²"),
    ]
    for col, title, unit in derived_cols:
        if col in all_df.columns:
            fig, ax = plt.subplots(figsize=(_lbl_width, 5))
            sns.boxplot(data=all_df, x="_meta_label", y=col, ax=ax)
            ax.set_xlabel("Label"); ax.set_ylabel(f"{title} ({unit})")
            ax.set_title(f"{title} by Label")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(os.path.join(comp_dir, f"{col}_boxplot_by_label.png"), dpi=150)
            plt.close(fig)

    # ── 윈도우 특징 비교 그래프 (window × stride 별) ──
    _WINDOWS = [1.0, 1.5, 2.0, 3.0]
    _STRIDES = [0.5, 1.0, 1.5, 2.0]
    for win_val in _WINDOWS:
        for stride_val in _STRIDES:
            ws_tag = f"_w{win_val}_s{stride_val}"
            wf_frames = []
            for p in all_parsed:
                wf_grid = getattr(p, '_wf_grid', None)
                if wf_grid and (win_val, stride_val) in wf_grid:
                    wf = wf_grid[(win_val, stride_val)]
                else:
                    wf = None
                if wf is not None and not wf.empty:
                    wfc = wf.copy()
                    wfc["_meta_label"] = p.header_metadata.get("label", "unknown")
                    wfc["_source_file"] = p.filename
                    wf_frames.append(wfc)

            if not wf_frames:
                continue
            wf_all = pd.concat(wf_frames, ignore_index=True)
            wf_feats = [
                ("S_v", "Std(a_v) per Window"),
                ("R_impact", "Impact Ratio per Window"),
                ("J_v", "Vertical Jerk per Window"),
                ("R_HF", "HF Energy Ratio per Window"),
                ("E_w_h", "Horiz Gyro Energy per Window"),
                ("step_freq", "Step Frequency per Window"),
            ]
            for col, title in wf_feats:
                if col not in wf_all.columns:
                    continue
                # boxplot
                fig, ax = plt.subplots(figsize=(_lbl_width, 5))
                sns.boxplot(data=wf_all, x="_meta_label", y=col, ax=ax)
                ax.set_xlabel("Label"); ax.set_ylabel(col)
                ax.set_title(f"{title} (win={win_val}s, stride={stride_val}s) — Boxplot")
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                fig.savefig(os.path.join(comp_dir, f"wf_{col}_boxplot{ws_tag}.png"), dpi=150)
                plt.close(fig)

                # violin
                fig, ax = plt.subplots(figsize=(_lbl_width, 5))
                sns.violinplot(data=wf_all, x="_meta_label", y=col, ax=ax, inner="quartile")
                ax.set_xlabel("Label"); ax.set_ylabel(col)
                ax.set_title(f"{title} (win={win_val}s, stride={stride_val}s) — Violin")
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                fig.savefig(os.path.join(comp_dir, f"wf_{col}_violin{ws_tag}.png"), dpi=150)
                plt.close(fig)

            # ── gyro_z Lateral Rotation Smoothness 관련 비교 (window 단위) ──
            gz_wf_feats = [
                ("gyro_z_lateral_rotation_smoothness", "Lateral Rotation Smoothness (F_LRS)"),
                ("gyro_z_low_high_ratio", "Gyro-Z Low/High Freq Ratio"),
                ("gyro_z_kurtosis", "Gyro-Z Kurtosis (spikiness)"),
                ("gyro_z_rms", "Gyro-Z RMS (A_z)"),
            ]
            for col, title in gz_wf_feats:
                if col not in wf_all.columns:
                    continue
                # boxplot
                fig, ax = plt.subplots(figsize=(_lbl_width, 5))
                sns.boxplot(data=wf_all, x="_meta_label", y=col, ax=ax)
                ax.set_xlabel("Label"); ax.set_ylabel(col)
                ax.set_title(f"{title} (win={win_val}s, stride={stride_val}s) — Boxplot")
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                fig.savefig(os.path.join(comp_dir, f"wf_{col}_boxplot{ws_tag}.png"), dpi=150)
                plt.close(fig)

                # violin
                fig, ax = plt.subplots(figsize=(_lbl_width, 5))
                sns.violinplot(data=wf_all, x="_meta_label", y=col, ax=ax, inner="quartile")
                ax.set_xlabel("Label"); ax.set_ylabel(col)
                ax.set_title(f"{title} (win={win_val}s, stride={stride_val}s) — Violin")
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                fig.savefig(os.path.join(comp_dir, f"wf_{col}_violin{ws_tag}.png"), dpi=150)
                plt.close(fig)


# ──────────────────────────────────────────────
# 7. 요약 테이블 생성
# ──────────────────────────────────────────────

def build_label_summary(all_parsed: List[ParsedCSV]) -> pd.DataFrame:
    """label별 요약 테이블 생성."""
    records = []
    for p in all_parsed:
        if p.dataframe is None or p.dataframe.empty:
            continue
        label = p.header_metadata.get("label", "unknown")
        df = p.dataframe
        dur = 0
        if "elapsed_ms" in df.columns:
            dur = df["elapsed_ms"].max() - df["elapsed_ms"].min()
        records.append({
            "label": label,
            "filename": p.filename,
            "num_samples": len(df),
            "duration_ms": dur,
            "acc_mag_mean": df["acc_mag"].mean() if "acc_mag" in df.columns else np.nan,
            "acc_mag_std": df["acc_mag"].std() if "acc_mag" in df.columns else np.nan,
            "gyro_mag_mean": df["gyro_mag"].mean() if "gyro_mag" in df.columns else np.nan,
            "gyro_mag_std": df["gyro_mag"].std() if "gyro_mag" in df.columns else np.nan,
        })

    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records)
    summary = raw.groupby("label").agg(
        file_count=("filename", "nunique"),
        total_samples=("num_samples", "sum"),
        total_duration_ms=("duration_ms", "sum"),
        acc_mag_mean=("acc_mag_mean", "mean"),
        acc_mag_std_mean=("acc_mag_std", "mean"),
        gyro_mag_mean=("gyro_mag_mean", "mean"),
        gyro_mag_std_mean=("gyro_mag_std", "mean"),
    ).reset_index()

    # round
    for col in summary.select_dtypes(include=[np.number]).columns:
        summary[col] = summary[col].round(6)

    return summary


# ──────────────────────────────────────────────
# 8. HTML 리포트 생성
# ──────────────────────────────────────────────

def generate_html_report(
    file_summaries: pd.DataFrame,
    label_summary: pd.DataFrame,
    all_parsed: List[ParsedCSV],
    out_dir: str,
    graphs_dir: str,
    comp_dir: str,
):
    """analysis_output/report.html 생성."""

    total_files = len(all_parsed)
    valid_files = sum(1 for p in all_parsed if p.dataframe is not None and not p.dataframe.empty)
    error_files = total_files - valid_files
    total_samples = int(file_summaries["num_samples"].sum()) if not file_summaries.empty else 0
    labels = sorted(file_summaries["meta_label"].unique()) if not file_summaries.empty else []

    # 그래프 파일 목록
    per_file_graphs = sorted([f for f in os.listdir(graphs_dir)
                              if f.endswith(".png")]) if os.path.isdir(graphs_dir) else []
    comp_graphs = sorted([f for f in os.listdir(comp_dir)
                          if f.endswith(".png")]) if os.path.isdir(comp_dir) else []

    def df_to_html(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>데이터 없음</p>"
        return df.to_html(index=False, classes="table", border=0, na_rep="—")

    # 관찰 포인트 생성
    observations = []
    if not file_summaries.empty:
        # acc_z 중력 근처 여부
        for _, row in file_summaries.iterrows():
            if "acc_z_mean" in row and not pd.isna(row.get("acc_z_mean")):
                if 9.5 < row["acc_z_mean"] < 10.1:
                    observations.append(
                        f"<code>{row['filename']}</code>: acc_z 평균이 {row['acc_z_mean']:.3f}로 "
                        f"중력(~9.81) 근처 → standing/sitting 등 정적 자세 가능성"
                    )
        # near constant
        near_const = file_summaries[file_summaries["near_constant"] == True]
        if not near_const.empty:
            observations.append(
                f"sensor 변동이 매우 작은 파일: {', '.join(near_const['filename'].tolist())}"
            )
        # label mismatch
        mismatch = file_summaries[file_summaries["label_mismatch"] == True]
        if not mismatch.empty:
            observations.append(
                f"label 불일치 파일: {', '.join(mismatch['filename'].tolist())}"
            )

    obs_html = "\n".join(f"<li>{o}</li>" for o in observations) if observations else "<li>특이사항 없음</li>"

    # Feature analysis 연결 포인트
    next_steps = [
        "시간 도메인 특징: mean, std, min, max, median, RMS, zero-crossing rate 등",
        "주파수 도메인 특징: FFT peak frequency, spectral energy, dominant frequency",
        "Magnitude 기반 특징: acc_mag, gyro_mag의 통계 + 변동성",
        "Window-based feature extraction (1~5초 sliding window 권장)",
        "Rule-based HAR: 정적 활동(standing/sitting)은 acc_mag ≈ 9.81, gyro_mag ≈ 0 활용",
        "동적 활동(walking/running)은 acc_mag 분산이 크고 gyro_mag 변동이 큼",
    ]
    next_html = "\n".join(f"<li>{s}</li>" for s in next_steps)

    # ── label → 파일 목록 매핑 (JS용 JSON) ──
    import json as _json
    label_files_map: Dict[str, List[str]] = {}
    for p in all_parsed:
        if p.dataframe is not None and not p.dataframe.empty:
            lbl = p.header_metadata.get("label", "unknown")
            base = os.path.splitext(p.filename)[0]
            label_files_map.setdefault(lbl, []).append(base)
    for k in label_files_map:
        label_files_map[k].sort()
    label_files_json = _json.dumps(label_files_map, ensure_ascii=False)

    graph_types_json = _json.dumps([
        {"suffix": "acc_xyz",  "title": "Accelerometer X / Y / Z"},
        {"suffix": "gyro_xyz", "title": "Gyroscope X / Y / Z"},
        {"suffix": "acc_mag",  "title": "Acceleration Magnitude"},
        {"suffix": "acc_mag_window", "title": "Acc-Mag Window Mean & Std", "strideDependent": True},
        {"suffix": "gyro_mag", "title": "Gyroscope Magnitude"},
        {"suffix": "gyro_mag_window", "title": "Gyro-Mag Window Mean & Std", "strideDependent": True},
        {"suffix": "gravity_xyz", "title": "Gravity Estimation (LPF)"},
        {"suffix": "dynamic_xyz", "title": "Dynamic Acceleration X / Y / Z"},
        {"suffix": "dyn_mag",  "title": "Dynamic Accel Magnitude"},
        {"suffix": "a_v_a_h",  "title": "Vertical / Horizontal Accel"},
        {"suffix": "w_v_w_h",  "title": "Vertical / Horizontal Gyro"},
        {"suffix": "window_features", "title": "Window Feature Trends", "strideDependent": True},
        {"suffix": "ewh_trend", "title": "Horizontal Gyro Energy (E_w_h) Trend", "strideDependent": True},
        {"suffix": "zero_crossing", "title": "Zero-Crossing Count Trend", "strideDependent": True},
        {"suffix": "gyro_z_rotation_profile", "title": "Gyro-Z Rotation Profile"},
        {"suffix": "window_feature_trends_gyroz", "title": "Gyro-Z Rotation Features (Window)", "strideDependent": True},
    ])

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>HAR Sensor Data Analysis Report</title>
<style>
    :root {{ --accent: #1a73e8; --bg: #f8f9fa; --card: #fff; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1400px; margin: 0 auto; padding: 20px; background: var(--bg); color: #333; }}
    h1 {{ color: var(--accent); border-bottom: 3px solid var(--accent); padding-bottom: 10px; }}
    h2 {{ color: #333; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
    h3 {{ color: #555; }}
    .summary-box {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
    .stat-card {{ background: var(--card); border-radius: 8px; padding: 20px; flex: 1; min-width: 180px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.12); text-align: center; }}
    .stat-card .number {{ font-size: 2em; font-weight: bold; color: var(--accent); }}
    .stat-card .label {{ color: #666; margin-top: 5px; }}
    .table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    .table th {{ background: #e8f0fe; font-weight: 600; position: sticky; top: 0; }}
    .table tr:nth-child(even) {{ background: #f8f9fa; }}
    .table-wrapper {{ overflow-x: auto; max-height: 500px; overflow-y: auto;
                      border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
    code {{ background: #e8f0fe; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
    .obs-list li {{ margin: 5px 0; }}
    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;
               color: #888; font-size: 0.85em; text-align: center; }}

    /* ── 비교 UI ── */
    .compare-controls {{ background: var(--card); border-radius: 8px; padding: 20px;
                         box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin: 20px 0;
                         display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .compare-side {{ display: flex; flex-direction: column; gap: 8px; }}
    .compare-side .side-label {{ font-weight: 700; font-size: 1.1em; }}
    .compare-side .side-label.left {{ color: #d93025; }}
    .compare-side .side-label.right {{ color: #1a73e8; }}
    .compare-side select {{ padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc;
                            font-size: 0.95em; background: white; }}
    .compare-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
    .compare-cell {{ background: var(--card); border-radius: 8px; padding: 12px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.12); text-align: center; }}
    .compare-cell img {{ width: 100%; border-radius: 4px; }}
    .compare-cell .graph-title {{ font-weight: 600; margin-bottom: 8px; font-size: 0.95em; }}
    .compare-cell .graph-title.left {{ color: #d93025; }}
    .compare-cell .graph-title.right {{ color: #1a73e8; }}
    .compare-cell .no-graph {{ color: #999; padding: 40px 0; font-style: italic; }}
    .graph-type-header {{ font-size: 1.05em; font-weight: 700; color: #444;
                          margin: 18px 0 8px 0; padding: 6px 12px;
                          background: #e8f0fe; border-radius: 6px; display: inline-block; }}

    /* 비교 그래프 (전체) */
    .comp-graph-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
                        gap: 15px; margin: 15px 0; }}
    .comp-graph-grid img {{ width: 100%; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
</style>
</head>
<body>

<h1>HAR Sensor Data Analysis Report</h1>
<p>생성일: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<!-- ═══════════════ 1. 개요 ═══════════════ -->
<h2>1. 전체 데이터셋 개요</h2>
<div class="summary-box">
  <div class="stat-card"><div class="number">{total_files}</div><div class="label">전체 파일 수</div></div>
  <div class="stat-card"><div class="number">{valid_files}</div><div class="label">유효 파일 수</div></div>
  <div class="stat-card"><div class="number">{error_files}</div><div class="label">오류 파일 수</div></div>
  <div class="stat-card"><div class="number">{total_samples:,}</div><div class="label">전체 샘플 수</div></div>
  <div class="stat-card"><div class="number">{len(labels)}</div><div class="label">고유 Label 수</div></div>
</div>
<h3>Label 분포</h3>
<ul>
{"".join(f'<li><code>{l}</code></li>' for l in labels) if labels else '<li>없음</li>'}
</ul>

<!-- ═══════════════ 2. 파일별 요약 ═══════════════ -->
<h2>2. 파일별 요약 테이블</h2>
<div class="table-wrapper">
{df_to_html(file_summaries[["filename", "meta_label", "body_labels", "phone_hand",
    "screen_direction", "num_samples", "elapsed_duration_ms",
    "footer_total_recording_time_ms", "footer_num_samples_after_trim",
    "sample_density_hz", "has_missing", "time_monotonic", "near_constant",
    "label_mismatch", "issues"]] if not file_summaries.empty else pd.DataFrame())}
</div>
<h3>센서 통계 (파일별)</h3>
<div class="table-wrapper">
{df_to_html(file_summaries[["filename", "meta_label"] + [c for c in file_summaries.columns
    if any(c.startswith(p) for p in ["acc_x_","acc_y_","acc_z_","acc_mag_",
    "gyro_x_","gyro_y_","gyro_z_","gyro_mag_"])]] if not file_summaries.empty else pd.DataFrame())}
</div>

<!-- ═══════════════ 3. Label별 요약 ═══════════════ -->
<h2>3. Label별 요약 테이블</h2>
<div class="table-wrapper">
{df_to_html(label_summary)}
</div>

<!-- ═══════════════ 4. 라벨 비교 (인터랙티브) ═══════════════ -->
<h2>4. 라벨 비교 분석</h2>
<p style="color:#555;">아래에서 <strong>Window</strong>와 <strong>Stride</strong>를 선택한 뒤, 비교할 <strong>두 라벨</strong>과 각 라벨의 <strong>대표 파일</strong>을 선택하면,
그래프를 좌/우 배치로 비교할 수 있습니다. Window/Stride 변경 시 윈도우 기반 그래프만 갱신됩니다.</p>

<div style="margin:12px 0; display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
  <div>
    <label for="windowSelect" style="font-weight:700; font-size:1.05em; margin-right:10px;">Window (sec):</label>
    <select id="windowSelect" onchange="renderComparison()" style="padding:8px 16px; border-radius:6px; border:1px solid #ccc; font-size:1em; background:white;">
      <option value="1.0">1.0 s</option>
      <option value="1.5">1.5 s</option>
      <option value="2.0" selected>2.0 s</option>
      <option value="3.0">3.0 s</option>
    </select>
  </div>
  <div>
    <label for="strideSelect" style="font-weight:700; font-size:1.05em; margin-right:10px;">Stride (sec):</label>
    <select id="strideSelect" onchange="renderComparison()" style="padding:8px 16px; border-radius:6px; border:1px solid #ccc; font-size:1em; background:white;">
      <option value="0.5" selected>0.5 s</option>
      <option value="1.0">1.0 s</option>
      <option value="1.5">1.5 s</option>
      <option value="2.0">2.0 s</option>
    </select>
  </div>
</div>

<div class="compare-controls">
  <div class="compare-side">
    <span class="side-label left">◀ Left (A)</span>
    <select id="labelA" onchange="onLabelChange('A')"><option value="">— 라벨 선택 —</option></select>
    <select id="fileA" onchange="renderComparison()"><option value="">— 파일 선택 —</option></select>
  </div>
  <div class="compare-side">
    <span class="side-label right">Right (B) ▶</span>
    <select id="labelB" onchange="onLabelChange('B')"><option value="">— 라벨 선택 —</option></select>
    <select id="fileB" onchange="renderComparison()"><option value="">— 파일 선택 —</option></select>
  </div>
</div>

<div id="comparison-area"></div>

<!-- ═══════════════ 5. 전체 비교 그래프 ═══════════════ -->
<h2>5. 전체 비교 그래프</h2>
<div class="comp-graph-grid">
{"".join(f'<img src="comparative/{g}" alt="{g}">' for g in comp_graphs)}
</div>

<!-- ═══════════════ 6. 관찰 포인트 ═══════════════ -->
<h2>6. 관찰 포인트 / 데이터 품질</h2>
<ul class="obs-list">
{obs_html}
</ul>

<!-- ═══════════════ 7. Feature Analysis ═══════════════ -->
<h2>7. Feature Analysis 연결 포인트</h2>
<ul>
{next_html}
</ul>

<div class="footer">
  HAR Sensor Data Analysis — auto-generated by analyze_sensor_data.py
</div>

<!-- ═══════════════ JavaScript ═══════════════ -->
<script>
const LABEL_FILES = {label_files_json};
const GRAPH_TYPES = {graph_types_json};
const LABELS = Object.keys(LABEL_FILES).sort();

// 초기화: 드롭다운 채우기
function initSelectors() {{
    ['A','B'].forEach(side => {{
        const sel = document.getElementById('label' + side);
        LABELS.forEach(l => {{
            const opt = document.createElement('option');
            opt.value = l; opt.textContent = l;
            sel.appendChild(opt);
        }});
    }});
    // 기본 선택: 첫 번째 / 두 번째 라벨
    if (LABELS.length >= 1) {{ document.getElementById('labelA').value = LABELS[0]; onLabelChange('A'); }}
    if (LABELS.length >= 2) {{ document.getElementById('labelB').value = LABELS[1]; onLabelChange('B'); }}
}}

function onLabelChange(side) {{
    const label = document.getElementById('label' + side).value;
    const fileSel = document.getElementById('file' + side);
    fileSel.innerHTML = '<option value="">— 파일 선택 —</option>';
    if (label && LABEL_FILES[label]) {{
        LABEL_FILES[label].forEach(f => {{
            const opt = document.createElement('option');
            opt.value = f; opt.textContent = f;
            fileSel.appendChild(opt);
        }});
        fileSel.value = LABEL_FILES[label][0];  // 첫 파일 자동 선택
    }}
    renderComparison();
}}

function renderComparison() {{
    const fileA = document.getElementById('fileA').value;
    const fileB = document.getElementById('fileB').value;
    const labelA = document.getElementById('labelA').value;
    const labelB = document.getElementById('labelB').value;
    const winSize = document.getElementById('windowSelect').value;
    const stride = document.getElementById('strideSelect').value;
    const area = document.getElementById('comparison-area');

    if (!fileA && !fileB) {{
        area.innerHTML = '<p style="color:#999; text-align:center; padding:30px;">라벨과 파일을 선택하세요.</p>';
        return;
    }}

    let html = '';
    GRAPH_TYPES.forEach(gt => {{
        // strideDependent 그래프는 suffix에 _w{{window}}_s{{stride}} 추가
        const suffix = gt.strideDependent ? gt.suffix + '_w' + winSize + '_s' + stride : gt.suffix;
        const titleExtra = gt.strideDependent ? ' [win=' + winSize + 's, stride=' + stride + 's]' : '';

        html += '<div class="graph-type-header">' + gt.title + titleExtra + '</div>';
        html += '<div class="compare-row">';

        // Left (A)
        html += '<div class="compare-cell">';
        if (fileA) {{
            html += '<div class="graph-title left">[A] ' + labelA + ' \u2014 ' + gt.title + '</div>';
            html += '<img src="per_file/' + fileA + '_' + suffix + '.png" onerror="handleImgError(this)">';
        }} else {{
            html += '<div class="no-graph">Left \ud30c\uc77c\uc744 \uc120\ud0dd\ud558\uc138\uc694</div>';
        }}
        html += '</div>';

        // Right (B)
        html += '<div class="compare-cell">';
        if (fileB) {{
            html += '<div class="graph-title right">[B] ' + labelB + ' \u2014 ' + gt.title + '</div>';
            html += '<img src="per_file/' + fileB + '_' + suffix + '.png" onerror="handleImgError(this)">';
        }} else {{
            html += '<div class="no-graph">Right \ud30c\uc77c\uc744 \uc120\ud0dd\ud558\uc138\uc694</div>';
        }}
        html += '</div>';

        html += '</div>';
    }});

    area.innerHTML = html;
}}

function handleImgError(img) {{
    var div = document.createElement('div');
    div.className = 'no-graph';
    div.textContent = '\uadf8\ub798\ud504 \uc5c6\uc74c';
    img.parentNode.replaceChild(div, img);
}}

initSelectors();
</script>

</body>
</html>"""

    report_path = os.path.join(out_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report → {report_path}")


# ──────────────────────────────────────────────
# 9. 메인 파이프라인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HAR Sensor CSV Analysis")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"),
                        help="센서 CSV 파일이 들어 있는 디렉토리 경로 (기본: ../data)")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "analysis_output"),
                        help="분석 결과 출력 디렉토리 (기본: ../analysis_output)")
    parser.add_argument("--skip-graphs", action="store_true",
                        help="그래프 생성을 건너뛰고 요약/리포트만 재생성")
    parser.add_argument("--window-sec", type=float, default=2.0,
                        help="윈도우 특징 추출 윈도우 길이 (초, 기본: 2.0)")
    parser.add_argument("--stride-sec", type=float, default=0.5,
                        help="윈도우 특징 추출 stride (초, 기본: 0.5)")
    parser.add_argument("--lpf-cutoff", type=float, default=0.3,
                        help="중력 추정 LPF 차단 주파수 (Hz, 기본: 0.3)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    graphs_dir = os.path.join(out_dir, "per_file")
    comp_dir = os.path.join(out_dir, "comparative")

    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"Data directory : {data_dir}")
    logger.info(f"Output directory: {out_dir}")

    # 출력 디렉토리 생성
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    # ── Step 1: CSV 파일 탐색 ──
    csv_files = find_csv_files(data_dir)
    logger.info(f"발견된 CSV 파일 수: {len(csv_files)}")
    if not csv_files:
        logger.warning("CSV 파일이 없습니다. data 폴더를 확인하세요.")
        return

    # ── Step 2: 파싱 + magnitude + 파생 신호 + 윈도우 특징 ──
    all_parsed: List[ParsedCSV] = []
    for fp in csv_files:
        logger.info(f"파싱 중: {fp}")
        parsed = parse_sensor_csv(fp)
        if parsed.parse_error:
            logger.warning(f"  ⚠ {parsed.filename}: {parsed.parse_error}")
        else:
            parsed.dataframe = add_magnitude_columns(parsed.dataframe)
            parsed.dataframe = add_derived_signal_columns(parsed.dataframe, lpf_cutoff=args.lpf_cutoff)
            # 윈도우 특징 (ParsedCSV에 임시 속성으로 첨부) — 여러 window × stride 조합
            _WINDOWS = [1.0, 1.5, 2.0, 3.0]
            _STRIDES = [0.5, 1.0, 1.5, 2.0]
            parsed._wf_grid = {}  # (window_sec, stride_sec) → DataFrame
            for _w in _WINDOWS:
                for _s in _STRIDES:
                    parsed._wf_grid[(_w, _s)] = compute_window_features(
                        parsed.dataframe, window_sec=_w, stride_sec=_s)
            # 하위 호환용 aliases
            parsed._window_features_map = {s: parsed._wf_grid[(2.0, s)] for s in _STRIDES}
            parsed._window_features = parsed._wf_grid[(2.0, 0.5)]
            _sample_counts = ', '.join(
                f'w{_w}s{_s}={len(parsed._wf_grid[(_w, _s)])}'
                for _w in _WINDOWS for _s in _STRIDES[:1]  # 로그는 대표 1개 stride만
            )
            logger.info(f"  ✓ {parsed.filename}: {len(parsed.dataframe)} samples, "
                        f"label={parsed.header_metadata.get('label', '??')}, "
                        f"wins=[{_sample_counts}] (×{len(_STRIDES)} strides)")
        all_parsed.append(parsed)

    # ── Step 3: 품질 검사 + 통계 ──
    file_summary_rows = []
    for parsed in all_parsed:
        quality = check_quality(parsed)
        stats = compute_file_stats(parsed)
        row = build_file_summary(parsed, quality, stats)
        file_summary_rows.append(row)

    file_summaries = pd.DataFrame(file_summary_rows)

    # ── Step 4: label별 요약 ──
    label_summary = build_label_summary(all_parsed)

    # ── Step 5: 파일별 그래프 ──
    if not args.skip_graphs:
        logger.info("파일별 그래프 생성 중...")
        for parsed in all_parsed:
            if parsed.parse_error:
                continue
            try:
                plot_file_graphs(parsed, graphs_dir)
            except Exception as e:
                logger.warning(f"  그래프 생성 실패 ({parsed.filename}): {e}")

        # ── Step 6: 비교 그래프 ──
        logger.info("비교 그래프 생성 중...")
        try:
            plot_comparative_graphs(all_parsed, out_dir)
        except Exception as e:
            logger.warning(f"  비교 그래프 생성 실패: {e}")
    else:
        logger.info("--skip-graphs: 그래프 생성 건너뜀")

    # ── Step 7: 요약 CSV 저장 ──
    file_csv_path = os.path.join(out_dir, "file_summary.csv")
    file_summaries.to_csv(file_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"파일별 요약 → {file_csv_path}")

    label_csv_path = os.path.join(out_dir, "label_summary.csv")
    label_summary.to_csv(label_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Label별 요약 → {label_csv_path}")

    # ── Step 7-b: 윈도우 특징 CSV 저장 (window × stride 별) ──
    _WINDOWS = [1.0, 1.5, 2.0, 3.0]
    _STRIDES = [0.5, 1.0, 1.5, 2.0]
    for win_val in _WINDOWS:
        for stride_val in _STRIDES:
            wf_rows = []
            for p in all_parsed:
                wf_grid = getattr(p, '_wf_grid', None)
                if wf_grid and (win_val, stride_val) in wf_grid:
                    wf = wf_grid[(win_val, stride_val)]
                else:
                    wf = None
                if wf is not None and not wf.empty:
                    wfc = wf.copy()
                    wfc.insert(0, "filename", p.filename)
                    wfc.insert(1, "label", p.header_metadata.get("label", "unknown"))
                    wf_rows.append(wfc)
            if wf_rows:
                wf_all = pd.concat(wf_rows, ignore_index=True)
                wf_csv_path = os.path.join(out_dir, f"window_feature_summary_w{win_val}_s{stride_val}.csv")
                wf_all.to_csv(wf_csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"윈도우 특징 요약 (win={win_val}s, stride={stride_val}s) → {wf_csv_path}  ({len(wf_all)} windows)")

    # ── Step 8: HTML 리포트 ──
    logger.info("HTML 리포트 생성 중...")
    try:
        generate_html_report(file_summaries, label_summary, all_parsed,
                             out_dir, graphs_dir, comp_dir)
    except Exception as e:
        logger.warning(f"  HTML 리포트 생성 실패: {e}")

    # ── 완료 ──
    logger.info("=" * 60)
    logger.info("분석 완료!")
    logger.info(f"  전체 파일: {len(all_parsed)}")
    logger.info(f"  유효 파일: {sum(1 for p in all_parsed if p.dataframe is not None)}")
    logger.info(f"  결과 폴더: {out_dir}")
    logger.info(f"  HTML 리포트: {os.path.join(out_dir, 'report.html')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
