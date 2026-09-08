import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# Input CSI recordings and output directory
# Download wifi_presence_detection_dsk.zip from:
#   https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wifi_presence_detection_dsi.zip
# Then extract the zip file and pass the path to the directory as IN_ROOT below
IN_ROOT  = Path(r"/path/to/wifi_presence_detection_dsi/")
OUT_ROOT = Path("preprocessed_wifi_presence_detection")
WIN_SEC     = 2.0  

LABEL_TO_FOLDER = {0: "class_0_no_presence", 1: "class_1_presence"}
LABEL_TO_PREFIX = {0: "no_presence", 1: "presence"}

# Folder name -> label for wifi_dataset_mini_final structure (classes/no_presence, classes/presence)
FOLDER_TO_LABEL = {"no_presence": 0, "presence": 1}

# Signal parameters
Fs              = 128.0
SESSION_GAP_SEC = 0.5   # gap threshold to detect session boundaries
DROP_COLS   = {"tx_mac", "rx_mac", "packet_no", "hw_seq"}

# 52 usable subcarriers — hardware index 26 is the DC subcarrier, so we skip it
CSI_USED_IDX = np.array(list(range(0, 26)) + list(range(27, 53)), dtype=int)
N_SC         = len(CSI_USED_IDX)  # 52

def find_sessions(t):
    """Return (start, end) index pairs for each continuous segment in t (seconds)."""
    dt     = np.diff(t)
    breaks = np.where((dt < 0) | (dt > SESSION_GAP_SEC))[0] + 1
    starts = np.concatenate([[0], breaks])
    ends   = np.concatenate([breaks, [len(t)]])
    return list(zip(starts.tolist(), ends.tolist()))


def interpolate_to_grid(tw, Xw_cplx, fs, win_sec):
    """Resample |CSI| magnitude onto a uniform time grid of (win_sec * fs) points."""
    if len(tw) < 2:
        return None

    order = np.argsort(tw)
    tw, Xw_cplx = tw[order], Xw_cplx[order]

    Xm = np.abs(Xw_cplx).astype(np.float64)
    T, S = Xm.shape

    # fill any NaN gaps with linear interpolation before resampling
    for s in range(S):
        bad = np.isnan(Xm[:, s])
        if bad.any():
            idx  = np.arange(T)
            good = ~bad
            Xm[:, s] = np.interp(idx, idx[good], Xm[good, s]) if good.any() else 0.0

    N  = int(round(win_sec * fs))
    tq = tw[0] + np.arange(N) / fs
    Xi = np.empty((N, S), dtype=np.float64)
    for s in range(S):
        Xi[:, s] = np.interp(tq, tw, Xm[:, s])

    return Xi.astype(np.float32)

def extract_csi(df):
    """Pull complex CSI columns and select the 52 usable subcarriers."""
    sub_cols = [c for c in df.columns if c.startswith("sub_")]
    if not sub_cols:
        return None

    idx_map = {}
    for c in sub_cols:
        try:
            idx_map[int(c.split("_")[1])] = c
        except ValueError:
            pass

    used = [idx_map[i] for i in CSI_USED_IDX if i in idx_map]
    if len(used) != N_SC:
        return None

    return df[used].replace("i", "j", regex=True).astype(complex).to_numpy()


def process_file(csv_path):
    """Drop metadata cols, extract 52-subcarrier CSI magnitudes, resample to uniform grid."""
    df = pd.read_csv(csv_path, low_memory=False)
    # if "timestamp" not in df.columns:
    #     return []
    ts_col = next((c for c in ("timestamp", "mcu_timestamp") if c in df.columns), None)
    if ts_col is None:
        return None

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore", inplace=True)

    t_raw = df[ts_col].to_numpy(dtype=float)
    # timestamps are microseconds when > 1e5, otherwise already in seconds
    t = (t_raw - t_raw[0]) / 1e6 if np.nanmax(t_raw) > 1e5 else t_raw - t_raw[0]

    Xc = extract_csi(df)
    if Xc is None:
        return None

    segments = []
    for s, e in find_sessions(t):
        if e - s < 2:
            continue
        Xi = interpolate_to_grid(t[s:e], Xc[s:e], Fs, WIN_SEC)
        if Xi is not None:
            segments.append(Xi)
    return np.concatenate(segments, axis=0) if segments else None


def main():
    classes_root     = OUT_ROOT / "classes"
    classes_root.mkdir(parents=True, exist_ok=True)

    if not IN_ROOT.exists():
        print(f"ERROR: Input root not found: {IN_ROOT}")
        return

    csv_files = []
    for class_dir in sorted((IN_ROOT / "classes").iterdir()):
        if not class_dir.is_dir() or class_dir.name not in FOLDER_TO_LABEL:
            continue
        for p in sorted(class_dir.glob("*.csv")):
            csv_files.append((p, class_dir.name))
    print(f"Found {len(csv_files)} CSV files\n")

    n_files = 0

    for csv_path, class_name in csv_files:
        label  = FOLDER_TO_LABEL[class_name]
        folder = classes_root / LABEL_TO_FOLDER[label]
        folder.mkdir(exist_ok=True)

        Xi = process_file(csv_path)
        if Xi is None:
            print(f"  [{class_name}] {csv_path.name}: skipped")
            continue

        fname = f"{csv_path.stem}.csv"
        pd.DataFrame(Xi, columns=[f"sub_{i}" for i in CSI_USED_IDX]).to_csv(folder / fname, index=False)
        n_files += 1
        if n_files % 10 == 0:
            print(f"  {n_files} files processed...")

    print(f"\nDone: {n_files} files -> {OUT_ROOT}")


if __name__ == "__main__":
    main()