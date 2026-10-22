#!/usr/bin/env python3
"""
Sum FITS CCD frames over spatial rows 760–1150 and save as a single CSV.

The output CSV has one shared pixel column plus one counts column per file.
If a DataInfo.txt is found in the same directory as the FITS files, column
headers are labelled with the delay in ns (e.g. "0.0ns"); otherwise the
two-digit file number is used (e.g. "00").

DataInfo.txt format (one entry per line after the title line):
    Reference <file_first>-<file_last>                  -> columns Ref_1, Ref_2, ...
    <start> -> <end> ns    <file_first>-<file_last>    ...

Usage:
    python bin/fits_to_csv.py Au/experiment_raw_data/
    python bin/fits_to_csv.py Au/experiment_raw_data/*.fits
    python bin/fits_to_csv.py Au/experiment_raw_data/ --plot
    python bin/fits_to_csv.py Au/experiment_raw_data/ --outdir Au/spectra/
    python bin/fits_to_csv.py Au/experiment_raw_data/ --rows 800 1200
    python bin/fits_to_csv.py Au/experiment_raw_data/ --drop-unmapped
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

ROW_MIN = 760
ROW_MAX = 1150
REFERENCE_FILES = set(range(1, 11))  # files 01-10 are always kept as reference


def load_fits(path):
    with fits.open(path) as f:
        data = f[0].data.squeeze().astype(float)
        header = f[0].header
    return data, header


def frame_time(header):
    """Return the FRAME timestamp string, or empty string if absent."""
    return header.get("FRAME", "")


def parse_datainfo(datainfo_path):
    """
    Parse DataInfo.txt and return a dict mapping integer file number -> (delay_ns, run_index).

    When the same delay value appears in more than one run, run_index is 1-based
    (1 for the first occurrence, 2 for the second, etc.).  If a delay value is
    unique across all runs, run_index is None (no suffix needed).

    Lines matching "<start> -> <end> ns  <first>-<last>  ..." are parsed as delay
    runs, and lines matching "Reference <first>-<last>" as reference files.
    Reference files map to ("Ref", n), numbered 1, 2, ... in file order.
    Other lines are ignored.
    """
    pattern = re.compile(r"([\d.]+)\s*->\s*([\d.]+)\s*ns\s+(\d+)-(\d+)")
    ref_pattern = re.compile(r"^\s*Reference\s+(\d+)-(\d+)", re.IGNORECASE)

    # First pass: collect all (file_num, delay) pairs in order, tracking runs
    runs = []  # list of (f_start, f_end, delays array)
    ref_files = []
    with open(datainfo_path) as f:
        for line in f:
            m = ref_pattern.search(line)
            if m:
                ref_files.extend(range(int(m.group(1)), int(m.group(2)) + 1))
                continue
            m = pattern.search(line)
            if not m:
                continue
            d_start, d_end = float(m.group(1)), float(m.group(2))
            f_start, f_end = int(m.group(3)), int(m.group(4))
            delays = np.linspace(d_start, d_end, f_end - f_start + 1)
            runs.append((f_start, f_end, delays))

    # Determine which delay labels appear more than once across runs
    # (compare rounded string to avoid float noise)
    label_counts = defaultdict(int)
    for f_start, f_end, delays in runs:
        seen_in_run = set()
        for d in delays:
            key = f"{d:g}ns"
            if key not in seen_in_run:
                label_counts[key] += 1
                seen_in_run.add(key)
    duplicated = {k for k, v in label_counts.items() if v > 1}

    # Second pass: assign run indices only for duplicated labels
    label_run_counter = defaultdict(int)
    delay_map = {}  # file_num -> (delay_ns, run_index or None)
    for f_start, f_end, delays in runs:
        # Find the run index for this block (based on first label of the run)
        first_label = f"{delays[0]:g}ns"
        if first_label in duplicated:
            label_run_counter[first_label] += 1
            run_idx = label_run_counter[first_label]
        else:
            run_idx = None
        for file_num, d in zip(range(f_start, f_end + 1), delays):
            delay_map[file_num] = (d, run_idx)

    for ref_idx, file_num in enumerate(ref_files, start=1):
        delay_map[file_num] = ("Ref", ref_idx)

    return delay_map


def col_label(file_num, delay_map):
    """Return a column label: Ref_N, delay string (with run suffix if needed), else file number."""
    if file_num in delay_map:
        d, run_idx = delay_map[file_num]
        if d == "Ref":
            return f"Ref_{run_idx}"
        label = f"{d:g}ns"
        if run_idx is not None:
            label += f"_{run_idx}"
        return label
    return f"{file_num:02d}"


def collect_fits(inputs):
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.fits")))
        elif p.suffix.lower() == ".fits":
            paths.append(p)
        else:
            print(f"Skipping {inp} (not a .fits file or directory)", file=sys.stderr)
    return [Path(p) for p in paths]


def process_group(paths, row_min, row_max, outdir, plot, drop_unmapped, time_header):
    """Build and save one CSV for a group of FITS files."""
    src_dir = paths[0].parent

    # Load delay map if DataInfo.txt exists alongside the FITS files
    datainfo_path = src_dir / "DataInfo.txt"
    delay_map = parse_datainfo(datainfo_path) if datainfo_path.exists() else {}
    if delay_map:
        print(f"Using delay map from {datainfo_path}")

    spectra = {}      # ordered: col_label -> spectrum array
    timestamps = {}   # col_label -> FRAME timestamp string
    pixels = None

    for path in paths:
        file_num = int(path.stem.split("_")[-1])
        if drop_unmapped and delay_map and file_num not in delay_map and file_num not in REFERENCE_FILES:
            continue
        data, header = load_fits(path)
        spectrum = data[row_min:row_max, :].sum(axis=0)
        if pixels is None:
            pixels = np.arange(spectrum.size)
        label = col_label(file_num, delay_map)
        spectra[label] = spectrum
        timestamps[label] = frame_time(header)

    # Output path
    dest_dir = Path(outdir) if outdir else src_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    first_stem = paths[0].stem          # e.g. "2023_07_19_00"
    csv_stem = "_".join(first_stem.split("_")[:-1])  # "2023_07_19"
    out_path = dest_dir / (csv_stem + ".csv")

    # Write CSV
    col_names = list(spectra.keys())
    comment = f"# rows {row_min}-{row_max} summed"
    if delay_map:
        comment += "; columns are delay in ns from DataInfo.txt (unmapped files use file number)"
    comment += "\n"

    rows = np.column_stack([pixels] + [spectra[c] for c in col_names])
    fmt = ["%d"] + ["%.6g"] * len(col_names)

    with open(out_path, "w") as f:
        f.write(comment)
        if time_header:
            time_line = "pixel," + ",".join(timestamps.get(c, "") for c in col_names)
            f.write(time_line + "\n")
        delay_line = "pixel," + ",".join(col_names)
        f.write(delay_line + "\n")
        np.savetxt(f, rows, delimiter=",", fmt=fmt)

    print(f"Saved: {out_path}  ({len(col_names)} spectra)")

    if plot:
        fig, ax = plt.subplots(figsize=(11, 5))
        for col, spectrum in spectra.items():
            ax.plot(pixels, spectrum, lw=0.7, label=col)
        ax.set_xlabel("Pixel (dispersion axis)")
        ax.set_ylabel("Summed counts")
        ax.set_title(f"{csv_stem}  |  rows {row_min}–{row_max}")
        ax.legend(fontsize=6, ncol=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Sum FITS frames over spatial rows and save as a combined CSV."
    )
    parser.add_argument("inputs", nargs="+",
                        help="One or more .fits files, or a directory of .fits files")
    parser.add_argument("--rows", nargs=2, type=int, metavar=("R0", "R1"),
                        default=[ROW_MIN, ROW_MAX],
                        help=f"Spatial row range to sum (default: {ROW_MIN} {ROW_MAX})")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for CSV (default: same as input)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot all spectra overlaid after saving")
    parser.add_argument("--drop-unmapped", action="store_true",
                        help="Exclude files not in DataInfo.txt, except files 01-10 (reference)")
    parser.add_argument("--time-header", action="store_true",
                        help="Add a second header row with the FRAME timestamp from each FITS file")
    args = parser.parse_args()

    paths = collect_fits(args.inputs)
    if not paths:
        sys.exit("No .fits files found.")

    row_min, row_max = sorted(args.rows)

    # Group files by parent directory so each directory gets its own CSV
    groups = defaultdict(list)
    for p in paths:
        groups[p.parent].append(p)

    for group_paths in groups.values():
        process_group(group_paths, row_min, row_max, args.outdir, args.plot,
                      args.drop_unmapped, args.time_header)


if __name__ == "__main__":
    main()
