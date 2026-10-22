#!/usr/bin/env python3
"""
Interactive GUI for exploring a single absorption spectrum at a time.

Controls:
  - Set (1 or 2): selects which run to display
  - Time delay: dropdown of available delays for the selected set
  - Reference: "average (Ref_*)" or any individual Ref_* column
  - Shift target: toggle whether the shift is applied to the reference or the signal
  - Pixel shift: ±N pixels applied to the selected target before dividing

Usage:
    python bin/absorption_single_gui.py Au/experiment_raw_data/2023_07_19.csv
    python bin/absorption_single_gui.py Au/experiment_raw_data/2023_07_19.csv --shift-signal
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # share Tk's event loop; the macOS backend can segfault alongside tkinter
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REF_AVERAGE = "average (Ref_*)"


def is_reference_col(name):
    return bool(re.fullmatch(r"Ref_\d+", name))


def is_delay_col(name):
    return bool(re.fullmatch(r"[\d.]+ns(_\d+)?", name))


def delay_set(col_name):
    m = re.search(r"_(\d+)$", col_name)
    return int(m.group(1)) if m else None


def load_csv(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]

    def _is_header(line):
        return any(c.isalpha() for c in line)

    if len(lines) >= 2 and _is_header(lines[1]):
        timestamps = [c.strip() for c in lines[0].strip().split(",")[1:]]
        col_names  = [c.strip() for c in lines[1].strip().split(",")[1:]]
        data = np.loadtxt(lines[2:], delimiter=",")
    else:
        timestamps = None
        col_names  = [c.strip() for c in lines[0].strip().split(",")[1:]]
        data = np.loadtxt(lines[1:], delimiter=",")

    pixels  = data[:, 0].astype(int)
    spectra = data[:, 1:]
    return pixels, col_names, spectra, timestamps


def _edge_taper(n, width=50):
    window = np.ones(n)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(width) / width))
    window[:width] = ramp
    window[n - width:] = ramp[::-1]
    return window


def fft_shift(arr, shift, taper_width=50):
    if shift == 0:
        return arr.copy()
    n = len(arr)
    taper = _edge_taper(n, width=min(taper_width, n // 4))
    freqs = np.fft.rfftfreq(n)
    phase = np.exp(-2j * np.pi * shift * freqs)
    return np.fft.irfft(np.fft.rfft(arr * taper) * phase, n=n)


EDGE_TRIM = 50

def compute_absorption(signal, reference, shift, shift_signal=False):
    if shift_signal:
        sig = fft_shift(signal, shift)
        t = sig / np.where(reference == 0, np.nan, reference)
    else:
        ref = fft_shift(reference, shift)
        t = signal / np.where(ref == 0, np.nan, ref)
    result = -np.log(np.where(t > 0, t, np.nan))
    return result[EDGE_TRIM:-EDGE_TRIM]


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class AbsorptionSingleGUI:
    def __init__(self, csv_path, shift_signal=False):
        self.src = Path(csv_path)
        self._shift_signal_default = shift_signal
        self.pixels, self.col_names, self.spectra, self.timestamps = load_csv(self.src)

        self.ref_cols   = [c for c in self.col_names if is_reference_col(c)]
        self.delay_cols = [c for c in self.col_names if is_delay_col(c)]
        self.sets = sorted({delay_set(c) for c in self.delay_cols})

        self._current_delay_cols = []
        self._build_tk()
        self._build_mpl()
        self._update_delay_list()
        self.draw()
        tk.mainloop()

    # -----------------------------------------------------------------------
    # Build tkinter control panel
    # -----------------------------------------------------------------------
    def _build_tk(self):
        self.root = tk.Tk()
        self.root.title("Absorption Single GUI")
        self.root.resizable(False, False)

        pad = {"padx": 8, "pady": 4}
        row = 0

        # --- Set selector ---
        tk.Label(self.root, text="Set:").grid(row=row, column=0, sticky="w", **pad)
        set_labels = [str(s) if s is not None else "1" for s in self.sets]
        self.set_var = tk.StringVar(value=set_labels[0])
        set_menu = ttk.Combobox(self.root, textvariable=self.set_var,
                                values=set_labels, state="readonly", width=6)
        set_menu.grid(row=row, column=1, sticky="w", **pad)
        self.set_var.trace_add("write", lambda *_: self._on_set_change())
        row += 1

        # --- Delay selector ---
        tk.Label(self.root, text="Time delay:").grid(row=row, column=0, sticky="w", **pad)
        self.delay_var = tk.StringVar()
        self.delay_menu = ttk.Combobox(self.root, textvariable=self.delay_var,
                                       state="readonly", width=12)
        self.delay_menu.grid(row=row, column=1, sticky="w", **pad)
        self.delay_var.trace_add("write", lambda *_: self.draw())
        row += 1

        # --- Reference selector ---
        tk.Label(self.root, text="Reference:").grid(row=row, column=0, sticky="w", **pad)
        ref_options = [REF_AVERAGE] + self.ref_cols
        self.ref_var = tk.StringVar(value=ref_options[0])
        self.ref_menu = ttk.Combobox(self.root, textvariable=self.ref_var,
                                values=ref_options, state="readonly", width=16)
        self.ref_menu.grid(row=row, column=1, sticky="w", **pad)
        self.ref_var.trace_add("write", lambda *_: self.draw())
        row += 1

        # --- Pixel shift ---
        tk.Label(self.root, text="Pixel shift:").grid(row=row, column=0, sticky="w", **pad)
        self.shift_var = tk.DoubleVar(value=0.0)
        shift_frame = tk.Frame(self.root)
        shift_frame.grid(row=row, column=1, sticky="w", **pad)
        tk.Button(shift_frame, text="−1",   width=3,
                  command=lambda: self._nudge_shift(-1)).pack(side="left")
        tk.Button(shift_frame, text="−0.1", width=4,
                  command=lambda: self._nudge_shift(-0.1)).pack(side="left")
        self.shift_label = tk.Label(shift_frame, text="0.0", width=6)
        self.shift_label.pack(side="left")
        tk.Button(shift_frame, text="+0.1", width=4,
                  command=lambda: self._nudge_shift(+0.1)).pack(side="left")
        tk.Button(shift_frame, text="+1",   width=3,
                  command=lambda: self._nudge_shift(+1)).pack(side="left")
        tk.Button(shift_frame, text="Reset", width=5,
                  command=lambda: self._set_shift(0.0)).pack(side="left", padx=(6, 0))
        row += 1

        tk.Label(self.root, text="  (or type):").grid(row=row, column=0, sticky="e", **pad)
        self.shift_entry = tk.Entry(self.root, width=8)
        self.shift_entry.insert(0, "0.0")
        self.shift_entry.grid(row=row, column=1, sticky="w", **pad)
        self.shift_entry.bind("<Return>", lambda _: self._apply_entry_shift())
        row += 1

        # --- Shift target toggle ---
        tk.Label(self.root, text="Shift target:").grid(row=row, column=0, sticky="w", **pad)
        self.shift_signal_var = tk.BooleanVar(value=self._shift_signal_default)
        shift_frame2 = tk.Frame(self.root)
        shift_frame2.grid(row=row, column=1, sticky="w", **pad)
        tk.Radiobutton(shift_frame2, text="Reference", variable=self.shift_signal_var,
                       value=False, command=self.draw).pack(side="left")
        tk.Radiobutton(shift_frame2, text="Signal",    variable=self.shift_signal_var,
                       value=True,  command=self.draw).pack(side="left")
        row += 1

        # --- Open file button ---
        tk.Button(self.root, text="Open CSV…", command=self._open_file).grid(
            row=row, column=0, columnspan=2, pady=10)

    def _nudge_shift(self, delta):
        self._set_shift(round(self.shift_var.get() + delta, 10))

    def _set_shift(self, value):
        self.shift_var.set(value)
        self.shift_label.config(text=f"{value:g}")
        self.shift_entry.delete(0, tk.END)
        self.shift_entry.insert(0, f"{value:g}")
        self.draw()

    def _apply_entry_shift(self):
        try:
            self._set_shift(float(self.shift_entry.get()))
        except ValueError:
            pass

    def _on_set_change(self):
        self._update_delay_list()
        self.draw()

    def _update_delay_list(self):
        selected_set_str = self.set_var.get()
        selected_suffix = None if selected_set_str == "1" and None in self.sets \
            else int(selected_set_str)
        delays_for_set = [c for c in self.delay_cols if delay_set(c) == selected_suffix]
        self._current_delay_cols = delays_for_set
        display = [re.sub(r"_\d+$", "", c) for c in delays_for_set]
        self.delay_menu["values"] = display
        if display:
            self.delay_var.set(display[0])

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open spectra CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.src = Path(path)
            self.pixels, self.col_names, self.spectra, self.timestamps = load_csv(self.src)
            self.ref_cols   = [c for c in self.col_names if is_reference_col(c)]
            self.ref_menu["values"] = [REF_AVERAGE] + self.ref_cols
            if self.ref_var.get() not in self.ref_menu["values"]:
                self.ref_var.set(REF_AVERAGE)
            self.delay_cols = [c for c in self.col_names if is_delay_col(c)]
            self.sets = sorted({delay_set(c) for c in self.delay_cols})
            set_labels = [str(s) if s is not None else "1" for s in self.sets]
            self.set_var.set(set_labels[0])
            self._update_delay_list()
            self.draw()

    # -----------------------------------------------------------------------
    # Build matplotlib figure
    # -----------------------------------------------------------------------
    def _build_mpl(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.subplots_adjust(bottom=0.12)
        self.line, = self.ax.plot([], [], lw=1.5, color="steelblue")
        self.ax.axvline(1315, color="black", lw=0.8, ls="--", alpha=0.6)
        self.ax.set_xlabel("Pixel (dispersion axis)")
        self.ax.set_ylabel("Absorbance")
        self.ax.grid(True, alpha=0.3)
        plt.ion()
        plt.show(block=False)

    # -----------------------------------------------------------------------
    # Compute and redraw
    # -----------------------------------------------------------------------
    def draw(self):
        delay_display = self.delay_var.get()
        if not delay_display or not self._current_delay_cols:
            return

        idx = self.delay_menu["values"].index(delay_display) \
              if delay_display in self.delay_menu["values"] else 0
        delay_col = self._current_delay_cols[idx]
        signal = self.spectra[:, self.col_names.index(delay_col)]

        ref_choice = self.ref_var.get()
        if ref_choice == REF_AVERAGE:
            ref_indices = [i for i, c in enumerate(self.col_names) if is_reference_col(c)]
            reference = self.spectra[:, ref_indices].mean(axis=1)
            ref_desc = "mean(Ref_*)"
        else:
            ref_idx = self.col_names.index(ref_choice)
            reference = self.spectra[:, ref_idx]
            ref_desc = ref_choice

        shift = self.shift_var.get()
        shift_signal = self.shift_signal_var.get()
        absorption = compute_absorption(signal, reference, shift, shift_signal)
        pixels_trimmed = self.pixels[EDGE_TRIM:-EDGE_TRIM]

        self.line.set_data(pixels_trimmed, absorption)
        self.ax.relim()
        self.ax.autoscale_view()

        # Build title including timestamp if available
        col_idx = self.col_names.index(delay_col)
        if self.timestamps is not None:
            ts = self.timestamps[col_idx]
            ts_short = ts.split("T")[-1] if "T" in ts else ts
            delay_str = f"{delay_display} ({ts_short})"
        else:
            delay_str = delay_display

        set_str = self.set_var.get()
        shift_str = ""
        if shift != 0:
            target = "signal" if shift_signal else "ref"
            shift_str = f"  {target} shift={shift:+g}px"
        self.ax.set_title(
            f"{self.src.stem}  |  set {set_str}  |  {delay_str}"
            f"  |  ref: {ref_desc}{shift_str}"
        )
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive GUI for exploring a single absorption spectrum."
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="Combined spectra CSV from fits_to_csv.py")
    parser.add_argument("--shift-signal", action="store_true",
                        help="Start with the shift applied to the signal instead of the reference")
    args = parser.parse_args()

    if args.input is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Open spectra CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        if not path:
            sys.exit("No file selected.")
        args.input = path

    if not Path(args.input).exists():
        sys.exit(f"File not found: {args.input}")

    AbsorptionSingleGUI(args.input, shift_signal=args.shift_signal)


if __name__ == "__main__":
    main()
