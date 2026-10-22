#!/usr/bin/env python3
"""
Interactive GUI for exploring absorption spectra with a per-spectrum pixel shift.

Controls:
  - Set (1 or 2): selects which run to display
  - Time delays: checkboxes, each with its own pixel shift entry
  - Reference: "average (Ref_*)" or any individual Ref_* column
  - Shift target: toggle whether shifts are applied to the reference or signal

Usage:
    python bin/absorption_shift_gui.py Au/experiment_raw_data/2023_07_19.csv
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
# Shared helpers
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
        col_names = [c.strip() for c in lines[1].strip().split(",")[1:]]
        data = np.loadtxt(lines[2:], delimiter=",")
    else:
        timestamps = None
        col_names = [c.strip() for c in lines[0].strip().split(",")[1:]]
        data = np.loadtxt(lines[1:], delimiter=",")

    pixels = data[:, 0].astype(int)
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

class AbsorptionShiftGUI:
    def __init__(self, csv_path, shift_signal=False):
        self.src = Path(csv_path)
        self._shift_signal_default = shift_signal
        self.pixels, self.col_names, self.spectra, self.timestamps = load_csv(self.src)

        self.ref_cols = [c for c in self.col_names if is_reference_col(c)]
        self.delay_cols = [c for c in self.col_names if is_delay_col(c)]
        self.sets = sorted({delay_set(c) for c in self.delay_cols})

        self._current_delay_cols = []
        self._delay_vars = {}
        self._delay_shift_entries = {}
        self._delay_checkboxes = []
        self._delay_shift_widgets = []

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
        self.root.title("Absorption Shift GUI")
        self.root.resizable(True, True)

        pad = {"padx": 8, "pady": 4}

        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, width=340)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)
        inner.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(
            int(-1 * (event.delta / 120)), "units"))

        screen_h = self.root.winfo_screenheight()
        canvas.config(height=int(screen_h * 0.8))

        row = 0

        # --- Set selector ---
        tk.Label(inner, text="Set:").grid(row=row, column=0, sticky="w", **pad)
        set_labels = [str(s) if s is not None else "1" for s in self.sets]
        self.set_var = tk.StringVar(value=set_labels[0])
        set_menu = ttk.Combobox(inner, textvariable=self.set_var,
                                values=set_labels, state="readonly", width=6)
        set_menu.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        self.set_var.trace_add("write", lambda *_: self._on_set_change())
        row += 1

        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        # --- Delay checkboxes + per-delay shift ---
        tk.Label(inner, text="Delay", font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", **pad)
        tk.Label(inner, text="Shift (px)", font=("", 10, "bold")).grid(
            row=row, column=1, columnspan=2, sticky="w", **pad)
        row += 1

        self._cb_frame = tk.Frame(inner)
        self._cb_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=8)
        row += 1

        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        # --- Reference selector ---
        tk.Label(inner, text="Reference:").grid(row=row, column=0, sticky="w", **pad)
        ref_options = [REF_AVERAGE] + self.ref_cols
        self.ref_var = tk.StringVar(value=ref_options[0])
        self.ref_menu = ttk.Combobox(inner, textvariable=self.ref_var,
                                values=ref_options, state="readonly", width=16)
        self.ref_menu.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        self.ref_var.trace_add("write", lambda *_: self.draw())
        row += 1

        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        # --- Shift target toggle ---
        tk.Label(inner, text="Shift target:").grid(row=row, column=0, sticky="w", **pad)
        self.shift_signal_var = tk.BooleanVar(value=self._shift_signal_default)
        shift_frame2 = tk.Frame(inner)
        shift_frame2.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        tk.Radiobutton(shift_frame2, text="Reference", variable=self.shift_signal_var,
                       value=False, command=self.draw).pack(side="left")
        tk.Radiobutton(shift_frame2, text="Signal", variable=self.shift_signal_var,
                       value=True, command=self.draw).pack(side="left")
        row += 1

        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        # --- Open / Save buttons ---
        btn_frame = tk.Frame(inner)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        tk.Button(btn_frame, text="Open CSV…", command=self._open_file).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Save shifts…", command=self._save_shifts).pack(side="left", padx=4)

        self._inner = inner

    def _build_delay_checkboxes(self, *_):
        """Rebuild per-delay checkbox + shift entry rows inside _cb_frame."""
        for w in getattr(self, "_delay_checkboxes", []):
            w.destroy()
        for w in getattr(self, "_delay_shift_widgets", []):
            w.destroy()
        self._delay_checkboxes = []
        self._delay_shift_widgets = []
        self._delay_vars = {}
        self._delay_shift_entries = {}

        for r, col in enumerate(self._current_delay_cols):
            display = re.sub(r"_\d+$", "", col)
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(self._cb_frame, text=display, variable=var,
                                command=self.draw)
            cb.grid(row=r, column=0, sticky="w", pady=1)

            entry = tk.Entry(self._cb_frame, width=7)
            entry.insert(0, "0.0")
            entry.grid(row=r, column=1, sticky="w", padx=(4, 0), pady=1)
            entry.bind("<Return>", lambda _: self.draw())

            # Nudge buttons
            btn_frame = tk.Frame(self._cb_frame)
            btn_frame.grid(row=r, column=2, sticky="w", padx=(2, 0), pady=1)
            tk.Button(btn_frame, text="−", width=2,
                      command=lambda e=entry: self._nudge_entry(e, -0.1)).pack(side="left")
            tk.Button(btn_frame, text="+", width=2,
                      command=lambda e=entry: self._nudge_entry(e, +0.1)).pack(side="left")

            self._delay_vars[col] = var
            self._delay_shift_entries[col] = entry
            self._delay_checkboxes.append(cb)
            self._delay_shift_widgets.extend([entry, btn_frame])

    def _nudge_entry(self, entry, delta):
        try:
            val = round(float(entry.get()) + delta, 10)
        except ValueError:
            val = delta
        entry.delete(0, tk.END)
        entry.insert(0, f"{val:g}")
        self.draw()

    def _get_shift(self, col):
        try:
            return float(self._delay_shift_entries[col].get())
        except ValueError:
            return 0.0

    def _on_set_change(self):
        self._update_delay_list()
        self.draw()

    def _update_delay_list(self):
        selected_set_str = self.set_var.get()
        selected_suffix = None if selected_set_str == "1" and None in self.sets \
            else int(selected_set_str)
        self._current_delay_cols = [c for c in self.delay_cols
                                    if delay_set(c) == selected_suffix]
        self._build_delay_checkboxes()

    def _save_shifts(self):
        path = filedialog.asksaveasfilename(
            title="Save shifts",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=self.src.stem + "_shifts.csv",
        )
        if not path:
            return

        has_timestamps = self.timestamps is not None
        with open(path, "w") as f:
            header = "delay,timestamp,shift_px" if has_timestamps else "delay,shift_px"
            f.write(header + "\n")
            for col in self._current_delay_cols:
                delay_label = re.sub(r"_\d+$", "", col)
                shift = self._get_shift(col)
                if has_timestamps:
                    ts = self.timestamps[self.col_names.index(col)]
                    f.write(f"{delay_label},{ts},{shift:g}\n")
                else:
                    f.write(f"{delay_label},{shift:g}\n")

        print(f"Saved shifts to {path}")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open spectra CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.src = Path(path)
            self.pixels, self.col_names, self.spectra, self.timestamps = load_csv(self.src)
            self.ref_cols = [c for c in self.col_names if is_reference_col(c)]
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
        self.ax.set_xlabel("Pixel (dispersion axis)")
        self.ax.set_ylabel("Absorbance")
        self.ax.grid(True, alpha=0.3)
        plt.ion()
        plt.show(block=False)

    # -----------------------------------------------------------------------
    # Compute and redraw
    # -----------------------------------------------------------------------
    def draw(self):
        selected_cols = [col for col, var in self._delay_vars.items() if var.get()]

        ref_choice = self.ref_var.get()
        if ref_choice == REF_AVERAGE:
            ref_indices = [i for i, c in enumerate(self.col_names) if is_reference_col(c)]
            reference = self.spectra[:, ref_indices].mean(axis=1)
            ref_desc = "mean(Ref_*)"
        else:
            ref_idx = self.col_names.index(ref_choice)
            reference = self.spectra[:, ref_idx]
            ref_desc = ref_choice

        shift_signal = self.shift_signal_var.get()
        pixels_trimmed = self.pixels[EDGE_TRIM:-EDGE_TRIM]

        self.ax.cla()
        self.ax.set_xlabel("Pixel (dispersion axis)")
        self.ax.set_ylabel("Absorbance")
        self.ax.grid(True, alpha=0.3)

        colours = plt.cm.plasma(np.linspace(0, 0.8, max(len(selected_cols), 1)))
        for col, colour in zip(selected_cols, colours):
            shift = self._get_shift(col)
            signal = self.spectra[:, self.col_names.index(col)]
            absorption = compute_absorption(signal, reference, shift, shift_signal)
            delay_label = re.sub(r"_\d+$", "", col)
            if self.timestamps is not None:
                ts = self.timestamps[self.col_names.index(col)]
                ts_short = ts.split("T")[-1] if "T" in ts else ts
                label = f"{delay_label} ({ts_short}) shift={shift:+g}"
            else:
                label = f"{delay_label} shift={shift:+g}" if shift != 0 else delay_label
            self.ax.plot(pixels_trimmed, absorption, lw=1.5, color=colour, label=label)

        self.ax.axvline(1315, color="black", lw=0.8, ls="--", alpha=0.6)
        if selected_cols:
            self.ax.legend(fontsize=8, framealpha=0.7)

        set_str = self.set_var.get()
        target = "signal" if shift_signal else "ref"
        self.ax.set_title(
            f"{self.src.stem}  |  set {set_str}  |  ref: {ref_desc}  |  shifting {target}"
        )
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Absorption GUI with per-spectrum pixel shift."
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="Combined spectra CSV from fits_to_csv.py")
    parser.add_argument("--shift-signal", action="store_true",
                        help="Start with shifts applied to the signal instead of the reference")
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

    AbsorptionShiftGUI(args.input, shift_signal=args.shift_signal)


if __name__ == "__main__":
    main()
