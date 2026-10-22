"""Browse all imaging runs and positions under a data folder in napari.

All runs found under --input_dir (recursively), in any format supported by
formats.py (OME-TIFF or MetaXpress), are stacked lazily with dask into
(run, position, C, Y, X), shown with one layer per channel, so the napari
sliders select run and position. For MetaXpress runs, the positions are all
sites of all wells, ordered by well. For MetaXpress runs, a side panel also
has sliders for well, site and timepoint, kept in sync with the position slider.

Runs can have different numbers of positions and channels: channels are
matched by name, and missing positions or channels are shown as empty images.

Optionally, label images are shown as a labels layer, looked up from the same
relative subfolder in --labels_dir, or from --labels_dir itself if it doesn't
have that subfolder. A label file belongs to a position if its name starts
with the position name or the name of one of its image files, followed by "_"
and any suffix: e.g. plate1_A01_s1_seg.tif from segment_stardist.py and
segment_cellpose.py, or plate1_A01_s1_w1<GUID>_cell_labels.tiff from
CellProfiler. If a folder has several label files per position (e.g. cells and
nuclei), choose one with --labels_suffix. --labels_dir can list several
folders (e.g. to compare models), giving one labels layer per folder.

Usage:
    python scripts/view_positions.py --input_dir /path/to/data/20261008
    python scripts/view_positions.py --input_dir /path/to/data/20261008 --labels_dir /path/to/results/20261008
    python scripts/view_positions.py --input_dir /path/to/data/20261008 \
        --labels_dir /path/to/results_a/20261008 /path/to/results_b/20261008
    python scripts/view_positions.py --input_dir /path/to/data/plate1 \
        --labels_dir /path/to/cellprofiler/labels --labels_suffix _cell_labels.tiff
"""
import argparse
import re
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import tifffile
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QLabel, QSlider, QWidget

from formats import existing_dir, find_runs

DEFAULT_COLORMAPS = ["cyan", "green", "red", "magenta", "yellow", "gray"]
# Colormap by (case-insensitive) substring of the channel name; others get DEFAULT_COLORMAPS
CHANNEL_COLORMAPS = {"dapi": "cyan", "hoechst": "cyan", "fitc": "green", "gfp": "green",
                     "dsred": "red", "texas red": "red", "texasred": "red", "rfp": "red", "mcherry": "red", "tritc": "red", "cy5": "magenta"}


def colormap_for(name, i):
    for key, cmap in CHANNEL_COLORMAPS.items():
        if key in name.lower():
            return cmap
    return DEFAULT_COLORMAPS[i % len(DEFAULT_COLORMAPS)]


def read_padded(run, pos, channel_map, shape_cyx, dtype):
    """Read one position into an array of shape_cyx; channel_map[i] is the output channel of input channel i."""
    img = run.read(pos)
    out = np.zeros(shape_cyx, dtype=dtype)
    y, x = img.shape[-2:]
    for c_in, c_out in enumerate(channel_map):
        out[c_out, :y, :x] = img[c_in]
    return out


def read_labels_padded(f, shape_yx):
    labels = tifffile.imread(f)
    out = np.zeros(shape_yx, dtype=np.uint32)
    out[: labels.shape[0], : labels.shape[1]] = labels
    return out


class PlateNavigator(QWidget):
    """Well, site and timepoint sliders for MetaXpress runs, kept in sync with the position slider.

    Each slider steps through the values present in the current run. Moving one
    jumps to the position with that value that matches the other two sliders
    best; moving the position slider updates all three.
    """

    FIELDS = ("well", "site", "timepoint")

    def __init__(self, viewer, runs):
        super().__init__()
        self.viewer, self.runs = viewer, runs
        # Values of each field per run, e.g. ["A02", "B01", ...] for the wells
        self.run_choices = [{f: sorted({getattr(p, f) for p in r.positions} - {None}) for f in self.FIELDS}
                            for r in runs]
        self.sliders, self.value_labels = {}, {}
        layout = QGridLayout(self)
        for i, field in enumerate(self.FIELDS):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(lambda _, f=field: self.on_slider(f))
            value_label = QLabel()
            value_label.setMinimumWidth(40)
            layout.addWidget(QLabel(field), i, 0)
            layout.addWidget(slider, i, 1)
            layout.addWidget(value_label, i, 2)
            self.sliders[field], self.value_labels[field] = slider, value_label
        layout.setRowStretch(len(self.FIELDS), 1)
        self.updating = False
        viewer.dims.events.current_step.connect(self.on_dims)
        self.on_dims()

    def on_dims(self, event=None):
        """Show the well, site and timepoint of the current position."""
        i_run, i_pos = self.viewer.dims.current_step[:2]
        positions = self.runs[i_run].positions
        pos = positions[i_pos] if i_pos < len(positions) else None
        self.updating = True
        try:
            for field, choices in self.run_choices[i_run].items():
                slider = self.sliders[field]
                slider.setRange(0, max(len(choices) - 1, 0))
                slider.setEnabled(len(choices) > 1)
                value = getattr(pos, field) if pos is not None else None
                if value in choices:
                    slider.setValue(choices.index(value))
                self.value_labels[field].setText(str(value) if value is not None else "–")
        finally:
            self.updating = False

    def on_slider(self, field):
        """Go to the position with the chosen value that matches the other sliders best."""
        if self.updating:
            return
        i_run = self.viewer.dims.current_step[0]
        choices = self.run_choices[i_run]
        if not choices[field]:
            return
        target = {f: choices[f][self.sliders[f].value()] for f in self.FIELDS if choices[f]}
        candidates = [i for i, p in enumerate(self.runs[i_run].positions) if getattr(p, field) == target[field]]
        best = max(candidates, key=lambda i: sum(getattr(self.runs[i_run].positions[i], f) == v
                                                 for f, v in target.items()))
        self.viewer.dims.set_current_step(1, best)


def build_stacks(input_dir):
    """Lazily stack all runs into (R, P, C, Y, X) images.

    Also returns the runs and the merged channel names.
    """
    runs = find_runs(input_dir)
    if not runs:
        raise FileNotFoundError(f"No runs (OME-TIFF or MetaXpress) under {input_dir}")

    channel_names = []
    for r in runs:
        channel_names += [n for n in r.channel_names if n not in channel_names]
    n_pos = max(len(r.positions) for r in runs)
    shape_yx = tuple(max(r.shape[i] for r in runs) for i in (1, 2))
    shape_cyx = (len(channel_names), *shape_yx)
    dtype = np.result_type(*[r.dtype for r in runs])

    read_image = dask.delayed(read_padded)
    image_runs = []
    for r in runs:
        channel_map = [channel_names.index(n) for n in r.channel_names]
        planes = [da.from_delayed(read_image(r, pos, channel_map, shape_cyx, dtype), shape=shape_cyx, dtype=dtype)
                  for pos in r.positions]
        planes += [da.zeros(shape_cyx, dtype=dtype)] * (n_pos - len(planes))
        image_runs.append(da.stack(planes))

    return da.stack(image_runs), runs, channel_names


def find_label_file(pos, files, suffix):
    """The label file of a position, or None: its name starts with the position name or an image file
    name, followed by "_" (so that _s1_ doesn't match _s10_), and ends with suffix."""
    prefixes = {pos.name} | {re.sub(r"\.ome$", "", f.stem) for f in pos.files if f is not None}
    matches = sorted({f for f in files for p in prefixes
                      if f.name.startswith(p + "_") and f.name.endswith(suffix)})
    if len(matches) > 1:
        raise SystemExit(f"Several label files for position {pos.name} in {matches[0].parent}: "
                         f"{', '.join(f.name for f in matches)}\nChoose one with --labels_suffix")
    return matches[0] if matches else None


def build_label_stack(labels_dir, input_dir, runs, n_pos, shape_yx, suffix):
    """Lazily stack the label images of all runs into (R, P, Y, X), matching build_stacks.

    Also returns the number of label images found per run.
    """
    read_labels = dask.delayed(read_labels_padded)
    label_runs, n_labels = [], []
    for r in runs:
        label_dir = labels_dir / r.folder.relative_to(input_dir)
        if not label_dir.is_dir():
            label_dir = labels_dir   # e.g. CellProfiler writes all label images into one folder
        files = [f for f in label_dir.iterdir() if f.suffix.lower() in (".tif", ".tiff")]
        label_files = [find_label_file(pos, files, suffix) for pos in r.positions]
        n_labels.append(sum(f is not None for f in label_files))
        planes = [da.from_delayed(read_labels(f, shape_yx), shape=shape_yx, dtype=np.uint32)
                  if f is not None else da.zeros(shape_yx, dtype=np.uint32)
                  for f in label_files]
        planes += [da.zeros(shape_yx, dtype=np.uint32)] * (n_pos - len(planes))
        label_runs.append(da.stack(planes))
    return da.stack(label_runs), n_labels


def layer_names(dirs):
    """Short, unique layer names for the folders: the shortest path tails that tell them apart."""
    for depth in range(1, max(len(d.parts) for d in dirs) + 1):
        names = [str(Path(*d.parts[-depth:])) for d in dirs]
        if len(set(names)) == len(names):
            return names
    return [str(d) for d in dirs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=existing_dir, required=True,
                        help="folder with imaging run subfolders (or a single run folder)")
    parser.add_argument("--labels_dir", type=existing_dir, nargs="+", action="extend", default=[],
                        help="segmentation output folder(s), with the same subfolder structure or all label images in one folder; "
                             "each is shown as its own labels layer")
    parser.add_argument("--labels_suffix", default="",
                        help="only use label files ending with this, e.g. _cell_labels.tiff (default: any)")
    args = parser.parse_args()

    images, runs, channel_names = build_stacks(args.input_dir)
    names = layer_names(args.labels_dir) if args.labels_dir else []
    labels = {name: build_label_stack(d, args.input_dir, runs, images.shape[1], images.shape[-2:],
                                       args.labels_suffix)
              for name, d in zip(names, args.labels_dir)}

    print(f"Stack shape (run, position, C, Y, X) = {images.shape}, channels: {channel_names}")
    for i, r in enumerate(runs):
        rel = r.folder.relative_to(args.input_dir)
        msg = f"  run {i}: {rel} ({r.format}, {len(r.positions)} positions, {', '.join(r.channel_names)})"
        for name, (_, n_labels) in labels.items():
            msg += f", labels '{name}' for {n_labels[i]}/{len(r.positions)}"
        print(msg)

    # Contrast limits from the first position that has an image of each channel,
    # so napari doesn't read every file to compute them
    contrast_limits = []
    for c, name in enumerate(channel_names):
        r, pos, ci = next((r, pos, ci) for r in runs if name in r.channel_names
                          for ci in [r.channel_names.index(name)]
                          for pos in r.positions if pos.files[ci] is not None)
        plane = r.read(pos, [ci])[0]  # unpadded, so the padding doesn't lower the limits
        lo, hi = np.percentile(plane, (1, 99.9))
        contrast_limits.append((lo, max(hi, lo + 1)))  # napari needs lo < hi, e.g. for a blank image

    # Runs can have different pixel sizes (10x vs 20x), so everything is shown in pixel units
    viewer = napari.Viewer(title=args.input_dir.name)
    viewer.add_image(
        images,
        channel_axis=2,
        name=channel_names,
        colormap=[colormap_for(n, i) for i, n in enumerate(channel_names)],
        blending="additive",
        contrast_limits=contrast_limits,
    )
    for name, (label_stack, _) in labels.items():
        viewer.add_labels(label_stack, name=name, opacity=0.5)

    viewer.dims.axis_labels = ["run", "position", "y", "x"]

    # Show the current run, position and file name on the canvas
    viewer.canvas.overlays.text.visible = True

    def update_overlay(event=None):
        i_run, i_pos = viewer.dims.current_step[:2]
        r = runs[i_run]
        text = f"run {i_run}: {r.folder.relative_to(args.input_dir)}"
        if r.pixel_size:
            text += f" ({r.pixel_size[1]} µm/px)"
        if i_pos < len(r.positions):
            text += f"\n{r.positions[i_pos].label}"
        else:
            text += f"\n(no position {i_pos} in this run)"
        viewer.canvas.overlays.text.text = text

    viewer.dims.events.current_step.connect(update_overlay)
    update_overlay()

    # Well/site/timepoint sliders, if there are MetaXpress runs
    if any(p.well is not None for r in runs for p in r.positions):
        viewer.window.add_dock_widget(PlateNavigator(viewer, runs), name="well / site / timepoint", area="left")

    napari.run()


if __name__ == "__main__":
    main()
