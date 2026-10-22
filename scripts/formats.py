"""Readers for the supported microscope file formats.

A run is one acquisition folder. Supported formats:

- OME-TIFF (spinning disk): one multichannel <name>_t0_p<N>.ome.tiff per
  position, channel names and pixel size in a .companion.ome file.
- MetaXpress (ImageXpress slide scanning): one single-channel TIFF per well,
  site and wavelength, t<T>_<well>_s<site>_w<W>_z<Z>.tif, with channel names,
  pixel size and the file list in a .xdce file.
- MetaXpress without .xdce (older ImageXpress): one single-channel TIFF per
  well, site and wavelength, <plate>_<well>_s<site>_w<W><GUID>.tif, with the
  channel name and pixel size in each TIFF's metadata (_thumb files skipped).

find_runs() detects the format of each folder, so one data folder can contain
runs of both formats. Every run has the same interface:

    run.folder          acquisition folder
    run.channel_names   list of channel names
    run.pixel_size      (y, x) in µm, or None
    run.shape, .dtype   image shape (C, Y, X) and dtype
    run.positions       list of Position(name, label, files, well, site, timepoint)
    run.read(pos, channels=None) -> (C, Y, X) array

Position.name is unique within the run and is used for output file names
(<name>_seg.tif), Position.files has the file of each channel (the same file
for all channels in OME-TIFF, None for a missing MetaXpress image). well
(e.g. "B01"), site and timepoint (both from 1) are only set for MetaXpress.
"""
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tifffile


class Position(NamedTuple):
    name: str
    label: str
    files: list
    well: str = None
    site: int = None
    timepoint: int = None


def _tiff_shape_dtype(f):
    with tifffile.TiffFile(f) as tif:
        return tif.series[0].shape, tif.series[0].dtype


class OmeTiffRun:
    """Folder with <name>_t0_p<N>.ome.tiff files (C, Y, X) and a .companion.ome file."""

    format = "OME-TIFF"
    POSITION_RE = re.compile(r"_p(\d+)\.ome\.tiff?$")

    @classmethod
    def is_run(cls, folder):
        return any(cls.POSITION_RE.search(f.name) for f in folder.glob("*.ome.tif*"))

    def __init__(self, folder):
        self.folder = folder
        files = sorted((int(m.group(1)), f) for f in folder.glob("*.ome.tif*")
                       if (m := self.POSITION_RE.search(f.name)))
        shape, self.dtype = _tiff_shape_dtype(files[0][1])
        self.shape = (1, *shape) if len(shape) == 2 else tuple(shape)

        names, self.pixel_size = self._read_companion()
        if not names or len(names) != self.shape[0]:
            names = [f"ch{i}" for i in range(self.shape[0])]
        self.channel_names = names

        self.positions = []
        for p, f in files:
            # Backticks in some file names are awkward in output file names
            name = re.sub(r"\.ome\.tiff?$", "", f.name).replace("`", "")
            self.positions.append(Position(name, f"p{p}: {f.name}", [f] * self.shape[0]))

    def _read_companion(self):
        """Channel names and XY pixel size (µm) from the .companion.ome file, if present."""
        companions = list(self.folder.glob("*.companion.ome"))
        if not companions:
            return None, None
        root = ET.parse(companions[0]).getroot()
        ns = {"ome": root.tag.split("}")[0].strip("{")}
        pixels = root.find(".//ome:Pixels", ns)
        if pixels is None:
            return None, None
        names = [c.get("Name") for c in pixels.findall("ome:Channel", ns)]
        size_x = pixels.get("PhysicalSizeX")
        size_y = pixels.get("PhysicalSizeY")
        pixel_size = (float(size_y), float(size_x)) if size_x and size_y else None
        return names, pixel_size

    def read(self, pos, channels=None):
        img = tifffile.imread(pos.files[0])
        if img.ndim == 2:
            img = img[np.newaxis]
        return img if channels is None else img[list(channels)]


class MetaXpressRun:
    """Folder with single-channel t<T>_<well>_s<site>_w<W>_z<Z>.tif files and a .xdce file."""

    format = "MetaXpress"
    WAVE_RE = re.compile(r"_w\d+")

    @classmethod
    def is_run(cls, folder):
        return any(folder.glob("*.xdce"))

    def __init__(self, folder):
        self.folder = folder
        root = ET.parse(next(folder.glob("*.xdce"))).getroot()

        waves = sorted(root.iter("Wavelength"), key=lambda w: int(w.get("index")))
        self.channel_names = [w.find("EmissionFilter").get("name") for w in waves]
        cal = root.find(".//ObjectiveCalibration")
        self.pixel_size = (float(cal.get("pixel_height")), float(cal.get("pixel_width"))) if cal is not None else None

        # Group the per-channel images into positions (time, well, site, z)
        groups = {}
        for img in root.iter("Image"):
            ident = img.find("Identifier")
            row = int(img.find("Well/Row").get("number"))
            col = int(img.find("Well/Column").get("number"))
            key = (int(ident.get("time_index")), row, col, int(ident.get("field_index")), int(ident.get("z_index")))
            f = folder / img.get("filename")
            groups.setdefault(key, [None] * len(self.channel_names))
            if f.exists():
                groups[key][int(ident.get("wave_index"))] = f

        # Keep only channels that have images, e.g. when only the FITC files were copied
        keep = [c for c in range(len(self.channel_names)) if any(files[c] for files in groups.values())]
        self.channel_names = [self.channel_names[c] for c in keep]

        multiple_times = len({key[0] for key in groups}) > 1
        self.positions = []
        for (t, row, col, field, z), files in sorted(groups.items()):
            files = [files[c] for c in keep]
            existing = [f for f in files if f is not None]
            if not existing:
                continue
            name = self.WAVE_RE.sub("", existing[0].stem, count=1)  # t1_B01_s1_w2_z1 -> t1_B01_s1_z1
            well = f"{chr(ord('A') + row - 1)}{col:02d}"
            label = f"well {well}, site {field + 1}"
            if multiple_times:
                label += f", timepoint {t + 1}"
            self.positions.append(Position(name, label, files, well, field + 1, t + 1))
        if not self.positions:
            raise FileNotFoundError(f"None of the images listed in the .xdce file found in {folder}")

        shape, self.dtype = _tiff_shape_dtype(next(f for f in self.positions[0].files if f is not None))
        self.shape = (len(self.channel_names), *shape)

    def read(self, pos, channels=None):
        channels = range(len(self.channel_names)) if channels is None else channels
        planes = [tifffile.imread(pos.files[c]) if pos.files[c] is not None
                  else np.zeros(self.shape[1:], dtype=self.dtype) for c in channels]
        return np.stack(planes)


class MetaXpressPlainRun:
    """Folder with single-channel <plate>_<well>_s<site>_w<W><GUID>.tif files and no .xdce file."""

    format = "MetaXpress (no .xdce)"
    # <plate>_<well>_s<site>_w<W> followed by an optional 36-character GUID; thumbnails don't match
    FILE_RE = re.compile(r"^(?P<name>(?P<plate>.+)_(?P<well>[A-P]\d{2})_s(?P<site>\d+))_w(?P<w>\d)"
                         r"(?P<guid>[0-9A-Fa-f-]{36})?\.tiff?$")

    @classmethod
    def is_run(cls, folder):
        return not any(folder.glob("*.xdce")) and any(cls.FILE_RE.match(f.name) for f in folder.glob("*.tif*"))

    def __init__(self, folder):
        self.folder = folder
        groups = {}   # (well, site, name) -> {wavelength: file}
        for f in folder.glob("*.tif*"):
            m = self.FILE_RE.match(f.name)
            if m:
                groups.setdefault((m["well"], int(m["site"]), m["name"]), {})[int(m["w"])] = f
        waves = sorted({w for files in groups.values() for w in files})

        # Channel names and pixel size from the MetaMorph metadata of one image per wavelength
        self.channel_names = []
        self.pixel_size = None
        for w in waves:
            f = next(files[w] for files in groups.values() if w in files)
            with tifffile.TiffFile(f) as tif:
                desc = tif.pages[0].description
            name = re.search(r'id="_IllumSetting_"[^>]*value="([^"]*)"', desc)
            self.channel_names.append(name.group(1) if name else f"w{w}")
            cal = re.search(r'id="spatial-calibration-x"[^>]*value="([^"]*)"', desc)
            if cal and self.pixel_size is None:
                self.pixel_size = (float(cal.group(1)), float(cal.group(1)))

        self.positions = [Position(name, f"well {well}, site {site}", [files.get(w) for w in waves], well, site, 1)
                          for (well, site, name), files in sorted(groups.items())]
        shape, self.dtype = _tiff_shape_dtype(next(f for f in self.positions[0].files if f is not None))
        self.shape = (len(self.channel_names), *shape)

    read = MetaXpressRun.read


FORMATS = [MetaXpressRun, MetaXpressPlainRun, OmeTiffRun]


def open_run(folder):
    """Run object for an acquisition folder, or None if the format is not recognized."""
    for fmt in FORMATS:
        if fmt.is_run(folder):
            return fmt(folder)
    return None


def existing_dir(arg):
    """argparse type: a folder that exists, as a Path."""
    path = Path(arg)
    if path.is_dir():
        return path
    msg = f"folder not found: {arg}"
    if arg.endswith('"'):
        # Windows: in "C:\some folder\" the backslash escapes the closing quote, which ends up in the argument
        msg += ' (remove the trailing backslash: "C:\\some folder", not "C:\\some folder\\")'
    raise argparse.ArgumentTypeError(msg)


def find_runs(input_dir):
    """All runs under input_dir (including itself), sorted by folder."""
    folders = [input_dir] + sorted(p for p in input_dir.rglob("*") if p.is_dir())
    return [run for run in map(open_run, folders) if run is not None]
