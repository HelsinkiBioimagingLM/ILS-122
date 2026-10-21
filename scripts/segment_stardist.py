"""Batch StarDist nucleus segmentation of MetaXpress (ImageXpress) images.

Finds all single-channel TIFFs of one wavelength under --input_dir
(recursively), named <plate>_<well>_s<site>_w<W>[<GUID>].tif (thumbnails
are skipped), segments them with a StarDist 2D model and writes label images
to --output_dir, reproducing the subfolder structure:

    <input_dir>/<sub>/plate1_A03_s3_w1<GUID>.tif  ->  <output_dir>/<sub>/plate1_A03_s3_seg.tif

A summary CSV with the object count per image is written to
<output_dir>/segmentation_summary.csv.

With --n_workers N, N worker processes segment images in parallel, each with
its own copy of the model. Workers are spread over the available GPUs and use
only the GPU memory they need; the CPU threads (used by StarDist's
non-maximum suppression) are divided between the workers.

Run in the stardist env of microscopy-ai-envs:
    pixi run --manifest-path ../microscopy-ai-envs/pixi.toml -e stardist \\
        python scripts/segment_stardist.py --input_dir /path/to/data --output_dir /path/to/results --n_workers 4
"""
import argparse
import csv
import multiprocessing as mp
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import tifffile

# <plate>_<well>_s<site>_w<W> followed by an optional 36-character GUID
NAME_RE = re.compile(r"^(?P<name>.+)_w(?P<w>\d)(?P<guid>[0-9A-Fa-f-]{36})?\.tiff?$")


def find_images(input_dir, wavelength):
    """(file, position name) of all images of the wavelength, sorted by path."""
    images = []
    for f in sorted(input_dir.rglob("*.tif*")):
        m = NAME_RE.match(f.name)
        if m and m["w"] == str(wavelength):
            images.append((f, m["name"]))
    return images


def summary_row(f, out, n_objects, args):
    return {"input": str(f.relative_to(args.input_dir)), "output": str(out.relative_to(args.output_dir)),
            "n_objects": n_objects}


# Model of the current process, set by init_worker()
_model = None
_init_error = None
_predict_kwargs = {}


def init_worker(model_name, model_dir, predict_kwargs, n_gpus, n_threads, counter):
    """Load the model in a worker; workers take the GPUs in turn.

    A failure is stored and reported for each image, because an exception here
    would make multiprocessing.Pool restart the worker forever.
    """
    global _model, _init_error, _predict_kwargs
    _predict_kwargs = predict_kwargs
    try:
        with counter.get_lock():
            k = counter.value
            counter.value += 1
        # Must be set before TensorFlow is imported (no effect in the main process, which imported it already)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        if n_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(k % n_gpus)
        import tensorflow as tf
        from stardist.models import StarDist2D
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)  # otherwise the first worker takes all GPU memory
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        if model_dir:
            _model = StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent))
        else:
            _model = StarDist2D.from_pretrained(model_name)
    except Exception:
        _init_error = "model loading failed:\n" + traceback.format_exc()


def segment_one(task):
    """Segment one image and write the labels. Returns (task, n_objects, error)."""
    if _init_error:
        return task, None, _init_error
    f, out, p_low, p_high = task[1:]
    try:
        from csbdeep.utils import normalize
        img = normalize(tifffile.imread(f), p_low, p_high, axis=(0, 1))
        labels, _ = _model.predict_instances(img, **_predict_kwargs)
        labels = labels.astype(np.uint32)
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out, labels, compression="zlib")
        return task, int(labels.max()), None
    except Exception:
        return task, None, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--wavelength", type=int, default=1, help="wavelength to segment, the W in _w<W> (default: 1)")
    parser.add_argument("--model", default="2D_versatile_fluo", help="pretrained StarDist model (default: 2D_versatile_fluo)")
    parser.add_argument("--model_dir", type=Path, help="own trained model folder (overrides --model)")
    parser.add_argument("--prob_thresh", type=float, help="probability threshold (default: the model's)")
    parser.add_argument("--nms_thresh", type=float, help="overlap threshold of non-maximum suppression (default: the model's)")
    parser.add_argument("--n_tiles", type=int, nargs=2, help="predict in Y X tiles to save GPU memory (e.g. 2 2)")
    parser.add_argument("--p_low", type=float, default=1, help="lower percentile for normalization (default: 1)")
    parser.add_argument("--p_high", type=float, default=99.8, help="upper percentile for normalization (default: 99.8)")
    parser.add_argument("--overwrite", action="store_true", help="re-segment images that already have output")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="number of parallel worker processes, each with its own model (default: 1)")
    args = parser.parse_args()

    images = find_images(args.input_dir, args.wavelength)
    if not images:
        raise SystemExit(f"No _w{args.wavelength} images found under {args.input_dir}")
    n_total = len(images)
    print(f"Found {n_total} _w{args.wavelength} images under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "segmentation_summary.csv"
    rows = {}   # task index -> summary row, written in input order

    # Collect the images to segment; existing results are only counted
    tasks = []
    for i, (f, name) in enumerate(images, 1):
        out = args.output_dir / f.parent.relative_to(args.input_dir) / f"{name}_seg.tif"
        if out.exists() and not args.overwrite:
            print(f"[{i}/{n_total}] skip (exists): {f.relative_to(args.input_dir)}")
            rows[i] = summary_row(f, out, int(tifffile.imread(out).max()), args)
        else:
            tasks.append((i, f, out, args.p_low, args.p_high))

    if tasks:
        import tensorflow as tf
        n_gpus = len(tf.config.list_physical_devices("GPU"))
        devices = f"{n_gpus} GPU(s)" if n_gpus else "CPU"
        n_workers = max(1, min(args.n_workers, len(tasks)))
        n_threads = max(1, (os.cpu_count() or 1) // n_workers)
        print(f"Segmenting {len(tasks)} images with {n_workers} worker(s) on {devices}")

        predict_kwargs = {k: v for k, v in [("prob_thresh", args.prob_thresh), ("nms_thresh", args.nms_thresh),
                                            ("n_tiles", tuple(args.n_tiles) if args.n_tiles else None)] if v is not None}
        predict_kwargs["show_tile_progress"] = False
        ctx = mp.get_context("spawn")  # CUDA can't be used in forked processes
        init_args = (args.model, args.model_dir, predict_kwargs, n_gpus, n_threads, ctx.Value("i", 0))
        if n_workers == 1:
            init_worker(*init_args)
            results = map(segment_one, tasks)
            pool = None
        else:
            pool = ctx.Pool(n_workers, initializer=init_worker, initargs=init_args)
            results = pool.imap_unordered(segment_one, tasks)

        n_failed = 0
        start = time.time()
        try:
            for k, (task, n_objects, error) in enumerate(results, 1):
                idx, f, out = task[:3]
                rel = f.relative_to(args.input_dir)
                if error:
                    n_failed += 1
                    print(f"[{k}/{len(tasks)}] FAILED: {rel}\n{error}")
                    continue
                print(f"[{k}/{len(tasks)}] {n_objects} objects: {rel}")
                rows[idx] = summary_row(f, out, n_objects, args)
        finally:
            if pool is not None:
                pool.terminate()
        elapsed = time.time() - start
        print(f"Segmented {len(tasks) - n_failed} images in {elapsed:.0f} s "
              f"({elapsed / len(tasks):.1f} s/image, including model loading)")
        if n_failed:
            print(f"{n_failed} images failed (see above); run again to retry them")

    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["input", "output", "n_objects"])
        writer.writeheader()
        writer.writerows(rows[i] for i in sorted(rows))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
