"""Batch Cellpose 3 cell segmentation with a cell marker and a nuclear helper channel.

Finds all runs under --input_dir (recursively), in any format supported by
formats.py, and segments every position with Cellpose. The two channels are
combined on the fly into an RGB image with the cell marker (--channel) in red
and the nuclear helper channel (--helper_channel) in blue, as in the cellpose
GUI with "chan to segment: red, chan2: blue". Label images are written to
--output_dir, reproducing the subfolder structure:

    <input_dir>/<run>/...  ->  <output_dir>/<run>/<position name>_seg.tif

A summary CSV with the object count per image is written to
<output_dir>/segmentation_summary.csv.

With --n_workers N, N worker processes segment images in parallel, each with
its own copy of the model. Workers are spread over the available GPUs, so the
GPU memory limits how many workers fit; the CPU threads are divided between
the workers.

Run in the cellpose3 env of microscopy-ai-envs:
    pixi run --manifest-path ../microscopy-ai-envs/pixi.toml -e cellpose3 \\
        python scripts/segment_cellpose.py --input_dir /path/to/data --output_dir /path/to/results --diameter 60 --n_workers 2
"""
import argparse
import csv
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import tifffile
import torch

from cellpose import models

from formats import existing_dir, find_runs

# Cellpose channel numbers: 1 = red, 3 = blue
CELLPOSE_CHANNELS = [1, 3]


def resolve_channel(channel, run):
    """Channel index from an index ("1") or a channel name ("DAPI"), None if the run doesn't have it."""
    if channel.isdigit():
        idx = int(channel)
        return idx if idx < len(run.channel_names) else None
    if channel in run.channel_names:
        return run.channel_names.index(channel)
    return None


def composite(run, pos, c_main, c_helper):
    """(Y, X, 3) RGB image with the main channel in red and the helper channel in blue."""
    main, helper = run.read(pos, [c_main, c_helper]).astype(np.float32)
    rgb = np.zeros((*main.shape, 3), dtype=np.float32)
    rgb[..., 0] = main
    rgb[..., 2] = helper
    return rgb


def summary_row(f, out, n_objects, args):
    return {"input": str(f.relative_to(args.input_dir)), "output": str(out.relative_to(args.output_dir)),
            "n_objects": n_objects}


# Model of the current process, set by init_worker()
_model = None
_init_error = None
_eval_kwargs = {}


def init_worker(model_type, pretrained_model, eval_kwargs, n_gpus, n_threads, counter):
    """Load the model in a worker; workers take the GPUs in turn.

    A failure is stored and reported for each image, because an exception here
    would make multiprocessing.Pool restart the worker forever.
    """
    global _model, _init_error, _eval_kwargs
    _eval_kwargs = eval_kwargs
    try:
        with counter.get_lock():
            k = counter.value
            counter.value += 1
        torch.set_num_threads(n_threads)
        device = torch.device(f"cuda:{k % n_gpus}") if n_gpus else None
        if pretrained_model:
            _model = models.CellposeModel(gpu=bool(n_gpus), device=device, pretrained_model=str(pretrained_model))
        else:
            # Cellpose (not CellposeModel) includes the size model, used when --diameter is 0
            _model = models.Cellpose(gpu=bool(n_gpus), device=device, model_type=model_type)
    except Exception:
        _init_error = "model loading failed:\n" + traceback.format_exc()


def segment_one(task):
    """Segment one image and write the labels. Returns (task, n_objects, error)."""
    if _init_error:
        return task, None, _init_error
    run, pos, c_main, c_helper, out = task[1:]
    try:
        img = composite(run, pos, c_main, c_helper)
        labels = _model.eval(img, channels=CELLPOSE_CHANNELS, channel_axis=-1, **_eval_kwargs)[0]
        labels = labels.astype(np.uint32)
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out, labels, compression="zlib")
        return task, int(labels.max()), None
    except Exception:
        return task, None, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=existing_dir, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--channel", default="TEXASRED",
                        help="cell marker channel, shown as red: name from the metadata or an index (default: TEXASRED, i.e. _w3)")
    parser.add_argument("--helper_channel", default="DAPI",
                        help="nuclear helper channel, shown as blue: name or index (default: DAPI, i.e. _w1)")
    parser.add_argument("--diameter", type=float, default=30,
                        help="expected cell diameter in pixels; 0 estimates it per image (default: 30)")
    parser.add_argument("--model_type", default="cyto3", help="pretrained Cellpose model (default: cyto3)")
    parser.add_argument("--pretrained_model", type=Path, help="own trained model file (overrides --model_type)")
    parser.add_argument("--flow_threshold", type=float, default=0.4, help="flow error threshold (default: 0.4)")
    parser.add_argument("--cellprob_threshold", type=float, default=0.0, help="cell probability threshold (default: 0.0)")
    parser.add_argument("--min_size", type=int, default=15, help="remove objects smaller than this in pixels (default: 15)")
    parser.add_argument("--overwrite", action="store_true", help="re-segment images that already have output")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="number of parallel worker processes, each with its own model (default: 1)")
    args = parser.parse_args()
    if args.pretrained_model and args.diameter == 0:
        parser.error("--diameter 0 (estimate) needs a pretrained --model_type with a size model")

    runs = find_runs(args.input_dir)
    if not runs:
        raise SystemExit(f"No runs found under {args.input_dir}")
    n_total = sum(len(r.positions) for r in runs)
    print(f"Found {len(runs)} runs with {n_total} positions under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "segmentation_summary.csv"
    rows = {}   # task index -> summary row, written in input order

    # Collect the images to segment; existing results are only counted
    tasks = []
    i = 0
    for run in runs:
        run_rel = run.folder.relative_to(args.input_dir)
        print(f"{run_rel} ({run.format}, channels: {', '.join(run.channel_names)})")
        c_main = resolve_channel(args.channel, run)
        c_helper = resolve_channel(args.helper_channel, run)
        if c_main is None or c_helper is None:
            print(f"  skip run (no channel {args.channel!r} or {args.helper_channel!r})")
            i += len(run.positions)
            continue

        for pos in run.positions:
            i += 1
            out = args.output_dir / run_rel / f"{pos.name}_seg.tif"
            missing = [run.channel_names[c] for c in (c_main, c_helper) if pos.files[c] is None]
            if missing:
                print(f"[{i}/{n_total}] skip (no {', '.join(missing)} image): {run_rel / pos.name}")
                continue
            if out.exists() and not args.overwrite:
                print(f"[{i}/{n_total}] skip (exists): {pos.files[c_main].relative_to(args.input_dir)}")
                rows[i] = summary_row(pos.files[c_main], out, int(tifffile.imread(out).max()), args)
            else:
                tasks.append((i, run, pos, c_main, c_helper, out))

    if tasks:
        n_gpus = torch.cuda.device_count()
        devices = f"{n_gpus} GPU(s)" if n_gpus else "CPU"
        n_workers = max(1, min(args.n_workers, len(tasks)))
        n_threads = max(1, (os.cpu_count() or 1) // n_workers)
        print(f"Segmenting {len(tasks)} images with {n_workers} worker(s) on {devices}")

        eval_kwargs = {"diameter": args.diameter or None, "flow_threshold": args.flow_threshold,
                       "cellprob_threshold": args.cellprob_threshold, "min_size": args.min_size}
        ctx = mp.get_context("spawn")  # CUDA can't be used in forked processes
        init_args = (args.model_type, args.pretrained_model, eval_kwargs, n_gpus, n_threads, ctx.Value("i", 0))
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
                idx, run, pos, c_main = task[:4]
                rel = pos.files[c_main].relative_to(args.input_dir)
                if error:
                    n_failed += 1
                    print(f"[{k}/{len(tasks)}] FAILED: {rel}\n{error}")
                    continue
                print(f"[{k}/{len(tasks)}] {n_objects} objects: {rel}")
                rows[idx] = summary_row(pos.files[c_main], out=task[-1], n_objects=n_objects, args=args)
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
