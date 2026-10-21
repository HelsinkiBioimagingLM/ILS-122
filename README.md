# ILS-122

## Environment

The environment for `view_positions.py` (napari) is defined in `pixi.toml`
and locked in `pixi.lock`, for Linux and Windows. With pixi:

```bash
pixi shell
python scripts/view_positions.py --input_dir ... --labels_dir ...
```

### Without pixi (Anaconda / Miniconda)

`conda-locks/conda-lock.yml` has the same package versions as `pixi.lock`.
A GitHub Actions workflow (`.github/workflows/update-conda-locks.yml`)
regenerates it with `pixi-to-conda-lock` whenever `pixi.lock` changes.

Install it with [conda-lock](https://github.com/conda/conda-lock), kept in its
own environment so that the base environment stays untouched:

```bash
conda create -n conda-lock -c conda-forge conda-lock
conda activate conda-lock
conda-lock install -n ILS-122 conda-locks/conda-lock.yml
conda activate ILS-122
python scripts/view_positions.py --input_dir ... --labels_dir ...
```

To regenerate the lock file locally (needs pixi):

```bash
pixi exec --spec pixi-to-conda-lock --spec pyyaml -- bash -c \
    "pixi-to-conda-lock --output conda-locks pixi.lock && python scripts/normalize-conda-locks.py pixi.lock conda-locks/*conda-lock.yml"
```
