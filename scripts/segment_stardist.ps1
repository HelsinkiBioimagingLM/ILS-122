# Paths relative to the repository, so this can be run from any folder
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$data_root = "L:\lsdcc\lmu-tmp\users\h\hajaalin\ILS-122"

pixi run --manifest-path "$repo\..\microscopy-ai-envs\pixi.toml" -e stardist `
    python "$repo\scripts\segment_stardist.py" `
    --input_dir "$data_root\ILS122 2022 Group1" `
    --output_dir "$data_root\ILS122 2022 Group1 stardist" `
    --n_workers 2
