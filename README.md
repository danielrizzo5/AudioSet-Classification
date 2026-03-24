# AudioSet-Classification

[AudioSet](https://research.google.com/audioset/) data prep and multi-label classification with **HuggingFace CLAP**, PyTorch, and Lightning. Offline feature caches use `ClapFeatureExtractor` (48 kHz); training uses `ClapModel`’s audio encoder as a frozen-then-unfrozen backbone plus a projection head (`BackboneFinetuning` unfreezes the encoder in the last ~10% of epochs).

## Setup

```bash
uv sync
```

`uv sync` pulls **torch**, **torchaudio**, and **torchcodec** (torchaudio loads audio via TorchCodec).

Place AudioSet CSVs under `dev-data/audioset-csv/`:

- `balanced_train_segments.csv`
- `eval_segments.csv`
- `class_labels_indices.csv`

## Data and training pipeline

Use the same `--clap-model` for `data features` and `train` (default: `laion/clap-htsat-fused`).

1. **Inspect CSVs**

   ```bash
   uv run audioset data inspect
   ```

2. **Download audio** (requires [ffmpeg](https://ffmpeg.org/) on `PATH`, e.g. `brew install ffmpeg`)

   ```bash
   uv run audioset data download --split train
   uv run audioset data download --split eval
   ```

   Optional: `--max-clips N` for a small run.

3. **Write JSONL manifests** (only clips with existing WAVs are included). Eval rows are shuffled (default `--seed 42`) before splitting into `val.jsonl` / `test.jsonl` via `--val-fraction`.

   ```bash
   uv run audioset data manifest
   ```

4. **Precompute CLAP inputs** (one command per split)

   ```bash
   uv run audioset data features --split train --clap-model laion/clap-htsat-fused
   uv run audioset data features --split val --clap-model laion/clap-htsat-fused
   uv run audioset data features --split test --clap-model laion/clap-htsat-fused
   ```

5. **Train**

   ```bash
   uv run audioset train --clap-model laion/clap-htsat-fused --max-epochs 10
   ```

Checkpoints and TensorBoard logs go under `training-outputs/` (git-ignored). The first train run may download CLAP weights from the Hugging Face Hub.

## Development

```bash
just format    # ruff check --fix && ruff format
just lint      # ruff check
just typecheck # pyright
just test      # format, lint, typecheck, pytest
```
