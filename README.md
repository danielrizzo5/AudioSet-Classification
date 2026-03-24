# AudioSet-Classification

[AudioSet](https://research.google.com/audioset/) data prep and multi-label classification with **HuggingFace CLAP**, PyTorch, and Lightning. Offline feature caches use `ClapFeatureExtractor` (48 kHz); training uses `ClapModel`’s audio encoder as a frozen-then-unfrozen backbone plus a projection head (`BackboneFinetuning` unfreezes the encoder in the last ~10% of epochs).

## Setup

```bash
uv sync
```

`uv sync` pulls **torch**, **torchaudio**, and **torchcodec** (`torchaudio.load` decodes via TorchCodec).

Place AudioSet CSVs under `dev-data/audioset-csv/`:

- `balanced_train_segments.csv`
- `eval_segments.csv`
- `class_labels_indices.csv`
- `ontology.json` (hierarchy for analysis plots — e.g. `curl -L -o dev-data/audioset-csv/ontology.json https://raw.githubusercontent.com/audioset/ontology/master/ontology.json`)

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

6. **Optional: CLAP embeddings + UMAP** (install analysis deps: `uv sync --group analysis`)

   Embeddings read precomputed `.pt` from step 4 (same `--features-dir` / split as `data features`).

   ```bash
   uv run audioset data embeddings --split train --clap-model laion/clap-htsat-fused
   uv run audioset analysis umap --embeddings dev-data/embeddings/clap/train.npz
   # Or combine train/val/test into one UMAP + one set of tier plots (default dir):
   uv run audioset analysis umap --all-splits
   # Optional: ``--combined-stem my_run`` names cache/PNGs ``my_run_tier_*.png``.
   ```

   Writes `dev-data/embeddings/clap/{split}.npz` and a cached 2D UMAP array under `dev-data/analysis/umap/` (filename includes `n_neighbors`, `min_dist`, and row count) plus tier PNGs. A single `--embeddings` run uses that file’s stem (e.g. `train_tier_*.png`). **`--all-splits`** stacks every present `train` / `val` / `test` npz, fits **one** UMAP, and writes **`all_splits_tier_*.png`** (or `--combined-stem`). Defaults match the observer-style setup: `n_neighbors=200`, `min_dist=0.002`, seed **42**, scatter `s=8`, `alpha=0.8`, **tab20** colors (with blends after 20 categories). Override via `audioset analysis umap --help` (`--seed` is an alias for `--random-state`). Each PNG only changes tier-based colors. `representative_label_id` in the npz is the **first** manifest `label_id` for hierarchy coloring. With `--all-splits`, any missing split file is skipped with a warning; if none are found, the command exits with an error.

Checkpoints and TensorBoard logs go under `training-outputs/` (git-ignored). The first train run may download CLAP weights from the Hugging Face Hub.

## Development

```bash
just format    # ruff check --fix && ruff format
just lint      # ruff check
just typecheck # pyright
just test      # format, lint, typecheck, pytest
```
