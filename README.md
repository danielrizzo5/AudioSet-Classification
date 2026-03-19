# AudioSet-Classification

Python package for loading [AudioSet](https://research.google.com/audioset/) data and training multi-label classification models with PyTorch Lightning.

## Setup

```bash
uv sync
```

## Usage

```bash
# Inspect data (CSV + ontology)
audioset data inspect --data-dir /path/to/audioset

# Train with synthetic embeddings (no feature download needed)
audioset train --data-dir /path/to/audioset --synthetic --max-segments 100 --max-epochs 5

# Train with real features (requires .pt embeddings in data_dir/features/)
audioset train --data-dir /path/to/audioset --split balanced_train
```

## Data layout

Place AudioSet files in `data_dir/`:

- `class_labels_indices.csv` – ontology (required)
- `balanced_train_segments.csv`, `eval_segments.csv`, `unbalanced_train_segments.csv` – segment metadata
- `features/XX/ytid_start_end.pt` – per-segment embeddings (torch.save; use `--synthetic` for testing without them)

## Development

```bash
just format   # ruff check --fix && ruff format
just lint     # ruff check
just typecheck  # pyright
just test     # format, lint, typecheck, pytest
```
