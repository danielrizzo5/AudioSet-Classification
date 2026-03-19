# AudioSet-Classification

Monorepo for [AudioSet](https://research.google.com/audioset/) data processing and multi-label classification with PyTorch Lightning.

## Packages

- **src/audioset_classification/** – Lightning training, classifier, data loading
- **src/audioset_data/** – Data processing, conversion (TFRecord → .pt)

## Setup

```bash
uv sync
```

Single environment: both `audioset` and `audioset-data` CLIs available.

## Usage

```bash
# Training
audioset data inspect --data-dir /path/to/audioset
audioset train --data-dir /path/to/audioset --synthetic --max-segments 100 --max-epochs 5

# Data processing
audioset-data convert --tfrecord-dir /path/to/tfrecord --output-dir /path/to/output
```

## Development

```bash
just format   # ruff check --fix && ruff format
just lint     # ruff check
just typecheck  # pyright
just test     # format, lint, typecheck, pytest
```
