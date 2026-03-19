# AudioSet-Classification

Monorepo for [AudioSet](https://research.google.com/audioset/) data processing and multi-label classification with PyTorch Lightning.

## Setup

```bash
uv sync
```

## Usage

```bash
# Data: inspect, download, convert
audioset data inspect --data-dir /path/to/audioset
audioset data download --data-dir /path/to/audioset
audioset data convert --tfrecord-dir /path/to/tfrecord --output-dir /path/to/output

# Training
audioset train --data-dir /path/to/audioset --synthetic --max-segments 100 --max-epochs 5
```

## Development

```bash
just format   # ruff check --fix && ruff format
just lint     # ruff check
just typecheck  # pyright
just test     # format, lint, typecheck, pytest
```
