# TraceCompass

Official implementation for the paper **"TraceCompass: Span-Level Attribution for
Retrieval-Augmented Generation"** (Transactions on Representation Learning, 2025).

Status: released (v1.0.0). The code below reproduces the main faithfulness
experiments from the paper.

## Layout

```
src/            attribution and retrieval utilities
scripts/        training and evaluation entry points
configs/        experiment configuration
```

## Quick start

```
pip install -r requirements.txt
python scripts/evaluate.py --config configs/default.yaml
```
