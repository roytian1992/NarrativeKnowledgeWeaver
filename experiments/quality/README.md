`prepare_quality_experiment.py` is an isolated data-prep script for the SCROLLS `QUALITY` benchmark.

It does three things:

1. scans `articles/*.json` and `qas/*.csv`
2. splits by article into train/eval
3. exports minimal NKW-style article JSON files:

```json
[
  {
    "id": "article_name",
    "title": "title",
    "subtitle": "",
    "content": "..."
  }
]
```

Example:

```bash
source ~/.bashrc && conda activate screenplay
cd /vepfs-mlp2/c20250513/241404044/users/roytian/NarrativeKnowledgeWeaver
python experiments/quality/prepare_quality_experiment.py
```
