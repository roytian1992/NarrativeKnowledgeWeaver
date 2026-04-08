from __future__ import annotations

import argparse
import csv
from pathlib import Path
from random import Random
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils.general_utils import dump_json, ensure_dir, load_json, word_len


def load_article(article_path: Path) -> Dict[str, Any]:
    payload = load_json(str(article_path))
    if not isinstance(payload, dict):
        raise ValueError(f"Article JSON must be an object: {article_path}")
    return payload


def load_qas(qa_csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with qa_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = str((row or {}).get("question", "") or "").strip()
            answer_choice = str((row or {}).get("answer_choice", "") or "").strip().upper()
            answer_text = str((row or {}).get("answer_text", "") or "").strip()
            if not question:
                continue
            rows.append(
                {
                    "question": question,
                    "answer_choice": answer_choice,
                    "answer_text": answer_text,
                }
            )
    return rows


def extract_content(article_payload: Dict[str, Any]) -> str:
    content = str(article_payload.get("content", "") or "").strip()
    if content:
        return content
    return str(article_payload.get("article_opening", "") or "").strip()


def build_nkw_payload(article_name: str, article_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    title = str(article_payload.get("title", "") or "").strip() or article_name
    content = extract_content(article_payload)
    return [
        {
            "id": article_name,
            "title": title,
            "subtitle": "",
            "content": content,
        }
    ]


def scan_dataset(dataset_root: Path) -> List[Dict[str, Any]]:
    articles_dir = dataset_root / "articles"
    qas_dir = dataset_root / "qas"
    rows: List[Dict[str, Any]] = []
    for article_path in sorted(articles_dir.glob("*.json")):
        article_name = article_path.stem
        qa_path = qas_dir / f"{article_name}.csv"
        if not qa_path.exists():
            continue
        article_payload = load_article(article_path)
        content = extract_content(article_payload)
        qas = load_qas(qa_path)
        rows.append(
            {
                "article_name": article_name,
                "article_path": str(article_path),
                "qa_path": str(qa_path),
                "title": str(article_payload.get("title", "") or "").strip(),
                "content_words": word_len(content, lang="auto"),
                "qa_count": len(qas),
            }
        )
    return rows


def split_articles(rows: List[Dict[str, Any]], train_articles: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (int(row["content_words"]), int(row["qa_count"]), str(row["article_name"])))
    if train_articles <= 0:
        return [], ordered
    if train_articles >= len(ordered):
        return ordered, []

    bucket_count = min(5, len(ordered))
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(bucket_count)]
    for idx, row in enumerate(ordered):
        buckets[idx * bucket_count // len(ordered)].append(row)

    rnd = Random(seed)
    chosen_names: List[str] = []
    remaining = train_articles
    non_empty_buckets = [bucket for bucket in buckets if bucket]
    for bucket_idx, bucket in enumerate(non_empty_buckets):
        buckets_left = len(non_empty_buckets) - bucket_idx
        take = max(1, round(remaining / buckets_left))
        take = min(take, len(bucket), remaining)
        shuffled = list(bucket)
        rnd.shuffle(shuffled)
        chosen_names.extend(str(row["article_name"]) for row in shuffled[:take])
        remaining -= take
        if remaining <= 0:
            break

    if remaining > 0:
        selected = set(chosen_names)
        pool = [row for row in ordered if str(row["article_name"]) not in selected]
        rnd.shuffle(pool)
        chosen_names.extend(str(row["article_name"]) for row in pool[:remaining])

    chosen = set(chosen_names[:train_articles])
    train_rows = [row for row in ordered if str(row["article_name"]) in chosen]
    eval_rows = [row for row in ordered if str(row["article_name"]) not in chosen]
    return train_rows, eval_rows


def export_articles(rows: List[Dict[str, Any]], split_name: str, out_dir: Path) -> List[Dict[str, Any]]:
    exported: List[Dict[str, Any]] = []
    split_dir = out_dir / "converted_articles" / split_name
    ensure_dir(str(split_dir))
    for row in rows:
        article_name = str(row["article_name"])
        article_payload = load_article(Path(str(row["article_path"])))
        out_path = split_dir / f"{article_name}.json"
        dump_json(str(out_path), build_nkw_payload(article_name, article_payload))
        exported.append(
            {
                "article_name": article_name,
                "article_path": str(row["article_path"]),
                "qa_path": str(row["qa_path"]),
                "converted_path": str(out_path),
                "title": str(row["title"]),
                "content_words": int(row["content_words"]),
                "qa_count": int(row["qa_count"]),
            }
        )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare isolated QUALITY experiment artifacts.")
    parser.add_argument(
        "--dataset-root",
        default="/vepfs-mlp2/c20250513/241404044/users/roytian/benchmarks/SCROLLS/quality/quality_processed",
        help="QUALITY dataset root containing articles/ and qas/.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/quality/artifacts",
        help="Output directory for split manifest and converted article JSON files.",
    )
    parser.add_argument("--train-articles", type=int, default=10, help="Number of train articles.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for article split.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(str(out_dir))

    rows = scan_dataset(dataset_root)
    train_rows, eval_rows = split_articles(rows, train_articles=args.train_articles, seed=args.seed)
    train_exported = export_articles(train_rows, "train", out_dir)
    eval_exported = export_articles(eval_rows, "eval", out_dir)

    manifest = {
        "dataset_root": str(dataset_root),
        "train_articles": len(train_exported),
        "eval_articles": len(eval_exported),
        "seed": int(args.seed),
        "train": train_exported,
        "eval": eval_exported,
    }
    dump_json(str(out_dir / "split_manifest.json"), manifest)
    print(f"Prepared QUALITY experiment under: {out_dir}")
    print(f"train_articles={len(train_exported)} eval_articles={len(eval_exported)}")


if __name__ == "__main__":
    main()
