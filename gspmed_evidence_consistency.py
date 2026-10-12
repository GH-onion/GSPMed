import re
import json
import csv
import sys
from pathlib import Path
import ast
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

import torch
import pandas as pd


# ---------------------------
# Cleaning utilities
# ---------------------------
def clean_answer_str(s: str) -> str:
    """Remove trailing '\n{...}' (often tool traces / metadata)."""
    return re.sub(r'\n\{.*$', '', s, flags=re.DOTALL)

def clean_data_dict(data: dict) -> dict:
    """Clean all nested strings in data2."""
    cleaned = {}
    for qid, info in data.items():
        cleaned[qid] = {}
        for lvl, lists in info.items():
            cleaned_lists = []
            for inner in lists:
                cleaned_inner = [clean_answer_str(s) for s in inner]
                cleaned_lists.append(cleaned_inner)
            cleaned[qid][lvl] = cleaned_lists
    return cleaned

def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------
# Self-consistency vote
# ---------------------------
_PUNCT = re.compile(r"[^\w\s]+")

def normalize_pred(s: str) -> str:
    """
    Normalize answer string for self-consistency voting.
    - lowercase
    - strip
    - remove punctuation
    - collapse whitespace
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def self_consistency_vote(pred_list: List[str]) -> str:
    """
    Majority vote within a layer using only GSPMed predictions.
    If tie, return the first appeared among top candidates (stable tie-break).
    """
    norm = [normalize_pred(p) for p in pred_list if p is not None and str(p).strip() != ""]
    if len(norm) == 0:
        return ""

    cnt = Counter(norm)
    max_count = max(cnt.values())
    top = {k for k, v in cnt.items() if v == max_count}

    # stable tie-break: return earliest prediction whose normalized form is in top
    for p in pred_list:
        if normalize_pred(p) in top:
            return normalize_pred(p)

    return norm[0]


# ---------------------------
# Parsing model records
# ---------------------------
def extract_answer_from_record(record_str: str, drop_idx: List[int] = []):
    """
    Parse record like:
      "[Agenti] 回答_j: {'thought': '...', 'Answer': 'No'}"
    Return: (idx, pred_str_lower)
    If idx in drop_idx -> (-2, None)
    """
    try:
        idx_part = record_str.split(":", 1)[0].split("回答")[-1].strip()
        if idx_part != "":
            idx = int(idx_part.replace("_", ""))
            if drop_idx and idx in drop_idx:
                return -2, None
        else:
            idx = -1

        json_part = record_str.split(":", 1)[1].strip()

        # Preferred structured dict
        try:
            data = ast.literal_eval(json_part)
            pred = data.get("Answer", "")
            return idx, str(pred).strip()
        except Exception:
            # fallback: try to extract after "Answer:"
            if "Answer:" in json_part:
                pred = json_part.split("Answer:", 1)[-1]
            else:
                pred = json_part
            pred = pred.replace("'", "").replace(".", "").strip()
            return idx, pred

    except Exception as e:
        return -1, ""


# ---------------------------
# Main merge + write
# ---------------------------
def merge_and_write_selfconsistency(
    json1_path: str,
    json2_path: str,
    dataset: str,
    csv_prefix: str = ".",
):
    """
    json1_path: GT/metadata json (qid -> img_id, atype, answer)
    json2_path: GSPMed multi-agent raw outputs
    dataset: 'rad' / 'slake' / 'pvqa'
    """
    data1 = load_json(json1_path)
    data2 = clean_data_dict(load_json(json2_path))

    metadata = {
        qid: {
            "img_id": info.get("img_id", ""),
            "answer_type": info.get("atype", ""),
            "ref": str(info.get("answer", "")),
        }
        for qid, info in data1.items()
    }

    levels = ["captions", "keyword", "candidate", "sub-question"]

    # NOTE: keep your drop_idx logic (optional). It was used to skip descendants of dropped answers.
    # In pure self-consistency mode, we typically don't need to drop, but we keep structure for compatibility.
    drop_idx = defaultdict(lambda: defaultdict(list))

    for lvl in levels:
        rows = []
        for qid, info in data2.items():
            if lvl not in info:
                continue

            meta = metadata.get(qid, {"img_id": "", "answer_type": "", "ref": ""})

            # Each lvl is list-of-lists (multiple inner groups)
            for inner_list in info[lvl]:
                vote_nodes = []

                for i, rec in enumerate(inner_list):
                    # candidate layer keeps your "drop from keyword" mechanism
                    if lvl == "candidate":
                        idx, pred = extract_answer_from_record(rec, drop_idx[qid]["keyword"])
                    else:
                        idx, pred = extract_answer_from_record(rec)

                    if idx == -2:
                        continue

                    vote_nodes.append(pred)

                # Self-consistency vote inside this layer group
                final_pred = self_consistency_vote(vote_nodes)

                rows.append({
                    "qid": torch.tensor([int(qid)]),
                    "image": meta["img_id"],
                    "answer_type": meta["answer_type"],
                    "ref": meta["ref"],
                    "predict": final_pred
                })

        csv_file = Path(csv_prefix) / f"predictions_GSPMed_{lvl}_{dataset}_selfcons.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["answer_type", "predict", "ref", "qid", "image"],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved {csv_file}, rows={len(rows)}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python3 gspmed2csv_selfconsistency.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]

    if dataset == "rad":
        merge_and_write_selfconsistency(
            json1_path="keyword_candidate/radvqa_results_biomed.json",
            json2_path="outputs/predictions_GSP_rad.json",
            dataset="rad",
            csv_prefix="result_csv_gspmed"
        )
    elif dataset == "slake":
        merge_and_write_selfconsistency(
            json1_path="keyword_candidate/slake_results_biomed.json",
            json2_path="outputs/predictions_GSP_slake.json",
            dataset="slake",
            csv_prefix="result_csv_gspmed"
        )
    elif dataset == "pvqa":
        merge_and_write_selfconsistency(
            json1_path="keyword_candidate/pathvqa_results_biomed.json",
            json2_path="outputs/pvqa_out.json",
            dataset="pvqa",
            csv_prefix="result_csv_gspmed"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
