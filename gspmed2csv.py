import re
import json
import csv
import sys
from pathlib import Path
import ast
from collections import defaultdict
from typing import List

import torch
import pandas as pd
import numpy as np

def clean_answer_str(s: str) -> str:
    """
    删除字符串中的 '\n{' 及其之后全部内容
    """
    return re.sub(r'\n\{.*$', '', s, flags=re.DOTALL)

def clean_data_dict(data: dict) -> dict:
    """
    对整个 data（外层 key 为 qid）进行遍历，
    按层级 key ('captions','keyword',...) 对字符串进行 clean。
    """
    cleaned = {}
    for qid, info in data.items():
        cleaned[qid] = {}
        for lvl, lists in info.items():
            # 每一层都是二维嵌套： list of lists
            cleaned_lists = []
            for inner in lists:
                cleaned_inner = [clean_answer_str(s) for s in inner]
                cleaned_lists.append(cleaned_inner)
            cleaned[qid][lvl] = cleaned_lists
    return cleaned

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_csv(path):
    df = pd.read_csv(path, usecols=["qid", "predict"], dtype={"qid": str})
    df = df.set_index("qid")
    pred_dict = df["predict"].to_dict()
    return pred_dict

def extract_answer_from_record(record_str, drop_idx=[]):
    """
    从 "[Agenti] 回答_j: {"thought": "...", "Answer": "No"}" 中提取 Answer 字段以及序号j
    丢弃从回答_<drop_idx>引申的答案
    """
    try:
        idx_part = record_str.split(":", 1)[0].split("回答")[-1].strip()
        if idx_part != '':
            idx = int(idx_part.replace("_", ""))
            if len(drop_idx) != 0:
                if idx in drop_idx:
                    return -2, None # signal of skipping
        else:
            idx = -1
        json_part = record_str.split(":", 1)[1].strip()
        try:
            # used for structured \"Answer: "
            data = ast.literal_eval(json_part)
            return idx, data.get("Answer", "").lower()
        except:
            data = json_part.split("Answer: ")[-1].replace("'","").replace(".", "")
            return idx, data.lower()
        
    except Exception as e:
        return f"{e}"

def merge_and_write(json1_path, json2_path, dataset, mmix_csv, vqam_csv, chicl_csv, csv_prefix="."):
    """
    Attributes:
        json1_path: GT path
        json2_path: GSPMed prediction path
        dataset: Current datasets
        mmix_csv: MMix prediction path
        vqam_csv: VQAMix prediction path
        chicl_csv: CHICL prediction path
    """
    # 读取数据
    data1 = load_json(json1_path)  # 包含 img_id, atype, key=qid
    data2 = load_json(json2_path)  # 包含层级数组
    data2 = clean_data_dict(data2)
    data_mmix = load_csv(mmix_csv)
    data_vqam = load_csv(vqam_csv)
    data_chicl = load_csv(chicl_csv)

    # 构建 qid → img_id & atype 映射
    metadata = {
        qid: {"img_id": info.get("img_id", ""), "answer_type": info.get("atype", ""), "ref": str(info.get("answer", ""))}
        for qid, info in data1.items()
    }

    levels = ["captions", "keyword", "candidate", "sub-question"]

    drop_idx = defaultdict(lambda: defaultdict(list))

    for l, lvl in enumerate(levels):
        rows = []
        for qid, info in data2.items():
            if lvl not in info:
                continue
            
            meta = metadata.get(qid, {"img_id": "", "answer_type": "", "ref": ""})
            for inner_list in info[lvl]:
                answer_cnt = 0
                vote_nodes = [] # used for opt1
                for i, rec in enumerate(inner_list):
                    try:
                        # get in-layer index and prediciton
                        if lvl == 'candidate':
                            idx, pred = extract_answer_from_record(rec, drop_idx[qid]['keyword'])
                        else:
                            idx, pred = extract_answer_from_record(rec)
                        # if len(pred) > 8:
                        #     breakpoint()
                        
                        # drop all answers generated from dropped previous answers
                        if idx == -2:
                            continue

                        # if in-layer index is not -1, we need to vote
                        # if meta["answer_type"]!="OPEN":
                        if dataset == 'rad':
                            voted_pred = node_vote(gsp_pred=pred, mmix_pred=data_mmix[qid], vqam_pred=data_chicl[qid], atype=meta["answer_type"])
                        elif dataset == 'slake':
                            voted_pred = node_vote(gsp_pred=pred, mmix_pred=data_mmix[qid], vqam_pred=data_chicl[qid], atype=meta["answer_type"])
                        elif dataset == 'pvqa':
                            voted_pred = node_vote(gsp_pred=pred, mmix_pred=data_vqam[qid], vqam_pred=data_chicl[qid], atype=meta["answer_type"])

                        if lvl == 'captions':
                            pred = voted_pred

                            rows.append({
                                "qid": torch.tensor([int(qid)]),
                                "image": meta["img_id"],
                                "answer_type": meta["answer_type"],
                                "ref": meta["ref"],
                                "predict": pred
                            })
                            continue
                        

                        # opt1: record all and vote once more
                        # for answers not predicted using captions
                        # vote with MMixup and VQAMix for each node
                        # then we got multiple voted_preds in one layer
                        # vote insed this layer to get a final pred for this qid
                        
                        # record vote_pred
                        vote_nodes.append(voted_pred)
                        
                        if i == len(inner_list)-1:
                            # last node
                            # vote all nodes in this layer
                            final_vote = layer_vote(vote_nodes)
                            answer_dict = {
                                "qid": torch.tensor([int(qid)]),
                                "image": meta["img_id"],
                                "answer_type": meta["answer_type"],
                                "ref": meta["ref"],
                                "predict": final_vote
                            }
                            rows.append(answer_dict)

                        # opt2: drop not equal
                        # if pred != voted_pred:
                        #     # in case all answers are dropped
                        #     if i == len(inner_list)-1 and answer_cnt==0:
                        #         # if all pred is dropped, fall back to vqam
                        #         answer_dict = {
                        #             "qid": torch.tensor([int(qid)]),
                        #             "image": meta["img_id"],
                        #             "answer_type": meta["answer_type"],
                        #             "ref": meta["ref"],
                        #             "predict": data_vqam[qid]
                        #         }
                        #         rows.append(answer_dict)
                        #         answer_cnt += 1
                        #     else:
                        #         # when not equal to voted results, drop the answer
                        #         drop_idx[qid][lvl].append(i)
                        #     continue
                        # else:
                        #     # when equal to voted result
                        #     # check if the result is already saved
                        #     # if no, save voted result
                        #     answer_dict = {
                        #         "qid": torch.tensor([int(qid)]),
                        #         "image": meta["img_id"],
                        #         "answer_type": meta["answer_type"],
                        #         "ref": meta["ref"],
                        #         "predict": voted_pred
                        #     }
                        #     answer_cnt += 1
                        #     if answer_dict not in rows:
                        #         rows.append(answer_dict)

                        # if all models predict wrong answers in the same layer, save just one
                        

                    except:
                        print(f"{qid}")
                        breakpoint()
                        # get in-layer index and prediciton
                        if lvl == 'candidate':
                            idx, pred = extract_answer_from_record(rec, drop_idx[qid]['keyword'])
                        else:
                            idx, pred = extract_answer_from_record(rec)
                        
                        # drop all answers generated from dropped previous answers
                        if idx == -2:
                            continue

                        if dataset == 'rad':
                            voted_pred = node_vote(gsp_pred=pred, mmix_pred=data_mmix[qid], vqam_pred=data_chicl[qid], atype=meta["answer_type"])
                        elif dataset == 'slake':
                            voted_pred = node_vote(gsp_pred=pred, mmix_pred=data_mmix[qid], vqam_pred=data_chicl[qid], atype=meta["answer_type"])
                        elif dataset == 'pvqa':
                            voted_pred = node_vote(gsp_pred=pred, mmix_pred=data_vqam[qid], vqam_pred=data_chicl[qid], atype=meta["answer_type"])

                        if lvl == 'captions':
                            pred = voted_pred

                            rows.append({
                                "qid": torch.tensor([int(qid)]),
                                "image": meta["img_id"],
                                "answer_type": meta["answer_type"],
                                "ref": meta["ref"],
                                "predict": pred
                            })
                            continue

                        # record vote_pred
                        vote_nodes.append(voted_pred)
                        
                        if i == len(inner_list)-1:
                            # last node
                            # vote all nodes in this layer
                            final_vote = layer_vote(vote_nodes)
                            answer_dict = {
                                "qid": torch.tensor([int(qid)]),
                                "image": meta["img_id"],
                                "answer_type": meta["answer_type"],
                                "ref": meta["ref"],
                                "predict": final_vote
                            }
                            rows.append(answer_dict)


        csv_file = Path(csv_prefix) / f"predictions_GSPMed_{lvl}_{dataset}.csv"
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=["answer_type", "predict", "ref", "qid", "image"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {csv_file}, rows={len(rows)}")


def node_vote(gsp_pred: str, mmix_pred: str, vqam_pred:str, atype: str) -> str:
    '''vote for the prediction within a node

    Attributes:
        gsp_pred (str): prediction of GSPMed
        mmix_pred (str): prediction of M-Mixup
        vqam_pred (str): prediction of VQAMix
        atype (str): answer type

        vote_pred (str): voted predction, if no majority, fall back to GSPMed
    '''
    vote_list = [gsp_pred, mmix_pred, vqam_pred]
    vote_counts = {}
    for pred in vote_list:
        vote_counts[pred] = vote_counts.get(pred, 0) + 1

    # get the majority vote
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    top_pred, top_count = sorted_votes[0]

    # if majority
    if top_count >= 2:
        return top_pred
    else:
        # if no majority
        return vqam_pred
    

def layer_vote(vote_list: List[str]) -> str:
    '''vote for the prediction within a layer

    Attributes:
        vote_list (str): prediction of GSPMed

        vote_pred (str): voted predction, if no majority, fall back to GSPMed
    '''
    vote_counts = {}
    for pred in vote_list:
        vote_counts[pred] = vote_counts.get(pred, 0) + 1

    # get the majority vote
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    top_pred, top_count = sorted_votes[0]

    return top_pred


# 示例使用：
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python3 gspmed2csv.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]

    if dataset == 'rad':
        merge_and_write(
            json1_path="keyword_candidate/radvqa_results_biomed.json",
            json2_path="outputs/radvqa.json",
            dataset='rad',
            mmix_csv="result_csv/predictions_MMixup_rad.csv",
            vqam_csv="result_csv/predictions_VQAMix_re_rad.csv",
            chicl_csv="result_csv/predictions_CHICL_rad.csv",
            csv_prefix="result_csv_gspmed"
        )
    elif dataset == 'slake':
        merge_and_write(
            json1_path="keyword_candidate/slake_results_biomed.json",
            json2_path="outputs/predictions_GSP_slake.json",
            dataset='slake',
            mmix_csv="result_csv/predictions_MMixup_slake.csv",
            vqam_csv="result_csv/predictions_VQAMix_re_rad.csv",
            chicl_csv="result_csv/predictions_CHICL_slake.csv",
            csv_prefix="result_csv"
        )
    elif dataset == 'pvqa':
        merge_and_write(
            json1_path="keyword_candidate/pathvqa_results_biomed.json",
            json2_path="outputs/pvqa_out.json",
            dataset='pvqa',
            mmix_csv="result_csv/predictions_MMixup_slake.csv",
            vqam_csv="result_csv/preidictions_VQA_pvqa_origin_reorder.csv",
            chicl_csv="result_csv/predictions_CHICL_pvqa.csv",
            csv_prefix="result_csv"
        )
