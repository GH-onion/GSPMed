# llm_agents.py
# 四个模块，每个模块负责一种llm agent的调用
# 输入: prompt_list (list of str)
# 输出: response_list (list of str)

import os
import time
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

from agents.agent import Caller, Thought
from agents.prompt import *

# =========================
# NEW: token counting helpers
# =========================
def _safe_token_counter(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens for a text string. Prefer tiktoken; fallback to rough heuristic.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))
    except Exception:
        # fallback: ~4 chars per token (rough)
        return max(1, len(text) // 4)

# =========================
# NEW: cost tracker
# =========================
class CostTracker:
    """
    Track latency/calls/tokens for profiling computational cost and scalability.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_calls = 0
        self.total_in_tokens = 0
        self.total_out_tokens = 0
        self.latencies = []  # seconds per call

        # per-image stats
        self.per_image = defaultdict(lambda: {
            "calls": 0,
            "in_tokens": 0,
            "out_tokens": 0,
            "latencies": [],
            "by_agent": defaultdict(lambda: {"calls": 0, "in_tokens": 0, "out_tokens": 0, "latencies": []}),
            "fanout": {}  # scalability signals
        })

    def log_call(self, image_id: str, agent_name: str, in_text: str, out_text: str, latency_s: float):
        in_toks = _safe_token_counter(in_text, self.model_name)
        out_toks = _safe_token_counter(out_text, self.model_name)

        self.total_calls += 1
        self.total_in_tokens += in_toks
        self.total_out_tokens += out_toks
        self.latencies.append(latency_s)

        img = self.per_image[image_id]
        img["calls"] += 1
        img["in_tokens"] += in_toks
        img["out_tokens"] += out_toks
        img["latencies"].append(latency_s)

        ag = img["by_agent"][agent_name]
        ag["calls"] += 1
        ag["in_tokens"] += in_toks
        ag["out_tokens"] += out_toks
        ag["latencies"].append(latency_s)

    def set_fanout(self, image_id: str, fanout_dict: Dict[str, Any]):
        """
        Store scalability-relevant fanout information per sample.
        """
        self.per_image[image_id]["fanout"] = fanout_dict

    def summary(self) -> Dict[str, Any]:
        def _p95(xs):
            if not xs:
                return 0.0
            xs_sorted = sorted(xs)
            idx = int(0.95 * (len(xs_sorted) - 1))
            return xs_sorted[idx]

        total_tokens = self.total_in_tokens + self.total_out_tokens
        return {
            "model": self.model_name,
            "total_calls": self.total_calls,
            "total_in_tokens": self.total_in_tokens,
            "total_out_tokens": self.total_out_tokens,
            "total_tokens": total_tokens,
            "mean_latency_s": sum(self.latencies) / max(1, len(self.latencies)),
            "p95_latency_s": _p95(self.latencies),
        }

    def export_per_image(self) -> Dict[str, Any]:
        """
        Convert defaultdict to normal dict for JSON serialization.
        """
        out = {}
        for image_id, v in self.per_image.items():
            by_agent = {}
            for agent_name, av in v["by_agent"].items():
                by_agent[agent_name] = {
                    "calls": av["calls"],
                    "in_tokens": av["in_tokens"],
                    "out_tokens": av["out_tokens"],
                    "total_tokens": av["in_tokens"] + av["out_tokens"],
                    "mean_latency_s": (sum(av["latencies"]) / max(1, len(av["latencies"]))) if av["latencies"] else 0.0,
                }
            out[image_id] = {
                "calls": v["calls"],
                "in_tokens": v["in_tokens"],
                "out_tokens": v["out_tokens"],
                "total_tokens": v["in_tokens"] + v["out_tokens"],
                "mean_latency_s": (sum(v["latencies"]) / max(1, len(v["latencies"]))) if v["latencies"] else 0.0,
                "by_agent": by_agent,
                "fanout": v.get("fanout", {}),
            }
        return out


load_dotenv()

my_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url='https://api.kwwai.top/v1',
    timeout=120.0,
)

MODEL_NAME = "gpt-3.5-turbo"

# NEW: global tracker
tracker = CostTracker(model_name=MODEL_NAME)

# instance of caller agent
tot_agent = Caller(
    system_prompt=TREE_OF_THOUGHT_SYS_PROMPT,
    base_model=Thought,
    model_name=MODEL_NAME,
    parallel_tool_calls=False,
    client=my_client,
    max_tokens=3000,
)

# =========================
# NEW: unified measured call wrapper
# =========================
def measured_run(image_id: str, agent_name: str, task: str, question: str, hint: str,
                 prev_answer: Optional[List[str]] = None) -> str:
    """
    Wrap tot_agent.run with latency + token cost measurement.
    Since Caller.run may not expose OpenAI usage, we estimate tokens locally.
    """
    # build an approximate "input text" for token accounting (reproducible)
    prev = ""
    if prev_answer:
        # if prev_answer is nested lists, flatten
        flat = []
        for x in prev_answer:
            if isinstance(x, list):
                flat.extend([str(y) for y in x])
            else:
                flat.append(str(x))
        prev = "\n".join(flat)

    in_text = f"[TASK]\n{task}\n[QUESTION]\n{question}\n[HINT]\n{hint}\n[PREV]\n{prev}\n"
    t0 = time.perf_counter()
    out = tot_agent.run(task, question, [hint], prev_answer=prev_answer)
    t1 = time.perf_counter()
    tracker.log_call(image_id=image_id, agent_name=agent_name, in_text=in_text, out_text=str(out), latency_s=(t1 - t0))
    return out


# 模块1: Agent 1
def agent_caption(image_id: str, question: str, prompt_list: List[str],
                  answer_list: Optional[List[str]] = None) -> List[str]:
    responses = []
    task = ("Now your are given a question and a hint, the hint is a caption describing the medical image. "
            "Your task: is to directly answer the question based on your thoughts and the given hint.")
    for prompt in prompt_list:
        response = measured_run(image_id, "Agent1_caption", task, question, prompt, prev_answer=answer_list)
        response = f"[Agent1] 回答: {response}"
        responses.append(response)
    return responses


# 模块2: Agent 2
def agent_keyword(image_id: str, question: str, prompt_list: List[str],
                  answer_list: Optional[List[str]] = None) -> List[str]:
    responses = []
    task = ("Now your are given a question and a hint, the hint is a list of keywords related to the medical image. "
            "Your task: is to directly answer the question based on your thoughts and the given hint.")
    for prompt in prompt_list:
        if answer_list:
            for i, ans in enumerate(answer_list):
                # NOTE: keep your original behavior; prev_answer stays answer_list for consistency
                response = measured_run(image_id, "Agent2_keyword", task, question, prompt, prev_answer=answer_list)
                response = f"[Agent2] 回答_{i}: {response}"
                responses.append(response)
        else:
            response = measured_run(image_id, "Agent2_keyword", task, question, prompt, prev_answer=answer_list)
            response = f"[Agent2] 回答: {response}"
            responses.append(response)
    return responses


# 模块3: Agent 3
def agent_candidate(image_id: str, question: str, prompt_list: List[str],
                    answer_list: Optional[List[str]] = None) -> List[str]:
    responses = []
    task = ("Now your are given a question and a hint, the hint is a list of knowledge scope related to the medical image. "
            "Your task: is to directly answer the question based on your thoughts and the given hint.")
    for prompt in prompt_list:
        if answer_list:
            for i, ans in enumerate(answer_list):
                response = measured_run(image_id, "Agent3_candidate", task, question, prompt, prev_answer=answer_list)
                response = f"[Agent3] 回答_{i}: {response}"
                responses.append(response)
        else:
            response = measured_run(image_id, "Agent3_candidate", task, question, prompt, prev_answer=answer_list)
            response = f"[Agent3] 回答: {response}"
            responses.append(response)
    return responses


# 模块4: Agent 4
def agent_subq(image_id: str, question: str, prompt_list: List[str],
               answer_list: Optional[List[str]] = None) -> List[str]:
    responses = []
    task = ("Now your are given a question and a hint, the hint is a subquestion and its corresponding answer related to the medical image. "
            "Your task: is to directly answer the question based on your thoughts and the given hint.")
    for prompt in prompt_list:
        if answer_list:
            for i, ans in enumerate(answer_list):
                response = measured_run(image_id, "Agent4_subq", task, question, prompt, prev_answer=answer_list)
                response = f"[Agent4] 回答_{i}: {response}"
                responses.append(response)
        else:
            response = measured_run(image_id, "Agent4_subq", task, question, prompt, prev_answer=answer_list)
            response = f"[Agent4] 回答: {response}"
            responses.append(response)
    return responses


def load_prompts(json_path, image_name):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if image_name not in data:
        raise ValueError(f"{image_name} not found in prompt file.")
    return data[image_name]


def load_prompts_from_folder(folder_path):
    prompts_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            image_id = filename[:-5]
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            prompts_dict[image_id] = prompts
    return prompts_dict


def load_candidate_prompts(candidate_json_path):
    with open(candidate_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts_dict = {}
    for image_id, info in data.items():
        if "top_10_preds" in info:
            prompts_dict[image_id] = info["top_10_preds"]
    return prompts_dict


def load_captions_prompts(captions_json_path):
    with open(captions_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts_dict = {}
    for image_id, info in data.items():
        if "caption" in info:
            prompts_dict[image_id] = [info["caption"]]
    return prompts_dict


def find_imageid_in_jsons(folder_path, image_id, field):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            if image_id in data and field in data[image_id]:
                value = data[image_id][field]
                return [value] if isinstance(value, str) else value
    return None


def find_imageid_in_json(file_path, image_id, field):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if image_id in data and field in data[image_id]:
        value = data[image_id][field]
        return [value] if isinstance(value, str) else value
    return None


def append_to_json(results_dict: Dict[str, Any], json_path: str):
    path = Path(json_path)
    if path.exists():
        existing = json.loads(path.read_text())
    else:
        existing = {}
    for k, v in results_dict.items():
        existing[k] = v
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


def check_json(image_id: str, json_path: str) -> bool:
    path = Path(json_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({}), encoding="utf-8")
        return False
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
        return image_id in data.keys()


# =========================
# NEW: save profiling result
# =========================
def save_profile(profile_path: str):
    profile = {
        "summary": tracker.summary(),
        "per_image": tracker.export_per_image(),
    }
    Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
    Path(profile_path).write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("用法: python3 llm_agents.py <source_dir> <target_json> [profile_json]")
        print("示例: python3 llm_agents.py radvqa outputs/radvqa.json outputs/radvqa_profile.json")
        sys.exit(1)

    dataset = sys.argv[1]
    output_json = sys.argv[2]
    profile_json = sys.argv[3] if len(sys.argv) == 4 else output_json.replace(".json", "_profile.json")

    method = "biomed"

    candidate_json_path = os.path.join("./keyword_candidate", f'{dataset.lower()}_results_{method}.json')
    with open(candidate_json_path, 'r') as f:
        candidate_data = json.load(f)

    all_image_ids = list(candidate_data.keys())

    keyword_json_path = os.path.join("./keyword", f'{dataset.lower()}_keyword.json')

    if dataset == "radvqa":
        subquestion_path = "subquestion/RadVQA_1"
        caption_path = "classified_caption/RadVQA_5"
    elif dataset == "pathvqa":
        subquestion_path = "subquestion/PathVQA_1"
        caption_path = "classified_caption/PathVQA_5"
    elif dataset == "slake":
        subquestion_path = "subquestion/Slake_1"
        caption_path = "classified_caption/Slake_5"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    last_id = None

    for image_id in all_image_ids:
        all_responses = {}
        if check_json(image_id, output_json):
            continue
        try:
            all_responses[image_id] = {}

            # get question
            question = find_imageid_in_jsons(caption_path, image_id, "question")
            if not question:
                continue
            # question is list sometimes; normalize
            if isinstance(question, list):
                question_str = question[0]
            else:
                question_str = str(question)

            # NEW: record fanout/scalability signals for this sample
            fanout = {
                "n_caption_prompts": 0,
                "n_keywords_after_prune": 0,
                "n_candidates_after_prune": 0,
                "n_subquestions": 0,
                "expected_calls_upperbound": 0,  # rough accounting
            }

            # 1. captions
            captions_prompt = find_imageid_in_jsons(caption_path, image_id, "caption")
            if not captions_prompt:
                continue
            fanout["n_caption_prompts"] = len(captions_prompt)

            answers1 = []
            a1 = agent_caption(image_id, question_str, captions_prompt)
            answers1.append(a1)
            all_responses[image_id]["captions"] = answers1

            # 2. keyword
            keyword_probs = find_imageid_in_json(keyword_json_path, image_id, "top_10_probs")
            keyword_prompt = find_imageid_in_json(keyword_json_path, image_id, "top_10_preds")
            if not keyword_prompt:
                continue

            prune_keywords = []
            for i, item in enumerate(zip(keyword_prompt, keyword_probs)):
                kw, prob = item
                if prob > 0.9:
                    prune_keywords.append(kw)
                elif i == (len(keyword_probs) - 1) and len(prune_keywords) == 0:
                    prune_keywords.append(keyword_prompt[0])
            fanout["n_keywords_after_prune"] = len(prune_keywords)

            answers2 = []
            for ans in answers1:
                answers2.append(agent_keyword(image_id, question_str, prune_keywords, ans))
            all_responses[image_id]["keyword"] = answers2

            # 3. candidate
            candidate_probs = find_imageid_in_json(candidate_json_path, image_id, "top_10_probs")
            candidate_prompt = find_imageid_in_json(candidate_json_path, image_id, "top_10_preds")
            if not candidate_prompt:
                continue

            prune_candidate = []
            for i, item in enumerate(zip(candidate_prompt, candidate_probs)):
                kw, prob = item
                if prob > 0.9:
                    prune_candidate.append(kw)
                elif i == (len(candidate_probs) - 1) and len(prune_candidate) == 0:
                    prune_candidate.append(candidate_prompt[0])
            fanout["n_candidates_after_prune"] = len(prune_candidate)

            answers3 = []
            for ans in answers2:
                answers3.append(agent_candidate(image_id, question_str, prune_candidate, ans))
            all_responses[image_id]["candidate"] = answers3

            # 4. sub-question
            subq_prompt = find_imageid_in_jsons(subquestion_path, image_id, "subquestion")
            if not subq_prompt:
                continue
            fanout["n_subquestions"] = len(subq_prompt)

            answers4 = []
            for ans in answers3:
                answers4.append(agent_subq(image_id, question_str, subq_prompt, ans))
            all_responses[image_id]["sub-question"] = answers4

            # NEW: rough expected calls upperbound for scalability explanation
            # (this mirrors your nested loops structure; exact calls are measured by tracker.total_calls)
            # captions: n_caption
            # keyword: n_caption * n_keywords * (len(answer_list) ~ n_caption responses)
            # candidate: depends on answers2 size, etc. Here we only store a rough structural signal.
            fanout["expected_calls_upperbound"] = (
                fanout["n_caption_prompts"]
                + fanout["n_caption_prompts"] * fanout["n_keywords_after_prune"]
                + fanout["n_caption_prompts"] * fanout["n_keywords_after_prune"] * fanout["n_candidates_after_prune"]
                + fanout["n_caption_prompts"] * fanout["n_keywords_after_prune"] * fanout["n_candidates_after_prune"] * fanout["n_subquestions"]
            )
            tracker.set_fanout(image_id, fanout)

            append_to_json(all_responses, output_json)
            
            last_id = image_id

            # ===== NEW: flush profile after each sample =====
            save_profile(profile_json)

            # ===== NEW: print per-sample cost summary =====
            per_img = tracker.export_per_image().get(image_id, {})
            print("\n=== Sample Done ===")
            print(f"image_id: {image_id}")
            print(f"calls: {per_img.get('calls', 0)} | total_tokens: {per_img.get('total_tokens', 0)} | "
                  f"mean_latency_s: {per_img.get('mean_latency_s', 0.0):.3f}")
            # optional: breakdown by agent
            by_agent = per_img.get("by_agent", {})
            if by_agent:
                for ag, av in by_agent.items():
                    print(f"  - {ag}: calls={av['calls']}, tokens={av['total_tokens']}, mean_lat={av['mean_latency_s']:.3f}s")

            last_id = image_id

        except Exception as e:
            print(f"Error occurs when generating answer for {image_id}! {e}")
            continue

    # NEW: save profiling json
    save_profile(profile_json)

    # Print overall profiling summary
    print("\n=== Profiling Summary ===")
    print(json.dumps(tracker.summary(), indent=2, ensure_ascii=False))

    # keep your original print
    if last_id is not None:
        for key, value in all_responses[last_id].items():
            print(key, len(value))
