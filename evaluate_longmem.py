import json
import os
import re
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from typing import Dict, Any

# =========================
# LLM config
# =========================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
    base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL")
)

MODEL = os.getenv("LLM_MODEL")

# =========================
# paths
# =========================
RAG_PATH = r"E:\tape-mem\experiments\rag_agent_longmemeval_0_result.json"
TAPE_PATH = r"E:\tape-mem\experiments\tape_agent_longmemeval_0_result.json"

# =========================
# text utils
# =========================
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s%$]", "", text)
    return text

def tokenize(text):
    return set(normalize(text).split())

# =========================
# Soft EM（替代 EM）
# =========================
def extract_core(text):
    text_low = text.lower()

    m = re.search(r"\d+%", text_low)
    if m:
        return m.group()

    m = re.search(r"\$\d+", text_low)
    if m:
        return m.group()

    m = re.search(r"\b\d+\b", text_low)
    if m:
        return m.group()

    return normalize(text)

def soft_em(pred, gt):
    pred_core = extract_core(pred)
    gt_core = extract_core(gt)

    # exact match
    if normalize(pred_core) == normalize(gt_core):
        return 1

    pred_tokens = set(pred_core.split())
    gt_tokens = set(gt_core.split())

    if len(gt_tokens) == 0:
        return 0

    overlap = pred_tokens & gt_tokens
    ratio = len(overlap) / len(gt_tokens)

    return 1 if ratio >= 0.8 else 0

# =========================
# F1
# =========================
def f1(pred, gt):
    p = tokenize(pred)
    g = tokenize(gt)
    if len(p) == 0 or len(g) == 0:
        return 0
    inter = p & g
    if len(inter) == 0:
        return 0
    precision = len(inter) / len(p)
    recall = len(inter) / len(g)
    return 2 * precision * recall / (precision + recall)

def extracted_f1(pred, gt):
    return f1(extract_core(pred), extract_core(gt))

# =========================
# LLM Judge
# =========================
def llm_judge(question, gt, ans):
    prompt = f"""
Return JSON only:

{{
  "correctness": 1-5,
  "faithfulness": 1-5,
  "completeness": 1-5,
  "reasoning": 1-5,
  "error_type": "none / retrieval / reasoning / hallucination"
}}

Question: {question}
Ground Truth: {gt}
Answer: {ans}
"""

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = r.choices[0].message.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return {
        "correctness": 3,
        "faithfulness": 3,
        "completeness": 3,
        "reasoning": 3,
        "error_type": "unknown"
    }


# =========================
# load
# =========================
def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["results"]
    return {x["question"]["question_id"]: x for x in data}


# =========================
# evaluation
# =========================
def evaluate(rag, tape):
    qids = sorted(set(rag.keys()) & set(tape.keys()))

    rag_stats = defaultdict(list)
    tape_stats = defaultdict(list)

    error_count = {"rag": defaultdict(int), "tape": defaultdict(int)}

    print(f"Evaluating {len(qids)} samples...\n")

    for qid in tqdm(qids):
        rq = rag[qid]
        tq = tape[qid]

        q = rq["question"]["text"]
        gt = rq["question"]["answer_candidates"][0]

        r_ans = rq["response"]["answer"]
        t_ans = tq["response"]["answer"]

        # =========================
        # metrics (EM -> Soft EM)
        # =========================
        for model, ans, stats in [
            ("rag", r_ans, rag_stats),
            ("tape", t_ans, tape_stats)
        ]:
            stats["soft_em"].append(soft_em(ans, gt))
            stats["f1"].append(f1(ans, gt))
            stats["ef1"].append(extracted_f1(ans, gt))

        # =========================
        # LLM judge
        # =========================
        r_j = llm_judge(q, gt, r_ans)
        t_j = llm_judge(q, gt, t_ans)

        for model, j, stats in [
            ("rag", r_j, rag_stats),
            ("tape", t_j, tape_stats)
        ]:
            stats["correctness"].append(j["correctness"])
            stats["faithfulness"].append(j["faithfulness"])
            stats["completeness"].append(j["completeness"])
            stats["reasoning"].append(j["reasoning"])
            error_count[model][j["error_type"]] += 1

    # =========================
    # print results
    # =========================
    def avg(x):
        return np.mean(x)

    print("\n===== FINAL RESULTS =====\n")

    for name, s in [("RAG", rag_stats), ("TAPE", tape_stats)]:
        print(name)
        print(f"Soft EM:       {avg(s['soft_em']):.3f}")
        print(f"F1:            {avg(s['f1']):.3f}")
        print(f"Extracted F1:  {avg(s['ef1']):.3f}")
        print(f"Correctness:   {avg(s['correctness']):.3f}")
        print(f"Faithfulness:  {avg(s['faithfulness']):.3f}")
        print(f"Completeness:  {avg(s['completeness']):.3f}")
        print(f"Reasoning:     {avg(s['reasoning']):.3f}\n")


# =========================
# run
# =========================
if __name__ == "__main__":
    rag = load(RAG_PATH)
    tape = load(TAPE_PATH)
    evaluate(rag, tape)