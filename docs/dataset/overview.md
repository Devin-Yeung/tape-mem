# MemoryAgentBench Dataset Overview

This document focuses on the dataset layout and practical usage. It does not
cover benchmark reproduction or scoring pipelines because this repository does
not currently ship dedicated evaluation code.

## Quick Summary

MemoryAgentBench evaluates four memory-related capabilities:

- Accurate Retrieval: retrieve specific facts from long histories.
- Test-Time Learning: learn a task from interaction history and apply it later.
- Long-Range Understanding: build a global understanding of long documents.
- Conflict Resolution: update stale facts when later evidence contradicts them.

At the dataset level, every example follows the same high-level pattern:

```python
{
    "context": str,
    "questions": list[str],
    "answers": list[list[str]],
    "metadata": {
        "source": str,
        # other fields are source-specific and are often null
    },
}
```

Two details are easy to miss:

- `questions` and `answers` are aligned by position.
- `answers[i]` is always a list of acceptable gold strings, even when there is
  only one candidate answer.

## Split Inventory

| Split                      | Examples | What it tests                                         | Sources in local snapshot                                                                                                                                                                                                      |
|----------------------------|---------:|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Accurate_Retrieval`       |       22 | fact lookup from long contexts                        | `ruler_qa1_197K`, `ruler_qa2_421K`, `eventqa_full`, `eventqa_65536`, `eventqa_131072`, `longmemeval_s*`                                                                                                                        |
| `Test_Time_Learning`       |        6 | learning from demonstrations in context               | `recsys_redial_full`, `icl_banking77_5900shot_balance`, `icl_clinic150_7050shot_balance`, `icl_nlu_8296shot_balance`, `icl_trec_coarse_6600shot_balance`, `icl_trec_fine_6400shot_balance`                                     |
| `Long_Range_Understanding` |      110 | summarization and multi-question global understanding | `infbench_sum_eng_shots2`, `detective_qa`                                                                                                                                                                                      |
| `Conflict_Resolution`      |        8 | resolving stale or conflicting facts                  | `factconsolidation_mh_6k`, `factconsolidation_mh_32k`, `factconsolidation_mh_64k`, `factconsolidation_mh_262k`, `factconsolidation_sh_6k`, `factconsolidation_sh_32k`, `factconsolidation_sh_64k`, `factconsolidation_sh_262k` |

## Common Row Structure

Each row packs one long `context` together with many aligned QA pairs.
This follows the benchmark's "inject once, query multiple times" design.

### Core fields

- `context`: one large string containing the full material the agent is allowed
  to remember. Depending on the source, this may be a long article, a chunked
  event history, dialogue transcripts, or a synthetic long context.
- `questions`: the prompts asked against that context.
- `answers`: a list of gold answer candidate lists. The outer list lines up with
  `questions`. The inner list contains one or more acceptable strings.
- `metadata`: extra source-specific fields. Only `source` is consistently
  present across all splits.

### Metadata fields seen in the snapshot

Not every source fills every metadata field. The full schema allows:

- `source`
- `qa_pair_ids`
- `demo`
- `keypoints`
- `previous_events`
- `haystack_sessions`
- `question_dates`
- `question_ids`
- `question_types`

In practice:

- `qa_pair_ids` and `source` are present for every row.
- `demo` and `keypoints` are used by `Long_Range_Understanding`.
- `previous_events` appears in the EventQA-based AR rows.
- `haystack_sessions`, `question_dates`, `question_ids`, and `question_types`
  appear in the `longmemeval_s*` AR rows.

## Split Details

### Accurate Retrieval

This split contains 22 examples:

- `ruler_qa1_197K`: 1 example, 100 questions
- `ruler_qa2_421K`: 1 example, 100 questions
- `eventqa_full`: 5 examples, 100 questions each
- `eventqa_65536`: 5 examples, 100 questions each
- `eventqa_131072`: 5 examples, 100 questions each
- `longmemeval_s*`: 5 examples, 60 questions each

Why this split is slightly unusual:

- Most AR sub-datasets use exactly one gold string per question.
- `ruler_qa1_197K` is the exception. Every question there stores 3 or 4 answer
  candidates, and many are duplicated or near-duplicate surface forms.

Examples from `ruler_qa1_197K`:

- `1066` vs `In 1066`
- `Bayeux Tapestry` vs `the Bayeux Tapestry`
- `King Ethelred II` vs `Ethelred II`

Practical implication:

- Evaluation should treat `answers[i]` as a candidate set and accept a
  prediction if it matches any normalized candidate string.
- A small number of `ruler_qa1_197K` candidates are not clean paraphrases, so
  downstream evaluation code should be explicit about normalization and
  tie-breaking rules.

### Test-Time Learning

This split contains 6 examples:

- `recsys_redial_full`: 200 questions
- the remaining 5 sources: 100 questions each

The payload shape is the same as AR, but the semantics are different:

- the `context` encodes demonstrations or prior interaction turns
- the later `questions` test whether the model learned a task from that context
- answers are usually labels, classes, or item identifiers rather than free-form
  spans from the context

The `recsys_redial_full` example is notably large and uses recommendation item
IDs as answers.

### Long-Range Understanding

This split contains 110 examples:

- `infbench_sum_eng_shots2`: 100 examples, 1 question each
- `detective_qa`: 10 examples, between 6 and 10 questions each

This is the most heterogeneous split:

- `infbench_sum_eng_shots2` looks like long-document summarization. The single
  question typically asks for a long summary, and the single answer is a long
  reference summary.
- `detective_qa` is multi-question comprehension over long contexts.

This split is also the one that makes the heaviest use of metadata:

- `demo` provides in-context demonstration material
- `keypoints` stores auxiliary reference points for evaluation or inspection

### Conflict Resolution

This split contains 8 examples. Every example has 100 questions, and the
sources vary by difficulty and context length:

- `factconsolidation_sh_*`: single-hop conflict resolution
- `factconsolidation_mh_*`: multi-hop conflict resolution
- suffixes such as `6k`, `32k`, `64k`, and `262k` indicate different context
  length settings

Compared with AR, the challenge here is not only locating a fact but deciding
which fact is still current after later evidence updates or overrides earlier
statements.

## Practical Usage Notes

### Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset("ai-hyz/MemoryAgentBench")

accurate_retrieval = dataset["Accurate_Retrieval"]
test_time_learning = dataset["Test_Time_Learning"]
long_range_understanding = dataset["Long_Range_Understanding"]
conflict_resolution = dataset["Conflict_Resolution"]
```

### Iterate safely over questions and answers

Because the dataset is row-oriented, most consumers will want to flatten each
row into per-question records:

```python
for row in dataset["Accurate_Retrieval"]:
    for question, answer_candidates in zip(row["questions"], row["answers"], strict=True):
        source = row["metadata"]["source"]
        # Compare your prediction against any normalized candidate in
        # `answer_candidates`.
```

### Recommended evaluation assumption

Treat each `answers[i]` as a set of acceptable gold strings, not as a single
canonical answer. This matters most for `ruler_qa1_197K`, but using the same
rule across all splits keeps the logic uniform.

### Recommended inspection assumption

Treat `metadata` as sparse and source-specific. Code should use defensive access
patterns such as `row["metadata"].get("question_types")` instead of assuming
every field exists for every source.
