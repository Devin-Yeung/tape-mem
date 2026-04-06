# Accurate Retrieval: Context Size & Scalability

This page documents the context size and scalability characteristics of the
**Accurate Retrieval (AR)** split in MemoryAgentBench.

## Context Size Overview

| Source          | Questions | Answers |     Chars |       Tokens |
| --------------- | --------: | ------: | --------: | -----------: |
| ruler_qa1_197K  |       100 |     100 |   985,698 | 202 K tokens |
| ruler_qa2_421K  |       100 |     100 | 1,881,772 | 430 K tokens |
| eventqa_full    |       100 |     100 | 2,306,899 | 536 K tokens |
| eventqa_full    |       100 |     100 | 3,171,853 | 736 K tokens |
| eventqa_full    |       100 |     100 | 2,591,136 | 602 K tokens |
| eventqa_full    |       100 |     100 | 1,694,961 | 403 K tokens |
| eventqa_full    |       100 |     100 | 1,957,500 | 450 K tokens |
| eventqa_65536   |       100 |     100 |   285,324 |  66 K tokens |
| eventqa_65536   |       100 |     100 |   285,199 |  66 K tokens |
| eventqa_65536   |       100 |     100 |   279,622 |  66 K tokens |
| eventqa_65536   |       100 |     100 |   272,813 |  66 K tokens |
| eventqa_65536   |       100 |     100 |   282,936 |  66 K tokens |
| eventqa_131072  |       100 |     100 |   573,520 | 131 K tokens |
| eventqa_131072  |       100 |     100 |   567,840 | 131 K tokens |
| eventqa_131072  |       100 |     100 |   569,210 | 131 K tokens |
| eventqa_131072  |       100 |     100 |   548,401 | 131 K tokens |
| eventqa_131072  |       100 |     100 |   566,706 | 131 K tokens |
| longmemeval_s\* |        60 |      60 | 1,600,183 | 355 K tokens |
| longmemeval_s\* |        60 |      60 | 1,589,693 | 353 K tokens |
| longmemeval_s\* |        60 |      60 | 1,715,268 | 384 K tokens |
| longmemeval_s\* |        60 |      60 | 1,588,305 | 354 K tokens |
| longmemeval_s\* |        60 |      60 | 1,646,919 | 362 K tokens |

## Scalability Tiers

The AR split covers a wide range of context lengths, from ~66 K tokens up to ~736 K tokens:

| Tier        | Sources                      | Context Range    |
| ----------- | ---------------------------- | ---------------- |
| Small       | eventqa_65536                | 66 K tokens      |
| Medium      | eventqa_131072               | 131 K tokens     |
| Large       | ruler_qa1_197K               | 202 K tokens     |
| X-Large     | ruler_qa2_421K               | 430 K tokens     |
| Ultra-Large | eventqa_full (5 examples)    | 403–736 K tokens |
| Extended    | longmemeval_s\* (5 examples) | 353–384 K tokens |

## Key Observations

- **Fixed question counts**: Each source maintains a consistent number of questions
  per example (100 for most sources, 60 for longmemeval_s\*).
- **Variable context lengths**: Even within the same source (e.g., eventqa_full),
  context sizes vary significantly, indicating different document lengths in the
  underlying data.
- **Token density**: The character-to-token ratio is approximately 4.8–4.9x,
  consistent with English text.
