# GSMedQA: Graph-Structured Prompting with Multimodal Priors for Medical Visual Question Answering

Official implementation and released experimental artifacts for **GSMedQA**, a zero-shot medical visual question answering framework based on graph-structured prompting and multimodal domain priors.

> **Note**: The codebase uses the internal name `GSPMed` in several scripts and output files. `GSMedQA` and `GSPMed` refer to the same project in this repository.

## Highlights

- Zero-shot medical visual question answering without task-specific fine-tuning.
- Graph-structured reasoning over four complementary multimodal priors.
- Support for VQA-RAD, SLAKE, and PathVQA.
- Expert-guided reasoning-path evaluation and hierarchical voting.
- Released intermediate priors, raw reasoning outputs, and prediction files.
- Optional profiling of API calls, token usage, and latency.

## Method

GSMedQA organizes four types of pretrained multimodal priors into a multi-stage reasoning process:

1. **Image descriptions** generated from medical images.
2. **Knowledge scopes** retrieved as image- and question-related concepts.
3. **Candidate answers** obtained from a vision-language label space.
4. **Problem decomposition** represented by fine-grained sub-questions and answers.

The LLM progressively reasons over these priors. The resulting predictions are aggregated with domain-expert outputs through node-level and layer-level voting.

```text
Multimodal priors
       |
       v
Caption -> Knowledge -> Candidate -> Sub-question
       |
       v
Multi-branch LLM reasoning
       |
       v
Expert-guided voting
       |
       v
Final prediction
```

## Repository Structure

```text
.
├── agents/
│   ├── agent.py                   # OpenAI-compatible caller and model wrappers
│   └── prompt.py                  # System prompts
├── classified_caption/            # Image-description priors
├── keyword/                       # Knowledge-scope priors
├── keyword_candidate/             # Candidate answers and dataset metadata
├── subquestion/                   # Decomposed questions and answers
├── outputs/                       # Raw multi-stage reasoning outputs
├── result_csv/                    # Expert and GSMedQA predictions
├── result_csv_gspmed/             # Self-consistency and development outputs
├── llm_agents.py                  # Main inference entry point
├── llm_agents_0129.py             # Inference with cost and latency profiling
├── gspmed2csv.py                  # Expert voting and CSV export
└── gspmed_evidence_consistency.py # GSMedQA-only self-consistency voting
```

## Installation

We recommend Python 3.10 or 3.11.

```bash
git clone <repository-url>
cd GSPMed_code

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openai python-dotenv pydantic torch pandas numpy transformers swarms tiktoken
```

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The current implementation uses an OpenAI-compatible endpoint configured in `llm_agents.py` and `llm_agents_0129.py`. Update `base_url` if a different provider is used.

Do not commit API keys, account names, passwords, or `.env` files. Credentials that were previously stored in an older README should be rotated before reuse.

## Data Preparation

The released directory already contains the precomputed priors used by the inference pipeline.

| Prior | Location |
|---|---|
| Image descriptions | `classified_caption/<dataset>/` |
| Knowledge scopes | `keyword/*.json` |
| Candidate answers and metadata | `keyword_candidate/*.json` |
| Sub-question decomposition | `subquestion/<dataset>/` |

Supported dataset identifiers are:

| Dataset | Inference argument | Post-processing argument |
|---|---|---|
| VQA-RAD | `radvqa` | `rad` |
| SLAKE | `slake` | `slake` |
| PathVQA | `pathvqa` | `pvqa` |

The released metadata contains 451 VQA-RAD examples, 1,061 SLAKE examples, and 6,761 PathVQA examples.

The scripts consume precomputed priors rather than raw medical images. Generation code for the four prior sources is not included in the current release.

## Inference

Run the four-stage reasoning pipeline from the repository root:

```bash
python llm_agents.py radvqa outputs/predictions_GSP_rad_new.json
python llm_agents.py slake outputs/predictions_GSP_slake_new.json
python llm_agents.py pathvqa outputs/predictions_GSP_pathvqa_new.json
```

Inference results are saved incrementally. Samples already present in the target JSON file are skipped, allowing an interrupted run to resume.

Full-dataset inference may issue a large number of paid API requests. We recommend running a small subset first and writing new results to a separate output file.

### Runtime Profiling

Use the profiling entry point to record call counts, approximate token usage, and latency:

```bash
python llm_agents_0129.py \
  radvqa \
  outputs/radvqa_new.json \
  outputs/radvqa_new_profile.json
```

## Expert-Guided Voting

Convert raw reasoning outputs into final prediction CSV files:

```bash
python gspmed2csv.py rad
python gspmed2csv.py slake
python gspmed2csv.py pvqa
```

The input and output paths are currently configured in the `__main__` block of `gspmed2csv.py`. When processing a new inference run, update `json2_path` to point to the new output file.

The released configuration has several path conventions worth noting:

- The complete VQA-RAD reasoning output is `outputs/predictions_GSP_rad.json`.
- The default VQA-RAD path in `gspmed2csv.py` points to the smaller development file `outputs/radvqa.json` and should be changed for a full run.
- VQA-RAD results are written to `result_csv_gspmed/`, while SLAKE and PathVQA results are written to `result_csv/`.
- VQA-RAD and SLAKE use M-Mixup and CH-ICL expert predictions; the released PathVQA configuration uses VQAMix and CH-ICL.

## Self-Consistency

The repository also provides an optional GSMedQA-only majority-voting baseline that does not use external expert predictions:

```bash
python gspmed_evidence_consistency.py rad
python gspmed_evidence_consistency.py slake
python gspmed_evidence_consistency.py pvqa
```

Outputs are written as:

```text
result_csv_gspmed/predictions_GSPMed_<level>_<dataset>_selfcons.csv
```

This is an auxiliary experiment and is not a replacement for the expert-guided RPEM results reported in the main paper.

## Results

The following values are reported in the paper for the graph-structured model, **Ours-G**:

| Dataset | Closed | Open | Overall |
|---|---:|---:|---:|
| VQA-RAD | 81.62 | 68.16 | 76.27 |
| SLAKE | 85.82 | 86.36 | 86.15 |
| PathVQA | 88.00 | 41.35 | 66.45 |

Using normalized exact matching for closed-ended questions and bidirectional substring matching for open-ended questions, the released VQA-RAD and SLAKE CSV files reproduce the corresponding Ours-G values.

The currently released PathVQA prediction file reaches 88.00 closed, 36.56 open, and 62.36 overall under the same evaluation rule. It therefore does not reproduce the final PathVQA number in the paper. The final PathVQA prediction artifact or its exact post-processing configuration still needs to be added.

## Reproducibility Notes

- All Python source files pass syntax validation, but the repository does not yet include an automated end-to-end test.
- The LLM temperature is set to `0.4`; repeated API runs are not guaranteed to be identical.
- The paper describes explicit DFS traversal and path pruning. The released code realizes the reasoning process through nested branches and hierarchical voting, but does not serialize an explicit graph or path-level pruning trace.
- The concatenated-prior baseline, `Ours-P`, is not provided as a separate runnable entry point.
- The seven prior-ablation settings reported in the paper are not exposed through a configuration or command-line flag.
- A standalone evaluation script and the qualitative reasoning-graph visualization script are not included.
- `preidictions_*.csv` is a legacy misspelling retained because existing scripts refer to those filenames.
- `new_code_20260129/llm_agents_0129.py` duplicates the profiling script in the repository root. The root-level version should be treated as the primary entry point.

## Citation

If you find this project useful, please cite the paper. Replace the placeholder below with the final BIBM 2026 bibliographic record when it becomes available.

```bibtex
@inproceedings{guo2026gsmedqa,
  title     = {GSMedQA: Graph-Structured Prompting with Multimodal Priors for Medical Visual Question Answering},
  author    = {Guo, Yuting and Yang, Zhenyu and Tan, Lixing and Yan, Yujing and Song, Wenfeng and Hou, Xia},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine},
  year      = {2026}
}
```

## Acknowledgements

This project builds on pretrained language and vision-language models, including LLaVA-Med, BioMedCLIP, M-Mixup, CH-ICL, and related MedVQA research. We thank their authors for making their work available to the community.

