# Optional experiments and reproduction notes

**Static review only: none of the revised entry points, optional proposal smoke script, installation commands, QA runs, training, or model inference below have been executed.** Commands are instructions for a later, explicitly chosen run. Model/data downloads and paid calls were not performed during this cleanup.

The root method is independent of datasets and evaluation. This folder retains the original evaluation and Gemma scripts, all dataset-specific prompts, the original dependency pins, and legacy retrieval helpers. The cached QA demo has been moved from root `main.py` to `cached_qa.py` and repaired by inspection.

## Files

| Path | Purpose / status |
| --- | --- |
| `cached_qa.py`, `config.py` | Prompt-based rewriting/filtering/reading of supplied snippets; no memory or trigger |
| `evaluation.py` | Original scoring code, preserved byte-for-byte |
| `finetune_gemma_rewriter.py` | Original Gemma Rewriter+ LoRA training source, preserved byte-for-byte |
| `infer_gemma_rewriter.py` | Original CUDA/4-bit/FlashAttention Flask service, preserved byte-for-byte |
| `Prompt/` | All ten original prompts, preserved byte-for-byte |
| `shell/` | Relocatable demo and optional GPU recipes with local-path prerequisites |
| `legacy_helpers.py` | Archival webpage, BM25 and local-service material; incomplete optional paths; original Bing source remains at root |
| `unverified/` | Quarantined inferred cache/trigger/composition proposal and its toy smoke script; not default core, not run |
| `examples/smoke.jsonl` | Newly authored synthetic schema example; not benchmark data |
| `requirements.txt` | Original research package list; preserved byte-for-byte, not a complete GPU environment |
| `requirements-demo.txt` | Five packages used by the cached demo and original evaluator |

## Small local checks, if run later

Only the cached-demo local validation commands below correspond to the default source. The optional proposal and its toy smoke script are described separately in [unverified/README.md](unverified/README.md); they are excluded from the default implementation and have not been run.

```bash
python experiments/cached_qa.py --check --data_path experiments/examples/smoke.jsonl
bash experiments/shell/ERM4.sh --check --data_path experiments/examples/smoke.jsonl
```

`--check` inspects local prompts/data and existing results without API calls or writes. It requires only the Python standard library. Without `--allow_api_calls`, the cached demo refuses remote LLM execution. Neither a model nor a data file is downloaded by the demo.

## Cached-snippet QA demo, if chosen later

1. Use a fresh virtual environment with Python 3.10+, then install `experiments/requirements-demo.txt`. The exact pinned environment has not been installed or verified here.
2. Provide local JSONL at `Records/demo/data_<id>.jsonl`, or pass `--data_path`. Dataset IDs are CAmbigNQ=1, ambignq=2, nq=3, popqa=4, hotpot=5, 2wikimqa=6. The original repository links [demo/training material on Google Drive](https://drive.google.com/drive/folders/1UYkFJqfuNbJJZUad-psssL4uSn4ttuAY?usp=sharing). Availability, contents, and licensing of those files were not checked; nothing was downloaded.
3. Each row must contain `original_question` (nonempty string), `answer` (nonempty list of strings), and `external_knowledge` (title-to-text object). See `examples/smoke.jsonl` for the schema only.
4. Set `OPENAI_API_KEY` in your shell. [.env.example](../.env.example) is an example, not an automatically loaded config. Rewriting and filtering use OpenAI; `--use_silicon_flow` switches only the reader and additionally needs `SILICONFLOW_API_KEY`.
5. Validate first. A live run is a separate choice and may incur charges:

```bash
python experiments/cached_qa.py --dataset popqa --check
# Optional live run; not executed in this task:
python experiments/cached_qa.py --dataset popqa --exp_name my_run --allow_api_calls --model_name YOUR_AVAILABLE_CHAT_MODEL
```

Results append to `Records/demo/data_<id>_<dataset>_<exp_name>.jsonl`, or `--output_path`; answered question strings are skipped on resume. Use a new output/experiment name when changing models or settings. Reports append under `Records/`. Do not use an input file as the output. Generated results are ignored by Git.

The cached demo retains upstream always-on rewriting and filtering; options only accept the matching `rewriter+` / `filter` labels. The former source logged variant flags without changing execution, so unsupported combinations are now rejected rather than treated as implemented ablations. Historical `page_num`, `device`, `maxlength`, BM25, and similarity/popularity options remain unused metadata in this cached demo. Seeds do not guarantee reproducible remote model responses.

To preserve the upstream experiment semantics, the filter still receives the **original** question, and the reader receives the joined rewritten question. The upstream filter class also still treats model-request exceptions as `contradiction`; this known protocol limitation is retained for author review, not silently replaced by a new method. Invalid result formats and missing paths now fail explicitly before recording a question.

## Gemma and full-paper reproduction

The paper uses Gemma-2B LoRA modules, an external retriever and a historical GPT-3.5 snapshot. The cached demo uses prompt-based API modules and pre-retrieved snippets. Its Gemma Flask service is not connected to the core adapters or cached demo; it emits a textual format rather than the JSON consumed by those adapters. No dedicated Knowledge Filter fine-tuning entry is included.

The retained GPU scripts assume CUDA and 4-bit quantization; inference also explicitly requests FlashAttention 2. The original requirements omit the needed `bitsandbytes` and `flash-attn` dependencies; their CUDA/PyTorch compatibility needs a separately verified environment. These scripts have not been tested on this Mac, and no weights/adapters are bundled. They are not a Mac quickstart.

The shell recipes now require existing local paths via `ERM4_BASE_MODEL`, `ERM4_TRAIN_DATA`, `ERM4_VAL_DATA`, `ERM4_OUTPUT_DIR` (training), or `ERM4_LORA_WEIGHTS` (inference); use a new training output directory. Review the source and hardware environment before using them. The original Python sources can still download from a model/dataset identifier if called directly; no such calls were made here.

Bing Search v7 is archival: [Microsoft retired Bing Search APIs on 11 August 2025](https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement). A current retrieval provider would change the reproduction conditions and must be documented separately. The legacy page/BM25/embedding helpers have undefined or commented-out dependencies, and root `bing.py`'s standalone branch references `pprint` without importing it; these paths remain unverified research material.

Full reproduction needs the exact data samples, search snapshots, trained adapters, prompts, model versions and a verified evaluation protocol. They are not supplied by this structural cleanup. The retained evaluator uses substring Hit Rate and a nonstandard, character-count F1 after answer rectification; average length is `None`. Its values should not be represented as independently verified paper metrics.

[Paper-to-code map and reported results](../docs/PAPER_AND_CODE.md) · [Review record and remaining blockers](../docs/LOCAL_REVIEW.md)
