# ERM4 — Enhancing Retrieval and Managing Retrieval

Code for **Enhancing Retrieval and Managing Retrieval: A Four-Module Synergy for Improved Quality and Efficiency in RAG Systems**, ECAI 2024, pp. 2258–2265.

Yunxiao Shi, Xing Zi, Zijing Shi, Haimin Zhang, Qiang Wu, Min Xu.

[Paper / DOI](https://ebooks.iospress.nl/doi/10.3233/FAIA240748) · [arXiv:2407.10670](https://arxiv.org/abs/2407.10670) · [BibTeX](citations.bib) · [Citation metadata](CITATION.cff)

ERM4 studies retrieval-augmented generation (RAG) for open-domain question answering: clarified questions and multiple queries, NLI-based knowledge filtering, cached knowledge, and triggering external retrieval.

**Cleanup status: static review only. Revised code has not been run.** The public source provides component implementations and a cached-snippet experiment, with gaps listed below; it does not contain a verified complete four-module reproduction. Default prompt adapters use a selected chat API, whereas the paper uses instruction-tuned Gemma-2B for rewriting and filtering.

## Core interfaces

| Paper module | Available interface / gap |
| --- | --- |
| Query Rewriter+ | `QuestionRewriter.rewrite(question)` |
| Knowledge Filter | `await KnowledgeFilter.filter(question, knowledge)` |
| Memory Knowledge Reservoir | No standalone cache writer/update module is supplied upstream |
| Retrieval Trigger | `RetrievalTrigger.check_retrieval_need(queries)` reads saved experience records; the original embedding helper is missing |

`KnowledgeRetriever` retains the original Bing v7 snippet semantics; that provider is retired. The original full-page/BM25 branch is incomplete and now fails explicitly. `Reader` retains the original reading behavior. No new reservoir, automatic cache write, provider substitution, or complete pipeline is in the default core.

## Minimal use

Python 3.10+. Core dependencies are in `requirements.txt`; install them in a virtual environment only when you choose to run the code. No dependency setup was performed here.

```python
from Components import QuestionRewriter, KnowledgeFilter, RetrievalTrigger, Reader

# Construction reads a local prompt; invoking model-backed methods makes remote calls.
rewriter = QuestionRewriter(
    "experiments/Prompt/question_rewritter_plus_prompt_popqa.txt",
    model_name="YOUR_AVAILABLE_CHAT_MODEL",
)
```

Component configuration is explicit and does not parse experiment CLI arguments on import. Supply original-format experience JSONL and an embedding callback to use the trigger; no embedding model is automatically selected/downloaded. Keys belong in your environment; [.env.example](.env.example) is an example only. Core interfaces are in [`Components.py`](Components.py); the runnable historical experiment entry is [`experiments/cached_qa.py`](experiments/cached_qa.py), separately documented and unexecuted.

## Experiments and limitations

Training, evaluation, dataset prompts and experiment launch scripts live in [`experiments/`](experiments/README.md). The cached demo uses supplied snippets and does not activate memory or the trigger. See [paper/code mapping](docs/PAPER_AND_CODE.md) and [review record](docs/LOCAL_REVIEW.md) for retained behavior and missing assets.

A proposal drafted during this cleanup is quarantined in [`experiments/unverified/`](experiments/unverified/README.md). It is opt-in, never imported by default, has not been run, and is not the released ERM4 implementation or verified reproduction.

## Citation, license and contact

Use the ECAI entry `Shi2024ERM4` in [citations.bib](citations.bib); `CITATION.cff` prefers that paper. [ERAGent, arXiv:2405.06683](https://arxiv.org/abs/2405.06683), is a related earlier draft whose metadata links the same ECAI DOI. Its broader personalization claims are outside this paper's scope; its separate citation is retained.

The existing [CC BY-NC-SA 4.0 license](LICENSE) is unchanged. Project contact: Yunxiao Shi, Yunxiao.Shi@student.uts.edu.au.
