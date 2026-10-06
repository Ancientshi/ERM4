# Paper identity, method mapping, and reported results

Primary sources: [official IOS Press chapter](https://ebooks.iospress.nl/doi/10.3233/FAIA240748), [ERM4 arXiv record](https://arxiv.org/abs/2407.10670), and [paper full text](https://arxiv.org/html/2407.10670v1). Source review: 2026-10-06. The title, author order, year, pages, volume and DOI in `citations.bib` and `CITATION.cff` follow the publisher record.

- Title: Enhancing Retrieval and Managing Retrieval: A Four-Module Synergy for Improved Quality and Efficiency in RAG Systems.
- Authors, in order: Yunxiao Shi; Xing Zi; Zijing Shi; Haimin Zhang; Qiang Wu; Min Xu.
- ECAI 2024; Frontiers in Artificial Intelligence and Applications, volume 392; IOS Press; pp. 2258–2265; DOI 10.3233/FAIA240748.
- ERM4 preprint: arXiv:2407.10670, submitted 15 July 2024.

## Paper-to-code map

| Paper | Default source | Limits |
| --- | --- | --- |
| §3.1 Query Rewriter+ | `QuestionRewriter` | Original prompt response parsing; chat adapter, not tuned Gemma |
| §3.2 Knowledge Filter | `KnowledgeFilter` | Original entailment-only selection, including error-as-contradiction fallback |
| §3.3 Memory Knowledge Reservoir | Saved `filtered knowledge` entries read by the trigger | No standalone writer/update module or complete cache workflow supplied upstream |
| §3.4 Retrieval Trigger | `RetrievalTrigger` | Original experience-file aggregation and scikit-learn cosine/popularity comparison; missing embedding helper must be supplied explicitly |
| Basic retriever / reader | `KnowledgeRetriever`, `Reader` | Original Bing snippet/reader behavior; incomplete full-page/BM25 path fails explicitly |
| QA experiment | `experiments/cached_qa.py` | Original always-on prompt rewriting/filtering with supplied snippets; no memory/trigger calls |

The default trigger keeps upstream thresholds tau=0.4 and theta=3; no calibration or provider/embedding model is inferred. It reads the same legacy `ExperienceSave/<exp_name>/experience_pool_<user_id>.jsonl` format, aggregates `filtered knowledge` title/content pairs in record order, counts titles with cosine similarity >= tau, and requests external retrieval when count < theta. It does not write or update memory. Its absent embedding helper now produces an explicit missing-dependency error unless a caller supplies the dependency.

A first cleanup draft inferred a standalone in-memory reservoir and full composition. Following scope review, all of that code was removed from the default path and quarantined in `experiments/unverified/`, together with its generic prompts, replacement trigger/search adapter and toy smoke script. The optional proposal changes implementation/retrieval behavior and is **not** a verified paper implementation. See its [explicit difference list](../experiments/unverified/README.md).

All ten original prompts remain unchanged in `experiments/Prompt/`; no new generic prompt becomes the default. The cached experiment preserves the upstream choice to filter with the original question and read with the rewritten question. Legacy reader profile/multi-round arguments are retained for compatibility only, not advertised as ERM4-validated personalization.

## Reported quality results, not locally reproduced

Selected rows from [Table 1](https://arxiv.org/html/2407.10670v1#S4): Direct compared with Rewriter+-Retriever-Filter-Reader. Values are the paper's F1 and Hit Rate (%) for the sampled QA setting, not a universal accuracy gain.

| Dataset | Direct F1 | Method F1 | Direct Hit Rate | Method Hit Rate |
| --- | ---: | ---: | ---: | ---: |
| CAmbigNQ | 37.38 | 43.39 | 55.67 | 64.33 |
| NQ | 41.50 | 52.68 | 42.00 | 51.33 |
| PopQA | 35.24 | 41.77 | 42.33 | 51.33 |
| AmbigNQ | 45.21 | 53.47 | 46.67 | 55.67 |
| HotPot | 46.33 | 57.59 | 41.33 | 50.00 |
| 2WikiMQA | 41.85 | 46.83 | 42.33 | 49.33 |

## Reported efficiency setting

[§6 / Table 2](https://arxiv.org/html/2407.10670v1#S6) uses 100 AmbigNQ questions to populate memory, then 200 semantically similar questions; retriever n=5 and popularity threshold theta=3. At tau=1.0 versus tau=0.6:

| Measure | tau=1.0 | tau=0.6 | Interpretation |
| --- | ---: | ---: | --- |
| Mean time per question | 7.45 s | 3.97 s | About 47% lower latency (46.71%, calculated from table) |
| External knowledge instances | 15.00 | 4.39 | About 71% fewer instances (70.73%, calculated) |
| Hit Rate | 53.5% | 53.0% | 0.5 percentage-point decrease |

These are experiment-specific counts/times, not demonstrated monetary/API cost savings or a guarantee of unchanged quality. The old README's “from 14% to 21%” accuracy wording has been replaced by dataset-specific metrics. No benchmark claim is made for this local revision.

## ERAGent relationship

[ERAGent, arXiv:2405.06683](https://arxiv.org/abs/2405.06683), submitted 6 May 2024, lists the same six authors and calls itself a draft. Its current arXiv metadata explicitly links DOI 10.3233/FAIA240748 as a related resource and lists ECAI 2024, volume 392. It covers additional experiential learning and personalization. This supports treating it as a related earlier draft; the metadata does not justify applying every broader claim to the ECAI ERM4 paper. Its original separate BibTeX key is preserved.

The user supplied a follow-up observation that a personal homepage lists “Qiang Xu.” The publisher and both arXiv author lists say **Qiang Wu**. The homepage was not independently inspected, edited, or published in this task; correcting it is a separate author-approved action.
