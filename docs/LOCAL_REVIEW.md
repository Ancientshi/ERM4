# Local review record — final scope correction

**Static review only; revised code has not been run.** The user's latest instructions prohibit project execution, tests, installation, and model/data downloads, and require preserving the original method semantics.

## State

- Remote: `https://github.com/Ancientshi/ERM4`, default `master`.
- Baseline: `a732ed7eab514d6fe5651a45295f2ec0d551e16e` (2024-12-07).
- Checkout: `/Users/ferdinand/Documents/Codex/2026-10-07/task/ERM4`.
- Branch: `codex/erm4-local-review`. The user subsequently explicitly authorized a normal push to the default branch after static review; the actual publication outcome/commit is recorded in the task report. No public PR, release, deployment, visibility change or personal-homepage publication is authorized by that push.
- Task directory was initially empty; no user working tree was modified. No applicable repository `AGENTS.md` / `.agents/skills` rules found; local memories were not read/changed.
- Original LICENSE unchanged. Original evaluation/Gemma sources, research dependency pins, ten prompts and Bing source retained byte-for-byte.

## Default method and scope correction

The first cleanup draft inferred an in-memory reservoir, replacement trigger and full orchestration. That was broader than structural cleanup. **All inferred implementation is now isolated in `experiments/unverified/`; neither the default core nor cached experiment imports it.** It is explicitly a proposal, not original released source or verified reproduction.

The root component logic keeps the original response parsing, NLI entailment selection (including request-exception-as-contradiction), historical experience JSONL aggregation, scikit-learn cosine/count rule, default tau=0.4 and theta=3, Bing v7 snippet retrieval and reader behavior. Configuration is constructor arguments instead of imported experiment CLI globals. The absent embedding function must be supplied as a callback or produces an explicit missing-dependency error; no embedding implementation/model is inferred. Full-page/BM25 retrieval now fails explicitly before network calls because its upstream implementation was incomplete.

No default standalone reservoir writer, automatic memory update or complete four-module pipeline has been added. Root `main.py` is moved to the experiment entry; there is no newly inferred default root orchestrator. Core use is through `Components.py`. The cached experiment retains original-question NLI and rewritten-question reading, always-on rewriting/filtering, and supplied external snippets; unsupported labels are rejected rather than treated as new ablations.

## Changes that affect behavior, rather than pure moves

- `Components.py`: explicit constructor configuration; lazy sklearn dependency; clearer errors for missing embedding/full-page helpers. Duplicate reader branches are consolidated without changing ordinary prompt/result behavior. Legacy profile/multi-round branches retained but not advertised as ERM4 claims.
- `utils.py`: preserves environment credentials; no global API-key mutation/cache-environment override; explicit timeout and sanitized errors; async adapter delegates to the same Requests transport. It no longer pauses for input or prints provider response bodies. These transport changes are unexecuted repairs, not benchmark evidence.
- `experiments/cached_qa.py` / `config.py`: relocatable paths, input/output guards, explicit API-call opt-in, resume/result-format checks and truthful supported labels. Normal QA data flow follows upstream. Generated reports remain optional experiment output.
- `experiments/shell/*`: portable paths, quoted arguments and local-model/data prerequisites; existing training output directories rejected to avoid overwrite.
- `experiments/unverified/*`: newly inferred behavior, optional only; see its README for differences. Toy smoke uses tau=0.6/theta=1 and makes no claim about paper calibration.

All byte-preserving moves and every changed/new file are enumerated in the task-level `ERM4-review-summary.md` included in the review ZIP.

## Execution and verification

Before the no-run instruction: public-code clone; one original `main.py --help` call; original Python AST/source and shell `bash -n` checks; queries of already installed packages. The first clone attempt failed sandbox DNS; an approved retry succeeded. There were no QA/API/paid calls, model/data downloads, installation, inference, training or data uploads.

After the no-run instruction: only text/Git/source parsing and official web reads; no project import or entry/test execution. Final static evidence covers syntax, module/prompt/shell targets, default exclusion of the proposal, original-file equality, metadata and local links. The review ZIP is an explicitly requested document delivery, not a dataset upload.

| Item | Status |
| --- | --- |
| Official paper metadata and author order | Verified against IOS Press / arXiv |
| Original help and syntax | Passed before no-run; not evidence for revised behavior |
| Revised syntax/path/content checks | Static only; see final evidence file |
| Revised CLI, project imports, smoke, QA/resume | Not run |
| Dependency setup, API/model responses, Gemma/GPU | Not run |
| Full paper metrics, ablations and efficiency | Not run |
| CFF parser/full schema tool | YAML/fields checked; full schema tool not run |
| Credential patterns | Current files and 8 historical commits scanned without printing values; limited coverage |

## Remaining gaps and author approval

1. No standalone reservoir writer, defined embedding helper or complete original four-module pipeline is shipped. No code is presented as completing those gaps by inference.
2. Exact sampled data/search snapshots/adapters and Knowledge Filter training entry are absent; the Gemma service text output is not connected to JSON adapters.
3. Bing v7 is retired; replacing it changes retrieval conditions. Original page/BM25 helpers and standalone Bing `pprint` branch remain incomplete archival source.
4. Upstream original-question filtering, error-as-contradiction and evaluator substring Hit Rate/character-count F1 need an author-reviewed protocol decision. This cleanup preserves them, rather than silently changing scientific behavior.
5. Review constructor/API and transport changes, optional proposal isolation and software-author attribution. Preferred paper citation has six verified authors; root software author attribution needs author confirmation.
6. Only after explicit later approval should dependency setup/offline behavior checks occur. Downloads, paid calls and GPU runs need separate scope/budget decisions. The user has now explicitly approved a normal default-branch push after static review; no force push/history deletion is allowed. PR/release/deployment and personal-homepage changes are outside that authorization.
7. Repository topics/canonical links, reported homepage author typo and periodic discoverability checks remain proposed, not published or automated. SEO/GEO gains are unmeasured.
