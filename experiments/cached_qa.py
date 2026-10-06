"""Prompt-based QA using pre-retrieved snippets; not the full four-module pipeline."""
import asyncio
import json
import os
from pathlib import Path
import random
import time

import sys

# Allow both python experiments/cached_qa.py and python -m experiments.cached_qa.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.config import args

DATASET_IDS = {'CAmbigNQ': '1', 'ambignq': '2', 'nq': '3', 'popqa': '4', 'hotpot': '5', '2wikimqa': '6'}


def get_data():
    path = Path(args.data_path) if args.data_path else Path(args.root_path) / f'Records/demo/data_{DATASET_IDS[args.dataset]}.jsonl'
    if not path.is_file():
        raise ValueError(f'Data file not found: {path}. See experiments/README.md or use --data_path.')
    dataset = []
    with path.open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                valid = (
                    isinstance(row, dict)
                    and isinstance(row.get('original_question'), str)
                    and bool(row['original_question'].strip())
                    and isinstance(row.get('answer'), list)
                    and bool(row['answer'])
                    and all(isinstance(a, str) and a.strip() for a in row['answer'])
                    and isinstance(row.get('external_knowledge'), dict)
                    and all(isinstance(k, str) and isinstance(v, str) for k, v in row['external_knowledge'].items())
                )
                if not valid:
                    raise ValueError('expected original_question: string, answer: nonempty string list, external_knowledge: title-to-text object')
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid data at {path}:{number}: {error}') from None
            dataset.append(row)
    if not dataset:
        raise ValueError(f'Data file is empty: {path}')
    return dataset


def get_result():
    path = Path(args.output_path) if args.output_path else Path(args.root_path) / f'Records/demo/data_{DATASET_IDS[args.dataset]}_{args.dataset}_{args.exp_name}.jsonl'
    result = []
    if path.exists():
        with path.open(encoding='utf-8') as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('original_question'), str):
                    raise ValueError(f'Invalid result at {path}:{number}')
                response = row.get('ERM4')
                if not isinstance(response, list) or not response or not all(isinstance(a, str) for a in response):
                    raise ValueError(f'Invalid ERM4 answer list at {path}:{number}')
                result.append(row)
    return result, str(path)


def validate_inputs():
    # These choices were previously logged without affecting execution.
    if args.trigger != 'none' or args.retriever != 'retriever' or args.reader != 'reader' or args.rewriter != 'rewriter+' or args.filter != 'filter':
        raise ValueError('This cached demo always rewrites and filters; supports --rewriter rewriter+ --filter filter --trigger none --retriever retriever --reader reader only.')
    for path in (args.question_rewritter_prompt, args.knowledge_filter_prompt, args.reader_prompt):
        if not Path(path).is_file():
            raise ValueError(f'Prompt file not found: {path}')
    dataset = get_data()
    _, output_path = get_result()
    input_path = Path(args.data_path) if args.data_path else Path(args.root_path) / f'Records/demo/data_{DATASET_IDS[args.dataset]}.jsonl'
    if input_path.resolve() == Path(output_path).resolve():
        raise ValueError('Input and output paths must differ.')
    return dataset


def answer_question():
    dataset = validate_inputs()
    if not args.allow_api_calls:
        raise ValueError('Remote LLM calls are disabled. Use --check for local validation; --allow_api_calls enables potentially paid requests.')
    required = {'SILICONFLOW_API_KEY' if args.use_silicon_flow else 'OPENAI_API_KEY'}
    required.add('OPENAI_API_KEY')
    for name in sorted(required):
        if not os.environ.get(name):
            raise ValueError(f'Set {name} in the environment; keys are never stored in this repository.')

    from Components import QuestionRewriter, KnowledgeFilter, Reader
    from tqdm import tqdm
    random.seed(args.seed)
    rewriter = QuestionRewriter(args.question_rewritter_prompt, model_name=args.model_name)
    knowledge_filter = KnowledgeFilter(args.knowledge_filter_prompt, model_name=args.model_name)
    reader = Reader(args.reader_prompt, model_name=args.model_name, silicon_flow_qa=args.use_silicon_flow)
    results, result_path = get_result()
    answered = {row['original_question'] for row in results}
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    for row in tqdm(dataset):
        question = row['original_question']
        if question in answered:
            continue
        rewritten = ' '.join(rewriter.rewrite(question)[0])
        knowledge = row['external_knowledge']
        # Preserve the upstream demo: NLI receives the original question.
        knowledge = asyncio.run(knowledge_filter.filter(question, knowledge))
        response = reader.read(rewritten, knowledge)
        if not isinstance(response, list) or not response or not all(isinstance(a, str) and a.strip() for a in response):
            raise ValueError('Reader must return a nonempty JSON array of answer strings; no result was written for this question.')
        output = dict(row, ERM4=response)
        with open(result_path, 'a', encoding='utf-8') as target:
            target.write(json.dumps(output, ensure_ascii=False) + '\n')
        answered.add(question)
    print(f'Done: {result_path}')
    return True


def eval(key='ERM4'):
    from experiments.evaluation import my_eval_question_answering
    _, result_path = get_result()
    em, length, precision, recall, f1, hit_rate = my_eval_question_answering(result_path, key)
    report = f'Exact Match: {em}; Precision: {precision}; Recall: {recall}; F1: {f1}; Hit Rate: {hit_rate}; Avg.Length: {length}'
    print(report)
    record_dir = Path(args.root_path) / 'Records'
    record_dir.mkdir(parents=True, exist_ok=True)
    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with (record_dir / 'all_results.txt').open('a', encoding='utf-8') as target:
        target.write(f'{args.exp_name}_{key}_{args.dataset}_{local_time}:\n{report}\n')
    with (record_dir / 'report.csv').open('a', encoding='utf-8') as target:
        target.write(f'{args.exp_name}_{key},{args.dataset},{args.rewriter},None,{args.trigger},{args.retriever},{args.filter},{args.reader},None,{em},{precision},{recall},{f1},{hit_rate}\n')


if __name__ == '__main__':
    try:
        if args.check:
            print(f'Validated {len(validate_inputs())} question(s); no API calls or result writes.')
        else:
            answer_question()
            eval()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
