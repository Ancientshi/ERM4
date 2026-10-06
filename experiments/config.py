import argparse
from pathlib import Path

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def print_args(args):
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description='ERM4 cached-knowledge QA demo (ECAI 2024)')
parser.add_argument('--root_path', default=str(PROJECT_ROOT), type=str, help='Project/data root; defaults to this checkout.')
parser.add_argument('--question_rewritter_prompt', default=None, type=str, help='Query Rewriter+ prompt path (legacy spelling retained).')
parser.add_argument('--knowledge_filter_prompt', default=None, type=str, help='Knowledge Filter prompt path.')
parser.add_argument('--reader_prompt', default=None, type=str, help='LLM reader prompt path.')
parser.add_argument('--data_path', default=None, help='Input JSONL; defaults to Records/demo/data_<dataset id>.jsonl.')
parser.add_argument('--output_path', default=None, help='Result JSONL; existing answered questions are resumed.')
parser.add_argument('--check', action='store_true', help='Validate local data and prompts, then exit without API calls.')
parser.add_argument('--allow_api_calls', action='store_true', help='Explicitly enable remote LLM requests, which may incur charges.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')
parser.add_argument('--exp_name', default='default', type=str, help='The name of the experiment.')
parser.add_argument('--page_num', default=5, type=int, help='The number of pages to be retrieved.')
parser.add_argument('--device', default=0, type=int, help='GPU device.')
parser.add_argument('--dataset', default='popqa', choices=['CAmbigNQ','ambignq','nq','popqa','hotpot','2wikimqa'], help='Dataset name; selects default data and prompts.')
parser.add_argument('--maxlength', default=2048, type=int, help='The max length of page content.')
parser.add_argument('--cosine_similarity_threshold', default=0.4, type=float, help='The threshold of cosine similarity.')
parser.add_argument('--popularity_threshold', default=3, type=int, help='The threshold of popularity.')
parser.add_argument('--bm25_threshold', default=2, type=int, help='The threshold of bm25.')

parser.add_argument('--rewriter', default='rewriter+', choices=['rewriter+'], help='Cached demo always applies its selected rewrite prompt, not Gemma.')
parser.add_argument('--trigger', default='none', type=str, help='choose in {none,trigger}')
parser.add_argument('--retriever', default='retriever', type=str, help='choose in {none,retriever,retriever+}')
parser.add_argument('--filter', default='filter', choices=['filter'], help='Cached demo always applies its prompt-based NLI filter.')
parser.add_argument('--reader', default='reader', type=str, help='choose in {reader,reader+}')
parser.add_argument('--model_name', default='gpt-3.5-turbo', type=str, help='LLM model name.')
parser.add_argument('--use_silicon_flow', action='store_true', help='Whether to use silicon flow.')

args = parser.parse_args()
args.root_path = str(Path(args.root_path).expanduser().resolve())
if args.question_rewritter_prompt is None:
    args.question_rewritter_prompt = str(PROJECT_ROOT / 'experiments' / 'Prompt' / f'question_rewritter_plus_prompt_{args.dataset}.txt')
if args.knowledge_filter_prompt is None:
    args.knowledge_filter_prompt = str(PROJECT_ROOT / 'experiments' / 'Prompt' / 'knowledge_filter_prompt.txt')
if args.reader_prompt is None:
    args.reader_prompt = str(PROJECT_ROOT / 'experiments' / 'Prompt' / ('reader_prompt_hotpot.txt' if args.dataset == 'hotpot' else 'reader_prompt.txt'))
