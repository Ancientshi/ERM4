import argparse
import math

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

parser = argparse.ArgumentParser(description='E-RAGent Framework')
parser.add_argument('--root_path', default='''fill you path to this project''', type=str, help='The absolute path of this project.')
parser.add_argument('--question_rewritter_prompt', default='''question_rewritter_prompt.txt''', type=str, help='The prompt for query rewritter.')
parser.add_argument('--knowledge_filter_prompt', default='''knowledge_filter_prompt.txt''', type=str, help='The prompt for knowledge filter.')
parser.add_argument('--reader_prompt', default='''reader_prompt.txt''', type=str, help='The prompt for LLM reader.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')
parser.add_argument('--exp_name', default='default', type=str, help='The name of the experiment.')
parser.add_argument('--page_num', default=5, type=int, help='The number of pages to be retrieved.')
parser.add_argument('--device', default=0, type=int, help='GPU device.')
parser.add_argument('--dataset', default='popqa', type=str, help='The name of the dataset.')
parser.add_argument('--maxlength', default=2048, type=int, help='The max length of page content.')
parser.add_argument('--cosine_similarity_threshold', default=0.4, type=float, help='The threshold of cosine similarity.')
parser.add_argument('--popularity_threshold', default=3, type=int, help='The threshold of popularity.')
parser.add_argument('--bm25_threshold', default=2, type=int, help='The threshold of bm25.')

parser.add_argument('--rewriter', default='none', type=str, help='choose in {none,rewriter,rewriter+}')
parser.add_argument('--trigger', default='none', type=str, help='choose in {none,trigger}')
parser.add_argument('--retriever', default='retriever', type=str, help='choose in {none,retriever,retriever+}')
parser.add_argument('--filter', default='none', type=str, help='choose in {none,filter}')
parser.add_argument('--reader', default='reader', type=str, help='choose in {reader,reader+}')
parser.add_argument('--model_name', default='gpt-3.5-turbo', type=str, help='LLM model name.')
parser.add_argument('--use_silicon_flow', action='store_true', help='Whether to use silicon flow.')

args = parser.parse_args()
print_args(args)