import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
from rouge import Rouge
import concurrent.futures
import re

def hits(ans, res):
    n = 0
    res = normalize_answer(res)
    for a in ans:
        a = normalize_answer(a)
        n += res.count(a)
    return n

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(s, ' ', text) # dont remove a for mmlu

    def white_space_fix(text):
        if type(text) == list:
            return [white_space_fix(t) for t in text]
        else:
            cleaned_text = re.sub(r'\s+', ' ', text)
            return cleaned_text

    #Remove punctuation from text.
    def remove_punc(text):
        if type(text) == list:
            return [remove_punc(t) for t in text]
        else:
            punctuation = string.punctuation
            
            translator = str.maketrans('', '', punctuation)
            text=str(text)
            cleaned_text = text.translate(translator)
            
            return cleaned_text

    def lower(text):
        if type(text) == list:
            return [t.lower() for t in text]
        else:
            text=str(text)
            return text.lower()


    s=lower(s)
    s=remove_punc(s)
    s=white_space_fix(s)
    return s


def exact_match_score(prediction, ground_truth):
    if type(ground_truth) == list: #physics
        ground_truth = ','.join(ground_truth)
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):     
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def my_ems(prediction, ground_truths):
    answers=[normalize_answer(answer) for answer in ground_truths]
    outputs=[normalize_answer(output) for output in prediction]
    
    pairs=[]
    for answer in answers:
        for output in outputs:
            pairs.append((answer,output))

    exact_match_score_list=[exact_match_score(i, j) for i,j in pairs]
    if len(exact_match_score_list)==0:
        return 0
    else:
        return max(exact_match_score_list)

def my_f1(prediction, ground_truth):
    if type(ground_truth) == list: 
        ground_truth = ','.join(ground_truth)
    if type(prediction) == list: 
        prediction = ','.join(prediction)
    prediction_tokens = normalize_answer(prediction)
    ground_truth_tokens = normalize_answer(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0,0,0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def f1_score(prediction, ground_truth):
    if type(ground_truth) == list: #physics
        ground_truth = ','.join(ground_truth)
    prediction_tokens = normalize_answer(prediction)
    ground_truth_tokens = normalize_answer(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


def eval_recall(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def eval_question_answering(infile, end=None):

    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    f1_scores = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0] if line['output'] else ''
        if end:
            output = output.split(end)[0]
            output = output.split('\n')[0] # added 

        if ems(output, answer): # EM evaluation
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

        f1_scores.append(f1(output, answer))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)
    F1 = round(np.mean(f1_scores), 4)
    em_f1 = round(f1_scores.count(1)/len(lines), 4)
    print(exact_match_count, len(lines))
    print(em, F1, em_f1)
    return em, lens, F1

def hit(lines,key='response'):

    hit_count = 0
    for line in lines:
        line = json.loads(line)
        answers = line['answer']
        outputs = line[key]

        answers=[normalize_answer(answer) for answer in answers]
        outputs=[normalize_answer(output) for output in outputs]
        
        outputs_str=','.join(outputs)
        for answer in answers:
            if answer in outputs_str:
                hit_count+=1
                break
    
    hit_rate = round(hit_count/len(lines), 4)
    return hit_rate
            
            
def my_eval_question_answering(infile,key='response'):
    lines = open(infile, 'r').readlines()
    
    hit_rate = hit(lines,key)
    exact_match_count = 0
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line[key]
        answer=[normalize_answer(answer) for answer in answer]
        output=[normalize_answer(output) for output in output]
        answer=[a for a in answer if len(a)>0]
        output=[o for o in output if len(o)>0]
        
        rectified_output=[]
        for a in answer:
            for o in output:
                if a in o:
                    rectified_output.append(a)
                else:
                    rectified_output.append('None')
        output=rectified_output

        if my_ems(output, answer): # EM evaluation
            exact_match_count += 1
        
        precision, recall, f1 = my_f1(output, answer)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    em = round(exact_match_count/len(lines), 4)
    Precision=round(np.mean(precision_scores), 4)
    Recall=round(np.mean(recall_scores), 4)
    F1 = round(np.mean(f1_scores), 4)
    em_f1 = round(f1_scores.count(1)/len(lines), 4)
    
    return em, None, Precision, Recall, F1, hit_rate


def eval_fact_checking(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def eval_dialogue_system(infile):

    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens

