"""OPTIONAL INFERRED PROPOSAL, NOT THE UPSTREAM IMPLEMENTATION.

Prompt-based adapters remain available. They are not the paper's tuned Gemma
models. The memory/trigger and their composition in composition.py are a compact local
implementation of the published method, added during this cleanup and untested.
"""
import json
import math
from pathlib import Path

from utils import GPT_QA, async_GPT_QA

PROMPTS = Path(__file__).resolve().parent / 'Prompt'


def _strings(value, name):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f'{name} must be a nonempty list of strings.')
    return value


class QuestionRewriter:
    """Query Rewriter+: clarify the question and produce multiple search queries."""
    def __init__(self, rewriter_prompt=None, *, model=GPT_QA, model_name='gpt-3.5-turbo'):
        self.prompt = Path(rewriter_prompt or PROMPTS / 'question_rewriter_plus.txt').read_text(encoding='utf-8')
        self.model = model
        self.model_name = model_name

    def rewrite(self, original_question):
        prompt = self.prompt.replace('{Original_Question}', original_question)
        response = json.loads(self.model(prompt, model_name=self.model_name, t=0.0))
        rewritten = response.get('Rewritten Question', response.get('Rewritten Input'))
        return _strings(rewritten, 'Rewritten Question'), _strings(response.get('Query'), 'Query')


class KnowledgeFilter:
    """Retain only knowledge labeled entailment by the injected async NLI model."""
    def __init__(self, rewriter_prompt=None, *, model=async_GPT_QA, model_name='gpt-3.5-turbo'):
        self.prompt = Path(rewriter_prompt or PROMPTS / 'knowledge_filter.txt').read_text(encoding='utf-8')
        self.model = model
        self.model_name = model_name

    async def filter(self, rewritten_question, external_knowledge):
        filtered = {}
        for title, content in external_knowledge.items():
            prompt = self.prompt.replace('{Question}', rewritten_question).replace('{External_Knowledge}', content)
            response = json.loads(await self.model(prompt, model_name=self.model_name, t=0.0))
            label = response.get('NLI result')
            if label not in {'entailment', 'contradiction', 'neutral'}:
                raise ValueError('Invalid NLI result: expected entailment, contradiction, or neutral.')
            if label == 'entailment':
                filtered[title] = content
        return filtered


class MemoryKnowledgeReservoir:
    """In-memory title/content cache; new content replaces entries with the same title."""
    def __init__(self, knowledge=None):
        self.knowledge = {}
        self.update(knowledge or {})

    def update(self, knowledge):
        if not isinstance(knowledge, dict) or not all(isinstance(title, str) and isinstance(content, str) for title, content in knowledge.items()):
            raise ValueError('Knowledge must be a title-to-content dictionary of strings.')
        self.knowledge.update(knowledge)


def _cosine(left, right):
    if len(left) != len(right) or len(left) == 0:
        raise ValueError('Query and title embeddings must have matching nonzero dimensions.')
    denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


class RetrievalTrigger:
    """Fetch externally if fewer than theta cached titles meet cosine threshold tau.

    embed(texts) must return one numerical vector per text. No embedding model is
    selected, loaded, or downloaded here. A decision is True for external search,
    or a list of cached titles when enough matches exist (the upstream convention).
    """
    def __init__(self, memory, embed, *, cosine_similarity_threshold=0.4, popularity_threshold=3):
        if not -1 <= cosine_similarity_threshold <= 1 or popularity_threshold < 1:
            raise ValueError('Require -1 <= cosine threshold <= 1 and popularity threshold >= 1.')
        self.memory = memory
        self.embed = embed
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.popularity_threshold = popularity_threshold

    def check_retrieval_need(self, query):
        if not query:
            return []
        titles = list(self.memory.knowledge)
        if not titles:
            return [True] * len(query)
        title_vectors = list(self.embed(titles))
        query_vectors = list(self.embed(query))
        if len(title_vectors) != len(titles) or len(query_vectors) != len(query):
            raise ValueError('Embedding callback must return one vector per input text.')
        decisions = []
        for vector in query_vectors:
            matches = [title for title, title_vector in zip(titles, title_vectors)
                       if _cosine(vector, title_vector) >= self.cosine_similarity_threshold]
            decisions.append(matches if len(matches) >= self.popularity_threshold else True)
        return decisions


class KnowledgeRetriever:
    """Provider-neutral interface: search(query) returns a title-to-content dict."""
    def __init__(self, search):
        self.search = search

    def retrieve(self, queries):
        knowledge = MemoryKnowledgeReservoir()
        for query in queries:
            knowledge.update(self.search(query))
        return knowledge.knowledge


class Reader:
    """Basic RAG reader; no user profile or personalized-agent behavior."""
    def __init__(self, reader_prompt=None, *, model=GPT_QA, model_name='gpt-3.5-turbo', silicon_flow_qa=False):
        self.prompt = Path(reader_prompt or PROMPTS / 'reader.txt').read_text(encoding='utf-8')
        self.model = model
        self.model_name = model_name
        self.silicon_flow_qa = silicon_flow_qa

    def read(self, question, external_knowledge):
        context = '\n\n'.join(f'[Source {number}]\nTitle: {title}\nContent: {content}'
                               for number, (title, content) in enumerate((external_knowledge or {}).items()))
        prompt = self.prompt.replace('{Question}', question).replace('{External_Knowledge}', context or 'None')
        response = self.model(prompt, model_name=self.model_name, t=0.0, siliconflow=self.silicon_flow_qa)
        answers = json.loads(response)
        if not isinstance(answers, list):
            raise ValueError('Reader must return a JSON array of answer strings.')
        return _strings(answers, 'Reader response')
