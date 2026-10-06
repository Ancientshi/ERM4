"""Upstream ERM4 component logic with explicit configuration, not a full pipeline.

No reservoir writer is supplied upstream. Trigger embeddings and full-page BM25
are incomplete; this cleanup does not infer their implementations. The optional
proposal under experiments/unverified/ is not imported by these components.
"""
import json
import os
from pathlib import Path

from bing import searchbing
from utils import GPT_QA, async_GPT_QA


class QuestionRewriter:
    def __init__(self, rewriter_prompt='', *, model_name='gpt-3.5-turbo'):
        self.prompt = Path(rewriter_prompt).read_text(encoding='utf-8')
        self.model = GPT_QA
        self.model_name = model_name

    def rewrite(self, original_question):
        prompt = self.prompt.replace('{Original_Question}', original_question)
        response = json.loads(self.model(prompt, model_name=self.model_name, t=0.0))
        try:
            rewritten_question = response['Rewritten Input']
        except KeyError:
            rewritten_question = response['Rewritten Question']
        return rewritten_question, response['Query']


class RetrievalTrigger:
    """Original experience-file trigger; no new reservoir or automatic cache write.

    The upstream sentence2embedding helper is commented out. An explicit embed
    callback can supply that missing dependency, but no model is chosen here.
    Cosine/popularity comparisons and default thresholds retain upstream values.
    """
    def __init__(self, user_id=None, *, root_path=None, exp_name='default',
                 cosine_similarity_threshold=0.4, popularity_threshold=3, embed=None):
        self.root_path = str(root_path or Path(__file__).resolve().parent)
        self.exp_name = exp_name
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.popularity_threshold = popularity_threshold
        self.embed = embed
        self.experience_pool = None
        self.set_user_id(user_id)
        self.load_experience_pool()

    def set_user_id(self, user_id):
        self.user_id = user_id
        self.path_experience_pool = os.path.join(self.root_path, f'ExperienceSave/{self.exp_name}/experience_pool_{self.user_id}.jsonl')

    def load_experience_pool(self):
        if os.path.exists(self.path_experience_pool):
            with open(self.path_experience_pool, encoding='utf-8') as source:
                self.experience_pool = [json.loads(line) for line in source]
        else:
            self.experience_pool = None

    def check_retrieval_need(self, query):
        if self.experience_pool is None:
            return [True] * len(query)
        all_title_content = {}
        for experience in self.experience_pool:
            all_title_content.update(experience['filtered knowledge'])
        if self.embed is None:
            raise NotImplementedError('The upstream sentence2embedding helper is not implemented. Supply embed(texts) explicitly.')
        from sklearn.metrics.pairwise import cosine_similarity
        all_title = list(all_title_content)
        title_embeddings = self.embed(all_title)
        query_embeddings = self.embed(query)
        similarity = cosine_similarity(query_embeddings, title_embeddings)
        decisions = []
        for row in similarity:
            titles = [all_title[index] for index, score in enumerate(row)
                      if score >= self.cosine_similarity_threshold]
            decisions.append(titles if len(titles) >= self.popularity_threshold else True)
        return decisions


class KnowledgeRetriever:
    """Original Bing v7 snippet retrieval, retained despite provider retirement."""
    def __init__(self, *, bm25_threshold=2):
        self.bm25_threshold = bm25_threshold

    def retrieve(self, query, summary=True, maxlength=3000, pagenum=5):
        if not summary:
            # The upstream BM25 helper was commented out and page extraction
            # referenced an unimported BeautifulSoup. Do not silently replace it.
            raise NotImplementedError('The upstream full-page/BM25 path is incomplete; archival source is in experiments/legacy_helpers.py.')
        knowledge = {}
        for item in query:
            results = searchbing(item, pagenum)
            for result in results['webPages']['value']:
                knowledge[result['name']] = result['snippet']
        return knowledge


class KnowledgeFilter:
    def __init__(self, rewriter_prompt='', *, model_name='gpt-3.5-turbo'):
        self.prompt = Path(rewriter_prompt).read_text(encoding='utf-8')
        self.model = async_GPT_QA
        self.model_name = model_name

    async def async_run(self, rewritten_question, external_knowledge):
        labels = {}
        for title, content in external_knowledge.items():
            prompt = self.prompt.replace('{Question}', rewritten_question).replace('{External_Knowledge}', content)
            # Preserve the original error-as-contradiction behavior; it is a
            # known protocol limitation, not an inferred algorithm repair.
            try:
                response = await self.model(prompt, model_name=self.model_name, t=0.0)
            except Exception:
                labels[title] = 'contradiction'
                continue
            labels[title] = json.loads(response)['NLI result']
        return {title: external_knowledge[title] for title, label in labels.items()
                if label == 'entailment'}

    async def filter(self, rewritten_question, external_knowledge):
        return await self.async_run(rewritten_question, external_knowledge)


class Reader:
    """Original reader behavior; legacy profile argument is not an ERM4 claim."""
    def __init__(self, reader_prompt='', silicon_flow_qa=False, *, model_name='gpt-3.5-turbo'):
        self.prompt = Path(reader_prompt).read_text(encoding='utf-8')
        self.model = GPT_QA
        self.model_name = model_name
        self.silicon_flow_qa = silicon_flow_qa
        self.reader_prompt_path = str(reader_prompt)

    def read(self, question, external_knowledge, historical_qa=None, user_profile=None, api_key=None):
        prompt = self.prompt
        if user_profile is not None:
            prompt = prompt.replace('{User Profile}', json.dumps(user_profile))
        if external_knowledge is None:
            context = 'None'
        else:
            context = ''
            for number, (title, content) in enumerate(external_knowledge.items()):
                context += f'\t[Source {number}]\n\t\tTitle: {title}\n\t\tContent: {content}\n\n'
        prompt = prompt.replace('{Question}', question).replace('{External_Knowledge}', context)
        response = self.model(prompt, model_name=self.model_name, t=0.0,
                              historical_qa=historical_qa, siliconflow=self.silicon_flow_qa, api_key=api_key)
        if 'multi_round' not in self.reader_prompt_path:
            try:
                response = json.loads(response)
            except (ValueError, TypeError):
                pass
        return response
