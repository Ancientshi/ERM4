"""Optional, unexecuted offline wiring check; no model/data downloads or API calls.

Run later, if authorized: python experiments/unverified/smoke_test.py
This checks deterministic callbacks, not model quality or paper reproduction.
"""
import asyncio
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.unverified.components import (KnowledgeFilter, KnowledgeRetriever, MemoryKnowledgeReservoir,
                        QuestionRewriter, Reader, RetrievalTrigger)
from experiments.unverified.composition import answer_question


async def smoke():
    search_calls = []

    def rewrite(prompt, **kwargs):
        return json.dumps({'Rewritten Question': ['How many modules are in ERM4?'], 'Query': ['ERM4 modules']})

    async def nli(prompt, **kwargs):
        return json.dumps({'NLI result': 'entailment' if 'ERM4 uses four modules.' in prompt else 'neutral'})

    def read(prompt, **kwargs):
        assert 'ERM4 uses four modules.' in prompt
        assert 'cooking' not in prompt
        return '["four"]'

    def search(query):
        search_calls.append(query)
        return {'ERM4 modules': 'ERM4 uses four modules.', 'Unrelated': 'This snippet is about cooking.'}

    memory = MemoryKnowledgeReservoir()
    trigger = RetrievalTrigger(memory, lambda texts: [[1.0, 0.0] for _ in texts],
                               cosine_similarity_threshold=0.6, popularity_threshold=1)
    components = dict(rewriter=QuestionRewriter(model=rewrite), trigger=trigger,
                      retriever=KnowledgeRetriever(search), knowledge_filter=KnowledgeFilter(model=nli),
                      reader=Reader(model=read))
    assert await answer_question('How many modules?', **components) == ['four']
    assert memory.knowledge == {'ERM4 modules': 'ERM4 uses four modules.'}
    assert search_calls == ['ERM4 modules']
    assert await answer_question('How many modules?', **components) == ['four']
    assert search_calls == ['ERM4 modules'], 'Second question should reuse cached knowledge.'
    memory.update({'ERM4 modules': 'Updated content'})
    assert memory.knowledge['ERM4 modules'] == 'Updated content'
    print('Offline wiring check passed (deterministic callbacks only).')


if __name__ == '__main__':
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network blocked')), \
         patch('requests.sessions.Session.request', side_effect=AssertionError('HTTP blocked')):
        asyncio.run(smoke())
