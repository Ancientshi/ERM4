"""OPTIONAL INFERRED PROPOSAL, NOT A VERIFIED ERM4 PIPELINE.

No CLI, dataset loading, evaluation, training, or API calls occur on import.
"""


async def answer_question(question, *, rewriter, trigger, retriever, knowledge_filter, reader):
    """Rewrite, consult memory/trigger, retrieve if needed, filter, cache, and read.

    The trigger and pipeline must share the same MemoryKnowledgeReservoir.
    Component callbacks can be local or remote. This function does not choose a
    provider or enforce budgets; callers control them. See experiments/ for the
    historical cached-snippet QA demo, which does not use memory or the trigger.
    """
    rewritten_questions, queries = rewriter.rewrite(question)
    rewritten_question = ' '.join(rewritten_questions)
    retrieval_decisions = trigger.check_retrieval_need(queries)
    knowledge = {}
    external_queries = []
    for query, decision in zip(queries, retrieval_decisions):
        if decision is True:
            external_queries.append(query)
        else:
            knowledge.update({title: trigger.memory.knowledge[title] for title in decision})
    if external_queries:
        knowledge.update(retriever.retrieve(external_queries))
    filtered = await knowledge_filter.filter(rewritten_question, knowledge)
    trigger.memory.update(filtered)
    return reader.read(rewritten_question, filtered)
