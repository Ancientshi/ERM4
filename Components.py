from datetime import datetime
import json
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from config import args
from bing import searchbing
import asyncio

cosine_similarity_threshold=args.cosine_similarity_threshold
popularity_threshold=args.popularity_threshold
bm25_threshold=args.bm25_threshold

class QuestionRewriter:
    def __init__(self,rewriter_prompt=''):
        with open(rewriter_prompt, 'r') as f:
            self.prompt=f.read()
        self.model=GPT_QA
        
    def rewrite(self, original_question):
        formated_prompt=self.prompt.replace('{Original_Question}', original_question)
        response = self.model(formated_prompt, model_name=args.model_name,t=0.0)
        response_dict=json.loads(response)
        try:
            rewritten_question=response_dict['Rewritten Input']
        except:
            rewritten_question=response_dict['Rewritten Question']
        query=response_dict['Query']
        return rewritten_question, query

class RetrievalTrigger:
    def __init__(self,user_id=None):
        self.experience_pool=None
        self.set_user_id(user_id)
        self.load_experience_pool()
        
    def set_user_id(self, user_id):
        self.user_id=user_id
        self.path_experience_pool=os.path.join(args.root_path, 'ExperienceSave/%s/experience_pool_%s.jsonl'%(args.exp_name, self.user_id))
        
    def load_experience_pool(self):
        if os.path.exists(self.path_experience_pool):
            with open(self.path_experience_pool, 'r') as f:
                experience_pool_jsonl = f.readlines()
            self.experience_pool=[json.loads(line) for line in experience_pool_jsonl]
        else:
            self.experience_pool=None
            
    def check_retrieval_need(self, query):
        if self.experience_pool==None:
            return [True]*len(query)
        
        all_title_content={}
        for experience in self.experience_pool:
            all_title_content.update(experience['filtered knowledge'])

        
        all_title=list(all_title_content.keys())
        all_title_embedding=sentence2embedding(all_title)
        query_embedding=sentence2embedding(query)
        similarity=cosine_similarity(query_embedding, all_title_embedding)
        
        
        need_retrieval_list=[]
        for i in range(len(query)):
            popularity_sum=0
            title_list=[]
            for j in range(len(similarity[i])):
                if similarity[i][j]>=cosine_similarity_threshold:
                    popularity_sum+=1
                    title_list.append(all_title[j])
            if popularity_sum>=popularity_threshold:
                need_retrieval_list.append(title_list)
            else:
                need_retrieval_list.append(True)
             
        return need_retrieval_list


class KnowledgeRetriever:
    def __init__(self):
        pass
        
    def retrieve(self, query, summary=True,maxlength=3000,pagenum=5):
        external_knowledge={}
        
        for a_query in query:
            search_results = searchbing(a_query,pagenum)
            for result in search_results["webPages"]["value"]:
                title = result["name"]
                url = result["url"]
                if summary==False:
                    page_content=extract_page(url)
                    page_content=filter_content_bm25(page_content,a_query,threshold=bm25_threshold,maxlength = maxlength)
                    external_knowledge[title]=page_content
                else:
                    snippet = result["snippet"]
                    external_knowledge[title]=snippet
        return external_knowledge

class KnowledgeFilter:
    def __init__(self, rewriter_prompt=''):
        with open(rewriter_prompt, 'r') as f:
            self.prompt = f.read()
        self.model = async_GPT_QA

    async def async_run(self, rewritten_question, external_knowledge):
        NLI_result_dict = {}
        for title, content in external_knowledge.items():
            formatted_prompt = self.prompt.replace('{Question}', rewritten_question)
            formatted_prompt = formatted_prompt.replace('{External_Knowledge}', content)
            try:
                response = await self.model(formatted_prompt, model_name=args.model_name,t=0.0)
            except:
                NLI_result='contradiction'
                NLI_result_dict[title] = NLI_result
                continue
            response_dict = json.loads(response)
            NLI_result = response_dict['NLI result']
            NLI_result_dict[title] = NLI_result

        filtered_knowledge = {}
        for key, value in NLI_result_dict.items():
            if value == 'entailment':
                filtered_knowledge[key] = external_knowledge[key]

        return filtered_knowledge
    
    async def filter(self, rewritten_question, external_knowledge):
        filtered_knowledge = await self.async_run(rewritten_question, external_knowledge)
        return filtered_knowledge
    
class Reader:
    def __init__(self,reader_prompt='',silicon_flow_qa=False):
        with open(reader_prompt, 'r') as f:
            self.prompt=f.read()
        self.model=GPT_QA
        self.silicon_flow_qa=silicon_flow_qa
        self.reader_prompt_path=reader_prompt

    def read(self, question, external_knowledge,historical_qa=None, user_profile=None,api_key=None):
        if user_profile==None:
            if external_knowledge==None:
                formated_prompt=self.prompt.replace('{Question}', question)
                formated_prompt=formated_prompt.replace('{External_Knowledge}', 'None')
                response = self.model(formated_prompt, model_name=args.model_name,t=0.0, historical_qa=historical_qa, siliconflow=self.silicon_flow_qa,api_key=api_key)
            else:
                external_knowledge_prompt = ''
                for i,(title, content) in enumerate(external_knowledge.items()):
                    external_knowledge_prompt += f'\t[Source {i}]\n'
                    external_knowledge_prompt += f'\t\tTitle: {title}\n'
                    external_knowledge_prompt += f'\t\tContent: {content}\n\n'
                formated_prompt=self.prompt.replace('{Question}', question)
                formated_prompt=formated_prompt.replace('{External_Knowledge}', external_knowledge_prompt)
                response = self.model(formated_prompt,model_name=args.model_name,t=0.0, historical_qa=historical_qa, siliconflow=self.silicon_flow_qa,api_key=api_key)
        else:
            user_profile_json=json.dumps(user_profile)
            formated_prompt=self.prompt.replace('{User Profile}', user_profile_json)
            if external_knowledge==None:
                formated_prompt=formated_prompt.replace('{Question}', question)
                formated_prompt=formated_prompt.replace('{External_Knowledge}', 'None')
                response = self.model(formated_prompt, model_name=args.model_name,t=0.0, historical_qa=historical_qa, siliconflow=self.silicon_flow_qa,api_key=api_key)
            else:
                external_knowledge_prompt = ''
                for i,(title, content) in enumerate(external_knowledge.items()):
                    external_knowledge_prompt += f'\t[Source {i}]\n'
                    external_knowledge_prompt += f'\t\tTitle: {title}\n'
                    external_knowledge_prompt += f'\t\tContent: {content}\n\n'
                formated_prompt=formated_prompt.replace('{Question}', question)
                formated_prompt=formated_prompt.replace('{External_Knowledge}', external_knowledge_prompt)
                response = self.model(formated_prompt, model_name=args.model_name,t=0.0, historical_qa=historical_qa, siliconflow=self.silicon_flow_qa,api_key=api_key)
            
        #if self.reader_prompt_path begin with multi_round
        if 'multi_round' in self.reader_prompt_path:
            pass
        else:
            try:
                response=json.loads(response)
            except:
                response=response
            
        return response