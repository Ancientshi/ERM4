import openai
import requests
import asyncio
# from aiohttp import ClientSession
import os
import sys
# from sentence_transformers import SentenceTransformer, util
import requests
# from bs4 import BeautifulSoup
import re
# from rank_bm25 import BM25Okapi
import string
import numpy as np

cache_dir = "./model"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

os.environ['OPENAI_API_KEY'] = ''


# Initialize the model and move it to GPU
'''
sentence2embedding_model = SentenceTransformer('all-mpnet-base-v2').to('cuda')

def sentence2embedding(texts):
    embeddings = sentence2embedding_model.encode(texts, convert_to_tensor=True)
    return embeddings.to('cpu').detach().numpy()
'''

def query_rewriter_plus(question, t=0.001,port=8001):
    url = f'http://127.0.0.1:{port}/build'
    
    data={}
    data["original_question"]=question
    data['temperature']=t
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()['output']
        else:
            return "error"
    except Exception as e:
        return "error"
 


def GPT_QA(prompt, model_name="gpt-3.5-turbo-16k", t=0.0,historical_qa=None,siliconflow=False,api_key=None):
    if siliconflow:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        if api_key is not None:
            openai.api_key =api_key
        else:
            openai.api_key =os.environ["SILICONFLOW_API_KEY"]
    else:
        url = "https://api.openai.com/v1/chat/completions"
        if api_key is not None:
            openai.api_key =api_key
        else:
            openai.api_key =os.environ["OPENAI_API_KEY"]
        
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"}
    messages=[]
    if historical_qa!=None:
        for (q,a) in historical_qa:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": t,
        "n": 1,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
    except:
        print("Error: Connection error")
        answer="Connection error"
    try: 
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(response.json())
        aa=input('pause')
        print(f"Error: {e}")
        print("response",response)
        answer="Connection error"
    return answer


async def async_GPT_QA(prompt, model_name="gpt-3.5-turbo-16k",t=0.0,historical_qa=None):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"}
    messages=[]
    if historical_qa!=None:
        for (q,a) in historical_qa:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": t,
        "n": 1,
    }
    async with ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            json_response = await response.json()
            answer = json_response["choices"][0]["message"]["content"]
            return answer



def extract_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers, timeout=3)
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')

        text = soup.get_text()

        # find all paragraphs
        paragraphs = text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences.extend(re.split(r'\.\s+', p))
        
        cleaned_sentences = [re.sub(r'[^a-zA-Z0-9\s,;?]', '', s).strip() + '.' for s in sentences if s.strip()]
        page_content = ' '.join(cleaned_sentences)

        page_content = re.sub(r'\.+', '.', page_content)
        page_content = re.sub(r'\s+', ' ', page_content)
        return page_content
        

    except requests.RequestException as e:
        print(f"Request Error: {e}")
        return "No page content found."


'''
toker = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
def filter_content_bm25(page_content, query, threshold=3,maxlength = 3000):
    sentences = page_content.split(". ")
    
    tokenized_sentences = [toker.encode(sentence.lower()) for sentence in sentences]
    tokenized_query = toker.encode(query.lower())

    bm25 = BM25Okapi(tokenized_sentences)

    scores = bm25.get_scores(tokenized_query)
    
    # import matplotlib.pyplot as plt
    # plt.hist(scores, bins=100)
    # plt.savefig('score_distribution.png')


    percentile_score = np.percentile(scores, 80)
    threshold=max(percentile_score,threshold)
    relevant_sentences = [sentences[i] for i in range(len(sentences)) if scores[i] >= threshold and len(sentences[i])>5]
    
    page_content = '. '.join(relevant_sentences)
    
    words_list=page_content.split(' ')
    words_count = len(words_list)
    maxlength=min(maxlength-300,words_count)
    page_content = ' '.join(words_list[:maxlength])

    # page_content_tokens=toker.encode(page_content)
    # if len(page_content_tokens) > maxlength:
    #     print('Number of page_content_tokens: %s, shorten the page content.'%(len(page_content_tokens)))
    #     page_content_tokens = page_content_tokens[:maxlength]
    #     page_content = toker.decode(page_content_tokens)

    return page_content
'''
