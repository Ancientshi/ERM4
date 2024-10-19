import time
from tqdm import tqdm
from evaluation import my_eval_question_answering
import random
import numpy as np
from Components import *
import concurrent.futures

random.seed(args.seed)
np.random.seed(args.seed)


os.environ["SILICONFLOW_API_KEY"]= ''

datset2id={'CAmbigNQ':'1','ambignq':'2','nq':'3','popqa':'4','hotpot':'5','2wikimqa':'6'}
id=datset2id[args.dataset]

rewriter_option=args.rewriter
trigger_option=args.trigger
retriever_option=args.retriever
filter_option=args.filter
reader_option=args.reader

rewriter=QuestionRewriter(args.question_rewritter_prompt)
reader = Reader(args.reader_prompt,silicon_flow_qa=args.use_silicon_flow)
knowledge_filter=KnowledgeFilter(args.knowledge_filter_prompt)
#In our provided examples, the retrieved knowledge from Bing search is already prepared, so there is no need to include the Retriever. This is for the convenience of quickly running the demo.



def get_data():
    datapath=os.path.join(args.root_path, f'Records/demo/data_{id}.jsonl') 
    with open(datapath, 'r') as f:
        data = f.readlines()
    dataset=[json.loads(line) for line in data]
    return dataset

def get_result():
    exp_name=args.exp_name
    dataset=args.dataset
    result_path=os.path.join(args.root_path, f"Records/demo/data_{id}_{dataset}_{exp_name}.jsonl")
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result=f.readlines()
        result=[json.loads(line) for line in result]
        return result, result_path
    else:
        return [], result_path
                   
def answer_question():  
    Question_Answer_list=get_data()
    Result_list, result_path=get_result()
    
    answered_questions=[result["original_question"] for result in Result_list]
    Question_Answer_list=[QA for QA in Question_Answer_list if QA["original_question"] not in answered_questions]
    
    
    for QA in tqdm(Question_Answer_list):
        original_question=QA["original_question"]
        original_question=QA["original_question"]
        rewritten_question_list, queries=rewriter.rewrite(original_question)
        rewritten_question=' '.join(rewritten_question_list)
            
        answer=QA["answer"]
        all_knowledge=QA["external_knowledge"]        
        
        async def get_filtered_knowledge():
            return await knowledge_filter.filter(original_question,all_knowledge)
        filtered_knowledge = asyncio.run(get_filtered_knowledge())
        

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(reader.read, rewritten_question, filtered_knowledge, None, None, None),
            ]
            
        results = [future.result() for future in futures]  # 这会按照提交的顺序等待每个任务完成，并收集结果
        
        QA['ERM4']=results[0]

        
        #判断，六个都不是Connection error，那么就写入文件，否则不写入
        if all([result!='Connection error' for result in results]):
            Result_list.append(QA)
            with open(result_path, 'a') as f:
                f.write(json.dumps(QA)+'\n')
        else:
            print("Connection error, not write to file")
         
    print("Done!")
    return True

def eval(key='response'):
    Result_list, result_path=get_result()
    emscore, length, precision, recall, f1, hit_rate = my_eval_question_answering(result_path,key)
    print(f'Exact Match: {emscore}; Precision: {precision}; Recall: {recall}; F1: {f1}; Hit Rate: {hit_rate}; Avg.Length: {length}')
    local_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(os.path.join(args.root_path, 'Records/all_results.txt'), 'a') as f:
        f.write(f'{args.exp_name}_{key}_{args.dataset}_{local_time}:\nExact Match: {emscore}; Precision: {precision}; Recall: {recall}; F1: {f1}; Hit Rate: {hit_rate}; Avg.Length: {length}\n')
    with open(os.path.join(args.root_path, 'Records/report.csv'), 'a') as f:
        router_option=None
        learner_option=None
        f.write(f'{args.exp_name}_{key},{args.dataset},{rewriter_option},{router_option},{trigger_option},{retriever_option},{filter_option},{reader_option},{learner_option},{emscore},{precision},{recall},{f1},{hit_rate}\n')

    

if __name__ == "__main__":
    answer_question()
    eval(key='ERM4')


#这个实验做得是
# 1. 用原始问题去问GPT，不用外部知识
# 2. 用重写问题去问GPT，不用外部知识
# 3. 用原始问题去问GPT，用外部知识
# 4. 用重写问题去问GPT，用外部知识
# 5. 用原始问题去问GPT，用过滤后的外部知识
# 6. 用重写问题去问GPT，用过滤后的外部知识

#对应的应该是Ablation Study