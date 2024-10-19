question_rewritter_prompt=/Volumes/Data/work_study/Research/我的论文/RAG/RAG_github/Prompt/question_rewritter_plus_prompt_2wikimqa.txt
knowledge_filter_prompt=/Volumes/Data/work_study/Research/我的论文/RAG/RAG_github/Prompt/knowledge_filter_prompt.txt
reader_prompt=/Volumes/Data/work_study/Research/我的论文/RAG/RAG_github/Prompt/reader_prompt.txt
exp_name=ERM4

page_num=30
bm25_threshold=3
# datset2id={'CAmbigNQ':'1','ambignq':'2','nq':'3','popqa':'4','hotpot':'5','2wikimqa':'6'}
dataset=2wikimqa
maxlength=3000
rewriter=rewriter
trigger=none
retriever=retriever
filter=filter
reader=reader   

python3 ../main.py \
    --question_rewritter_prompt $question_rewritter_prompt \
    --knowledge_filter_prompt $knowledge_filter_prompt \
    --reader_prompt $reader_prompt \
    --exp_name $exp_name \
    --page_num $page_num \
    --dataset $dataset \
    --maxlength $maxlength \
    --rewriter $rewriter \
    --trigger $trigger \
    --retriever $retriever \
    --filter $filter \
    --reader $reader \
    --bm25_threshold $bm25_threshold \
    --model_name 'gpt-3.5-turbo' \



