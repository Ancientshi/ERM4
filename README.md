
# Enhancing Retrieval and Managing Retrieval: A Four-Module Synergy for Improved Quality and Efficiency in RAG Systems

This repository contains the source code and implementation for the paper "Enhancing Retrieval and Managing Retrieval: A Four-Module Synergy for Improved Quality and Efficiency in RAG Systems." The project introduces a framework that optimizes retrieval processes in Retrieval-Augmented Generation systems, enhancing both the quality and efficiency of information retrieval for Open-Domain Question Answering tasks.

## Overview

Retrieval-Augmented Generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval to improve the relevance and accuracy of responses. However, traditional RAG systems often face issues such as low retrieval quality, irrelevant knowledge, and redundant retrievals. Our approach introduces a four-module synergy to tackle these limitations:

1. **Query Rewriter+:** Generate more nuanced and multi-faceted queries, enhancing search coverage and clarifying intent.
2. **Knowledge Filter:** A module that filters out irrelevant information using natural language inference (NLI) tasks, ensuring that only relevant knowledge is retrieved.
3. **Memory Knowledge Reservoir:** A caching mechanism that speeds up retrieval for recurring queries by utilizing previously retrieved external knowledge.
4. **Retrieval Trigger:** A calibration-based mechanism that determines when to initiate external knowledge retrieval based on the confidence level of existing information.

<img alt="image" src="https://github.com/user-attachments/assets/9eee72d0-ac96-4b8e-91cd-19ceb3d217d2">

Our four-module synergy addresses these issues by improving response accuracy from 14% to 21% compared to directly querying the LLM and achieving around a 8%~12% improvement over the traditional RAG pipeline. Additionally, we can reduce response time cost by 46% and external knowledge retrieval cost by 71% without compromising response quality.

## Motivation

The current limitations of RAG systems include the following:

- **Information Plateau:** A single query limits the scope of retrieval, leading to less comprehensive information.
- **Ambiguity in Query Interpretation:** Misaligned phrasing often results in unreliable responses.
- **Irrelevant Knowledge:** Excessive retrieval can bring irrelevant information, reducing response quality.
- **Redundant Retrieval:** Repeated questions result in inefficient use of computational resources.

## Datasets

The following datasets were used for our experiments:
- **CAmbigNQ:** A curated version of the AmbigNQ dataset with clarified questions, designed to address ambiguities.
- **NQ (Natural Questions):** A dataset of real-world search engine queries.
- **PopQA:** Focuses on less popular topics from Wikidata.
- **AmbigNQ:** Contains ambiguous questions transformed into closely related queries.
- **2WIKIMQA & HotPotQA:** Datasets requiring logical reasoning and multi-hop question answering.

We provide demo dataset for Q&A, and Fine-Tuning Gemma-2B in Records.

## Key Findings

1. **Query Rewriting**: Clarifying ambiguous questions significantly improves retrieval precision.
2. **Multi-Query Retrieval**: Employing multiple, semantically varied queries enhances the amount of relevant information retrieved, overcoming the information plateau.
3. **Knowledge Filtering**: The Knowledge Filter reduces noise from irrelevant data, increasing the accuracy and reliability of RAG systems.
4. **Efficiency**: The use of the Memory Knowledge Reservoir accelerates repeated retrievals, reducing time cost by 46% at optimal configurations.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ancientshi/ERM4.git
   ```

2. Install dependencies:

   ```bash
   cd ERM4
   pip install -r requirements.txt
   ```
3. Download Demo Datasets from https://drive.google.com/drive/folders/1UYkFJqfuNbJJZUad-psssL4uSn4ttuAY?usp=sharing, move under ERM4.

4. Run the demo (knowledge retrieved from Bing search is pre-prepared for ease of use):

   ```bash
   cd shell
   bash ERM4.sh
   ```

5. Fine-tune Gemma-2B
   ```
   cd shell
   bash instruct_fine_tune_gemma.sh
   ```

6. Deploy the trained GEMMA-2B service for Flask to support API calls
   ```
   cd shell
   bash infer_gemma_rewriter.sh
   ```

## Usage

The provided code includes a demo that illustrates how our four-module synergy works within a RAG system. The example retrieval process uses pre-fetched data from Bing searches to streamline the execution. 

## Key Considerations:
### Prompt Design

In the demo, we provide a set of pre-designed prompts for each query. It’s important to note that these prompts may influence the results to some degree. If you're interested in further experimentation, we encourage adjusting these prompts to suit the format and style of each specific dataset. The provided prompts are meant to serve as a reference, and customizing them may yield different retrieval outcomes.

### Reproducibility and Research Development

The source code is shared to promote advancements in the field and facilitate future research. However, we do not guarantee a 100% replication of the exact results reported in our paper. This is due to the rapid evolution of the RAG landscape, the dynamic nature of large language model (LLM) APIs, dynamics in search engine behaviors, and LLM's fine-tuning difference—all of which introduce considerable variance. Nevertheless, the key point is that the findings and conclusions should align with those in our work.

We hope our guideline can help you can continue to explore RAG systems, and contribute to the evolving discourse in this domain.


## Contact

For any questions or contributions, please reach out to the project lead:

- **Yunxiao Shi**  
  Email: Yunxiao.Shi@student.uts.edu.au

## Citation
If you find our work useful and would like to reference it, please cite our paper as follows:

```bibtext
@incollection{Shi2024,
  author    = {Yunxiao Shi and Xing Zi and Zijing Shi and Haimin Zhang and Qiang Wu and Min Xu},
  title     = {Enhancing Retrieval and Managing Retrieval: A Four-Module Synergy for Improved Quality and Efficiency in RAG Systems},
  booktitle = {ECAI 2024},
  publisher = {IOS Press},
  year      = {2024},
  pages     = {2258--2265},
  doi       = {10.3233/FAIA240748},
  url       = {https://ebooks.iospress.nl/doi/10.3233/FAIA240748}
}
```
```bibtext
@misc{shi2024eragentenhancingretrievalaugmentedlanguage,
      title={ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization}, 
      author={Yunxiao Shi and Xing Zi and Zijing Shi and Haimin Zhang and Qiang Wu and Min Xu},
      year={2024},
      eprint={2405.06683},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.06683}, 
}
```
## License

This project is licensed under the by-nc-sa 4.0 License.
