Here's the English translation of your README file:

---

# Few-Shot Named Entity Recognition Using Chain-of-Thought Approach

## Task Introduction
**Named Entity Recognition (NER)**: An important task in natural language processing that aims to identify meaningful entities (such as person names, locations, and organizations) from text. Traditional NER methods typically rely on large amounts of annotated training data, but in practice, obtaining such data is often difficult and time-consuming. Thus, Few-Shot Named Entity Recognition (Few-shot NER) has emerged, which trains models using only a small number of annotated examples.

**Chain-of-Thought (CoT)**: The Chain-of-Thought (CoT) method is a technique that enhances large language models' (LLMs) ability to solve complex problems by simulating human step-by-step reasoning. Its core idea is to have the model generate intermediate reasoning steps before producing the final answer, thereby breaking down problems, reducing errors, and improving interpretability. Key points include:

**Few-Shot Learning**: A machine learning approach that trains models using only a small number of annotated examples. Unlike traditional supervised learning, few-shot learning often relies on techniques like transfer learning and meta-learning to achieve good performance with limited data.

**Meta-Learning**: During the meta-training phase, the dataset is decomposed into different meta-tasks to learn the model's generalization ability across varying categories. In the meta-testing phase, the model can classify entirely new categories without modifying its existing architecture.

## Implementation Steps
We use the support set data as examples in the prompt, then input the text to be recognized and perform inference using a large language model. The specific steps are:
1. Obtain the model's predictions for entities in the support set.
2. Based on the entity predictions and ground-truth labels in the support set, guide the model to generate reasoning steps that serve as examples for the prediction set.
3. Use the prediction set data as input and perform inference with the large language model to obtain the final predictions.

## Implementation
### 1. Dataset
**Few-NERD** is a large-scale, fine-grained, manually annotated named entity recognition dataset containing:
• 8 coarse-grained types
• 66 fine-grained types
• 188,200 sentences
• 491,711 entities
• 4,601,223 tokens

The dataset is primarily designed for few-shot NER tasks and provides three benchmark tasks:
• Supervised Learning (Few-NERD (SUP))
• Few-shot Learning (Few-NERD (INTRA))
• Few-shot Learning (Few-NERD (INTER))

Download: [Few-NERD: A Few-shot Named Entity Recognition Dataset](https://nlp.csai.tsinghua.edu.cn/news/few-nerd%E4%B8%80%E4%B8%AAfew-shot%E5%9C%BA%E6%99%AF%E7%9A%84%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB%E6%95%B0%E6%8D%AE%E9%9B%86/)

Directory Structure:
```shell
data/
    ├── episode_save
    │   ├── inter
    │   └── intra
    └── episode
        ├── inter
        └── intra
```

### 2. Environment Setup
```bash
conda create -n few-nerd python=3.8
conda activate few-nerd
pip install -r requirements.txt
```

### 3. Running the Code
#### 3.1 Dataset Conversion
```bash
python utils/episode_data_dealer.py
```

Data Structure:
```json
{
  "support": [
    {
      "text": "Example sentence with entities.",
      "entity_offset": [
        {"text": "entity1", "type": "type1", "start": 0, "end": 7}
      ]
    }
  ],
  "query": [...],  # Same format as "support"
  "types": ["type1", "type2", ...]
}
```

#### 3.2 Inference
```bash
python main.py --subset intra --n_way 10 --k_shot 5 \
               --model_path /path/to/model \
               --data_dir ./custom_data \
               --output_dir ./custom_results 
```

---
