# 使用思维链的方法进行少样本命名实体识别
## 任务简介
**命名实体识别（NER）**：自然语言处理中的一项重要任务，旨在从文本中识别出具有特定意义的实体，如人名、地名、组织机构等。传统的NER方法通常依赖于大量标注数据进行训练，但在实际应用中，获取大量标注数据往往困难且耗���。因此，少样本命名实体识别（Few-shot NER）应运而生，旨在通过少量标注样本来训练模型。

**思维链（CoT）**：思维链（Chain-of-Thought，CoT）方法是一种通过模拟人类逐步推理过程来提升大语言模型（LLM）复杂问题解决能力的技术。其核心思想是让模型在生成最终答案前，先输出中间推理步骤，从而分解问题、减少错误，并提高可解释性。以下是关键要点：

**少样本学习（Few-shot Learning）**：少样本学习是一种机器学习方法，旨在通过少量标注样本来训练模型。与传统的监督学习方法不同，少样本学习通常依赖于迁移学习、元学习等技术，以便在有限的数据上实现良好的性能。
Meta learning （元学习）中，在 meta training 阶段将数据集分解为不同的 meta task，去学习类别变化的情况下模型的泛化能力，在 meta testing 阶段，面对全新的类别，不需要变动已有的模型，就可以完成分类。

**实现步骤**：
我们需要将支持集数据作为prompt中的示例，然后将待识别的文本作为输入，使用大语言模型进行推理。具体步骤如下：
- 获取模型对支持集实体的预测
- 根据实体预测与支持集真实标签，使模型进行推理，给出用以作为预测集示例的推理过程
- 将预测集数据作为输入，使用大语言模型进行推理，得到最终的预测结果
## 实现
### 1. 数据集
**Few-NERD**是一个大规模、细粒度的手动标注命名实体识别数据集，包含8个粗粒度类型、66个细粒度类型、188,200个句子、491,711个实体和4,601,223个标记。该数据集主要用于小样本命名实体识别（Few-shot NER）任务，提供了三种基准任务：监督学习（Few-NERD (SUP)）、小样本学习（Few-NERD (INTRA)）和小样本学习（Few-NERD (INTER)）

下载：[Few-NERD：一个Few-shot场景的命名实体识别数据集](https://nlp.csai.tsinghua.edu.cn/news/few-nerd%E4%B8%80%E4%B8%AAfew-shot%E5%9C%BA%E6%99%AF%E7%9A%84%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB%E6%95%B0%E6%8D%AE%E9%9B%86/)
数据路径
```shell
data/
    ├── episode_save
    │   ├── inter
    │   └── intra
    └── episode
        ├── inter
        └── intra
```

### 2. 运行环境
```bash
conda create -n few-nerd python=3.8
conda activate few-nerd
pip install -r requirements.txt

```

### 3. 运行代码
#### 3.1 转化数据集
```bash
python utils/episode_data_dealer.py
```
数据结构
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


#### 3.2 推理
```bash
python main.py --subset intra --n_way 10 --k_shot 5 \
               --model_path /path/to/model \
               --data_dir ./custom_data \
               --output_dir ./custom_results 
```
