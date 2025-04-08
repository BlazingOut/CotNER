import pickle
import re
import numpy as np
from pathlib import Path


from transformers import BertTokenizer, BertModel
import torch


def clean_text(text):
    # 替换下划线和连字符为空格，并去除单词末尾的's
    cleaned = re.sub(r"[-_()（）]|('s\b)", " ", text)
    # 分割单词并过滤空字符串
    return cleaned.split()

class Word2VectEmbed:
    def __init__(self):
        # 初始化word2vec
        
        file_path = 'utils/glove.6B.50d.pkl'
        with open(file_path, 'rb') as f:
            embeddings_index = pickle.load(f)
        self.embeddings_dict = embeddings_index

    def get_phrase_embedding(self, phrase, mode='mean'):
        # 获取单词的 GloVe 嵌入
        # words = re.split(r'[ _-]+', phrase)
        words = clean_text(phrase)
        
        # print('phrase:', phrase)
        # print('words:', words)
        total_embedding = list()
        for word in words:
            word = word.lower()
            embedding_ = self.embeddings_dict.get(word)
            if embedding_ is None:
                return None
            total_embedding.append(embedding_)
        if mode == 'mean':
            total_array = np.array(total_embedding)
            embeddings = np.mean(total_array, axis=0)
            return embeddings
        # 增加最后一个词的权重
        elif mode == 'weighted mean':
            total_array = np.array(total_embedding)
            embeddings = np.mean(total_array, axis=0)
            embeddings += total_array[-1]
            return embeddings

    def get_word_embedding(self, word):
        # 获取单词的 GloVe 嵌入
        embedding = self.embeddings_dict.get(word)
        if embedding is None:
            return None
        return embedding

class BertEmbed:
    def __init__(self):
        current_path = Path(__file__).parent
        pretrained_model_name = current_path  / 'bert-base-cased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name)
        self.model.eval()

    def get_phrase_embedding(self, text):
        merked_text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(merked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)

        # 转化为张量
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            # BERT模型有多个输出，第一个是所有隐藏状态的最后一层，第二个是[CLS] token的池化输出
            hidden_states, cls_output = outputs[:2]
        # 获取每个token的最后一层隐藏状态作为词嵌入
        token_embeddings = hidden_states[0][0]
        return token_embeddings.numpy()

