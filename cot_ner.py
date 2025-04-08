from tqdm import tqdm
import json
import re

from models.llm_pipeline import Llm

from metrics import calculate_ner_metrics, NERErrorAnalyzer
from utils.data_utils import load_jsonl_data


class EntityRecognizer:
    def __init__(self, model_path):
        # 初始化模型和相关工具
        self.model = Llm(model_name=model_path)  # 加载LLM模型

        # 读取答案格式模板和示例数据
        with open('cot_answer_format.txt', 'r', encoding='utf-8') as f:
            self.answer_format = f.read()  # 读取思维链(CoT)答案格式模板
        with open('cot_demo.json', 'r', encoding='utf-8') as f:
            self.support_demo = json.load(f)['support']  # 加载支持集示例

    def get_support_locate_predict(self, file_path, save_path):
        """获取支持集的实体定位预测结果"""
        with open('cot_demo.json', 'r', encoding='utf-8') as f:
            support_locate_demo = json.load(f)['support_locate']  # 加载实体定位示例

        data = load_jsonl_data(file_path)  # 加载输入数据
        f_output = open(save_path, 'w', encoding='utf-8')  # 打开输出文件

        # 遍历每个示例(episode)
        for ep in tqdm(data):
            ep_pred_entities = []  # 存储当前示例的预测实体
            types_list = json.dumps(ep['types'])  # 将实体类型列表转为JSON字符串

            # 处理支持集中的每个项目
            for item in ep['support']:
                prompt = support_locate_demo.copy()  # 复制示例模板
                # 添加用户输入(实体类型和文本)
                prompt.append({"role": "user", "content": f"*Categories* {types_list}\n*Text* {item['text']}"})
                response = self.model.message_format_pipeline(prompt)  # 获取模型响应

                # 获取真实实体和输出实体
                truth_entity = [entity['text'] for entity in item['entity_offset']]
                output_entity = {entity['text']: entity['type'] for entity in item['entity_offset']}

                try:
                    # 尝试解析模型响应为JSON
                    pred_entity = json.loads(response)
                    if not isinstance(pred_entity, list):
                        raise ValueError('Not a list')
                    for entity in pred_entity:
                        if not isinstance(entity, str):
                            raise ValueError("Not a string")
                except json.JSONDecodeError as e:
                    print("解析失败", e)
                except ValueError:
                    print("Not a list")
                else:
                    # 将预测实体不在真实实体中的标记为'none'
                    for entity in pred_entity:
                        if entity not in truth_entity:
                            output_entity[entity] = 'none'

                ep_pred_entities.append(output_entity)  # 添加当前项目的预测结果
                f_output.write(json.dumps(ep_pred_entities) + '\n')  # 写入文件

    def get_support_demo(self, supports, types, pred_entities):
        """生成支持集的思维链(CoT)示例"""
        cot_format = '- {entity}: {reason}\n'  # 思维链格式模板

        types_list = json.dumps(types)  # 实体类型列表转为JSON字符串
        demo = []  # 存储生成的示例

        # 遍历支持集和预测实体
        for item, pred_item in zip(supports, pred_entities):
            prompt = self.support_demo.copy()  # 复制支持集示例模板
            # 添加用户输入(实体类型、文本和预测实体)
            prompt.append({"role": "user",
                           "content": f"*Categories* {types_list}\n*Text* {item['text']}\n*entities* {json.dumps(pred_item)}"})
            response = self.model.message_format_pipeline(prompt)  # 获取模型响应

            try:
                # 尝试从响应中提取JSON格式的思维链解释
                pattern = r'```json(.*?)```'
                matches = re.findall(pattern, response, re.DOTALL)
                if not matches:
                    raise ValueError("未找到有效的 ```json ``` 代码块")
                json_response = matches[0].strip()
                description = json.loads(json_response)
                if not isinstance(description, list):
                    raise ValueError("Not a dict")
            except (json.JSONDecodeError, ValueError) as e:
                continue  # 解析失败则跳过
            else:
                # 构建思维链文本
                cot_chain = ""
                for des in description:
                    cot_chain += cot_format.format(entity=des['entity'], reason=des['reason'])

                # 准备真实实体答案
                entity_answer = [{"entity": entity['text'], "type": entity['type']} for entity in item['entity_offset']]
                locate_list = list(pred_item.keys())  # 预测实体列表
                json_answer = json.dumps(entity_answer, indent=4)  # 真实实体转为JSON

                # 格式化最终思维链答案
                cot_answer = self.answer_format.format(locate_list=locate_list, reasoning_text=cot_chain,
                                                       json_answer=json_answer)

                # 添加用户输入和助手响应到示例中
                demo.append({"role": "user", "content": f"*Text* {item['text']}"})
                demo.append({"role": "assistant", "content": cot_answer})
        return demo

    def load_the_dataset(self, file_path, support_locate_path, save_file):
        """加载数据集并生成查询集的预测结果"""
        # 系统提示模板
        system_prompt_1 = """*TASK* The task is named entity recognition. Recognize the entity mentions in texts that belong to certain types. The target entity types are """
        system_prompt_2 = """. Please think step by step, then output a JSON format list, where each element is an entity mention formatted as {"entity": entity_name, "type": entity_type}.\n"""

        data = load_jsonl_data(file_path)  # 加载输入数据

        # 设置输出文件路径
        support_save_path = f'saves/suppport_{save_file}'
        query_save_path = f'saves/query_{save_file}'

        # 加载支持集预测结果
        support_pred_entities = load_jsonl_data(support_locate_path)
        f_support = open(support_save_path, 'w', encoding='utf-8')  # 支持集输出文件
        f_query = open(query_save_path, 'w', encoding='utf-8')  # 查询集输出文件

        # 遍历每个示例和支持集预测结果
        for ep, support_pred_entity in tqdm(zip(data, support_pred_entities)):
            # 生成支持集思维链示例
            support_demo = self.get_support_demo(ep['support'], ep['types'], support_pred_entity)
            types_list = json.dumps(ep['types'])  # 实体类型列表转为JSON

            # 构建系统提示
            ep_demo = [{"role": "system", "content": system_prompt_1 + types_list + system_prompt_2}]
            ep_demo.extend(support_demo)  # 添加支持集示例

            ep_query = []  # 存储查询集预测结果
            # 处理查询集中的每个项目
            for item in ep['query']:
                prompt = ep_demo.copy()  # 复制示例模板
                prompt.append({"role": "user", "content": f"*Text* {item['text']}"})  # 添加用户输入
                response = self.model.message_format_pipeline(prompt)  # 获取模型响应

                try:
                    # 尝试从响应中提取JSON格式的预测实体
                    pattern = r'```json(.*?)```'
                    matches = re.findall(pattern, response, re.DOTALL)
                    if not matches:
                        raise ValueError("未找到有效的 ```json ``` 代码块")
                    json_response = matches[0].strip()
                    pred_entity = json.loads(json_response)
                except (json.JSONDecodeError, ValueError) as e:
                    print(e)
                    pred_entity = []  # 解析失败则设为空列表

                ep_query.append(pred_entity)  # 添加当前查询项的预测结果

            print(ep_query)
            f_query.write(json.dumps(ep_query) + '\n')  # 写入查询集预测结果
            f_support.write(json.dumps(support_demo) + '\n')  # 写入支持集示例

def calculate_dataset(data_path, query_pred_file_path='saves/query_test_5_1.jsonl',):
    data = load_jsonl_data(data_path)
    pred = load_jsonl_data(query_pred_file_path)
    truth_entities, pred_entities = [], []
    for ep, pred_ep in zip(data, pred):
        for item, pred_item in zip(ep['query'], pred_ep):
            truth_entity = {entity['text']: entity['type'] for entity in item['entity_offset']}
            pred_entity = {entity['entity']: entity['type'] for entity in pred_item}
            truth_entities.append(truth_entity)
            pred_entities.append(pred_entity)
    metrics = calculate_ner_metrics(truth_entities, pred_entities)
    print(metrics)

