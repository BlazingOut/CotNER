import json
import torch
from transformers import AutoTokenizer, pipeline

class Llm:
    def __init__(self, model_name):
        print('building pipeline ...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def pipeline_response(self, prompt_text):
        response = self.pipeline(
            prompt_text,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512 ,
            return_full_text=False
        )
        return response[0]['generated_text']

    def message_format_pipeline(self, chat_list):
        prompt_text = self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True)
        return self.pipeline_response(prompt_text)

    def alpaca_format_pipline(self, question_dict, template_key='alpaca'):
        """

        :param question_dict: dict. 必须包含'instruct', 'input', 和 'output'三个键的字典。
                        各个键对应的值分别是任务指令、输入数据及预期输出描述。
        :param template_key: str. 模板的键值，用于选择模板。默认为alpaca，特殊情况下选alpaca_cot
        :return:
        """
        with open('template.json', 'r', encoding='utf-8') as f:
            template = json.load(f)
            prompt_template = template[template_key]
        prompt_text = prompt_template.format(**question_dict)
        return self.pipeline_response(prompt_text)
    
if __name__ == '__main__':
    model_path = '/data/yyma/code/models/Qwen2.5-7B'
    llm = Llm(model_path)
    generation_prompts =['Directly answer: Who was the designer of Lahti Town Hall?',
                'Directly answer: What role does Denny Herzig play in football?',
                'Directly answer: What city did Marl Young live when he died?']
    prompt = 'the capital of France is'
    for text in generation_prompts:
        response = llm.pipeline_response(text)
        print(f'Prompt: {text}')
        print(f'Pre-Edit  Output: {response}')
