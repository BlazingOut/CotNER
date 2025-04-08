import json
from pathlib import Path

def load_demo_data(demo_tag):
    # load demo data
    demo_file_path = 'demo.json'
    with open(demo_file_path, 'r', encoding='utf-8') as f:
        demo_data = json.load(f)
    demo_data = demo_data[demo_tag]
    return demo_data

def load_jsonl_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data