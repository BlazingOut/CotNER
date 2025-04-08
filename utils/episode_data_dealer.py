import json


def process_sentence(tokens, labels):
    entities = []
    current_entity = None

    for idx, (token, label) in enumerate(zip(tokens, labels)):
        if label != 'O':
            if current_entity and current_entity['type'] == label:
                # 延续当前实体
                current_entity['tokens'].append(token)
                current_entity['offsets'].append(idx)
            else:
                # 保存旧实体，创建新实体
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'tokens': [token],
                    'type': label,
                    'offsets': [idx]
                }
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    # 处理最后一个实体
    if current_entity:
        entities.append(current_entity)

    # 转换为目标格式
    formatted_entities = []
    for ent in entities:
        formatted_entities.append({
            "text": ' '.join(ent['tokens']),
            "type": ent['type'],
            "offset": ent['offsets']
        })
    return {
        "text": ' '.join(tokens),
        "entity_offset": formatted_entities
    }

def process_episode_dataset(n_way, k_shot, subset, task):
    data_file = f'episode-data/{task}/{subset}_{n_way}_{k_shot}.jsonl'
    done_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    for item in raw_data:
        support_data = []
        query_data = []
        for tokens, labels in zip(item["support"]["word"], item["support"]["label"]):
            entities = process_sentence(tokens, labels)
            support_data.append(entities)
        for tokens, labels in zip(item["query"]["word"], item["query"]["label"]):
            entities = process_sentence(tokens, labels)
            query_data.append(entities)
        done_data.append({"support":support_data,
                          "query":query_data,
                          "types": item["types"]})
    save_file = f'episode/{task}/{subset}_{n_way}_{k_shot}.jsonl'
    with open(save_file, 'w', encoding='utf-8') as f:
        for item in done_data:
            f.write(json.dumps(item) + '\n')
    print(f'{save_file} done!')

if __name__ == '__main__':
    subsets = ['train', 'test', 'dev']
    tasks = ['inter', 'intra']
    for n_way in [5, 10]:
        for k_shot in [1, 5]:
            for subset in subsets:
                for task in tasks:
                    process_episode_dataset(n_way, k_shot, subset, task)
