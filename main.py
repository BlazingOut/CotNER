import argparse
import json
from tqdm import tqdm
import re
from cot_ner import EntityRecognizer, calculate_dataset


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Few-shot NER with LLM')
    parser.add_argument('--subset', type=str, required=True, choices=['inter', 'intra'],
                        help='Subset type: inter or intra')
    parser.add_argument('--n_way', type=int, required=True,
                        help='Number of entity types (N-way)')
    parser.add_argument('--k_shot', type=int, required=True,
                        help='Number of support examples per class (K-shot)')
    parser.add_argument('--model_path', type=str, default='path/to/your/model',
                        help='Path to the LLM model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset files')
    parser.add_argument('--output_dir', type=str, default='saves',
                        help='Directory to save output files')

    args = parser.parse_args()

    # 根据参数构造文件路径
    input_file = f"{args.data_dir}/{args.subset}/{args.n_way}_{args.k_shot}.jsonl"
    support_locate_file = f"{args.output_dir}/support_locate_{args.subset}_{args.n_way}_{args.k_shot}.jsonl"
    output_file = f"saves/{args.subset}_{args.n_way}_{args.k_shot}.jsonl"

    # 初始化实体识别器
    recognizer = EntityRecognizer(args.model_path)

    print(f"Processing {args.subset} subset with {args.n_way}-way {args.k_shot}-shot...")

    # 第一步：获取支持集的实体定位预测
    print("Step 1: Getting support set entity location predictions...")
    recognizer.get_support_locate_predict(input_file, support_locate_file)

    # 第二步：加载数据集并生成查询集预测
    print("Step 2: Generating query set predictions...")
    recognizer.load_the_dataset(input_file, support_locate_file, output_file)

    # 第三步：计算评估指标
    print("Step 3: Calculating evaluation metrics...")
    calculate_dataset(
        input_file,
        query_pred_file_path=f"{args.output_dir}/query_{output_file}"
    )


if __name__ == "__main__":
    main()