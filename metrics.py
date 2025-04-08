from typing import Dict, List, Set, Tuple, Union
from collections import defaultdict


def remove_leading_the(text: str) -> str:
    """移除实体开头的'the'前缀（大小写不敏感），保留原始大小写格式。

    Args:
        text: 待处理的文本字符串

    Returns:
        处理后的文本，已去除开头的'the'前缀（如果存在）

    Examples:
        >>> remove_leading_the("The Apple")
        'Apple'
        >>> remove_leading_the("Microsoft")
        'Microsoft'
    """
    stripped_text = text.strip()
    if stripped_text.lower().startswith("the "):
        return stripped_text[4:].strip()
    return stripped_text


def calculate_locate_metrics(
        true_samples: List[List[str]],
        pred_samples: List[List[str]]
) -> Dict[str, Union[float, int]]:
    """计算数据集级别的精确率、召回率和F1值（基于实体匹配）

    Args:
        true_samples: 每个样本的真实实体列表，格式如 [["实体1", "实体2"], ...]
        pred_samples: 每个样本的预测实体列表，格式与true_samples相同

    Returns:
        包含以下指标的字典:
        - precision: 精确率
        - recall: 召回率
        - f1: F1值
        - tp: 真正例数
        - fp: 假正例数
        - fn: 假反例数

    Raises:
        ValueError: 当输入样本数量不匹配时
    """
    if len(true_samples) != len(pred_samples):
        raise ValueError("真实样本和预测样本数量必须相同")

    metrics = defaultdict(int)

    for true_entities, pred_entities in zip(true_samples, pred_samples):
        # 标准化实体格式
        clean_true = {remove_leading_the(e).lower() for e in true_entities}
        clean_pred = {remove_leading_the(e).lower() for e in pred_entities}

        # 计算当前样本指标
        tp = len(clean_true & clean_pred)
        fp = len(clean_pred - clean_true)
        fn = len(clean_true - clean_pred)

        metrics["tp"] += tp
        metrics["fp"] += fp
        metrics["fn"] += fn

    # 计算最终指标
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def calculate_ner_metrics(
        true_samples: List[Dict[str, str]],
        pred_samples: List[Dict[str, str]]
) -> Tuple[float, float, float]:
    """计算考虑实体类型的NER任务指标

    Args:
        true_samples: 真实实体字典列表，格式如 [{"实体": "类型"}, ...]
        pred_samples: 预测实体字典列表，格式与true_samples相同

    Returns:
        (precision, recall, f1) 元组，数值已四舍五入到4位小数

    Raises:
        ValueError: 当输入样本数量不匹配时
    """
    if len(true_samples) != len(pred_samples):
        raise ValueError("真实样本和预测样本数量必须相同")

    tp = fp = fn = 0

    for true_ents, pred_ents in zip(true_samples, pred_samples):
        true_set = {(ent.lower(), typ.lower()) for ent, typ in true_ents.items()}
        pred_set = {(ent.lower(), typ.lower()) for ent, typ in pred_ents.items()}

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return (
        round(precision, 4),
        round(recall, 4),
        round(f1, 4)
    )


class NERErrorAnalyzer:
    """命名实体识别任务的错误分析工具，支持边界错误、漏检和过检统计"""

    def __init__(self):
        """初始化分析器，重置所有统计计数器"""
        self._reset_counters()

    def _reset_counters(self) -> None:
        """重置所有统计指标"""
        self.span_errors: List[str] = []  # 边界错误的实体
        self.missed_entities: List[str] = []  # 漏检的实体
        self.over_predicted: List[str] = []  # 过检的实体
        self.total_truth_entities = 0  # 真实实体总数
        self._truth_cache: List[List[str]] = []  # 缓存真实实体
        self._pred_cache: List[List[str]] = []  # 缓存预测实体

    def analyze_single_sample(self, truth_entities: List[str], pred_entities: List[str]) -> None:
        """分析单个样本的错误类型

        Args:
            truth_entities: 真实实体列表
            pred_entities: 预测实体列表
        """
        self._validate_input(truth_entities, pred_entities)

        self._truth_cache.append(truth_entities)
        self._pred_cache.append(pred_entities)
        self.total_truth_entities += len(truth_entities)

        span_errs, missed, over_pred = self._classify_errors(truth_entities, pred_entities)

        self.span_errors.extend(span_errs)
        self.missed_entities.extend(missed)
        self.over_predicted.extend(over_pred)

    def _validate_input(self, truth: List[str], pred: List[str]) -> None:
        """验证输入数据格式"""
        if not isinstance(truth, list) or not isinstance(pred, list):
            raise TypeError("输入必须为字符串列表")
        if any(not isinstance(e, str) for e in truth + pred):
            raise ValueError("列表元素必须为字符串")

    def _classify_errors(self, truth: List[str], pred: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """错误分类核心逻辑

        Returns:
            (边界错误列表, 漏检列表, 过检列表)
        """
        truth_set = set(truth)
        pred_set = set(pred)

        # 完全正确的实体
        correct_entities = truth_set & pred_set

        # 边界错误检测
        span_errors = []
        related_truths = set()

        for pred_entity in pred_set - correct_entities:
            matched = False
            for truth_entity in truth_set:
                if self._is_partial_match(pred_entity, truth_entity):
                    span_errors.append(pred_entity)
                    related_truths.add(truth_entity)
                    matched = True
                    break

            # 未匹配任何真实实体的预测视为过检
            if not matched:
                self.over_predicted.append(pred_entity)

        # 漏检实体 = 真实实体 - 正确实体 - 被部分匹配的实体
        missed = [
            e for e in truth
            if e not in correct_entities and e not in related_truths
        ]

        return span_errors, missed, self.over_predicted

    @staticmethod
    def _is_partial_match(a: str, b: str) -> bool:
        """判断两个实体是否存在部分匹配关系"""
        return (a in b or b in a) and a != b

    def generate_report(self) -> Dict[str, float]:
        """生成错误分析报告

        Returns:
            包含以下指标的字典:
            - span_error_rate: 边界错误占真实实体的比例
            - missed_rate: 漏检比例
            - over_prediction_rate: 过检比例
            - span_error_ratio: 边界错误占所有错误的比例
            - missed_ratio: 漏检占所有错误的比例
            - over_prediction_ratio: 过检占所有错误的比例
        """
        total_errors = len(self.span_errors) + len(self.missed_entities) + len(self.over_predicted)

        metrics = {
            'span_error_rate': len(self.span_errors) / self.total_truth_entities if self.total_truth_entities else 0,
            'missed_rate': len(self.missed_entities) / self.total_truth_entities if self.total_truth_entities else 0,
            'over_prediction_rate': len(
                self.over_predicted) / self.total_truth_entities if self.total_truth_entities else 0,
        }

        if total_errors > 0:
            metrics.update({
                'span_error_ratio': len(self.span_errors) / total_errors,
                'missed_ratio': len(self.missed_entities) / total_errors,
                'over_prediction_ratio': len(self.over_predicted) / total_errors,
            })
        else:
            metrics.update({
                'span_error_ratio': 0,
                'missed_ratio': 0,
                'over_prediction_ratio': 0,
            })

        return {k: round(v, 4) for k, v in metrics.items()}

    def print_detailed_report(self) -> None:
        """打印详细的错误分析报告"""
        report = self.generate_report()

        print("\n=== NER Error Analysis Report ===")
        print(f"Total Truth Entities: {self.total_truth_entities}")
        print(f"Total Errors: {len(self.span_errors) + len(self.missed_entities) + len(self.over_predicted)}")
        print("\nError Rates (relative to truth entities):")
        print(f"Span Detection Error: {report['span_error_rate']:.2%}")
        print(f"Missed Detection: {report['missed_rate']:.2%}")
        print(f"Over Prediction: {report['over_prediction_rate']:.2%}")

        print("\nError Distribution:")
        print(f"Span Errors: {report['span_error_ratio']:.2%} of all errors")
        print(f"Missed Entities: {report['missed_ratio']:.2%} of all errors")
        print(f"Over Predictions: {report['over_prediction_ratio']:.2%} of all errors")
