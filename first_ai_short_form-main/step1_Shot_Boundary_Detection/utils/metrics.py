# utils/metrics.py
import torch

def precision_recall_f1(preds, labels, threshold=0.5):
    '''
    일단 수치화를 위해 뭔가를 넣어봅시다.
    '''
    preds = (preds >= threshold).float()
    true_pos = ((preds == 1) & (labels == 1)).sum().item()
    false_pos = ((preds == 1) & (labels == 0)).sum().item()
    false_neg = ((preds == 0) & (labels == 1)).sum().item()
    precision = true_pos / (true_pos + false_pos + 1e-8)
    recall = true_pos / (true_pos + false_neg + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1
