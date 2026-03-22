import os
import re
from model.openllama import OpenLLAMAPEFTModel
from model.ImageBind.data import load_and_transform_news_text, load_choice_data
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
import yaml
from datasets import create_dataset, create_loader
import json
import torch.nn.functional as F
from transformers import GenerationConfig

from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from utils.multilabel_metrics import get_multi_label, AveragePrecisionMeter
import datetime

# ---------- 与训练 prompt 完全一致的向量顺序 ----------
# [face_swap, face_attribute, text_swap, text_attribute]
# 同 deepfake_dataset.py 的 describles_answ 和
# multilabel_metrics.py 的 get_multi_label
LABEL_NAMES = ['face_swap', 'face_attribute', 'text_swap', 'text_attribute']

parser = argparse.ArgumentParser("FKA_Owl", add_help=True)
parser.add_argument("--FKA_Owl_ckpt_path", default='./ckpt/train_DGM4/pytorch_model.pt')
parser.add_argument("--config", default='./fake_config/test_guardian.yaml')

command_args = parser.parse_args()
time1 = datetime.datetime.now()

args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '/data1/yaxiong/UniMMFakeDet/FAK-Owl/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '/data1/yaxiong/ckpt/vicuna_ckpt/7b_v0/',
    'delta_ckpt_path': '/data1/yaxiong/UniMMFakeDet/FAK-Owl/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'device': 'cuda'
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(command_args.FKA_Owl_ckpt_path, map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

# 预先获取 "0" 和 "1" 的 token id
TOKEN_0_ID = model.llama_tokenizer("0", add_special_tokens=False).input_ids[-1]
TOKEN_1_ID = model.llama_tokenizer("1", add_special_tokens=False).input_ids[-1]


def extract_element_probs(scores):
    """从生成 scores 中提取 4 个元素各自 P("1") 的连续概率。

    scores: tuple of tensors, 每个 shape (1, vocab_size), 逐 token 的 logits。
    生成格式为 "[1, 0, 1, 0]"，在对应 "0"/"1" 位置提取概率。
    """
    element_probs = []
    for step_logits in scores:
        top_token = step_logits[0].argmax().item()
        if top_token in (TOKEN_0_ID, TOKEN_1_ID):
            logits_01 = step_logits[0, [TOKEN_0_ID, TOKEN_1_ID]]
            p = F.softmax(logits_01, dim=0)
            element_probs.append(p[1].item())
    while len(element_probs) < 4:
        element_probs.append(0.0)
    return np.array(element_probs[:4])


def parse_prediction_list(text):
    """解析 "[x, y, z, w]" 为 4 元素二值数组。"""
    match = re.search(r'\[(\d)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\]', text)
    if match:
        return np.array([int(match.group(i)) for i in range(1, 5)])
    numbers = re.findall(r'[01]', text)
    if len(numbers) >= 4:
        return np.array([int(n) for n in numbers[:4]])
    return np.array([0, 0, 0, 0])


def predict(
    input,
    images,
    news_texts,
    max_length,
    top_p,
    temperature,
    history,
    modality_cache,
):
    generation_config = GenerationConfig.from_model_config(model.llama_model.config)
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = True
    generation_config.top_p = None
    generation_config.top_k = None

    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, scores = model.generate_logits({
        'prompt': prompt_text,
        'image_paths': images if images else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'news_text': news_texts if news_texts else [],
        'top_p': None,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    },
    generation_config=generation_config)

    return response, scores


# -------- 数据集加载 --------
config = yaml.load(open(command_args.config, 'r'), Loader=yaml.Loader)
val_dataset = create_dataset(config, is_train=False)
samplers = [None]
val_loader = create_loader([val_dataset],
                           samplers,
                           batch_size=[config['batch_size_val']],
                           num_workers=[4],
                           is_trains=[False],
                           collate_fns=[val_dataset.collate])[0]

device = torch.device(args['device'])

# -------- 收集容器 --------
all_true = []       # N×4 int
all_pred = []       # N×4 int  (硬预测)
all_prob = []       # N×4 float (连续概率 P("1"))
all_harm = []

ap_meter = AveragePrecisionMeter(difficult_examples=False)

cls_nums_all = 0

for i, batch in enumerate(val_loader):
    # prompt 直接来自 deepfake_dataset.py 的 conversation[0]['value']
    # 与训练 prompt 完全一致
    for conversation in batch['texts']:
        prompts = conversation[0]['value']

    images = batch['images']
    label = batch['class_names'][0]
    text = batch['captions']

    resp, scores = predict(prompts, images, text, 512, 0.1, 1.0, [], [])

    # ground truth: 同 get_multi_label，向量顺序 [face_swap, face_attr, text_swap, text_attr]
    gt_multilabel, _ = get_multi_label([label], device)
    gt_multilabel = gt_multilabel.cpu().numpy()[0]  # (4,) int

    # 硬预测
    pred_multilabel = parse_prediction_list(resp)

    # 连续概率 (用于 AUC / mAP)
    element_probs = extract_element_probs(scores)

    # HARM 分 (partial credit)
    harm_score = np.mean(pred_multilabel == gt_multilabel)
    all_harm.append(harm_score)

    all_true.append(gt_multilabel)
    all_pred.append(pred_multilabel)
    all_prob.append(element_probs)

    # AveragePrecisionMeter 累积
    ap_meter.add(
        torch.from_numpy(element_probs).unsqueeze(0).float(),
        torch.from_numpy(gt_multilabel).unsqueeze(0).long()
    )

    cls_nums_all += 1

all_true = np.array(all_true)   # (N, 4)
all_pred = np.array(all_pred)   # (N, 4)
all_prob = np.array(all_prob)   # (N, 4)

# ==================== 指标计算 ====================

# --- 1. HARM Score (partial credit) ---
avg_harm = np.mean(all_harm)

# --- 2. Exact Match ---
exact_match = np.all(all_true == all_pred, axis=1).sum()
exact_match_acc = exact_match / cls_nums_all

# --- 3. 二分类指标: 每个 label 的 ACC & AUC (基于连续概率) ---
per_label_acc = []
per_label_auc = []
for j in range(4):
    y_t = all_true[:, j]
    y_p = all_pred[:, j]
    y_prob = all_prob[:, j]
    per_label_acc.append(np.mean(y_t == y_p))
    if y_t.sum() > 0 and len(np.unique(y_t)) > 1:
        per_label_auc.append(roc_auc_score(y_t, y_prob))
    else:
        per_label_auc.append(float('nan'))

# --- 4. mAP (per-class AP 取均值) ---
per_class_ap = ap_meter.value()  # (4,) tensor
mAP = per_class_ap.mean().item()

# --- 5. CF1 / OF1 (多标签 P/R/F1) ---
OP, OR, OF1, CP, CR, CF1 = ap_meter.evaluation(all_prob, all_true)

# ==================== 打印结果 ====================

print("\n" + "=" * 60)
print("  Multi-label Classification Results (Guardian)")
print("=" * 60)

print(f"\n  HARM Score (partial credit) : {avg_harm:.4f}")
print(f"  Exact Match Accuracy        : {exact_match_acc:.4f} ({exact_match}/{cls_nums_all})")

print(f"\n  --- Multi-label Aggregate ---")
print(f"  mAP  : {mAP:.4f}")
print(f"  OF1  : {OF1:.4f}  (OP={OP:.4f}, OR={OR:.4f})")
print(f"  CF1  : {CF1:.4f}  (CP={CP:.4f}, CR={CR:.4f})")

print(f"\n  --- Per-label Binary (ACC / AUC) ---")
print(f"  {'Label':<25s} {'ACC':>8s} {'AUC':>8s} {'AP':>8s}")
for j, name in enumerate(LABEL_NAMES):
    auc_str = f"{per_label_auc[j]:.4f}" if not np.isnan(per_label_auc[j]) else "   N/A"
    ap_str = f"{per_class_ap[j].item():.4f}"
    print(f"  {name:<25s} {per_label_acc[j]:>8.4f} {auc_str:>8s} {ap_str:>8s}")

print(f"\n  --- Per-label Classification Report ---")
for j, name in enumerate(LABEL_NAMES):
    y_t = all_true[:, j]
    y_p = all_pred[:, j]
    print(f"\n  [{name}]")
    print(classification_report(y_t, y_p, target_names=['no', 'yes'], digits=4, zero_division=0))

time2 = datetime.datetime.now()
print(f"\nTime consumed: {time2 - time1}")
