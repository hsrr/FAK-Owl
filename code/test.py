import os
from model.openllama import OpenLLAMAPEFTModel
from model.ImageBind.data import  load_and_transform_news_text,load_choice_data
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

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from utils.multilabel_metrics import get_multi_label, AveragePrecisionMeter
import datetime

parser = argparse.ArgumentParser("FKA_Owl", add_help=True)
# paths
parser.add_argument("--FKA_Owl_ckpt_path", default='./ckpt/train_DGM4/pytorch_model.pt')
parser.add_argument("--config", default='./fake_config/test_guardian.yaml')

command_args = parser.parse_args()
time1 = datetime.datetime.now()
# init the model
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
        'news_text':news_texts if news_texts else [],
        'top_p': None,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    },
    generation_config=generation_config)

    return response, scores

config = yaml.load(open(command_args.config, 'r'), Loader=yaml.Loader)
val_dataset = create_dataset(config,is_train=False)
samplers = [None]
val_loader = create_loader([val_dataset],
                               samplers,
                               batch_size=[config['batch_size_val']],
                               num_workers=[4],
                               is_trains=[False],
                               collate_fns=[val_dataset.collate])[0]


cls_nums_all = 0
device = torch.device(args['device'])

LABEL_NAMES = ['face_swap', 'face_attribute', 'text_swap', 'text_attribute']

choices = ["A", "B", "C", "D", "E"]
choice_ids = [model.llama_tokenizer(choice).input_ids[1] for choice in choices]

all_multilabel_true = []
all_multilabel_pred = []
all_multilabel_scores = []

ap_meter = AveragePrecisionMeter()

for i, batch in enumerate(val_loader):

    for conversation in batch['texts']:
        prompts = (conversation[0]['value'])

    images = batch['images']
    label = batch['class_names'][0]
    text = batch['captions']

    resp, scores = predict(prompts, images, text, 512, 0.1, 1.0, [], [])

    gt_multilabel, _ = get_multi_label([label], device)
    gt_multilabel = gt_multilabel.cpu().numpy()[0]

    choise_scores = scores[0]
    choice_logits = choise_scores[:, choice_ids]
    choice_probs = F.softmax(choice_logits, dim=1)[0]

    prob_A, prob_B, prob_C, prob_D, prob_E = choice_probs.tolist()
    pred_multilabel = np.array([
        1 if prob_A > prob_E else 0,
        1 if prob_B > prob_E else 0,
        1 if prob_C > prob_E else 0,
        1 if prob_D > prob_E else 0,
    ])
    pred_scores = np.array([prob_A, prob_B, prob_C, prob_D])

    all_multilabel_true.append(gt_multilabel)
    all_multilabel_pred.append(pred_multilabel)
    all_multilabel_scores.append(pred_scores)

    ap_meter.add(
        torch.tensor(pred_scores).unsqueeze(0),
        torch.tensor(gt_multilabel).unsqueeze(0)
    )

    cls_nums_all += 1

all_multilabel_true = np.array(all_multilabel_true)
all_multilabel_pred = np.array(all_multilabel_pred)
all_multilabel_scores = np.array(all_multilabel_scores)

exact_match = np.all(all_multilabel_true == all_multilabel_pred, axis=1).sum()
exact_match_acc = exact_match / cls_nums_all

print("\n===== Multi-label Classification Results =====")
print(f"Exact Match Accuracy: {exact_match_acc:.4f} ({exact_match}/{cls_nums_all})")

# --- Per-class AP and MAP ---
ap_per_class = ap_meter.value()
mAP = ap_per_class.mean().item()
print(f"\n--- Mean Average Precision (MAP) ---")
for j, name in enumerate(LABEL_NAMES):
    print(f"  AP_{name}: {ap_per_class[j]:.4f}")
print(f"  MAP: {mAP:.4f}")

# --- OP/OR/OF1 (micro-average), CP/CR/CF1 (macro-average) ---
OP, OR, OF1, CP, CR, CF1 = ap_meter.evaluation(all_multilabel_scores, all_multilabel_true)
print(f"\n--- Overall (micro-average) ---")
print(f"  OP (Precision): {OP:.4f}")
print(f"  OR (Recall):    {OR:.4f}")
print(f"  OF1 (F1):       {OF1:.4f}")
print(f"\n--- Class-average (macro-average) ---")
print(f"  CP (Precision): {CP:.4f}")
print(f"  CR (Recall):    {CR:.4f}")
print(f"  CF1 (F1):       {CF1:.4f}")

# --- Per-label accuracy / AUC ---
print("\nPer-label results:")
for j, name in enumerate(LABEL_NAMES):
    y_t = all_multilabel_true[:, j]
    y_p = all_multilabel_pred[:, j]
    acc_j = np.mean(y_t == y_p)
    print(f"  {name}: Accuracy={acc_j:.4f}", end="")
    if y_t.sum() > 0 and len(np.unique(y_t)) > 1:
        auc_j = roc_auc_score(y_t, y_p)
        print(f", AUC={auc_j:.4f}", end="")
    print()

# --- Per-label F1 (sklearn classification_report) ---
print("\nPer-label classification report:")
for j, name in enumerate(LABEL_NAMES):
    y_t = all_multilabel_true[:, j]
    y_p = all_multilabel_pred[:, j]
    print(f"\n  [{name}]")
    print(classification_report(y_t, y_p, target_names=['no', 'yes'], digits=4, zero_division=0))

time2 = datetime.datetime.now()
print("time consumed: ", time2 - time1)


