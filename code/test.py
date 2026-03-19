import os
from model.openllama import OpenLLAMAPEFTModel
from model.ImageBind.data import  load_and_transform_news_text,load_choice_data
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import yaml
from datasets import create_dataset, create_loader
from datasets.deepfake_dataset import MULTICLASS_CHOICES, LABEL_TO_INDEX
import json
from transformers import GenerationConfig
from sklearn.metrics import accuracy_score, f1_score
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
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0/',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
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


y_true, y_pred = [], []
device = torch.device(args['device'])

choice_ids = [model.llama_tokenizer(choice).input_ids[1] for choice in MULTICLASS_CHOICES]

for i, batch in enumerate(val_loader):

    for conversation in batch['texts']:
        prompts = (conversation[0]['value'])

    images = batch['images']
    label = batch['class_names'][0]
    text = batch['captions']

    resp, scores = predict(prompts, images, text, 512, 0.1, 1.0, [], [])

    choise_scores = scores[0]
    if choise_scores.dim() == 1:
        choise_scores = choise_scores.unsqueeze(0)

    choice_logits = choise_scores[:, choice_ids]
    pred_class_idx = int(torch.argmax(choice_logits, dim=1).item())

    if label not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown label found in dataset: {label}")

    y_pred.append(pred_class_idx)
    y_true.append(LABEL_TO_INDEX[label])


y_true, y_pred = np.array(y_true), np.array(y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
acc = accuracy_score(y_true, y_pred)

print("Macro-F1:", macro_f1)
print("Weighted-F1:", weighted_f1)
print("ACC:", acc)
print("Num samples:", len(y_true))
time2 = datetime.datetime.now()
print("time consumed: ", time2 - time1)


