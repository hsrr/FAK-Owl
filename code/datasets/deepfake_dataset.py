import json
import os
import random

from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from datasets.utils import pre_caption
from torchvision.transforms.functional import hflip, resize
from random import random as rand

describles_answ = {}

describe_temple = "The following are multiple choice questions about fake news detection. \n\nThe caption of news is: "
describe_ques_latter = (
    ". The identity and emotion of the face, and the semantic and sentiment of the text "
    "should not be manipulated. For each manipulation type, determine whether it exists "
    "(1 for yes, 0 for no):\n"
    "1. Face swap\n"
    "2. Face attribute manipulation\n"
    "3. Text swap\n"
    "4. Text attribute manipulation\n"
    "Answer as [face_swap, face_attribute, text_swap, text_attribute]:"
)

describles_answ['orig'] = "[0, 0, 0, 0]"
describles_answ['face_swap'] = "[1, 0, 0, 0]"
describles_answ['face_attribute'] = "[0, 1, 0, 0]"
describles_answ['text_swap'] = "[0, 0, 1, 0]"
describles_answ['text_attribute'] = "[0, 0, 0, 1]"
describles_answ['face_swap&text_swap'] = "[1, 0, 1, 0]"
describles_answ['face_swap&text_attribute'] = "[1, 0, 0, 1]"
describles_answ['face_attribute&text_swap'] = "[0, 1, 1, 0]"
describles_answ['face_attribute&text_attribute'] = "[0, 1, 0, 1]"

class DGM4_Dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True):

        self.root_dir = '/data1/yaxiong/dataset'
        self.ann = []

        for f in ann_file:
            file = open(f, 'r', encoding='utf-8')
            for line in file.readlines():
                data = json.loads(line)
                self.ann.append(data)

        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann) / config['dataset_division'])]
            print('dataset_division')

        self.transform = transform
        self.max_words = max_words
        self.image_res = config['image_res']
        self.is_train = is_train

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        img_dir = ann['image']
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")

        if self.is_train:
            if rand() < 0.5:
                image = hflip(image)
            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)

        label = ann['fake_cls']
        caption = pre_caption(ann['text'], self.max_words)

        conversation = [
            {"from": "human", "value": describe_temple + caption + describe_ques_latter},
            {"from": "gpt", "value": describles_answ[label]},
        ]

        return image, conversation, label, caption

    def collate(self, instances):

        images = []
        texts = []
        class_names = []
        captions = []

        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[2])
            captions.append(instance[3])

        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            captions=captions,
        )