from distutils.command.config import config
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
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand
import numpy as np

MULTICLASS_LABELS = [
    "orig",
    "face_swap",
    "face_attribute",
    "text_swap",
    "text_attribute",
    "face_swap&text_swap",
    "face_swap&text_attribute",
    "face_attribute&text_swap",
    "face_attribute&text_attribute",
]
MULTICLASS_CHOICES = list("ABCDEFGHI")
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(MULTICLASS_LABELS)}
LABEL_TO_CHOICE = {label: MULTICLASS_CHOICES[idx] for idx, label in enumerate(MULTICLASS_LABELS)}

describles_answ = {label: f"{LABEL_TO_CHOICE[label]}. {label}." for label in MULTICLASS_LABELS}

describe_temple = "The following are multiple choice questions about fake news detection. \n\nThe caption of news is: "
describe_ques_latter = (
    ". Classify the manipulation type of this news into exactly one category.\n"
    "A. orig\n"
    "B. face_swap\n"
    "C. face_attribute\n"
    "D. text_swap\n"
    "E. text_attribute\n"
    "F. face_swap&text_swap\n"
    "G. face_swap&text_attribute\n"
    "H. face_attribute&text_swap\n"
    "I. face_attribute&text_attribute\n"
    "The answer is:"
)


class DGM4_DatasetMultiCls(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True):

        self.root_dir = "../data"
        self.ann = []

        for f in ann_file:
            file = open(f, "r", encoding="utf-8")
            for line in file.readlines():
                data = json.loads(line)
                self.ann.append(data)

        if "dataset_division" in config:
            self.ann = self.ann[: int(len(self.ann) / config["dataset_division"])]
            print("dataset_division")

        self.transform = transform
        self.max_words = max_words
        self.image_res = config["image_res"]
        self.is_train = is_train

    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)

    def __getitem__(self, index):

        ann = self.ann[index]
        img_dir = ann["image"]
        image_dir_all = f"{self.root_dir}/{img_dir}"

        try:
            image = Image.open(image_dir_all).convert("RGB")
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")

        W, H = image.size
        has_bbox = False
        mask = np.zeros((self.image_res, self.image_res, 1))
        try:
            x, y, w, h = self.get_bbox(ann["fake_image_box"])
            has_bbox = True
        except Exception:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)

        if has_bbox:
            if do_hflip:
                x = (W - x) - w

            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            mask_x = math.floor(x)
            mask_y = math.floor(y)
            mask_w = math.ceil(w)
            mask_h = math.ceil(h)

            mask[mask_y : mask_y + mask_h, mask_x : mask_x + mask_w, :] = 1

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor(
                [center_x / self.image_res, center_y / self.image_res, w / self.image_res, h / self.image_res],
                dtype=torch.float,
            )

        label = ann["fake_cls"]
        caption = pre_caption(ann["text"], self.max_words)
        fake_text_pos = ann["fake_text_pos"]

        fake_text_pos_list = torch.zeros(self.max_words)
        mask = torch.tensor(mask[None, ..., 0]).float()

        for i in fake_text_pos:
            if i < self.max_words:
                fake_text_pos_list[i] = 1

        conversation = []
        conversation.append({"from": "human", "value": describe_temple + caption + describe_ques_latter})
        conversation.append({"from": "gpt", "value": describles_answ[label]})

        return image, conversation, label, caption, fake_image_box, fake_text_pos_list, W, H, mask

    def collate(self, instances):

        images = []
        texts = []
        class_names = []
        masks = []
        captions = []
        fake_image_boxs = []
        fake_text_pos_list = []

        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[2])
            masks.append(instance[8])
            captions.append(instance[3])
            fake_image_boxs.append(instance[4])
            fake_text_pos_list.append(instance[5])

        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            captions=captions,
            fake_image_boxs=fake_image_boxs,
            fake_text_pos_list=fake_text_pos_list,
        )
