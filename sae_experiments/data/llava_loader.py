"""LLaVA dataset and data loader for VQA inference.

Extracted from root-level InformationFlow.py. Provides CustomDataset and
create_data_loader for running LLaVA inference on image-question pairs.
"""

import copy
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

try:
    from utils import create_mask_with_bbox
except ImportError:
    create_mask_with_bbox = None


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.image_processor_mask = copy.deepcopy(image_processor)
        self.model_name = get_model_name_from_path(self.model_config._name_or_path)
        self.task_name = task_name
        self.conv_mode = conv_mode

        if self.model_name == "llama3-llava-next-8b" or self.model_name == "llava-v1.6-vicuna-7b" or self.model_name == "llava-v1.5-7b" or self.model_name == "llava-v1.5-13b":
            self.image_processor_mask.do_normalize=False
            self.image_processor_mask.do_rescale=False
        elif self.model_name == "llava-next-qwen-32b":
            self.image_processor_mask.image_mean = (0, 0, 0)
            self.image_processor_mask.image_std = (1, 1, 1)
            self.image_processor_mask.rescale_factor = 1

    def __getitem__(self, index):

        line = self.questions[index]
        question = line["question"]
        question = question + " \nAnswer the question using a single word or phrase."
        _img_id_str = str(line["img_id"])
        _known_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        image_file = _img_id_str if any(_img_id_str.lower().endswith(e) for e in _known_exts) else _img_id_str + ".jpg"


        qs = DEFAULT_IMAGE_TOKEN + "\n" + question  #

        conv = copy.deepcopy(conv_templates[self.conv_mode])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if self.model_name == "llama3-llava-next-8b":
            prompt+=" \n"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")

        image_tensor = process_images([image], self.image_processor, self.model_config)
        image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]

        if self.task_name == "CompareAttr" or self.task_name == "ChooseRel" or self.task_name == "LogicalObj":
            bounding_boxes=[]
            bounding_boxes.append((int(line[f'object1 x']), int(line[f'object1 y']), int(line[f'object1 x'])+int(line[f'object1 w']), int(line[f'object1 y'])+int(line[f'object1 h'])))
            if line[f'object2 x'] !="-":
                bounding_boxes.append((int(line[f'object2 x']), int(line[f'object2 y']), int(line[f'object2 x'])+int(line[f'object2 w']), int(line[f'object2 y'])+int(line[f'object2 h'])))
        elif self.task_name=="ChooseAttr" or self.task_name=="ChooseCat" or self.task_name=="QueryAttr":
            bounding_boxes = [(int(line['central object x']), int(line['central object y']), int(line['central object x'])+int(line['central object w']), int(line['central object y'])+int(line['central object h']))]
        else:
            bounding_boxes = None

        if bounding_boxes is not None and create_mask_with_bbox is not None:
            mask = create_mask_with_bbox(image, bounding_boxes)
            mask_tensor = process_images([mask], self.image_processor_mask, self.model_config)
            mask_tensor = [_image.to(dtype=torch.float16) for _image in mask_tensor]
        else:
            mask_tensor=None

        image_sizes = [image.size]
        return input_ids, image_tensor, image_sizes, prompt, mask_tensor




    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, prompts, mask_tensors = zip(*batch)

    input_ids = input_ids[0]
    image_tensors = image_tensors[0]
    image_sizes=image_sizes[0]
    mask_tensors=mask_tensors[0]
    return input_ids, image_tensors, image_sizes, prompts, mask_tensors



def create_data_loader(questions, image_folder, batch_size, num_workers, tokenizer, image_processor, model_config, task_name, conv_mode):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader
