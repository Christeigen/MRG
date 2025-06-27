import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from encoder_layer import CLIPWithLoRA
from configs.config import DataConfig, EncoderConfig

class BaseDataset(Dataset):
    def __init__(self, tokenizer, split, config = DataConfig(), preprocess=None):
        self.image_dir = config.image_dir
        self.annotation_path = config.annotation_path
        self.max_seq_length = config.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.annotation = json.loads(open(self.annotation_path, 'r').read())

        self.examples = self.annotation[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

class Iuxray(BaseDataset):
    def __init__(self, tokenizer, split, config=None, preprocess=None, encoder=None):
        super().__init__(tokenizer, split, config or DataConfig(), preprocess)
        self.encoder = encoder

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert("RGB")
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert("RGB")

        if self.preprocess and self.encoder:
            image_1 = self.preprocess(image_1)
            image_2 = self.preprocess(image_2)

        image = torch.stack((image_1, image_2), 0)
        return (image_id, image, example['ids'], example['mask'], len(example['ids']))

class CustomDataLoader(DataLoader):
    def __init__(self, split, batch_size, num_workers, tokenizer, shuffle):
        self.encoder = CLIPWithLoRA(EncoderConfig())

        self.dataset = Iuxray(
            tokenizer=tokenizer,
            split=split,
            config=DataConfig(),
            preprocess=self.encoder.get_preprocess(),
            encoder=self.encoder
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers
        )

    @staticmethod
    def collate_fn(data, pad_token_id=0, max_seq_len=100):
        input_ids, images_features, reports_ids, reports_masks, seq_lengths = zip(*data)
        images_features = torch.stack(images_features, 0)

        # Fixed padding length (e.g., max_seq_len), not dynamic from longest in batch
        targets = torch.full((len(reports_ids), max_seq_len), pad_token_id, dtype=torch.long)
        targets_masks = torch.zeros((len(reports_ids), max_seq_len), dtype=torch.float)

        for i, (report_ids, report_masks) in enumerate(zip(reports_ids, reports_masks)):
            length = min(len(report_ids), max_seq_len)
            targets[i, :length] = torch.tensor(report_ids[:length], dtype=torch.long)
            targets_masks[i, :length] = torch.tensor(report_masks[:length], dtype=torch.float)

        return targets, images_features, targets_masks