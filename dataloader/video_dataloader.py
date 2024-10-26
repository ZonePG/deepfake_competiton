import glob
import os
import os.path

import numpy as np
import torch
import torchvision
from numpy.random import randint
from PIL import Image
from torch.utils import data

from dataloader.video_transform import GroupRandomHorizontalFlip
from dataloader.video_transform import GroupRandomSizedCrop
from dataloader.video_transform import GroupResize
from dataloader.video_transform import Stack
from dataloader.video_transform import ToTorchFormatTensor


class VideoRecord:
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(
        self, video_path, list_file, num_segments, duration, mode, transform, image_size
    ):
        self.video_path = video_path
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()

    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, class_idx]
        #
        tmp = [x.strip().split(" ") for x in open(self.list_file)]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        self.class_num = [0] * 2
        for i in range(len(self.video_list)):
            self.class_num[self.video_list[i].label] += 1
        self.max_num = max(self.class_num)
        self.class_weights = [self.max_num / self.class_num[i] for i in range(2)]

    def _get_train_indices(self, record):
        #
        # Split all frames into seg parts, then select frame in each part randomly
        #
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration
            ) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(
                randint(record.num_frames - self.duration + 1, size=self.num_segments)
            )
        else:
            offsets = np.pad(
                np.array(list(range(record.num_frames))),
                (0, self.num_segments - record.num_frames),
                "edge",
            )
        return offsets

    def _get_val_indices(self, record):
        #
        # Split all frames into seg parts, then select frame in the mid of each part
        #
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
            )
        else:
            offsets = np.pad(
                np.array(list(range(record.num_frames))),
                (0, self.num_segments - record.num_frames),
                "edge",
            )
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == "train":
            segment_indices = self._get_train_indices(record)
        elif self.mode == "val":
            segment_indices = self._get_val_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_frames_path = glob.glob(os.path.join(self.video_path, record.path, "*"))
        video_frames_path.sort()
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for _i in range(self.duration):
                seg_imgs = [
                    Image.open(os.path.join(video_frames_path[p])).convert("RGB")
                ]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        return images, record.label

    def __len__(self):
        return len(self.video_list)

    def get_labels(self):
        return [record.label for record in self.video_list]


def train_data_loader(list_file, num_segments, duration, image_size, args):
    train_transforms = torchvision.transforms.Compose(
        [
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
        ]
    )

    train_data = VideoDataset(
        video_path=args.video_path,
        list_file=list_file,
        num_segments=num_segments,
        duration=duration,
        mode="train",
        transform=train_transforms,
        image_size=image_size,
    )
    return train_data


def val_data_loader(list_file, num_segments, duration, image_size, args):
    test_transform = torchvision.transforms.Compose(
        [GroupResize(image_size), Stack(), ToTorchFormatTensor()]
    )

    val_data = VideoDataset(
        video_path=args.video_path,
        list_file=list_file,
        num_segments=num_segments,
        duration=duration,
        mode="val",
        transform=test_transform,
        image_size=image_size,
    )
    return val_data
