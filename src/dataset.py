import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# util functions to load the CompCars processed dataset
# also applies mean and normalization from ImageNet weights


# imagenet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CompCarsDataset(Dataset):
    # Dataset for compcars classification supporting make and model tasks

    def __init__(
        self,
        root_dir: str, # path to dataset/data folder
        split_file: str, #path to train.txt or test.txt
        transform: Optional[transforms.Compose] = None, #transforms to apply
        task: str = 'make', 
        # 'make' for 163-class
        # 'model' for fine-grained classification
        use_preprocessed: bool = False, #load from image_256 folder if true
        metadata_file: Optional[str] = None # optional for consistent label mappings
    ):
        

        self.root_dir = Path(root_dir)
        if use_preprocessed:
            self.image_dir = self.root_dir / 'image_256'
        else:
            self.image_dir = self.root_dir / 'image'
        self.transform = transform
        self.task = task

        # load image paths and labels from split file
        self.image_paths = []
        self.make_labels = []
        self.model_labels = []
        # build mappings from folder ids to class indices
        self.make_id_to_idx = {}
        self.model_id_to_idx = {}

        # parse split file
        self._load_split_file(split_file, metadata_file)

        # store class counts
        self.num_makes = len(self.make_id_to_idx)
        self.num_models = len(self.model_id_to_idx)

    def _load_split_file(self, split_file: str, metadata_file: Optional[str] = None) -> None:
        # parse the split file and build label mappings

        split_path = Path(split_file)
        with open(split_path, 'r') as f:
            lines = f.readlines()

        # if metadata file is present, use it's mappings for consistency with ProcessedCompCarsDataset
        if metadata_file is not None:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # build model_id_to_idx from metadata's class_to_idx
            for class_name, idx in metadata['class_to_idx'].items():
                parts = class_name.split('_')
                make_id = int(parts[0])
                model_id = int(parts[1])
                model_key = (make_id, model_id)
                self.model_id_to_idx[model_key] = idx

            # build make_id_to_idx (same logic as ProcessedCompCarsDataset)
            unique_makes = set()
            for class_name in metadata['class_to_idx'].keys():
                make_id = int(class_name.split('_')[0])
                unique_makes.add(make_id)

            sorted_makes = sorted(list(unique_makes))
            for idx, make_id in enumerate(sorted_makes):
                self.make_id_to_idx[make_id] = idx
        else:
            # original logic: build mappings from split file (tuple sorting)
            unique_makes = set()
            unique_models = set()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # format: make_id/model_id/year/image_name.jpg
                parts = line.split('/')
                make_id = int(parts[0])
                model_id = int(parts[1])

                unique_makes.add(make_id)
                # model is identified by make_id/model_id pair
                model_key = (make_id, model_id)
                unique_models.add(model_key)

            # sort and create mappings for consistent indexing
            sorted_makes = sorted(list(unique_makes))
            for idx, make_id in enumerate(sorted_makes):
                self.make_id_to_idx[make_id] = idx

            sorted_models = sorted(list(unique_models))
            for idx, model_key in enumerate(sorted_models):
                self.model_id_to_idx[model_key] = idx

        # load image paths and labels from split file
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split('/')
            make_id = int(parts[0])
            model_id = int(parts[1])

            # store image path
            image_path = self.image_dir / line
            self.image_paths.append(image_path)

            # store labels
            make_label = self.make_id_to_idx[make_id]
            self.make_labels.append(make_label)

            model_key = (make_id, model_id)
            model_label = self.model_id_to_idx[model_key]
            self.model_labels.append(model_label)

    def __len__(self) -> int:
        length = len(self.image_paths)
        return length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # return label based on task
        if self.task == 'make':
            label = self.make_labels[idx]
        elif self.task == 'model':
            label = self.model_labels[idx]
        elif self.task == 'hierarchical':
            # return both labels for hierarchical classification
            make_label = self.make_labels[idx]
            model_label = self.model_labels[idx]
            return image, make_label, model_label
        else:
            label = self.make_labels[idx]

        return image, label

    def get_class_counts(self) -> Dict[int, int]:
        # count samples per class based on current task
        if self.task == 'make':
            labels = self.make_labels
        else:
            labels = self.model_labels

        counts = {}
        for label in labels:
            if label not in counts:
                counts[label] = 0
            counts[label] = counts[label] + 1

        return counts

    def get_class_weights(self) -> torch.Tensor:
        # cCompute class weights for handling imbalance
        # weight[c] = n_total / (n_classes * n_c)
        counts = self.get_class_counts()
        n_total = len(self.image_paths)
        n_classes = len(counts)
        weights = []
        for class_idx in range(n_classes):
            n_c = counts[class_idx]
            weight = n_total / (n_classes * n_c)
            weights.append(weight)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return weights_tensor

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    # training transforms with data augmentation
    transform_list = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    composed = transforms.Compose(transform_list)
    return composed

def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    # Validation/test transforms without augmentation (but still resize and crop to match training input size)
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    composed = transforms.Compose(transform_list)
    return composed

# TO-DO: probably not used anymore.. god i wrote too much stuff
def get_preprocessed_transforms(image_size: int = 224) -> transforms.Compose:
    # transforms for pre-resized images (skips resize step)
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    composed = transforms.Compose(transform_list)
    return composed

# TO-DO: probably not used anymore
def get_preprocessed_val_transforms(image_size: int = 224) -> transforms.Compose:
    # validation transforms for pre-resized images (only center crop and normalization, no resize needed)

    transform_list = [
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    composed = transforms.Compose(transform_list)
    return composed


def get_dataloader( # helper function to create dataloader with optimal settings for training/evaluation
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    # create optimized dataloader for training/evaluation
    # multiprocessing options only work with num_workers > 0
    if num_workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=drop_last
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    return loader


def create_train_val_split(
    dataset: CompCarsDataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    # create stratified train/validation split indices

    np.random.seed(seed)

    # group indices by class
    class_indices = {}

    if dataset.task == 'make':
        labels = dataset.make_labels
    else:
        labels = dataset.model_labels

    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    # split each class proportionally
    for class_label in class_indices:
        indices = class_indices[class_label]
        np.random.shuffle(indices)

        n_val = int(len(indices) * val_ratio)
        if n_val < 1:
            n_val = 1
        if n_val >= len(indices):
            n_val = len(indices) - 1

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return train_indices, val_indices


class ProcessedCompCarsDataset(Dataset):
    # dataset for pre-processed compcars images (224x224, bbox cropped)

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        task: str = 'make'
    ):
        # root_dir: path to dataset/processed folder
        # split: 'train', 'val', or 'test'
        # transform: torchvision transforms to apply
        # task: 'make', 'model', or 'hierarchical'

        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.task = task

        # load metadata
        import json
        metadata_path = self.root_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # class mappings from metadata (model-level: makeId_modelId -> idx)
        self.class_to_idx = self.metadata['class_to_idx']
        self.idx_to_class = self.metadata['idx_to_class']

        # build make mappings from class names
        self.make_id_to_idx = {}
        unique_makes = set()
        for class_name in self.class_to_idx.keys():
            make_id = int(class_name.split('_')[0])
            unique_makes.add(make_id)

        sorted_makes = sorted(list(unique_makes))
        for idx, make_id in enumerate(sorted_makes):
            self.make_id_to_idx[make_id] = idx

        # store class counts
        self.num_makes = len(self.make_id_to_idx)
        self.num_models = len(self.class_to_idx)

        # load image paths and labels
        self.image_paths = []
        self.make_labels = []
        self.model_labels = []

        self._load_images()

    def _load_images(self) -> None:
        # scan split directory for images

        for class_folder in sorted(self.split_dir.iterdir()):
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name
            if class_name not in self.class_to_idx:
                continue

            # get model label from class name
            model_label = self.class_to_idx[class_name]

            # get make label from class name (makeId_modelId -> makeId)
            make_id = int(class_name.split('_')[0])
            make_label = self.make_id_to_idx[make_id]

            # load all images in this class folder
            for image_file in class_folder.iterdir():
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_paths.append(image_file)
                    self.make_labels.append(make_label)
                    self.model_labels.append(model_label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # return label based on task
        if self.task == 'make':
            return image, self.make_labels[idx]
        elif self.task == 'model':
            return image, self.model_labels[idx]
        elif self.task == 'hierarchical':
            return image, self.make_labels[idx], self.model_labels[idx]
        else:
            return image, self.make_labels[idx]

    def get_class_counts(self) -> Dict[int, int]:
        # count samples per class based on current task

        if self.task == 'make':
            labels = self.make_labels
        else:
            labels = self.model_labels

        counts = {}
        for label in labels:
            if label not in counts:
                counts[label] = 0
            counts[label] = counts[label] + 1

        return counts

    def get_class_weights(self) -> torch.Tensor:
        # compute class weights for handling imbalance

        counts = self.get_class_counts()
        n_total = len(self.image_paths)
        n_classes = len(counts)

        weights = []
        for class_idx in range(n_classes):
            n_c = counts.get(class_idx, 1)
            weight = n_total / (n_classes * n_c)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


def get_processed_train_transforms(image_size: int = 224) -> transforms.Compose:
    #training transforms for pre-processed 224x224 images (no resize needed)

    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    return transforms.Compose(transform_list)


def get_processed_val_transforms(image_size: int = 224) -> transforms.Compose:
    # Validation transforms for pre-processed images (minimal processing)

    transform_list = [
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    return transforms.Compose(transform_list)


if __name__ == '__main__':
    #Quick test of dataset loading

    root_dir = 'dataset/data'
    split_file = 'dataset/data/train_test_split/classification/train.txt'

    # test make classification
    train_transform = get_train_transforms()
    dataset = CompCarsDataset(
        root_dir=root_dir,
        split_file=split_file,
        transform=train_transform,
        task='make'
    )

    print(f"dataset size: {len(dataset)}")
    print(f"number of makes: {dataset.num_makes}")
    print(f"number of models: {dataset.num_models}")

    # test loading one sample
    image, label = dataset[0]
    print(f"image shape: {image.shape}")
    print(f"label: {label}")

    # test class weights
    weights = dataset.get_class_weights()
    print(f"class weights shape: {weights.shape}")
    print(f"weight range: [{weights.min():.3f}, {weights.max():.3f}]")
