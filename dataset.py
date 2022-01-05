import random
import pickle
import gzip

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms

seed = 42

class DatasetGenerator(Dataset):
    def __init__(self, images, labels, pad, transform=None):
        super(DatasetGenerator, self).__init__()
        
        self.images = images
        self.labels = labels
        
        self.pad = pad
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[:,:,index]
        label = self.labels[:,:,index]
        
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.transform(label)
        
        image = np.array(image, dtype=np.float64)
        image = np.pad(image, pad_width=self.pad)## pad for valid padding
        image *= 1/image.max()
        image = np.expand_dims(image, axis=0)## compatible with input size requirements
        
        label = np.array(label, dtype=np.int64)
        
        return image, label
    
    def __len__(self):
        return self.images.shape[2]

def _get_frames(data, resize, augs={}):
    ##box = data['box']
    label_idxs = data['frames']
    
    images = data['video'][:,:,label_idxs]
    labels = data['label'][:,:,label_idxs].astype(np.uint8)
    
    frames_augmented = []

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(min(images.shape[0], images.shape[1])),
        transforms.Resize(size=resize['output_size'])
        ])
    frames_augmented.append(DatasetGenerator(images, labels, resize['pad_size'], tf))
    
    for key, value in augs.items():## offline implementation, no combinatorial methods here!
        if key == 'rotate':
            tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(min(images.shape[0], images.shape[1])),
            transforms.RandomAffine(degrees=value),
            transforms.Resize(size=resize['output_size'])
            ])
        elif key == 'scale':
            tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(min(images.shape[0], images.shape[1])),
            transforms.RandomAffine(degrees=0,scale=value),
            transforms.Resize(size=resize['output_size'])
            ])
        elif key == 'translate':
            tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(min(images.shape[0], images.shape[1])),
            transforms.RandomAffine(degrees=0,translate=value),
            transforms.Resize(size=resize['output_size'])
            ])
        elif key == 'shear':
            tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(min(images.shape[0], images.shape[1])),
            transforms.RandomAffine(degrees=0,shear=value),
            transforms.Resize(size=resize['output_size'])
            ])
        ## TODO: deformation
        else:
            print('augmentation method %s not implemented, thus omitted'%key)
            continue

        frames_augmented.append(DatasetGenerator(images, labels, resize['pad_size'], tf))

    return ConcatDataset(frames_augmented)

def _get_dataset(train_path, resize, augs, val_size):
    with gzip.open(train_path, 'rb') as f:
        videos = pickle.load(f)
    
    ## TODO: using only expert
    ## TODO: val_size = 0
    train_videos, val_videos = train_test_split(videos, test_size=val_size, random_state=seed, shuffle=True)
    
    train_data = ConcatDataset([_get_frames(data, resize, augs=augs) for data in train_videos])
    val_data = ConcatDataset([_get_frames(data, resize) for data in val_videos])

    return train_data, val_data

def get_loader(train_path, resize, augs, val_size, batch_size, num_workers):
    train_data, val_data = _get_dataset(train_path, resize, augs, val_size)
    ## TODO: val_size = 0

    print('train size:', len(train_data))
    print('val size:', len(val_data))
    
    worker_init_fn = lambda worker_id: random.seed(torch.initial_seed() + worker_id)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn)
    
    return train_loader, val_loader