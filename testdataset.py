import pickle
import gzip

import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms

class DatasetGenerator(Dataset):
    def __init__(self, images, name, pad, transform=None):
        super(DatasetGenerator, self).__init__()
        
        self.images = images
        self.name = name
        
        self.pad = pad
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[:,:,index]
        origin_shape = np.array(self.images.shape, dtype=np.int16)##converted to tensor by default collate_fn
        
        if self.transform:
            image = self.transform(image)
        
        image = np.array(image, dtype=np.float32)
        output_shape = np.array(image.shape, dtype=np.int16)##converted to tensor by default collate_fn

        image = np.pad(image, pad_width=self.pad)## pad for valid padding
        image *= 1/image.max()
        image = np.expand_dims(image, axis=0)## compatible with input size requirements
        
        return image, self.name, index, origin_shape, output_shape
    
    def __len__(self):
        return self.images.shape[2]

def _get_frames(data, resize):
    images = data['video']
    name = data['name']
    
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(min(images.shape[0], images.shape[1])),
        transforms.Resize(size=resize['output_size']),
    ])
    
    return DatasetGenerator(images, name, resize['pad_size'], tf)


def _get_dataset(test_path, resize):
    with gzip.open(test_path, 'rb') as f:
        test_videos = pickle.load(f)
    
    test_data = ConcatDataset([_get_frames(data, resize) for data in test_videos])

    return test_data


def get_loader(test_path, resize):
    test_data = _get_dataset(test_path, resize)
    
    test_loader = DataLoader(test_data, batch_size=1, pin_memory=True, shuffle=False)
    
    return test_loader