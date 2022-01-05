import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
from time import time

from PIL import Image
import random
from itertools import chain

N_CHANNELS = 3
datapath = 'D:/Deep Learning Practice/FireDataset/Classify/fire_daraset_classification_compression'

def read_split_data(root: str, val_rate: float = 0.2, plot_image: bool = False):
    
    random.seed(0) 
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    data_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    
    data_class.sort()
    
    class_indices = dict((classname, index) for index, classname in enumerate(data_class))
        
    supported = [".jpg", ".JPG", ".png", ".PNG"]  
    
    images_path = []
    images_label = []
    
    for classname in data_class:

        class_folderpath = os.path.join(root, classname)

        images = [os.path.join(root, classname, imagefile) for imagefile in os.listdir(class_folderpath)
                  if os.path.splitext(imagefile)[-1] in supported]
        
        image_class = class_indices[classname]
        label = [image_class] * len(images)
        
        images_path.append(images)        
        images_label.append(label)
                                   
    images_path = list(chain.from_iterable(images_path))
    images_label = list(chain.from_iterable(images_label))
    print(f"{len(images_path)} images were found in the dataset.")
    print(f"{len(images_label)} lable were found in the dataset.")

    return images_path, images_label

class MyDataSet(Dataset):
    def __init__(self, image_path: list, image_labels: list, transforms = None):
        self.image_path = image_path
        self.image_labels = image_labels
        self.transforms = transforms
        
    def __getitem__(self, item):
        
        img = Image.open(self.image_path[item])
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        elif img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[item]} isn't RGB mode.")
            
        label = self.image_labels[item]
        
        if self.transforms != None:
            img = self.transforms(img)
            
        return img, label
        
    def __len__(self):
        return len(self.image_path)
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
        

images_path, image_label = read_split_data(datapath)

transforms = transforms.Compose([transforms.Resize(384),
                                 transforms.CenterCrop(384),
                                 transforms.ToTensor()])

dataset = MyDataSet(image_path = images_path,
                    image_labels = image_label,
                    transforms = transforms)


data_loader = DataLoader(dataset,
                         batch_size = 8,
                         shuffle    = False,
                         collate_fn = dataset.collate_fn)



MEAN = [0.051, 0.045, 0.041]
STD = [0.031, 0.03, 0.03]

before = time()
mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)
print('==> Computing mean and std..')
for inputs, _labels in tqdm(data_loader):
    for i in range(N_CHANNELS):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
mean.div_(len(dataset))
std.div_(len(dataset))

print(f"Mean: {mean}, Std: {std}")
print("time elapsed: ", time()-before)