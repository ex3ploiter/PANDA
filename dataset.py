from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                    #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                    ])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

mvtec_labels = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
                'wood', 'zipper']


class normalizC:
    def __init__(self,mu,std) -> None:
        self.mu=mu 
        self.std=std
    def normalize(self,X):
        return (X - self.mu)/self.

def get_dataloader(dataset='cifar', path='~/mydataset', normal_class_indx = 0, batch_size=8):
    if dataset == 'cifar10':
        ds_mean = (0.4914, 0.4822, 0.4465) 
        ds_std = (0.2471, 0.2435, 0.2616)   

        mu = torch.tensor(ds_mean).view(3,1,1).cuda()
        std = torch.tensor(ds_std).view(3,1,1).cuda()      
        normal_obj=normalizC(mu,std)
        
        return get_CIFAR10(normal_class_indx, batch_size, path),normal_obj
    elif dataset == 'mnist':
        ds_mean = (0.1307, 0.1307, 0.1307)
        ds_std = (0.3081, 0.3081, 0.3081) 
        
        mu = torch.tensor(ds_mean).view(3,1,1).cuda()
        std = torch.tensor(ds_std).view(3,1,1).cuda()      
        normal_obj=normalizC(mu,std),
        
               
        return get_MNIST(normal_class_indx, batch_size, path),normal_obj
    elif dataset == 'fashion':
        ds_mean =(0.2859, 0.2859, 0.2859)
        ds_std = (0.3530, 0.3530, 0.3530)
        
        mu = torch.tensor(ds_mean).view(3,1,1).cuda()
        std = torch.tensor(ds_std).view(3,1,1).cuda()
        normal_obj=normalizC(mu,std)      
        

        return get_FASHION_MNIST(normal_class_indx, batch_size, path),normal_obj
    elif dataset == 'svhn':
        ds_mean =(0.491373, 0.482353, 0.446667)
        ds_std =(0.247059, 0.243529, 0.261569)  
        
        mu = torch.tensor(ds_mean).view(3,1,1).cuda()
        std = torch.tensor(ds_std).view(3,1,1).cuda()  
        normal_obj=normalizC(mu,std)    

              
        return get_SVHN(normal_class_indx, batch_size, path),normal_obj
    elif dataset == 'mvtec':
        ds_mean=(0.485, 0.456, 0.406)
        ds_std=(0.229, 0.224, 0.225)    
        
        mu = torch.tensor(ds_mean).view(3,1,1).cuda()
        std = torch.tensor(ds_std).view(3,1,1).cuda()      
        normal_obj=normalizC(mu,std)
            
        return get_MVTEC(normal_class_indx, batch_size, path),normal_obj
    else:
        raise Exception("Dataset is not supported yet. ")


def get_CIFAR10(normal_class_indx, batch_size, path):
    trainset = CIFAR10(root=path, train=True, download=True, transform=transform_color)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root=path, train=False, download=True, transform=transform_color)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader



def get_MNIST(normal_class_indx, batch_size, path):
    
    trainset = MNIST(root=path, train=True, download=True, transform=transform_gray)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = MNIST(root=path, train=False, download=True, transform=transform_gray)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_FASHION_MNIST(normal_class_indx, batch_size, path):

    trainset = FashionMNIST(root=path, train=True, download=True, transform=transform_gray)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = FashionMNIST(root=path, train=False, download=True, transform=transform_gray)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_SVHN(normal_class_indx, batch_size, path):

    trainset = SVHN(root=path, split='train', download=True, transform=transform_color)
    trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx]
    trainset.labels  = [0 for t in trainset.labels]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = SVHN(root=path, split='test', download=True, transform=transform_color)
    testset.labels  = [int(t!=normal_class_indx) for t in testset.labels]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, normal=True):
        self.transform = transform
        if train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          self.image_files = image_files

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.image_files)


def get_MVTEC(normal_class_indx, batch_size, path):
    normal_class = mvtec_labels[normal_class_indx]

    trainset = MVTecDataset(path, normal_class, transform_color, train=True)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)

    testset = MVTecDataset(path, normal_class, transform_color, train=False)
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=batch_size)  

    return train_loader, test_loader


def download_and_extract_mvtec(path):
    import os
    import wget
    import tarfile
    so.extractall(path=os.environ['BACKUP_DIR'])
    url = 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
    filename = wget.download(url, out=path)
    with tarfile.open(os.path.join(path, filename)) as so:
        so.extractall(path=os.path.join(path, "mvtec_anomaly_detection"))