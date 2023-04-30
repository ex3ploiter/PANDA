import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
from torchvision.datasets import  ImageFolder
from tqdm import tqdm 


mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                    #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=True, progress=True)



class Normal_Model(torch.nn.Module):
    def __init__(self, resnet_type):
        super().__init__()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mu = torch.tensor(mean).view(3,1,1).cuda()
        std = torch.tensor(std).view(3,1,1).cuda()        
        self.norm = lambda x: ( x - mu ) / std

        if resnet_type == 18:
            self.backbone=ResNet.resnet18(pretrained=True, progress=True)
        elif resnet_type == 50:
            self.backbone=ResNet.wide_resnet50_2(pretrained=True, progress=True)
        elif resnet_type == 101:
            self.backbone=ResNet.resnet101(pretrained=True, progress=True)
        else:  #152
            self.backbone=ResNet.resnet152(pretrained=True, progress=True)

    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        
        return z1


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.backbone.conv1.parameters():
        p.requires_grad = False
    for p in model.backbone.bn1.parameters():
        p.requires_grad = False
    for p in model.backbone.layer1.parameters():
        p.requires_grad = False
    for p in model.backbone.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.backbone.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

def get_loaders(dataset, label_class, batch_size):
    if dataset in ['cifar10', 'fashion','BrainMRI','X-ray','Head-CT']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        
        elif dataset == "BrainMRI" or dataset == "X-ray" or dataset == "Head-CT":    
            if dataset == "BrainMRI" : # 2
                path1='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Training'
                path2='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Testing'
            elif dataset == "X-ray" : # 0
                path1='/mnt/new_drive/Sepehr/chest_xray/train'
                path2='/mnt/new_drive/Sepehr/chest_xray/test'

            elif dataset == "Head-CT" :# 1
#                 path1='/mnt/new_drive/Masoud_WorkDir/Transformaly_Test/head_ct/Train/'
#                 path2='/mnt/new_drive/Masoud_WorkDir/Transformaly_Test/head_ct/Test/'    
                path1='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/HEAD_CT/Train'
                path2='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/HEAD_CT/Test'
        
            transform = transform_color
            trainset = ImageFolder(root=path1, transform=transform)
            testset = ImageFolder(root=path2, transform=transform)       
                
            

        idx = np.array(trainset.targets) == label_class
        # testset.targets = [int(t != label_class) for t in testset.targets]
        testset.samples=[(pth,int(target!=label_class)) for (pth,target) in testset.samples]
        
        # trainset.data = trainset.data[idx]
        # trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        
        trainset.samples=[(pth,int(target!=label_class)) for (pth,target) in trainset.samples]
        trainset = torch.utils.data.Subset(trainset, idx)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()


def get_loaders_blackbox(dataset, label_class, batch_size):

    if dataset == "BrainMRI" or dataset == "X-ray" or dataset == "Head-CT":    
        
        # transform = transform_color if backbone == 152 else transform_resnet18
        if dataset == "BrainMRI" : # 2
            path1='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Training'
            path2='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Testing'
        elif dataset == "X-ray" : # 0
            path1='/mnt/new_drive/Sepehr/chest_xray/train'
            path2='/mnt/new_drive/Sepehr/chest_xray/test'
        
        elif dataset == "Head-CT" :# 0
            path1='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/HEAD_CT/Train'
            path2='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/HEAD_CT/Test'
        
        
        trainset = ImageFolder(root=path1, transform=transform_color)
        testset = ImageFolder(root=path2, transform=transform_color)

        
        trainset.samples=[(pth,int(target!=label_class)) for (pth,target) in trainset.samples]
        testset.samples=[(pth,int(target!=label_class)) for (pth,target) in testset.samples ]




        ds=torch.utils.data.ConcatDataset([trainset, testset])
        train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader


def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)




class Wrap_Model(torch.nn.Module):
    def __init__(self, model, train_loader):
        super().__init__()

        self.model = model

        self.train_feature_space = []
        with torch.no_grad():
            for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
                imgs = imgs.to('cuda')
                _, features = model(imgs)
                self.train_feature_space.append(features)
            self.train_feature_space = torch.cat(self.train_feature_space, dim=0).contiguous().cpu().numpy()
        


    def forward(self, x):
        test_adversarial_feature_space = []
        _, features = self.model(x)
        test_adversarial_feature_space.append(features.detach().cpu())
        test_adversarial_feature_space = torch.cat(test_adversarial_feature_space).detach().cpu().numpy()
        distances = knn_score(self.train_feature_space, test_adversarial_feature_space)
        
        return distances        