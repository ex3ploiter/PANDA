import numpy as np
import torch
from sklearn.metrics import roc_auc_score,accuracy_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torchattacks
import torch.nn.functional as F
from utils import *
from simba import *



class BB_Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mu = torch.tensor(mean).view(3,1,1).cuda()
        std = torch.tensor(std).view(3,1,1).cuda()        
        self.norm = lambda x: ( x - mu ) / std
        if backbone == 152:
            self.backbone = torchvision.models.resnet152(pretrained=True)
        else:
            self.backbone = torchvision.models.resnet18(pretrained=True)

        self.fc1=nn.Linear(1000,2)        
    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z1=self.fc1(z1)
        
        return z1


def train_model_blackbox(epoch, model, trainloader, device): 
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    soft = torch.nn.Softmax(dim=1)

    preds = []
    anomaly_scores = []
    true_labels = []
    running_loss = 0
    accuracy = 0

    
    with tqdm(trainloader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, targets) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            true_labels += targets.detach().cpu().numpy().tolist()

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()
            correct = (torch.tensor(preds) == torch.tensor(true_labels)).sum().item()
            accuracy = correct / len(preds)

            probs = soft(outputs).squeeze()
            anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()

            running_loss += loss.item() * data.size(0)

            tepoch.set_postfix(loss=running_loss / len(preds), accuracy=100. * accuracy)

        print("AUC : ",roc_auc_score(true_labels, anomaly_scores) )
        print("accuracy_score : ",accuracy_score(true_labels, preds, normalize=True) )

    return  model



def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))

    return model



def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    # with torch.no_grad():
    for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
        imgs = imgs.to(device)
        _, features = model(imgs)
        test_feature_space.append(features.detach().cpu())
    test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
    # test_labels = test_loader.dataset.targets
    test_labels=[j for (i,j) in test_loader.dataset.samples]

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)
    
    print("CLEAN AUC: ",auc)

    return auc, train_feature_space

def get_score_adv(model_normal, device, train_loader, test_loader):
    x_model = Wrap_Model(model_normal, train_loader)

    t = []
    l = []
    image_size = 224
    attack = SimBA(x_model, '', image_size)
    for (imgs, labels) in tqdm(test_loader, desc='Test set adversarial feature extracting'):
        imgs = imgs.to(device)
        labels = labels.to(device)
        #adv_imgs, adv_imgs_in, adv_imgs_out, labels= test_attack(imgs, labels)
        adv_data, _, _, _, _, _ = attack.simba_batch(
                imgs, labels, 10000, 224, 7, 1/255, linf_bound=0,
                order='rand', targeted=False, pixel_attack=True, log_every=0)
        t.append(x_model(adv_data))
        l.append(labels)

    t = np.concatenate(t)
    l = torch.cat(l).cpu().detach().numpy()
        
    auc=roc_auc_score(l, t)
    
    print("ADV AUC: ",auc)

    # return auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model=Normal_Model(args.resnet_type)
    model = model.to(device)

    ewc_loss = None




    

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    utils.freeze_parameters(model)
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
    model=train_model(model, train_loader, test_loader, device, args, ewc_loss)

    get_score(model, device, train_loader, test_loader)
    get_score_adv(model, device, train_loader, test_loader)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    main(args)