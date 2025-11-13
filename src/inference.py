# -*- coding: utf-8 -*-

import os
os.chdir('./coad_tiles')
best_model = '../read_tiles/model3.pth'
workdir = '../read_train_coad_test'
os.makedirs(workdir, exist_ok=True)
root = os.getcwd()
print("The Current working directory is :", root)

import sys
import numpy as np
import pandas as pd
import random
import os
import os.path as osp

import torch
import torchvision
from PIL import Image
import torch.utils.data as data
import torchvision
from torchvision.models.resnet import resnet34
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms

from torchvision import transforms, datasets
from torch.utils.data import  SubsetRandomSampler, Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

# saving the plots
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
plt.rcParams['pdf.fonttype'] = 42
plt.switch_backend('agg')


# Define training and test modules
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    
    for images, labels in dataloader:
        images,labels = images.to(device),labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predictions = torch.max(output, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss, train_correct
  
def valid_epoch_2(model, device, dataloader, loss_fn):
    valid_loss = 0.0
    model.eval()
    
    y_gts=[]
    y_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:

            images, labels = images.to(device),labels.to(device)

            y_gts.append(labels.cpu().numpy())
            
            output = model(images)
            probs = F.softmax(output, dim=1)
            y_probs.append(probs.cpu().numpy())
            
            loss=loss_fn(output,labels)
            valid_loss+=loss.item()*images.size(0)


    return valid_loss, y_gts, y_probs

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2,bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer = nn.Linear(512, 4) 
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x
    
if __name__ == '__main__':
    
    PIL_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomRotation(45)])

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        PIL_transform,
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet
    ])
    # 
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    import os
    def is_valid_file(pth):
        #if file name starts (not the whole path) with '.' then it is not a valid file, also must be .jpg
        filename = os.path.basename(pth)
        return not filename.startswith('.') and filename.lower().endswith(('.jpg', '.jpeg', '.png'))
     

    cell_dataset = datasets.ImageFolder(root, transform=transform_train, is_valid_file=is_valid_file)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    
    k_folds = 10
    num_epochs = 5
    batch_size = 16
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    y_gt_folds=[]
    y_pred_folds=[]
    y_probs_folds=[]

    ##### Test best model on test set #####
    resnet = resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    model = Net(resnet)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=5e-4,amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(best_model))

    test_loader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_labels, test_probs = valid_epoch_2(model,device,test_loader,criterion)

    test_labels = np.concatenate(test_labels, axis=0)
    test_probs = np.concatenate(test_probs, axis=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_correct = np.sum(test_preds == test_labels)

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset) * 100

    # Classification report for test set
    print('Classification report for test set')
    report = classification_report(test_labels, test_preds)
    print(report)
    with open(osp.join(workdir, 'classification_report_test.txt'), 'w') as f:
        f.write(report)
        
    # Confusion matrix for test set
    print('Confusion matrix for test set')
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    with open(osp.join(workdir, 'confusion_matrix_test.txt'), 'w') as f:
        print(cm, file=f)
        
    class_names = [f"CMS{c}" for c in range(1, 5)]
    
    macro_auc_test = roc_auc_score(test_labels, test_probs, average="macro", multi_class="ovr")
    micro_auc_test = roc_auc_score(test_labels, test_probs, average="micro", multi_class="ovr")

    plt.figure(figsize=(8, 6))
    fpr_macro = np.linspace(0, 1, 100) 
    tpr_macro = np.zeros_like(fpr_macro)
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((test_labels == i).astype(int), test_probs[:, i])
        tpr_macro += np.interp(fpr_macro, fpr, tpr)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    tpr_macro /= len(class_names)
    
    n_classes = len(class_names)
    test_labels_one_hot = label_binarize(test_labels, classes=np.arange(n_classes))
    fpr_micro, tpr_micro, _ = roc_curve(test_labels_one_hot.ravel(), test_probs.ravel())

    plt.plot(fpr_macro, tpr_macro, linestyle=':', color='deeppink', lw=2, label=f'Macro-average (AUC = {macro_auc_test:.2f})')
    plt.plot(fpr_micro, tpr_micro, linestyle=':', color='navy', lw=2, label=f'Micro-average (AUC = {micro_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.title('ROC Curve for Test Set (with Macro & Micro AUCs)')
    plt.legend(loc="lower right")
    plt.savefig(osp.join(workdir, 'roc_curve_test.png'))
    plt.savefig(osp.join(workdir, 'roc_curve_test.pdf'))
    plt.close()
    

    
    