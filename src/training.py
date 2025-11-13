# -*- coding: utf-8 -*-

import os
os.chdir('./coad_tiles')
root = os.getcwd()
print("The Current working directory is :", root)

import sys
import numpy as np
import pandas as pd
import random
import os

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
#

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
    targets = np.array([t for _, t in cell_dataset.imgs])
    
    train_val_idx, test_idx = train_test_split(np.arange(len(targets)), test_size=0.1, random_state=42, stratify=targets)
    class_names = cell_dataset.classes 
    train_val_counts = np.bincount(targets[train_val_idx])  # Training + Validation
    test_counts = np.bincount(targets[test_idx])  # Test set
    
    ##

    train_call_dataset = Subset(cell_dataset, train_val_idx)
    test_call_dataset = Subset(cell_dataset, test_idx)
    
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

    # Start printP
    
    best_model = None
    best_acc = 0.0
    
    train_val_indices = np.array(train_val_idx)
    fold_data = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices)):
        
        train_counts = np.bincount(targets[train_val_idx[train_idx]], minlength=len(class_names))
        val_counts = np.bincount(targets[train_val_idx[val_idx]], minlength=len(class_names))
        fold_data.append([fold + 1] + train_counts.tolist() + val_counts.tolist())
        
        #reset model
        resnet = resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        model = Net(resnet)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=5e-4,amsgrad=True)
        criterion = nn.CrossEntropyLoss()
      
        # Print
        print('--------------------------------')
        print(f'FOLD {fold+1}')
        print('--------------------------------')
        
        train_subset = Subset(cell_dataset, train_val_indices[train_idx])
        val_subset = Subset(cell_dataset, train_val_indices[val_idx])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(num_epochs):

            print(f'Starting epoch {epoch+1}')
            
            train_loss, train_correct = train_epoch(model,device,train_loader,criterion,optimizer)
            val_loss, val_labels, val_probs = valid_epoch_2(model,device,val_loader,criterion)
            
            val_labels = np.concatenate(val_labels, axis=0)
            val_probs = np.concatenate(val_probs, axis=0)
            val_preds = np.argmax(val_probs, axis=1)
            val_correct = np.sum(val_preds == val_labels)
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset) * 100
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                val_loss,
                                                                                                                train_acc,
                                                                                                                val_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(val_acc)
            
        y_gt_folds.append(val_labels)
        y_pred_folds.append(val_preds)
        y_probs_folds.append(val_probs)
        
        # Write models on disk
        foldNum = str(fold+1)
        model_name = 'model'+foldNum+'.pth'
        torch.save(model.state_dict(), model_name)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model_name
        
        # Write histories on disk
        df_history = pd.DataFrame(history)
        df_history.to_csv('history_'+str(fold+1)+'.csv', sep=',',
                          float_format='%f', index=False, header=False)
        
    columns = ['Fold'] + [f'Train_{cls}' for cls in class_names] + [f'Val_{cls}' for cls in class_names]
    df_folds = pd.DataFrame(fold_data, columns=columns)
    df_folds.to_csv('train_val_counts_per_fold.csv', index=False)
    df_overall = pd.DataFrame({
        'Class': class_names,
        'Train+Val': train_val_counts,
        'Test': test_counts
    })
    df_overall.to_csv('dataset_split_summary.csv', index=False)
        
    ############################################################################################################################################################################
    # Classification report per fold
    for i in range(k_folds):
        print(f'Classification report for fold {i+1}')
        report = classification_report(y_gt_folds[i], y_pred_folds[i])
        print(report)
        with open(f'classification_report_val_{i+1}.txt', 'w') as f:
            f.write(report)

    # Classification report for all folds
    y_gt_all = np.concatenate(y_gt_folds, axis=0)
    y_pred_all = np.concatenate(y_pred_folds, axis=0)
    y_probs_all = np.concatenate(y_probs_folds, axis=0)
    print('Classification report for all folds')
    report_all = classification_report(y_gt_all, y_pred_all)
    print(report_all)
    with open('classification_report_all.txt', 'w') as f:
        f.write(report_all)

    # Confusion matrix per fold
    for i in range(k_folds):
        print(f'Confusion matrix for fold {i+1}')
        cm = confusion_matrix(y_gt_folds[i], y_pred_folds[i])
        print(cm)
        with open(f'confusion_matrix_val_{i+1}.txt', 'w') as f:
            print(cm, file=f)

    # Confusion matrix for all folds
    print('Confusion matrix for all folds')
    cm_all = confusion_matrix(y_gt_all, y_pred_all)
    print(cm_all)
    with open('confusion_matrix_all.txt', 'w') as f:
        print(cm_all, file=f)

    # Define class names
    class_names = [f"CMS{c}" for c in range(1, 5)]
    n_classes = len(class_names)

    # ROC curve per fold 
    for i in range(k_folds):
        plt.figure(figsize=(8, 6))
        for j, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y_gt_folds[i] == j).astype(int), y_probs_folds[i][:, j])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'ROC Curve for Fold {i+1}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_val_{i+1}.png')
        plt.close()

    # Compute macro and micro average AUCs for validation set
    macro_auc_val = roc_auc_score(y_gt_all, y_probs_all, average="macro", multi_class="ovr")
    micro_auc_val = roc_auc_score(y_gt_all, y_probs_all, average="micro", multi_class="ovr")
    
    # ROC curve for validation set (all folds)
    plt.figure(figsize=(8, 6))
    #per-class ROC curves
    fpr_macro = np.linspace(0, 1, 100)
    tpr_macro = np.zeros_like(fpr_macro)
    for j, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_gt_all == j).astype(int), y_probs_all[:, j])
        tpr_macro += np.interp(fpr_macro, fpr, tpr)  # Interpolation to align different class FPRs
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    tpr_macro /= len(class_names)  # Average across all classes
    
    n_classes = len(class_names)
    y_gt_one_hot = label_binarize(y_gt_all, classes=np.arange(n_classes))
    fpr_micro, tpr_micro, _ = roc_curve(y_gt_one_hot.ravel(), y_probs_all.ravel())
        
    plt.plot(fpr_macro, tpr_macro, linestyle=':', color='deeppink', lw=2, label=f'Macro-average (AUC = {macro_auc_val:.2f})')
    plt.plot(fpr_micro, tpr_micro, linestyle=':', color='navy', lw=2, label=f'Micro-average (AUC = {micro_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.title('ROC Curve for Validation Set (All Fold)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_all.png')
    plt.savefig('roc_curve_all.pdf')
    plt.close()

    ##### Test best model on test set #####
    resnet = resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    model = Net(resnet)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=5e-4,amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(best_model))

    test_loader = DataLoader(test_call_dataset, batch_size=batch_size, shuffle=False)
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
    with open('classification_report_test.txt', 'w') as f:
        f.write(report)
        
    # Confusion matrix for test set
    print('Confusion matrix for test set')
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    with open('confusion_matrix_test.txt', 'w') as f:
        print(cm, file=f)
        
    #Compute macro and micro average AUCs
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
    plt.savefig('roc_curve_test.png')
    plt.savefig('roc_curve_test.pdf')
    plt.close()
    

    
    