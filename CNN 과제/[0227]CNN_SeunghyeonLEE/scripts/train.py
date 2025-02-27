import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(ROOT_DIR, "models"))

from CustomDataset import CustomImageDataset
from resnet import ResNet50

data_dir = os.path.join(ROOT_DIR, "data")
weights_dir = os.path.join(ROOT_DIR, "weights")
results_dir = os.path.join(ROOT_DIR, "results")
config_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
metadata_path = os.path.join(ROOT_DIR, "data", "metadata.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = min(4, os.cpu_count() // 2)

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 on Pneumonia Dataset")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation during training")
    return parser.parse_args()


def main():
    
    args = parse_args()
    augment = args.augment
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 하이퍼파라미터 설정
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if augment:
        batch_size, lr, epochs \
            = config["augmentation"]["hyperparameters"]["batch_size"], \
              float(config["augmentation"]["hyperparameters"]["learning_rate"]), \
              config["augmentation"]["hyperparameters"]["num_epochs"]
        model_weights = "best_model_augmentation.pth"
    else:
        batch_size, lr, epochs \
            = config["no_augmentation"]["hyperparameters"]["batch_size"], \
              float(config["no_augmentation"]["hyperparameters"]["learning_rate"]), \
              config["no_augmentation"]["hyperparameters"]["num_epochs"]
        model_weights = "best_model_no_augmentation.pth"
    
    # MetaData.csv를 로드해주고, train, val, test로 나눠줍니다
    meta_data = pd.read_csv('/content/drive/MyDrive/DSL/hw/0227/data/metadata.csv')
    
    # train : val : test = 6 : 2 : 2
    train, val_test = train_test_split(meta_data, train_size=0.6, random_state=2025)
    val, test = train_test_split(val_test, train_size=0.5, random_state=2025)
    
    # CustomDataset.py에서 가져온 CustomImageDataset 클래스에 train, val, test 데이터프레임을 패스하여 데이터셋을 만들어주세요
    train_dataset = CustomImageDataset(data_dir=data_dir, df=train, augment=augment)
    val_dataset   = CustomImageDataset(data_dir=data_dir, df=val)
    test_dataset  = CustomImageDataset(data_dir=data_dir, df=test)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset  , batch_size = batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset , batch_size = batch_size, shuffle=True, num_workers=num_workers)

    # model을 선언하고 GPU에 올려줍니다
    model = ResNet50()
    model.to(device)
    
    # 손실함수와 optimizer를 정의합니다 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Best Validation Loss를 양의 무한대로 초기화
    best_loss = float("inf")
    
    # 모든 epoch에서의 loss와 accuracy들을 저장할 리스트들
    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []
    
    for curr_epoch in range(epochs): # iterate over all EPOCHS
        running_loss, total_correct, total_samples = 0, 0, 0 # 현재 epoch에서의 train_loss와 train_accuracy들을 계산하기 위한 변수들
    
        for batch_idx, (img, label) in enumerate(train_loader): # iterate over all BATCHES
            # GPU에 올라가는 건 모델 뿐 아니라 데이터도 같이 올라가야합니다!
            img, label = img.to(device), label.to(device)
    
            # Forward Propagation
            pred = model(img)
            
            # 손실함수 계산
            label = label.view(-1,1)
            loss = criterion(pred, label.float()) # 구현 상의 차이로 label 텐서에 조작을 가해 모양을 맞춰줘야하는 경우가 생길 수 있습니다 ( 힌트 : tensor.view() )
    
            # 누적된 기울기 초기화 및 역전파
            optimizer.zero_grad()
            loss.backward()
    
            # 파라미터 업데이트
            optimizer.step()
    
            # 해당 Batch에서의 loss를 running_loss 변수에 누적
            running_loss += loss.item()
            
            # Accuracy 계산
            predicted = (torch.sigmoid(pred) > 0.5).float()
            label = label.float()
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)
    
        # 모든 Batch를 순회한 이후 최종적인 train_loss, train_accuracy를 계산해줍니다
        train_loss = running_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
    
        
        # Validation 시작
        model.eval()
        
        with torch.no_grad():  # 학습 중이 아니므로 역전파를 위한 gradient 연산은 필요하지 않습니다
            running_loss, total_correct, total_samples = 0, 0, 0 # 현재 epoch에서의 val_loss와 val_accuracy들을 계산하기 위한 변수들
            
            for img, label in val_loader: # iterate over all BATCHES
                # GPU에 올라가는 건 모델 뿐 아니라 데이터도 같이 올라가야합니다!
                img, label = img.to(device), label.to(device)
    
                # Forward Propagation
                pred = model(img)
    
                # 손실함수 계산
                label = label.view(-1,1)
                loss = criterion(pred, label.float())  # 구현 상의 차이로 label 텐서에 조작을 가해 모양을 맞춰줘야하는 경우가 생길 수 있습니다 ( 힌트 : tensor.view() )
    
                # 해당 Batch에서의 loss를 running_loss 변수에 누적
                running_loss += loss.item()
    
                # 예측값 및 Accuracy 계산
                predicted = (torch.sigmoid(pred) > 0.5).float()
                label = label.float()
                total_correct += (predicted == label).sum().item()
                total_samples += label.size(0)
    
            # 모든 Batch를 순회한 이후 최종적인 train_loss, train_accuracy를 계산해줍니다
            val_loss = running_loss / len(val_loader)
            val_accuracy = total_correct / total_samples
    
    
        model.train()  # 모델을 다시 학습모드로 설정합니다
    
        # 현재 epoch에서의 결과들 출력
        print(f"Epoch [{curr_epoch + 1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    
        # 리스트에 저장
        train_loss_list.append(train_loss)
        valid_loss_list.append(val_loss)
        train_accuracy_list.append(train_accuracy)
        valid_accuracy_list.append(val_accuracy)
        
        # 가장 작은 validation loss를 가졌을 때의 모델 파라미터를 저장해줍니다.
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, model_weights))
    
    # Train Loss & Validation Loss 그래프 시각화 및 저장
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss', marker='o', linestyle='-')
    plt.plot(range(1, epochs + 1), valid_loss_list, label='Validation Loss', marker='s', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if augment:
        plt.title('Train & Validation Loss - Augmentation')
    else:
        plt.title('Train & Validation Loss - No Augmentation')
    plt.legend()
    if augment:
        plt.savefig(os.path.join(results_dir, "train_validation_loss_augmentation.png"))
    else:
        plt.savefig(os.path.join(results_dir, "train_validation_loss_no_augmentation.png"))

    # Train Accuracy & Validation Accuracy 그래프 시각화 및 저장
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), train_accuracy_list, label='Train Accuracy', marker='o', linestyle='-')
    plt.plot(range(1, epochs + 1), valid_accuracy_list, label='Validation Accuracy', marker='s', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if augment:
        plt.title('Train & Validation Accuracy - Augmentation')
    else:
        plt.title('Train & Validation Accuracy - No Augmentation')
    plt.legend()
    if augment:
        plt.savefig(os.path.join(results_dir, "train_validation_accuracy_augmentation.png"))
    else:
        plt.savefig(os.path.join(results_dir, "train_validation_accuracy_no_augmentation.png"))


if __name__ == "__main__":
    main()

