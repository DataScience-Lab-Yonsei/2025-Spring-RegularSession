import os
import sys
import yaml
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(ROOT_DIR, "models"))

from CustomDataset import CustomImageDataset
from resnet import ResNet50

weights_dir = os.path.join(ROOT_DIR, "weights")
results_dir = os.path.join(ROOT_DIR, "results")
data_dir = os.path.join(ROOT_DIR, "data")
config_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
metadata_path = os.path.join(ROOT_DIR, "data", "metadata.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = min(4, os.cpu_count() // 2)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 on Pneumonia Dataset")
    parser.add_argument("--augment", action="store_true", help="Applied data augmentation during training")
    return parser.parse_args()


def main():

    args = parse_args()
    augment = args.augment

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if augment:
        batch_size = config["augmentation"]["hyperparameters"]["batch_size"]
        model_weights = "best_model_augmentation.pth"
    else:
        batch_size = config["no_augmentation"]["hyperparameters"]["batch_size"]
        model_weights = "best_model_no_augmentation.pth"

    meta_data = pd.read_csv('/content/drive/MyDrive/DSL/hw/0227/data/metadata.csv')
    _, val_test = train_test_split(meta_data, train_size=0.6, random_state=2025)
    _, test = train_test_split(val_test, train_size=0.5, random_state=2025)
    test_dataset  = CustomImageDataset(data_dir=data_dir, df=test)
    test_loader  = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)

    # 우선 ResNet50 모델 하나를 초기화 해줍니다
    trained_model = ResNet50()
    # 아까 저장해둔 weight를 로드해줍니다
    trained_model.load_state_dict(torch.load(os.path.join(weights_dir, model_weights), weights_only=True))
    # 모델을 evaluation 모드로 설정합니다
    trained_model.eval()
    trained_model.to(device)

    all_labels = []
    all_predictions = []
    
    with torch.no_grad():  # 학습 중이 아니므로 역전파를 위한 gradient 연산은 필요하지 않습니다
        for image, label in test_loader:
            # GPU에 올라가는 건 모델 뿐 아니라 데이터도 같이 올라가야합니다!
            image, label = image.to(device), label.to(device)
    
            # Forward Propagation
            pred = trained_model(image)
    
            # 예측값 계산
            predicted = (torch.sigmoid(pred) > 0.5).float()
    
            # 리스트에 저장
            all_labels.append(label)
            all_predictions.append(predicted)
    
    all_labels = torch.cat(all_labels).cpu().numpy().tolist()
    all_predictions = torch.cat(all_predictions).cpu().numpy().tolist()

    # Accuracy 계산 및 Confusion Matrix 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Confusion Matrix 시각화 및 저장
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if augment:
        ax.set_title(f'Confusion Matrix & Accuracy: {accuracy * 100:.2f}% - Augmentation')
        plt.savefig(os.path.join(results_dir, "test_accuracy_augmentation.png"))
    else:
        ax.set_title(f'Confusion Matrix & Accuracy: {accuracy * 100:.2f}% - No Augmentation')
        plt.savefig(os.path.join(results_dir, "test_accuracy_no_augmentation.png"))


if __name__ == "__main__":
    main()

