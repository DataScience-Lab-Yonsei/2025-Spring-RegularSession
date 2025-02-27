import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

"""
metaData.csv를 나누어 만든 train, val, test 데이터프레임에 담긴 이미지와 레이블을 반환해주는 CustomImageDataset을 만들어주세요
참고자료 : https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

class CustomImageDataset(Dataset):
    def __init__(self, data_dir: str, df: pd.DataFrame, augment=False):
        self.data_dir = data_dir  # 이미지가 저장된 기본 경로
        self.df = df.reset_index(drop=True)  # 인덱스 리셋
        self.augment = augment  # 데이터 증강 여부

        # 데이터 증강 (train 데이터에만 적용)
        if self.augment:
            self.transform = transforms.Compose([
                #### TO DO ####
                transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), # 크롭
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)) , #좌우이동
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # 블러
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # 대비 조절
                ###############
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.ToTensor()  # 정규화 없이 ToTensor()만 적용

    def __len__(self) -> int:
        # self.df에 포함된 전체 데이터포인트의 수를 반환하도록 해주세요
        return len(self.df)
        
    def __getitem__(self, idx: int):
        # 1) CSV에서 image_id, class_label 가져오기
        image_id = self.df.iloc[idx]["image_id"]          # 예: "IM-0001-0001"
        class_label = self.df.iloc[idx]["class_label"]    # 예: "NORMAL" or "PNEUMONIA"
        
        folder_name = class_label

        # 2) 폴더명 + image_id + ".jpeg"로 경로 구성
        image_path = os.path.join(self.data_dir, folder_name, image_id + ".jpeg")

        # 3) 이미지 로드
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # 4) 라벨을 숫자로 변환
        if folder_name == "NORMAL":
            label = 0.0
        else:
            label = 1.0

        label = torch.tensor(label, dtype=torch.float)

        return image, label
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


    
