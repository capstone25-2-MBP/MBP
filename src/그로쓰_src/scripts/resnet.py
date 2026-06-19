"""
ResNet 모델 관련 함수 및 설정
"""
import torch
import torch.nn as nn
from torchvision import models


class ResNetModel:
    """ResNet 모델 래퍼 클래스"""
    
    def __init__(self, pretrained=True):
        """
        ResNet 모델 초기화
        
        Args:
            pretrained (bool): ImageNet 사전학습 가중치 로드 여부
        """
        self.model = models.resnet50(pretrained=pretrained)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def get_model(self):
        """모델 반환"""
        return self.model
    
    def freeze_backbone(self):
        """백본 레이어 동결"""
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """백본 레이어 동결 해제"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def modify_classifier(self, num_classes):
        """분류기 레이어 수정
        
        Args:
            num_classes (int): 분류할 클래스 수
        """
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model.fc = self.model.fc.to(self.device)
    
    def save_checkpoint(self, filepath):
        """모델 체크포인트 저장
        
        Args:
            filepath (str): 저장할 파일 경로
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"ResNet 모델 저장됨: {filepath}")
    
    def load_checkpoint(self, filepath):
        """모델 체크포인트 로드
        
        Args:
            filepath (str): 로드할 파일 경로
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"ResNet 모델 로드됨: {filepath}")


def create_resnet_model(num_classes, pretrained=True):
    """ResNet 모델 생성
    
    Args:
        num_classes (int): 분류할 클래스 수
        pretrained (bool): ImageNet 사전학습 가중치 로드 여부
    
    Returns:
        ResNetModel: 생성된 모델 래퍼
    """
    model = ResNetModel(pretrained=pretrained)
    model.modify_classifier(num_classes)
    return model
