"""
Vision Transformer (ViT) 모델 관련 함수 및 설정
"""
import torch
import torch.nn as nn
from torchvision import models


class ViTModel:
    """Vision Transformer 모델 래퍼 클래스"""
    
    def __init__(self, architecture="vit_b_16", pretrained=True):
        """
        Vision Transformer 모델 초기화
        
        Args:
            architecture (str): 모델 아키텍처 
                ("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32")
            pretrained (bool): ImageNet 사전학습 가중치 로드 여부
        """
        if architecture == "vit_b_16":
            self.model = models.vit_b_16(pretrained=pretrained)
        elif architecture == "vit_b_32":
            self.model = models.vit_b_32(pretrained=pretrained)
        elif architecture == "vit_l_16":
            self.model = models.vit_l_16(pretrained=pretrained)
        elif architecture == "vit_l_32":
            self.model = models.vit_l_32(pretrained=pretrained)
        else:
            raise ValueError(f"지원하지 않는 아키텍처: {architecture}")
        
        self.architecture = architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def get_model(self):
        """모델 반환"""
        return self.model
    
    def freeze_backbone(self):
        """백본 레이어 동결"""
        for param in self.model.encoder.layers.parameters():
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
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, num_classes)
        self.model.heads.head = self.model.heads.head.to(self.device)
    
    def save_checkpoint(self, filepath):
        """모델 체크포인트 저장
        
        Args:
            filepath (str): 저장할 파일 경로
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"ViT 모델 저장됨: {filepath}")
    
    def load_checkpoint(self, filepath):
        """모델 체크포인트 로드
        
        Args:
            filepath (str): 로드할 파일 경로
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"ViT 모델 로드됨: {filepath}")


def create_vit_model(num_classes, architecture="vit_b_16", pretrained=True):
    """Vision Transformer 모델 생성
    
    Args:
        num_classes (int): 분류할 클래스 수
        architecture (str): 모델 아키텍처
        pretrained (bool): ImageNet 사전학습 가중치 로드 여부
    
    Returns:
        ViTModel: 생성된 모델 래퍼
    """
    model = ViTModel(architecture=architecture, pretrained=pretrained)
    model.modify_classifier(num_classes)
    return model
