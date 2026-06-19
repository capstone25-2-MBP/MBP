"""
EfficientNet 모델 관련 함수 및 설정
"""
import torch
import torch.nn as nn
from torchvision import models


class EfficientNetModel:
    """EfficientNet 모델 래퍼 클래스"""
    
    def __init__(self, architecture="efficientnet_b0", pretrained=True):
        """
        EfficientNet 모델 초기화
        
        Args:
            architecture (str): 모델 아키텍처 
                ("efficientnet_b0", "efficientnet_b1", ..., "efficientnet_b7")
            pretrained (bool): ImageNet 사전학습 가중치 로드 여부
        """
        if architecture == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=pretrained)
        elif architecture == "efficientnet_b1":
            self.model = models.efficientnet_b1(pretrained=pretrained)
        elif architecture == "efficientnet_b2":
            self.model = models.efficientnet_b2(pretrained=pretrained)
        elif architecture == "efficientnet_b3":
            self.model = models.efficientnet_b3(pretrained=pretrained)
        elif architecture == "efficientnet_b4":
            self.model = models.efficientnet_b4(pretrained=pretrained)
        elif architecture == "efficientnet_b5":
            self.model = models.efficientnet_b5(pretrained=pretrained)
        elif architecture == "efficientnet_b6":
            self.model = models.efficientnet_b6(pretrained=pretrained)
        elif architecture == "efficientnet_b7":
            self.model = models.efficientnet_b7(pretrained=pretrained)
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
        for param in self.model.features.parameters():
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
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        self.model.classifier[1] = self.model.classifier[1].to(self.device)
    
    def save_checkpoint(self, filepath):
        """모델 체크포인트 저장
        
        Args:
            filepath (str): 저장할 파일 경로
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"EfficientNet 모델 저장됨: {filepath}")
    
    def load_checkpoint(self, filepath):
        """모델 체크포인트 로드
        
        Args:
            filepath (str): 로드할 파일 경로
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"EfficientNet 모델 로드됨: {filepath}")


def create_efficientnet_model(num_classes, architecture="efficientnet_b0", pretrained=True):
    """EfficientNet 모델 생성
    
    Args:
        num_classes (int): 분류할 클래스 수
        architecture (str): 모델 아키텍처
        pretrained (bool): ImageNet 사전학습 가중치 로드 여부
    
    Returns:
        EfficientNetModel: 생성된 모델 래퍼
    """
    model = EfficientNetModel(architecture=architecture, pretrained=pretrained)
    model.modify_classifier(num_classes)
    return model
