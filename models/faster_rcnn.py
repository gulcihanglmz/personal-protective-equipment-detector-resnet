import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes, pretrained_backbone=True):
    # Pretrained Faster R-CNN + ResNet50-FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        pretrained_backbone=pretrained_backbone
    )
    # Head’i uyarlama
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    return model
