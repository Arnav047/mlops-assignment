# model.py
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_object_detection_model():
    #  R-CNN with a ResNet-50 backbone pre-trained on the COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # the number of features entering the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    #Background (0) and Pedestrian (1)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    
    return model