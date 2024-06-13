import torchvision.models.detection

def create_ssd_model(num_classes):
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.head.classification_head.num_classes = num_classes
    return model
