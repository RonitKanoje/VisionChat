import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.transforms import transforms

NUM_CLASSES = 80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = vgg16(weights=None)

    inFeatures = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(inFeatures, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),

        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),

        nn.Linear(512, NUM_CLASSES)
    )

    state_dict = torch.load("model/best_model.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # VGG input size
    transforms.ToTensor(),            # PIL → Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
