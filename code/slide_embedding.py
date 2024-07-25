import torch
import torch.nn as nn
from torchvision import transforms
from timm.models.swin_transformer import swin_tiny_patch4_window7_224


class PathoPreModel(nn.Module):
    def __init__(self, pretrained_path):
        super(PathoPreModel, self).__init__()
        self.model = swin_tiny_patch4_window7_224(pretrained=True)
        self.model.head = nn.Identity()  # 去掉分类头
        self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        state_dict = torch.load(pretrained_path)
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


pretrained_path = 'pathopre.pth'
model = PathoPreModel(pretrained_path)
model.eval()

from PIL import Image
import numpy as np


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    return preprocess(image).unsqueeze(0)  # 增加batch维度


def load_and_preprocess_wsi(wsi_path, patch_size=224, stride=224):
    wsi_image = Image.open(wsi_path).convert("RGB")
    w, h = wsi_image.size
    patches = []
    
    for i in range(0, w, stride):
        for j in range(0, h, stride):
            if i + patch_size <= w and j + patch_size <= h:
                patch = wsi_image.crop((i, j, i + patch_size, j + patch_size))
                patch_tensor = preprocess_image(patch)
                patches.append(patch_tensor)
    
    return patches


wsi_path = 'pathology_image.tif'
patches = load_and_preprocess_wsi(wsi_path)

embeddings = []

with torch.no_grad():
    for patch in patches:
        embedding = model(patch)
        embeddings.append(embedding.squeeze().cpu().numpy())


embeddings = np.array(embeddings)


print(embeddings.shape)
