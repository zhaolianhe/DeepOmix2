import openslide
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import h5py

# Load the whole slide image using OpenSlide
def load_slide(svs_file):
    slide = openslide.OpenSlide(svs_file)
    return slide

# Extract patches from the whole slide image
def extract_patches(slide, patch_size, level):
    w, h = slide.level_dimensions[level]
    patches = []
    for x in range(0, w, patch_size):
        for y in range(0, h, patch_size):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
            patches.append(patch)
    return patches

# Define a function to extract features from a model
def extract_features(patches, model, transform, device):
    model.eval()
    features = []
    for patch in patches:
        input_tensor = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(input_tensor)
            features.append(feature.cpu().numpy().flatten())
    return np.array(features)

# Save features in HDF5 format
def save_h5(features, output_path):
    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('features', data=features)

# Save features in PyTorch format
def save_pt(features, output_path):
    torch.save(torch.tensor(features), output_path)

def main():
    svs_file = 'path/to/slide.svs'
    h5_output = 'path/to/slide.h5'
    pt_output = 'path/to/slide.pt'
    patch_size = 224
    level = 0  # Level 0 corresponds to the highest resolution

    # Load the slide
    slide = load_slide(svs_file)

    # Extract patches
    patches = extract_patches(slide, patch_size, level)

    # Define the model for feature extraction (using a pretrained ResNet)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()  # Remove the classification layer
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract features
    features = extract_features(patches, model, transform, 'cuda' if torch.cuda.is_available() else 'cpu')

    # Save the features
    save_h5(features, h5_output)
    save_pt(features, pt_output)

if __name__ == "__main__":
    main()
