import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from Dataloader import get_dataloader
from model import MultimodalModel 
from PIL import Image

def load_model(model_path, input_dims, output_dim, task_type):
    model = MultimodalModel(input_dims, output_dim, task_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def get_features(model, dataloader, device='cpu'):
    model.to(device)
    features = []
    with torch.no_grad():
        for batch in dataloader:
            mutation_data = batch['mutation'].to(device)
            expression_data = batch['expression'].to(device)
            slide_data = batch['slide'].to(device)

            outputs = model(mutation_data, expression_data, slide_data)
            features.extend(outputs.cpu().numpy())
    return np.array(features)

def visualize_heatmap(features, slide_img_path, output_path, patch_size):
    slide_img = Image.open(slide_img_path)
    slide_width, slide_height = slide_img.size

  
    heatmap = features.reshape((slide_height // patch_size, slide_width // patch_size))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  

    plt.imshow(slide_img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  
    plt.colorbar()
    plt.savefig(output_path)
    plt.show()

def main(args):
   
    model_path = f"{args.results_dir}/model.pth"
    slide_pt_files = f"{args.data_dir}/slide_files.csv"
    slide_img_path = f"{args.data_dir}/slide_image.jpg"  
    output_path = f"{args.results_dir}/heatmap.png"


    input_dims = [mutation_dim, expression_dim, slide_dim] 
    output_dim = 1  

    dataloader = get_dataloader("path/to/mutation.csv", "path/to/expression.csv", slide_pt_files, "path/to/labels.csv", args.batch_size, shuffle=False)


    model = load_model(model_path, input_dims, output_dim, args.task)


    features = get_features(model, dataloader, device=args.device)


    visualize_heatmap(features, slide_img_path, output_path, patch_size=args.patch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize slide predictions")
    parser.add_argument('--task', type=str, required=True, choices=["survival_analysis", "therapy_response", "meta_state", "stage"], help="Task type")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory where model and results are saved")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the data")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for processing")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run the model on (cpu or cuda)")
    parser.add_argument('--patch_size', type=int, default=224, help="Size of each patch used in feature extraction")
    
    args = parser.parse_args()
    main(args)
