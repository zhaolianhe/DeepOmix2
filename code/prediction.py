import torch
from torch.utils.data import DataLoader
import argparse
from Dataloader import get_dataloader
from model import MultimodalModel  


def load_model(model_path, input_dims, output_dim, task_type):
    model = MultimodalModel(input_dims, output_dim, task_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataloader, device='cpu'):
    model.to(device)
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            mutation_data = batch['mutation'].to(device)
            expression_data = batch['expression'].to(device)
            slide_data = batch['slide'].to(device)

            outputs = model(mutation_data, expression_data, slide_data)
            predictions = torch.sigmoid(outputs) if model.task == 'therapy_response' else outputs
            all_predictions.extend(predictions.cpu().numpy())
    return all_predictions


def main(args):
 
    model_path = f"{args.results_dir}/model.pth"
    mutation_file = f"{args.data_dir}/mutation.csv"
    expression_file = f"{args.data_dir}/expression.csv"
    slide_pt_files = f"{args.data_dir}/slide_files.csv"
    labels_file = f"{args.data_dir}/labels.csv"  # 如果有真实标签进行对比


    task_config = {
        "survival_analysis": {"output_dim": 1},  # 回归
        "therapy_response": {"output_dim": 1},  # 二分类
        "meta_state": {"output_dim": num_meta_states},  # 多分类
        "stage": {"output_dim": num_stages}  # 多分类
    }
    output_dim = task_config[args.task]["output_dim"]

    input_dims = [mutation_dim, expression_dim, slide_dim]

    dataloader = get_dataloader(mutation_file, expression_file, slide_pt_files, labels_file, args.batch_size, shuffle=False)

    model = load_model(model_path, input_dims, output_dim, args.task)

    predictions = predict(model, dataloader, device=args.device)


    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal model prediction")
    parser.add_argument('--task', type=str, required=True, choices=["survival_analysis", "therapy_response", "meta_state", "stage"], help="Task type")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory where model is saved")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the data")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run the model on (cpu or cuda)")
    
    args = parser.parse_args()
    main(args)
