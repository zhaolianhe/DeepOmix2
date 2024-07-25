import torch
import torch.nn as nn
import torch.optim as optim

class AttentionFusion(nn.Module):
    def __init__(self, input_dims):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(sum(input_dims), 128),
            nn.ReLU(),
            nn.Linear(128, len(input_dims)),
            nn.Softmax(dim=1)
        )

    def forward(self, *inputs):
        concatenated = torch.cat(inputs, dim=1)
        attention_weights = self.attention(concatenated)
        fused = sum(w * inp for w, inp in zip(attention_weights, inputs))
        return fused

class MultimodalModel(nn.Module):
    def __init__(self, input_dims, output_dim, task):
        super(MultimodalModel, self).__init__()
        self.task = task
        self.attention_fusion = AttentionFusion(input_dims)
        self.fc = nn.Linear(input_dims[0], output_dim)

    def forward(self, mutation_data, expression_data, slide_data):
        fused_data = self.attention_fusion(mutation_data, expression_data, slide_data)
        output = self.fc(fused_data)
        return output


tasks = {
    "survival_analysis": {"output_dim": 1, "loss": nn.MSELoss()},
    "therapy_response": {"output_dim": 1, "loss": nn.BCEWithLogitsLoss()},
    "meta_state": {"output_dim": num_meta_states, "loss": nn.CrossEntropyLoss()},
    "stage": {"output_dim": num_stages, "loss": nn.CrossEntropyLoss()}
}


task_type = "therapy_response"  # 示例任务
task_config = tasks[task_type]
input_dims = [mutation_dim, expression_dim, slide_dim]
output_dim = task_config["output_dim"]

model = MultimodalModel(input_dims, output_dim, task_type)
criterion = task_config["loss"]
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    for mutation_data, expression_data, slide_data, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(mutation_data, expression_data, slide_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

