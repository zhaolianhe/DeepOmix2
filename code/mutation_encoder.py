import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and preprocess data
# Assume mutation_matrix.csv is your input file
mutation_matrix = pd.read_csv('mutation_matrix.csv', index_col=0)
data = mutation_matrix.values.astype('float32')

# Convert to PyTorch tensor
data_tensor = torch.tensor(data)

# Create DataLoader
batch_size = 32
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 2: Define the beta-VAE model
class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta):
        super(BetaVAE, self).__init__()
        self.beta = beta
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mean = nn.Linear(64, latent_dim)
        self.fc3_log_var = nn.Linear(64, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3_mean(h2), self.fc3_log_var(h2)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = torch.relu(self.fc4(z))
        h5 = torch.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# Hyperparameters
input_dim = data.shape[1]
latent_dim = 2
beta = 4

# Initialize model, optimizer, and loss function
model = BetaVAE(input_dim, latent_dim, beta)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
reconstruction_loss_fn = nn.MSELoss(reduction='sum')

# Step 3: Train the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        recon_x, mu, log_var = model(x)
        recon_loss = reconstruction_loss_fn(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + beta * kl_divergence
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset)}')

# Step 4: Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = []
    for batch in dataloader:
        x = batch[0]
        mu, log_var = model.encode(x)
        z = model.reparameterize(mu, log_var)
        embeddings.append(z)
    embeddings = torch.cat(embeddings)

# Convert embeddings to numpy array for further analysis
embeddings_np = embeddings.numpy()
