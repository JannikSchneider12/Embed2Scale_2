import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import json
import ast
    
class SimpleEmbeddingDataset(Dataset):
    def __init__(self, dataframe):
        self.embeddings = [torch.tensor(ast.literal_eval(emb)) for emb in dataframe['Embeddings']]
        self.labels = torch.tensor(dataframe['label'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
def create_dataloader(submission_df_path, annotations_json_path, batch_size):

    '''
    Create a torch.utils.data.DataLoader object based on a pandas dataframe that contains feature embeddings along with the ids
    and an annotation json file that contains the same ids along with the true labels

    Input:
    * submission_df_path: Path to pandas dataframe with 'feature_embeddings' column that contains feature embeddings and 'ids' column that contain the 
    corresponding ids
    * annotations_json_path: Path to annotations json file that contains the ids along with the true labels

    Return
    * train_dataloader: The dataloader that contains the train_samples and returns (embedding, label) 
    * val_dataloader: The dataloader that contains the test_samples and returns (embedding, label) 
    '''

    submission_df = pd.read_csv(submission_df_path)


    # Load annotations from JSON
    with open(annotations_json_path, 'r') as f:
        annotations = json.load(f)

    # Prepare a list to hold data for DataFrame construction
    annotation_data = []

    # Extract information from the annotations
    for split, data in annotations.items():
        for sample in data['samples']:
            annotation_data.append({
                'id': sample['id'],
                'label': sample['label'],
                'split': split
            })

    # Create a DataFrame from the annotations
    annotations_df = pd.DataFrame(annotation_data)

    # Merge the submission_df with annotations_df on the 'id' column
    combined_df = submission_df.merge(annotations_df, on='id', how='left')

    # Separate the combined DataFrame into train and validation sets
    train_samples = combined_df[combined_df['split'] == 'train']
    val_samples = combined_df[combined_df['split'] == 'val']

    # Create datasets
    train_dataset = SimpleEmbeddingDataset(train_samples)
    val_dataset = SimpleEmbeddingDataset(val_samples)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

class linear_probing_model(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_features=embedding_dim, out_features=1),
                                    nn.Sigmoid())

    def forward(self,X):

        X = self.layer1(X)

        return X.view(-1)

    
def train(model, train_dataloader, val_dataloader, device, optimizer, loss_fn, num_epochs): 

    model = model.to(device)
    model.train()

    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):

        train_loss_per_epoch = 0

        for X,y in train_dataloader:

            optimizer.zero_grad()

            X,y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y.float())

            loss.backward()

            optimizer.step()

            train_loss_per_epoch += loss.item()

        train_loss.append(train_loss_per_epoch/len(train_dataloader))

        model.eval()
        with torch.no_grad():

            val_loss_per_epoch = 0

            for X, y in val_dataloader:

                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y.float())

                val_loss_per_epoch += loss.item()

            val_loss.append(val_loss_per_epoch/len(val_dataloader))

      #  print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}')

    '''
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    '''

    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]

    return final_train_loss, final_val_loss

        





