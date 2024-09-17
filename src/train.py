#%%

import numpy as np

## Model import
from model import GNNModel

## Utils
from utils import train_one_epoch

## Progress bar
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.data as geom_data

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/"

# # Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

trainData=np.load("../data/train.npy", allow_pickle=True).item()
devData=np.load("../data/dev.npy", allow_pickle=True).item()
testData=np.load("../data/test.npy", allow_pickle=True).item()

graph_train_loader = geom_data.DataLoader(list(trainData.values()), batch_size=1, shuffle=True)
graph_val_loader = geom_data.DataLoader(list(devData.values()), batch_size=1)

#%%

EPOCHS = 500
epoch_number = 0
Patience = 10
count = 0
lr=1e-6

model = GNNModel(c_in=640, c_hidden=16, c_out=58, num_layers=16).to(device)
loss_fn = nn.BCELoss()
optimizer =  optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'{CHECKPOINT_PATH}/Trainer_{timestamp}')

best_vloss = 1_000_000.

for epoch in tqdm(range(EPOCHS)):
    print('EPOCH {} (count {}):'.format(epoch_number + 1, count))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer, model, graph_train_loader, optimizer, loss_fn, device)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(graph_val_loader):
            vinputs, vedge_index, vbatch_idx = vdata.x.to(device), vdata.edge_index.to(device), vdata.batch.to(device)
            voutputs = model(vinputs, vedge_index, vbatch_idx)
            vpred = voutputs.squeeze(dim=-1)
            vpred = (vpred > 0).float()      
            vlabels = vdata.y.float().to(device)
     
        # Compute the loss and its gradients
            vloss = loss_fn(vpred, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        count=0
        best_vloss = avg_vloss
        model_path = f'{CHECKPOINT_PATH}/Trainer_{timestamp}/model_{epoch_number}'
        torch.save(model.state_dict(), model_path)
    else:
        count+=1
    if count >= Patience:
        break

    epoch_number += 1
# %%

