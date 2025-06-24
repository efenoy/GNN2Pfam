#%%
## Standard libraries
import os
import numpy as np 
import torch

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "../data/GraphData/Dicts/CONS/"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../data/GraphData/saved_models/"

# Setting the seed
pl.seed_everything(42)

# import torch geometric
import torch_geometric.data as geom_data

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

#import model
from model import NodeLevelGNNCRF

import warnings

warnings.filterwarnings("ignore")

print(device)

trainData=np.load(f"{DATASET_PATH}/train.npy", allow_pickle=True).item()
devData=np.load(f"{DATASET_PATH}/dev.npy", allow_pickle=True).item()
testData=np.load(f"{DATASET_PATH}/test.npy", allow_pickle=True).item()

graph_train_loader = geom_data.DataLoader(list(trainData.values()), batch_size=1, shuffle=True)
graph_val_loader = geom_data.DataLoader(list(devData.values()), batch_size=1) # Additional loader if you want to change to a larger dataset
graph_test_loader = geom_data.DataLoader(list(testData.values()), batch_size=1)

#%%

def train_node_classifier(model_name, **model_kwargs):
    pl.seed_everything(42)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "CONS_NodeLevel_" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_f1")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         enable_progress_bar=True) # False because epoch size is 1
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNNCRF.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNNCRF(model_name=model_name, c_in=1280, c_out=58, **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = NodeLevelGNNCRF.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    train_result = trainer.test(model, dataloaders=graph_train_loader)#, verbose=False)
    test_result = trainer.test(model, dataloaders=graph_test_loader)#, verbose=False)
    result = {"test": test_result[0]["test_f1"], "train": train_result[0]["test_f1"]}
    return model, result

# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train f1: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val f1:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test f1:  {(100.0*result_dict['test']):4.2f}%")


#%%
def main():
    node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNNnode",
                                                            layer_name="GAT",                                                      
                                                            c_hidden=512,
                                                            num_layers=2,
                                                            edge_dim=16,
                                                            dp_rate=0.1)
    print_results(node_gnn_result)


if __name__ == "__main__":
    main()
# %%

