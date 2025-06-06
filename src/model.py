import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
from sklearn.metrics import f1_score

#%%
'''Graph level classification'''


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

class GNNModel(nn.Module):
    
    def __init__(self, c_in, c_hidden, c_out, num_layers=12, layer_name="GAT", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        #print("GNNModel")
        gnn_layer = gnn_layer_by_name[layer_name]
        
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels, 
                          out_channels=out_channels,
                          add_self_loops=False,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, 
                             out_channels=c_out,
                             add_self_loops=False,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        #print("GNNModel forward")
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)#, edge_attr)
            else:
                x = l(x)
        return x
    
class GraphGNNModel(nn.Module):
    
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        #print("GraphGNNModel")
        self.GNN = GNNModel(c_in=c_in, 
                            c_hidden=c_hidden, 
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        #print("GraphGNNModel forward")
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        x = self.head(x)
        return x
    
class GraphLevelGNN(pl.LightningModule):
    
    def __init__(self, **model_kwargs):
        super().__init__()
        #print("GraphLevelGNN")
        # Saving hyperparameters
        self.save_hyperparameters()
        
        self.model = GraphGNNModel(**model_kwargs)
        #self.loss_module = nn.BCEWithLogitsLoss()
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        #print("GraphLevelGNN forward")
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        
        # if self.hparams.c_out == 1:
        preds = (x > 10).float()
        #     data.y = data.y.float()
        # else:
        #     preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y.float())
        f1 = f1_score(preds.cpu(),data.y.cpu(),average='weighted')#(x.argmax(dim=-1) == data.y).sum().float() / x.shape[0]
        return loss, f1, x

    def configure_optimizers(self):
        #print("GraphLevelGNN optimizers")
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)#, eps=1e-5) 
        return optimizer

    def training_step(self, batch, batch_idx):
        #print("GraphLevelGNN trainer")
        loss, f1, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        #print("GraphLevelGNN val")
        loss, f1, _ = self.forward(batch, mode="val")
        self.log('val_loss', loss)
        self.log('val_f1', f1)

    def test_step(self, batch, batch_idx):
        #print("GraphLevelGNN test")
        _, f1, _ = self.forward(batch, mode="test")
        self.log('test_f1', f1)

#%%
'''Node level classification'''

class MLPModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)

class NodeLevelGNN(pl.LightningModule):

    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx, edge_attr = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.model(x, edge_index, edge_attr)
        x = x.squeeze(dim=-1)

        # Only calculate the loss on the nodes corresponding to the mask
        # if mode == "train":
        #     mask = data.train_mask
        # elif mode == "val":
        #     mask = data.val_mask
        # elif mode == "test":
        #     mask = data.test_mask
        # else:
        #     assert False, f"Unknown forward mode: {mode}"

        # loss = self.loss_module(x[mask], data.y[mask])
        # acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        preds = (x > 10).float()
        loss = self.loss_module(x, data.y.float())
        f1 = f1_score(preds.cpu(),data.y.cpu(),average='macro')#(x.argmax(dim=-1) == data.y).sum().float() / x.shape[0]
        return loss, f1, x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, f1, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, f1, _ = self.forward(batch, mode="val")
        self.log('val_loss', loss)
        self.log('val_f1', f1)

    def test_step(self, batch, batch_idx):
        _, f1, _ = self.forward(batch, mode="test")
        self.log('test_f1', f1)

#%%

from torchcrf import CRF

class GNNModelCRF(nn.Module):
    
    def __init__(self, c_in, c_hidden, c_out, num_layers=12, layer_name="GAT", dp_rate=0.1, edge_dim=None, **kwargs):
        """
        Inputs:
            c_in - Dimension of input node features
            c_hidden - Dimension of hidden features
            c_out - Number of output classes (domain labels)
            num_layers - Number of graph layers
            layer_name - Type of GNN layer
            dp_rate - Dropout rate
            edge_dim - Dimension of edge features (optional)
            kwargs - Additional arguments for the graph layer (e.g., heads for GAT)
        """
        super().__init__()

        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels = c_in

        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, 
                          out_channels=c_hidden,
                          edge_dim=edge_dim,  # Allow edge features
                          add_self_loops=False,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        layers += [gnn_layer(in_channels=in_channels, 
                             out_channels=c_out,
                             edge_dim=edge_dim,  # Final layer with edge features
                             add_self_loops=False,
                             **kwargs)]
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            if isinstance(layer, geom_nn.GATConv) or isinstance(layer, geom_nn.GraphConv):
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x)
        return x

class NodeLevelGNNCRF(pl.LightningModule):

    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = GNNModelCRF(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()  # Base loss (CRF will override this)
        
        # CRF layer (batch_first=True because we use batch dim first)
        self.crf = CRF(num_tags=model_kwargs['c_out'], batch_first=True)

    def forward(self, data, mode="train"):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #print(f"Data.y shape: {data.y.shape}")
        # Forward pass through GNN
        emissions = self.model(x, edge_index, edge_attr).unsqueeze(0)  # Ensure shape (batch_size, seq_length, num_tags)
        data.y = data.y.unsqueeze(0)
        if data.y.dim() == 3:  
            data.y = data.y.argmax(dim=-1)  # Convert from (batch, seq_length, num_tags) → (batch, seq_length)

        # print(f"Emissions shape: {emissions.shape}")
        # print(f"Data.y shape: {data.y.shape}")
        # Compute CRF loss
        loss = -self.crf(emissions, data.y)  # Shape: (1,)
        # print(f"Loss OK")

        # Viterbi decoding for predictions
        preds = self.crf.decode(emissions)[0]  # Best sequence
        # print(f"Viterbi OK")

        # Convert preds (list of lists) to a flat tensor
        predictions = torch.tensor(preds).view(-1)  # Shape: (batch_size * seq_length,)

        # Convert data.y to a flat tensor
        targets = data.y.view(-1).cpu()  # Shape: (batch_size * seq_length,)

        # Compute F1-score
        f1 = f1_score(predictions, targets, average='macro')
        # print(f"F1 OK")
        return loss, f1, preds, emissions

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, f1, preds, emissions = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, f1, preds, emissions = self.forward(batch, mode="val")
        self.log('val_loss', loss)
        self.log('val_f1', f1)

    def test_step(self, batch, batch_idx):
        _, f1, preds, emissions = self.forward(batch, mode="test")
        self.log('test_f1', f1)
        return preds, emissions

