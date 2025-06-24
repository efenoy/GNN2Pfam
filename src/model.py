import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
from sklearn.metrics import f1_score
from torchcrf import CRF

#%%
'''Graph level classification'''


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}
  
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

        # Forward pass through GNN
        emissions = self.model(x, edge_index, edge_attr).unsqueeze(0)  # Ensure shape (batch_size, seq_length, num_tags)
        data.y = data.y.unsqueeze(0)
        if data.y.dim() == 3:  
            data.y = data.y.argmax(dim=-1)  # Convert from (batch, seq_length, num_tags) → (batch, seq_length)

        # Compute CRF loss
        loss = -self.crf(emissions, data.y)  # Shape: (1,)

        # Viterbi decoding for predictions
        preds = self.crf.decode(emissions)[0]  # Best sequence

        # Convert preds (list of lists) to a flat tensor
        predictions = torch.tensor(preds).view(-1)  # Shape: (batch_size * seq_length,)

        # Convert data.y to a flat tensor
        targets = data.y.view(-1).cpu()  # Shape: (batch_size * seq_length,)

        # Compute F1-score
        f1 = f1_score(predictions, targets, average='macro')

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

