#%%
import torch_geometric.data as geom_data
import pandas as pd
import torch
from tqdm import tqdm
from model import GNNModel
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

Dataset=pd.DataFrame()
for Partition in ['train','test','dev']:
    tmpData=pd.read_csv(f'/home/fenoy/Documents/emb2Pfam/AuxData/Microdataset/{Partition}_microdataset.csv',index_col=0)
    tmpData=tmpData[(tmpData.PF=="PF13439")|(tmpData.PF=="PF12680")|(tmpData.PF=="PF01168")]
    tmpData["Partition"]=[Partition]*len(tmpData)
    Dataset=pd.concat([Dataset,tmpData])
Dataset.reset_index(drop=True,inplace=True)

UPFs=Dataset.PF.unique()
PFdict={}
n=0
for PF in UPFs:
    PFdict[PF]=n
    n+=1

Parts='/home/fenoy/Documents/emb2Pfam/GNN/data/Microdataset_parts/'

trainData={}
devData={}
testData={}

partDict={}

for graph in tqdm(os.listdir(f'{Parts}/Graphs/640/')):
    PID=graph.split(".")[0]
    if PID in Dataset.PID.values:
        partition=Dataset[Dataset.PID==PID].Partition.values[0]
        PF=Dataset[Dataset.PID==PID].PF.values[0]
        edges=[]
        Gg=torch.load(f'{Parts}/Graphs/640/{PID}.data')
        partDict[PID]=partition
        if partition=="train":
            trainData[PID]=Gg
        elif partition=="dev":
            devData[PID]=Gg
        else:
            testData[PID]=Gg

graph_test_loader = geom_data.DataLoader(list(testData.values()), batch_size=1)
#%%
saved_model = GNNModel(c_in=640, c_hidden=32, c_out=58, num_layers=32).to(device)
saved_model.load_state_dict(torch.load("/home/fenoy/Documents/emb2Pfam/GNN/saved_models/Trainer_20240916_104952/model_0"))

#%%
from sklearn.metrics import f1_score

F1s=[]
good=0
total=0
for i, vdata in enumerate(graph_test_loader):
            if len(vdata.edge_index)==0:
                continue
            vinputs, vedge_index, vbatch_idx = vdata.x.to(device), vdata.edge_index.to(device), vdata.batch.to(device)
            voutputs = saved_model(vinputs, vedge_index, vbatch_idx)
            outputs = saved_model(vinputs, vedge_index, vbatch_idx)
            vpred = outputs.squeeze(dim=-1)
            vpred = (vpred > 0).float()
            vlabels = vdata.y.float().to(device)
        # Compute the loss and its gradients
            good+=sum(sum(vlabels*vpred))
            total+=1
            F1s.append(f1_score(vlabels.squeeze().cpu(), vpred.squeeze().cpu(), zero_division=1.0))
print(good, total, good/total)
# %%
