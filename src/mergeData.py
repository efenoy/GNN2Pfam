#%%
# IMPORTS
# base imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Torch
import torch
from torch_geometric.data import Data

# Paralelization
from joblib import Parallel, delayed
import multiprocessing

import warnings

warnings.filterwarnings("ignore")

#%%
def Merger(Info, datadir, PFdict, ESMoutsize, ESMflavor, Criteria):
    PID, PF, Inicio, Fin = Info
    
    if not os.path.exists(f'{datadir}/GraphData/Graphs/{Criteria}/{PID}.graph'):
        node_attr=[]
        with open(f'{datadir}/GraphData/ESM/{ESMoutsize[ESMflavor]}/{PID}.attributes', 'r') as f:
            lines=f.readlines()
            for line in lines:
                na=line.strip().split(",")
                node_attr.append([float(x) for x in na])
        node_attr=torch.tensor(node_attr, dtype=torch.float)
        
        if node_attr.shape[0]>Fin:
            y=np.zeros((node_attr.shape[0],len(PFdict)))
            for i in range(Inicio,Fin):
                y[i,PFdict[PF]]=1
            y=torch.tensor(y, dtype=torch.long)

            edges=[]
            with open(f'{datadir}/GraphData/Adjacency/{Criteria}/{PID}.adj', 'r') as f:
                lines=f.readlines()
                for line in lines:
                    nodes=line.strip().split(",")
                    edges.append([int(x) for x in nodes])
                edge_index = torch.tensor(edges, dtype=torch.long)

            edge_attr=[]
            with open(f'{datadir}/GraphData/Edges/{Criteria}/{PID}.labels', 'r') as f:
                lines=f.readlines()
                for line in lines:
                    edgatt=[]
                    line=line.strip().replace("[","").replace("]","").split(", ")
                    for value in line:
                        edgatt.append(float(value))
                    edge_attr.append(edgatt)
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)

            Graph=Data(x=node_attr, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)
            torch.save(Graph,f'{datadir}/GraphData/Graphs/{Criteria}/{PID}.graph')

#%%

def main():
    datadir='../data/'
    Datos=pd.read_csv(f'{datadir}/MicroDataset.csv',index_col=0)

    if os.path.exists(f'{datadir}/PFtoIndex.npy'):
        PFdict=np.load(f'{datadir}/PFtoIndex.npy',allow_pickle=True).item()
    else:
        UPFs=Datos.PF.unique()
        PFdict={}
        n=0
        for PF in UPFs:
            PFdict[PF]=n
            n+=1
        np.save(f'{datadir}/PFtoIndex.npy', PFdict)

    # Embedding size of each model
    ESMoutsize = {"esm2_t48_15B_UR50D":5120, 
                "esm2_t36_3B_UR50D":2560, 
                "esm2_t33_650M_UR50D":1280, 
                "esm2_t30_150M_UR50D":640,
                "esm2_t12_35M_UR50D":480,
                "esm2_t6_8M_UR50D":320}

    #Select a model
    ESMflavor="esm2_t33_650M_UR50D"
    Criteria="CONS"

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(Merger)(info, datadir, PFdict, ESMoutsize, ESMflavor, Criteria) for i,info in tqdm(Datos[["PID","PF","Inicio","Fin"]].iterrows()))

if __name__ == "__main__":
    main()