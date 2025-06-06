#%%
##
# sbatch -n 16 -N 1 -o /usr/users/efenoy/GNNproject/sbatchOutputs/runMergeData.out -t 20:00:00 runMergeData.sh
# scancel #####
##


# IMPORTS
# base imports
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Bio related imports
import Bio.PDB

# Embedding related imports
import esm

# Torch related imports
import torch

# Libraries to paralelize
from joblib import Parallel, delayed
import multiprocessing

#%%
# VARIABLES

# Output layer of each ESM model
ESMlayer = {"esm2_t48_15B_UR50D":48, 
            "esm2_t36_3B_UR50D":36, 
            "esm2_t33_650M_UR50D":33, 
            "esm2_t30_150M_UR50D":30,
            "esm2_t12_35M_UR50D":12,
            "esm2_t6_8M_UR50D":6}

# Embedding size of each model
ESMoutsize = {"esm2_t48_15B_UR50D":5120, 
              "esm2_t36_3B_UR50D":2560, 
              "esm2_t33_650M_UR50D":1280, 
              "esm2_t30_150M_UR50D":640,
              "esm2_t12_35M_UR50D":480,
              "esm2_t6_8M_UR50D":320}

#Select a model
ESMflavor="esm2_t33_650M_UR50D"

# Load selected ESM
encoder, alphabet = esm.pretrained.load_model_and_alphabet(ESMflavor)
batch_converter = alphabet.get_batch_converter()
encoder.eval()  # Disable dropout for evaluation

# AA code change
d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

# AA identity
ID={
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

#%%
# AUX FUNCTIONS
def calc_residue_dist_CB(residue_one, residue_two):
    '''Computes distance between C, O and/or N atoms
        between two residues and returns the distance 
        of the closest pair'''
    dist=[]
    for a1 in residue_one.get_atoms():
        condition1 = str(residue_one)[9:12]=="GLY"
        for a2 in residue_two.get_atoms():
            condition2 = str(residue_two)[9:12]=="GLY"
            if condition1:
                if (str(a1)[6] in ["C"]):
                    coordenadas1=a1.coord
            else:
                if (str(a1)[6:8] in ["CB"]):
                    coordenadas1=a1.coord
            if condition2:
                if (str(a2)[6] in ["C"]):
                    coordenadas2=a2.coord
            else:
                if (str(a2)[6:8] in ["CB"]):
                    coordenadas2=a2.coord            
    dist.append(np.sqrt(np.sum((coordenadas1 - coordenadas2)**2)))
    return min(dist)

def calc_residue_dist_CONS(residue_one, residue_two):
    '''Computes distance between C, O and/or N atoms
        between two residues and returns the distance 
        of the closest pair'''
    dist=np.ones((4,4))*10
    Positions={"C":0, "N":1, "O":2, "S":3}
    for a1 in residue_one.get_atoms():
        atom1=str(a1)[6]
        if (atom1 in ["C","N","O","S"]):
            coordenadas1=a1.coord
            for a2 in residue_two.get_atoms():
                atom2=str(a2)[6]
                if (atom2 in ["C","N","O","S"]):
                    coordenadas2=a2.coord
                    distancia=np.sqrt(np.sum((coordenadas1 - coordenadas2)**2))
                    if dist[Positions[atom1],Positions[atom2]]>distancia:
                        dist[Positions[atom1],Positions[atom2]]=distancia
    return dist

def calc_residue_dist_COM(residue_one, residue_two):
    '''Calculates the distances between the centroids
        of two given residues'''
    coor1=[]
    coor2=[]
    for a1 in residue_one.get_atoms():
        coor1.append(a1.coord)
    for a2 in residue_two.get_atoms():
        coor2.append(a2.coord)
    
    com1=np.array([np.mean([i[0] for i in coor1]),np.mean([i[1] for i in coor1]),np.mean([i[2] for i in coor1])])
    com2=np.array([np.mean([i[0] for i in coor2]),np.mean([i[1] for i in coor2]),np.mean([i[2] for i in coor2])])
    
    dist=np.sqrt(np.sum((com1 - com2)**2))
    return dist

def calc_conections(chain_one, chain_two, max_dist) :
    '''Given two chains and an interaction distance it 
        returns an adjacency list of interacting AAs, 
        a score between 0 and 1 depending on the distance
        and the sequence of the 1st chain'''
    edges = []
    edgesprop = []
    seq = []
    for row, residue_one in enumerate(chain_one) :
        seq.append(d3to1[residue_one.resname])
        for col, residue_two in enumerate(chain_two) :
            if row>col:
                continue
            distance=calc_residue_dist_CONS(residue_one, residue_two)
            score=1-(distance/max_dist)
            if np.min(distance)<max_dist:
                if row!=col:
                    edges.append([col, row])
                    edgesprop.append(score)
                edges.append([row, col])
                edgesprop.append(score)
    return edges, edgesprop, ''.join(seq)

def PDBprocessor(indir, outdir, IDS):
    '''Writes the files needed to compile the graph networks used to
        train the GNN'''
    PID=IDS[0] 
    AFID=IDS[1]
    Criteria="CONS"
    AdjFileOk=os.path.exists(f"{outdir}/Adjacency/{Criteria}/{PID}.adj")
    EdgeFileOk=os.path.exists(f"{outdir}/Edges/{Criteria}/{PID}.labels")
    NodesFileOk=os.path.exists(f"{outdir}/Nodes/{PID}.labels")
    ESMFileOk=os.path.exists(f'{outdir}/ESM/{ESMoutsize[ESMflavor]}/{PID}.attributes')

    if not (AdjFileOk and EdgeFileOk and NodesFileOk and ESMFileOk):
        try:
            pdb_filename=f'{indir}/{AFID}.pdb'
            structure = Bio.PDB.PDBParser().get_structure(AFID, pdb_filename)
            model = structure[0]
            adj_list, score, sequence = calc_conections(model["A"], model["A"], 4)
        except:
            return
        if not AdjFileOk:
            if not os.path.isdir(f'{outdir}/Adjacency/{Criteria}/'):
                os.makedirs(f'{outdir}/Adjacency/{Criteria}/')
            with open(f'{outdir}/Adjacency/{Criteria}/{PID}.adj', 'w') as f:
                for nodes in adj_list:
                    f.write(f"{nodes[0]},{nodes[1]}\n")
        if not EdgeFileOk:
            if not os.path.isdir(f'{outdir}/Edges/{Criteria}/'):
                os.makedirs(f'{outdir}/Edges/{Criteria}/')
            with open(f'{outdir}/Edges/{Criteria}/{PID}.labels', 'w') as f:
                for edges in score:
                    f.write(f"{list(np.round(edges.flatten(),3))}\n")
        if not NodesFileOk:
            if not os.path.isdir(f'{outdir}/Nodes/'):
                os.makedirs(f'{outdir}/Nodes/')
            with open(f'{outdir}/Nodes/{PID}.labels', 'w') as f:
                for AA in sequence:
                    f.write(f"{ID[AA]}\n")
        if not ESMFileOk:
            if not os.path.isdir(f'{outdir}/ESM/{ESMoutsize[ESMflavor]}/'):
                os.makedirs(f'{outdir}/ESM/{ESMoutsize[ESMflavor]}/')
            data=(PID,sequence)
            _, _, batch_tokens = batch_converter([data])
            
            with torch.no_grad():
                results = encoder(batch_tokens, repr_layers=[ESMlayer[ESMflavor]])

            embedding=results["representations"][ESMlayer[ESMflavor]].squeeze()
            #np.save(f"/home/fenoy/Documents/emb2Pfam/GNN/data/Microdataset/{ESMflavor}/{PID}.npy",embedding)
        
            with open(f'{outdir}/ESM/{ESMoutsize[ESMflavor]}/{PID}.attributes', 'w') as f:
                for i in range(len(sequence)):
                    f.write(", ".join([str(x) for x in np.round(embedding[i+1].tolist(),6)]))
                    f.write("\n")


#%%
def main():
    #Data=pd.read_csv('/usr/users/efenoy/data/GraphData/MicroDataset.csv', index_col=0)
    print('Reading data')
    Data=pd.read_csv('/usr/users/efenoy/data/idmapping_miniset_pairs.tsv', sep=' ', names=['UID','Alphafold'])
    idir='/usr/users/efenoy/data/AFstructures/'
    odir='/usr/users/efenoy/data/GraphData/'

    print('Starting processing')
    t=time.time()
    #num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=8)(delayed(PDBprocessor)(idir,odir, ids) for ids in tqdm(Data[["UID","Alphafold"]].values))
    print(time.time()-t)

if __name__ == "__main__":
    main()
