#%% 
import pandas as pd
import os
from tqdm import tqdm
import urllib.request

from joblib import Parallel, delayed
import multiprocessing

PIDtoUPID = {}
with open('../data/idmapping_miniset_pairs.tsv') as f:
    lines = f.readlines()
    for line in lines:
        PID, UPID = line.split()
        PIDtoUPID[PID] = UPID
TrainSet=pd.read_csv('../data/Microset_train.csv', sep=',', names=['PID','Inicio', 'Fin', 'PF', 'Seed'], header=None)
TestSet=pd.read_csv('../data/Microset_test.csv', sep=',', names=['PID','Inicio', 'Fin', 'PF', 'Seed'], header=None)

def Retriever(r):
    if r.PID in PIDtoUPID.keys():
        if f'{PIDtoUPID[r.PID]}.pdb' not in os.listdir('../data/AFstructures'):
            try:
                urllib.request.urlretrieve(f'https://alphafold.ebi.ac.uk/files/AF-{PIDtoUPID[r.PID]}-F1-model_v4.pdb', f'../data/AFstructures/{PIDtoUPID[r.PID]}.pdb')
            except:
                pass
# %%
def main():
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(Retriever)(info) for i,info in tqdm(TrainSet.iterrows()))

if __name__ == "__main__":
    main()
# %%

