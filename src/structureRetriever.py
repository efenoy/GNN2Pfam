#%% 
import pandas as pd
import os
from tqdm import tqdm
import urllib.request

from joblib import Parallel, delayed
import multiprocessing

def Retriever(r):
    if f'{r['AlphaFold']}.pdb' not in os.listdir('../data/AFstructures'):
        try:
            urllib.request.urlretrieve(f'https://alphafold.ebi.ac.uk/files/AF-{r['AlphaFold']}-F1-model_v4.pdb', f'../data/AFstructures/{r['PID']}.pdb')
        except:
            pass
# %%
def main():
    dataSet=pd.read_csv('../data/MicroDataset.csv', sep=',', names=['PID','Inicio','Fin','PF','Seed','Partition','UID','Alphafold'], header=None)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(Retriever)(info) for i,info in tqdm(dataSet.iterrows()))

if __name__ == "__main__":
    main()
# %%

