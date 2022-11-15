import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.dummy import Pool

inDir="Z:/data/diverse_metals/diverseTop"
topTypes=os.listdir(inDir)

outDir="Z:/data/diverse_metals/diverseTopCSV"

df = pd.read_csv(r"Z:\data\diverse_metals\post-combustion-vsa-2-clean.csv")


try:
    os.mkdir(outDir)
except:
    pass


try:
    for k in topTypes:
        os.mkdir(outDir+f"/{k}")
        for j in range(3):
            os.mkdir(outDir+f"/{k}/B{j}")
except:
    pass

def extractMOF(fName):
    for k in topTypes:
        diags=np.load(f"{inDir}/{k}/{fName}_PH.npy",allow_pickle=True)
        for j in range(3):
            with open(f"{outDir}/{k}/B{j}/{fName}_PH.csv","w") as f:
                for pairs in diags[j]:
                    f.write(f"{pairs[0]},{pairs[1]}\n")


ncpus=6
pool=Pool(ncpus)
results=pool.map(extractMOF,list(df["filename"]))

pool.close()
pool.join()