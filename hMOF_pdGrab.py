import os
import sys
import typing
import pickle
import json
import numpy as np
import pandas as pd

from pdHash import PDhash


import pymatgen.core as mg
from pymatgen.io.cif import CifWriter
from pymatgen.io.cif import CifFile
from pymatgen.core.structure import Structure


import gudhi,gudhi.hera,gudhi.wasserstein,persim
import ripserplusplus as rpp

inDir="Z:/data/diverse_metals"
os.chdir(inDir)
hDir="Z:/data/hMOF"
df=pd.read_csv(f"{hDir}/id_prop.csv",index_col=0,header=None)
df.columns=["workCap"]
totalLen=len(df["workCap"])



## strat samps

np.random.seed(42)
trP=.7

numBins=5
regVars=["workCap"]
bounds={k:[(0,np.quantile(df[k],1/numBins))] for k in regVars}

k="workCap"
for j in range(numBins-2):
    bounds[k].append((bounds[k][j][1],np.quantile(df[k],(j+2)/numBins)))
bounds[k].append((bounds[k][-1][1],))

indexBounds = {k: [] for k in bounds.keys()}  # upper index for

for k in regVars:
    j = 0
    bj = 0
    for index, row in df.sort_values(by=[k]).iterrows():
        if bj >= len(bounds[k]) - 1:
            break
        elif row[k] > bounds[k][bj][1]:
            indexBounds[k].append(j)
            bj += 1
        j += 1

testBins = {k: [] for k in bounds.keys()}
for k in bounds.keys():
    j = 0
    testBins[k].append(list(
        np.random.choice(df.sort_values(by=[k]).index[:indexBounds[k][j]], size=round((1 - trP) * indexBounds[k][j]),
                         replace=False)))
    for j in range(1, len(indexBounds[k])):
        testBins[k].append(list(np.random.choice(df.sort_values(by=[k]).index[indexBounds[k][j - 1]:indexBounds[k][j]],
                                                 size=round((1 - trP) * (indexBounds[k][j] - indexBounds[k][j - 1])),
                                                 replace=False)))
    testBins[k].append(list(np.random.choice(df.sort_values(by=[k]).index[indexBounds[k][-1]:],
                                             size=round((1 - trP) * (totalLen - indexBounds[k][-1])), replace=False)))

trainBins = {k: [] for k in bounds.keys()}
for k in bounds.keys():
    j = 0
    trainBins[k].append(list(set(df.sort_values(by=[k]).index[:indexBounds[k][j]]) - set(testBins[k][j])))
    for j in range(1, len(indexBounds[k])):
        trainBins[k].append(
            list(set(df.sort_values(by=[k]).index[indexBounds[k][j - 1]:indexBounds[k][j]]) - set(testBins[k][j])))
    trainBins[k].append(list(set(df.sort_values(by=[k]).index[indexBounds[k][-1]:]) - set(testBins[k][-1])))




hinDir="Z:/data/diverse_metals/hMOF-1039C2-CO2"
hOut="Z:/data/diverse_metals/hMOF_PDhash_sameIndex"



failMOFs=set()
for binNum in range(4,numBins):
    pdStack = PDhash(res=.25, diags=None, maxHdim=2, persistThresh=0)
    hIndex=list(trainBins["workCap"][binNum])
    for i in range(len(hIndex)):
        fName=hIndex[i]
        cif_name=f"{hinDir}/{fName}"
        try:
            struct=Structure.from_file(cif_name,)
            rppdgm=rpp.run("--format distance --dim 2",struct.distance_matrix)
            npdgm=[np.array([[float(rppdgm[b][k][0]),float(rppdgm[b][k][1])] for k in range(len(rppdgm[b]))])for b in rppdgm.keys()]
            pdStack.addDiagRpp(npdgm,int(fName[fName.find('-')+1:fName.find(".cif")]))
        except:
            failMOFs.add(fName)
    with open(f"{hOut}/train_pdStack_b{binNum}_{numBins}.pkl","wb") as f:
        pickle.dump(pdStack,f)
    with open(f"{hOut}/train_pdStack_index_b{binNum}_{numBins}.pkl", "wb") as f:
        pickle.dump(hIndex, f)


print(f"Training complete!! failed MOFs:\n")
for i in failMOFs:
    print(i)
print("\n Beginning TestBins!")



failMOFs=set()
for binNum in range(numBins):
    pdStack = PDhash(res=.25, diags=None, maxHdim=2, persistThresh=0)
    hIndex=list(testBins["workCap"][binNum])
    for i in range(len(hIndex)):
        fName=hIndex[i]
        cif_name=f"{hinDir}/{fName}"
        try:
            struct=Structure.from_file(cif_name,)
            rppdgm=rpp.run("--format distance --dim 2",struct.distance_matrix)
            npdgm=[np.array([[float(rppdgm[b][k][0]),float(rppdgm[b][k][1])] for k in range(len(rppdgm[b]))])for b in rppdgm.keys()]
            pdStack.addDiagRpp(npdgm,int(fName[fName.find('-')+1:fName.find(".cif")]))
        except:
            failMOFs.add(fName)
    with open(f"{hOut}/test_pdStack_b{binNum}_{numBins}.pkl","wb") as f:
        pickle.dump(pdStack,f)
    with open(f"{hOut}/test_pdStack_index_b{binNum}_{numBins}.pkl", "wb") as f:
        pickle.dump(hIndex, f)


print(f"Testing complete!! failed MOFs:\n")
for i in failMOFs:
    print(i)


