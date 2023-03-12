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


#hDir="Z:/data/hMOF"
#df=pd.read_csv(f"{hDir}/id_prop.csv",index_col=0,header=None)
#df.columns=["workCap"]
#totalLen=len(df["workCap"])


df=pd.read_csv(f"{inDir}/hMOF_CO2_info.csv",index_col=0)

hIndexMap={int(fName[fName.find('-')+1:]):fName for fName in df.index}



## strat samps

np.random.seed(42)
trP=.8

numBins=20
#regVars=cols
regVars=["CO2_wc_01"]
dfLen=len(df[regVars[0]])
bounds={k:[np.quantile(df[k],j/numBins) for j in range(1,numBins)] for k in regVars}


testBins = {k: [] for k in bounds.keys()} #currently 20% split, can be broken in half for val
for k in bounds.keys():
    for j in range(numBins):
        testBins[k].append(list(
            np.random.choice(df.sort_values(by=[k]).index[j*dfLen//numBins:(j+1)*dfLen//numBins], size=round((1 - trP)*dfLen//numBins),
                             replace=False)))
trainBins = {k: [] for k in bounds.keys()}
for k in bounds.keys():
    for j in range(numBins):
        trainBins[k].append(list(set(df.sort_values(by=[k]).index[j*dfLen//numBins:(j+1)*dfLen//numBins]) - set(testBins[k][j])))




hinDir="Z:/data/diverse_metals/hMOF-1039C2-CO2"
hOut="Z:/data/diverse_metals/hMOF_PDhash_sameIndex"

regVar=regVars[0]
pdRes=0.1

failMOFs=set()
for binNum in reversed(range(numBins)):
    pdStack = PDhash(res=pdRes, diags=None, maxHdim=2, persistThresh=0)
    hIndex=list(trainBins[regVar][binNum])
    for i in range(len(hIndex)):
        fName=hIndex[i]
        cif_name=f"{hinDir}/{fName}.cif"
        try:
            struct=Structure.from_file(cif_name,)
            rppdgm=rpp.run("--format distance --dim 2",struct.distance_matrix)
            npdgm=[np.array([[float(rppdgm[b][k][0]),float(rppdgm[b][k][1])] for k in range(len(rppdgm[b]))])for b in rppdgm.keys()]
            pdStack.addDiagRpp(npdgm,int(fName[fName.find('-')+1:]))
        except:
            failMOFs.add(fName)
    with open(f"{hOut}/train_pdStack_res01_b{binNum}_{numBins}.pkl","wb") as f:
        pickle.dump(pdStack,f)
    with open(f"{hOut}/train_pdStack_res01_index_b{binNum}_{numBins}.pkl", "wb") as f:
        pickle.dump(hIndex, f)


print(f"Training complete!! failed MOFs:\n")
for i in failMOFs:
    print(i)
print("\n Beginning TestBins!")



failMOFs=set()
for binNum in reversed(range(numBins)):
    pdStack = PDhash(res=pdRes, diags=None, maxHdim=2, persistThresh=0)
    hIndex=list(testBins[regVar][binNum])
    for i in range(len(hIndex)):
        fName=hIndex[i]
        cif_name=f"{hinDir}/{fName}.cif"
        try:
            struct=Structure.from_file(cif_name,)
            rppdgm=rpp.run("--format distance --dim 2",struct.distance_matrix)
            npdgm=[np.array([[float(rppdgm[b][k][0]),float(rppdgm[b][k][1])] for k in range(len(rppdgm[b]))])for b in rppdgm.keys()]
            pdStack.addDiagRpp(npdgm,int(fName[fName.find('-')+1:]))
        except:
            failMOFs.add(fName)
    with open(f"{hOut}/test_pdStack_res01_b{binNum}_{numBins}.pkl","wb") as f:
        pickle.dump(pdStack,f)
    with open(f"{hOut}/test_pdStack_res01_index_b{binNum}_{numBins}.pkl", "wb") as f:
        pickle.dump(hIndex, f)


print(f"Testing complete!! failed MOFs:\n")
for i in failMOFs:
    print(i)


