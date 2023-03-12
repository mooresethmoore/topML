import os
import sys
import typing
import pickle
import json
import numpy as np
import pandas as pd

from pdHash import PDhash



from pymatgen.io.cif import CifWriter
from pymatgen.io.cif import CifFile
from pymatgen.core.structure import Structure


import gudhi,gudhi.hera,gudhi.wasserstein,persim
import ripserplusplus as rpp

inDir="Z:/data/diverse_metals"
os.chdir(inDir)

df=pd.read_csv(f"{inDir}/post-combustion-vsa-2-clean.csv",index_col=0)
mofNames=list(df.index)
cols=df.columns #[wc,sel]
totalLen=len(df.index)




hinDir=f"{inDir}/cifs"
hOut=f"{inDir}/diverseTopPDhash"



failMOFs=set()


res=.1

pdStack = PDhash(res=res, diags=None, maxHdim=2, persistThresh=0)
#hIndex=list(trainBins["workCap"][binNum])
for i in range(len(mofNames)):
    fName=mofNames[i]
    cif_name=f"{hinDir}/{fName}.cif"
    try:
        struct=Structure.from_file(cif_name,)

        rppdgm=rpp.run("--format distance --dim 2",struct.distance_matrix)
        npdgm=[np.array([[float(rppdgm[b][k][0]),float(rppdgm[b][k][1])] for k in range(len(rppdgm[b]))])for b in rppdgm.keys()]
        pdStack.addDiagRpp(npdgm,i)
    except:
        failMOFs.add(fName)
with open(f"{hOut}/arcMOF_pdHash_pbcXYZ_tenthRes.pkl","wb") as f:
    pickle.dump(pdStack,f)



print(f"GrabXYZ complete!! failed MOFs:\n")
for i in failMOFs:
    print(i)

