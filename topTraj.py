import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
from scipy.spatial import distance_matrix
import gudhi, gudhi.hera, gudhi.wasserstein, persim
import json
import plotly
from plotly.graph_objs import graph_objs as go
import ipywidgets as widgets
plotly.offline.init_notebook_mode(connected=True)
from plotly.offline import iplot
import mdtraj
from mdtraj import load



os.chdir("/mnt/z/data/micelles/Micelles_identification")
polyComp=os.listdir()
polyComp=[i for i in polyComp if i.find(".")==-1 and i!="High_conc"]


temps=[int(i)*10 for i in range(1,6)]
#temps

res = [1200, 1200]

res = [1200, 1200]


def genTimeCluster(groFile, alpha=.05, saveName=None, cam=dict(eye=dict(x=.75, y=2, z=.25)), titPref="",
                   dt=1):  # assume 3D for now
    if len(groFile.xyz.shape) == 3:
        t = 0
        for dat in groFile.xyz:
            ac = gudhi.AlphaComplex(dat)
            st = ac.create_simplex_tree()
            skel = list(st.get_skeleton(2))
            skel.sort(key=lambda s: s[1])
            points = np.array([ac.get_point(i) for i in range(st.num_vertices())])
            lims = [[np.floor(np.min(dat[:, i])), np.ceil(np.max(dat[:, i]))] for i in range(3)]

            sfig = [
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color="cornflowerblue",  # set color to an array/list of desired values
                        # colorscale='Viridis',   # choose a colorscale
                        opacity=.9
                    ),
                    name='H0'
                )
            ]

            b1s = np.array([s[0] for s in skel if len(s[0]) == 2 and s[1] <= alpha])

            linepts = {0: [], 1: [], 2: []}
            for i in b1s:
                linepts[0].append(points[i[0], 0])
                linepts[1].append(points[i[0], 1])
                linepts[2].append(points[i[0], 2])
                linepts[0].append(points[i[1], 0])
                linepts[1].append(points[i[1], 1])
                linepts[2].append(points[i[1], 2])

                linepts[0].append(None)
                linepts[1].append(None)
                linepts[2].append(None)

            if len(linepts[0]) > 0:
                lins = go.Scatter3d(
                    x=linepts[0],
                    y=linepts[1],
                    z=linepts[2],
                    mode='lines',
                    name='H1',
                    marker=dict(
                        size=3,
                        color="#d55e00",  # set color to an array/list of desired values
                        # colorscale='Viridis',   # choose a colorscale
                        opacity=.85
                    )
                )
                sfig.append(lins)
                triangles = np.array([s[0] for s in skel if len(s[0]) == 3 and s[1] <= alpha])
                if len(triangles) > 0:
                    mesh = go.Mesh3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        i=triangles[:, 0],
                        j=triangles[:, 1],
                        k=triangles[:, 2],
                        color="#009e73",
                        opacity=.75,
                        name='H2'
                    )
                    sfig.append(mesh)
            fig = go.Figure(data=sfig, layout=go.Layout(width=res[0], height=res[1],
                                                        title=f"{titPref}       :       Simplicial complex with radius <= {round(float(alpha), 5)}       \t\t\tFrame:  {t * dt} ns",
                                                        scene_camera=cam
                                                        ))
            if type(saveName) == str:
                fig.write_json(file=saveName + f"t{t}.json", engine="auto")
                fig.write_image(file=saveName + f"t{t}.png")
            t += 1
    else:
        print("err in .xyz grab on input")
        return


def genAlphaEvolution(dat,initial=.05,step=.25,maximum=20, saveName=None, cam=dict(eye=dict(x=.75, y=2, z=.25)), titPref="",): #assume 3D for now
    ac = gudhi.AlphaComplex(dat)
    st = ac.create_simplex_tree()
    skel=list(st.get_skeleton(2))
    skel.sort(key=lambda s: s[1])
    points = np.array([ac.get_point(i) for i in range(st.num_vertices())])
    #lims=[[np.floor(np.min(dat[:,i])),np.ceil(np.max(dat[:,i]))] for i in range(3)]

    frame=0
    alpha=initial



    pts=go.Scatter3d(
        x = points[:, 0],
        y = points[:, 1],
        z = points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color="cornflowerblue",                # set color to an array/list of desired values
            #colorscale='Viridis',   # choose a colorscale
            opacity=.9
        ),
        name='H0'

    )

    sfig=[pts]

    skelNum=len(points)

    while skelNum < len(skel) and alpha <=maximum:
        s=skel[skelNum]
        if len(s[0])==2 and s[1]<=alpha:


        alpha+=step
        skelNum+=1

    #b1s=np.array([s[0] for s in skel if len(s[0]) == 2 and s[1] <= alpha])
    #triangles = np.array([s[0] for s in skel if len(s[0]) == 3 and s[1] <= alpha])



    linepts={0:[],1:[],2:[]}
    for i in b1s:
        linepts[0].append(points[i[0],0])
        linepts[1].append(points[i[0],1])
        linepts[2].append(points[i[0],2])
        linepts[0].append(points[i[1],0])
        linepts[1].append(points[i[1],1])
        linepts[2].append(points[i[1],2])

        linepts[0].append(None)
        linepts[1].append(None)
        linepts[2].append(None)

    if len(linepts[0])>0:
        lins=go.Scatter3d(
            x=linepts[0],
            y=linepts[1],
            z=linepts[2],
            mode='lines',
            name='H1',
            marker=dict(
                size=3,
                color="#d55e00",                # set color to an array/list of desired values
                #colorscale='Viridis',   # choose a colorscale
                opacity=.9
            )
        )
        sfig.append(lins)
        if len(triangles)>0:
            mesh = go.Mesh3d(
                x = points[:, 0],
                y = points[:, 1],
                z = points[:, 2],
                i = triangles[:, 0],
                j = triangles[:, 1],
                k = triangles[:, 2],
                color="#009e73",
                opacity=.75,
                name='H2'
            )


            sfig.append(mesh)
    fig=go.Figure(sfig)
    fig.update_layout(width=800,height=800)
    #fig.show()


    if type(saveName) == str:
        fig.write_json(file=saveName + f"t{t}.json", engine="auto")
        fig.write_image(file=saveName + f"t{t}.png")

    else:
        print("err in .xyz grab on input")
        return

def saveMov(saveNameBase,res,fps,outName,tryRm=True, delFileCheck = lambda f: f.find(".png")!=-1 or f.find(".jpg")!=-1 or f.find(".json")!=-1):
    osout=os.system(f"ffmpeg -r {fps} -f image2 -s {res[0]}x{res[1]} -i {saveNameBase}t%d.png -vcodec libx264 -crf 18 {outName}.mp4")
    if osout==0 and tryRm:
        rootDir=saveNameBase[:-1* saveNameBase[::-1].find("/")]
        for f in os.listdir(rootDir):
            if delFileCheck(f):
                try:
                    os.remove(f"{rootDir}{f}")
                except:
                    print(f"del error! \t {rootDir}{f}\n\n")
                    break



def __main__():
    groName = "P85_ini_40"
    titPref = f"{groName[:-3]} @ {groName[-2:]}" + " \u00B0 C"
    groFile = mdtraj.load(f"{groName[:groName.find('_')]}/{groName}.gro")
    p0Index = [a.index for a in groFile.topology.atoms if a.name.find('PO') != -1]
    p0Gro = groFile.restrict_atoms(p0Index)

    # saveDir=r"C:\code\git\topML\frames\\"
    saveDir = f"/mnt/c/code/git/topML/frames/"
    try:
        os.mkdir(saveDir + f"init")
    except:
        pass

    dt = 4  # 4 ns between frame I believe
    alp = .05
    saveName = saveDir + "/init/" + f"{groName}_{alp}"
    genTimeCluster(p0Gro, alpha=alp, saveName=saveName, titPref=titPref, dt=dt)

    fps = 5
    tryRm = False
    outName = f"/mnt/c/code/git/topML/frames/{groName}_{alp}"
    saveMov(saveName, [1200, 1200], fps, outName, tryRm=tryRm)