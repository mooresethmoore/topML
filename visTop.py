import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
import gudhi, gudhi.hera, gudhi.wasserstein, persim
import ripserplusplus as rpp_py
from scipy.spatial import distance

import plotly
from plotly.graph_objs import graph_objs as go
import ipywidgets as widgets

#plotly.offline.init_notebook_mode(connected=True)
from plotly.offline import iplot

os.chdir("Z:/data/micelles/")

points=np.load(open('toMac/Ptraj0.npy',"rb"))
ac= gudhi.AlphaComplex(points)
st = ac.create_simplex_tree()


points = np.array([ac.get_point(i) for i in range(st.num_vertices())])
# We want to plot the alpha-complex with alpha=0.005 by default.
# We are only going to plot the triangles
triangles = np.array([s[0] for s in st.get_skeleton(2) if len(s[0]) == 3 and s[1] <= 0.05])

import plotly.io as pio
pio.renderers.default = "svg"

lims=[[np.floor(np.min(points[:,i])),np.ceil(np.max(points[:,i]))] for i in range(3)]
alpha = widgets.FloatSlider(
    value = 0.02,
    min = 0.0,
    max = 20,
    step = 0.05,
    description = 'Alpha:',
    readout_format = '.4f'
)

mesh = go.Mesh3d(
    x = points[:, 0],
    y = points[:, 1],
    z = points[:, 2],
    i = triangles[:, 0],
    j = triangles[:, 1],
    k = triangles[:, 2],
    color="green"
)


pts=go.Scatter3d(
    x = points[:, 0],
    y = points[:, 1],
    z = points[:, 2],
    mode='markers',
    marker=dict(
        size=1,
        color="cornflowerblue",                # set color to an array/list of desired values
        #colorscale='Viridis',   # choose a colorscale
        opacity=.9
    )
)

fig = go.FigureWidget(
    data = [mesh,pts],
    layout = go.Layout(
        title = dict(
            text = f'Simplicial Complex Representation of the P0 distribution'
        ),
        scene = dict(
            xaxis = dict(nticks = 15, range = lims[0]),
            yaxis = dict(nticks = 15, range = lims[1]),
            zaxis = dict(nticks = 15, range = lims[2])
        )
    )
)

def view_SC(alpha):
    if alpha < 0.0015:
        alpha = 0.0015
    triangles = np.array([s[0] for s in st.get_skeleton(2) if len(s[0]) == 3 and s[1] <= alpha])
    if len(triangles)>0:
        fig.data[0].i = triangles[:, 0]
        fig.data[0].j = triangles[:, 1]
        fig.data[0].k = triangles[:, 2]
    else:
        fig.data[0].i = []
        fig.data[0].j = []
        fig.data[0].k = []
    #fig.update_layout(title_text=f'Simplicial Complex Representation of the P0 distribution (epsilon={alpha.value})')
    iplot(fig)

control= widgets.interact(view_SC, alpha = alpha)

from IPython.display import display
display(control)