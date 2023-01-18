import pandas as pd
import numpy as np
import json, io, re, os
import pymatgen.core as mg
from pymatgen.io.cif import CifWriter
import plotly.graph_objects as go
from IPython.display import display, HTML
from scipy.spatial import Delaunay
display(HTML("<style>.container { width:90% !important; }</style>"))


def read_cif_lattice_param(fname): 
    cif_str = None
    with io.open(fname, "r", newline="\n") as cif:
        cif_str = cif.read().replace("(", "").replace(")", "")

    cif_list = cif_str.split("loop_")
    for cif_section in cif_list:
        if "_cell_length_a" in cif_section:
            a = float(cif_section.split("_cell_length_a")[1].split("\n")[0])
            b = float(cif_section.split("_cell_length_b")[1].split("\n")[0])
            c = float(cif_section.split("_cell_length_c")[1].split("\n")[0])

            alpha = float(cif_section.split("_cell_angle_alpha")[1].split("\n")[0])
            beta = float(cif_section.split("_cell_angle_beta")[1].split("\n")[0])
            gamma = float(cif_section.split("_cell_angle_gamma")[1].split("\n")[0])
            break
    return a, b, c, alpha, beta, gamma

def read_cif_xyz(fname): 
    cif_str = None
    with io.open(fname, "r", newline="\n") as cif:
        cif_str = cif.read().replace("(", "").replace(")", "")
    cif_list = cif_str.split("loop_")
    for cif_section in cif_list:
        if "_atom_site_fract_x" in cif_section:
            #atom_lines = cif_section.split("\n")
            columns = list(re.findall(r".*_atom_.*\n", cif_section))
            df_str = cif_section.replace("".join(columns), "").strip()
            columns = [x.strip() for x in columns]
            return pd.read_csv(io.StringIO(df_str), names=columns, sep=r"\s+")
    return None

def read_cif_bond(fname): 
    cif_str = None
    with io.open(fname, "r", newline="\n") as cif:
        cif_str = cif.read().replace("(", "").replace(")", "")
    cif_list = cif_str.split("loop_")
    for cif_section in cif_list:
        if "_geom_bond_atom_site_label_1" in cif_section:
            #bond_lines = cif_section.split("\n")
            columns = list(re.findall(r".*_geom_.*\n", cif_section))
            df_str = cif_section.replace("".join(columns), "").strip()
            columns = [x.strip() for x in columns]
            return pd.read_csv(io.StringIO(df_str), names=columns, sep=r"\s+")
    return None

def pairwise_distance_pbc(df1, df2, M):
    needed_cols = ["_atom_site_label", "_atom_site_type_symbol", 
                   "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    xyz_cols = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    site_col = "_atom_site_label"
    np1_expand = df1.loc[:, needed_cols].to_numpy().repeat(len(df2), axis=0)
    np2_expand = np.tile(df2.loc[:, needed_cols].to_numpy(), (len(df1), 1))
    df1_expand = pd.DataFrame(np1_expand, columns=needed_cols)
    df2_expand = pd.DataFrame(np2_expand, columns=needed_cols)
    dxdydz = df1_expand.loc[:, xyz_cols] - df2_expand.loc[:, xyz_cols]
    dxdydz[dxdydz>0.5] -= 1
    dxdydz[dxdydz<-0.5] += 1
    dist2 = ((dxdydz @ M) * (dxdydz @ M)).sum(axis=1)
    sub_df = pd.DataFrame(np.array([df1_expand.loc[:, site_col].to_list(), 
                                    df2_expand.loc[:, site_col].to_list(), 
                                    dist2.to_list()]).T, 
                          columns=["A", "B", "dist2"]).astype({"A": str, 
                                                               "B": str, 
                                                               "dist2": float})
    sub_df = sub_df.sort_values(by="dist2").reset_index(drop=True)
    return sub_df

def lat_param2vec(lat_param):
    a, b, c, alpha, beta, gamma = lat_param
    alpha_rad = alpha * np.pi / 180
    beta_rad = beta * np.pi / 180
    gamma_rad = gamma * np.pi / 180
    n2 = (np.cos(alpha_rad)-np.cos(gamma_rad)*np.cos(beta_rad))/np.sin(gamma_rad)
    M  = np.array([[a,                   0,                   0],
                   [b*np.cos(gamma_rad), b*np.sin(gamma_rad), 0], 
                   [c*np.cos(beta_rad),  c*n2,                c*np.sqrt(np.sin(beta_rad)**2-n2**2)]])
    return M


def read_cgd(filename, node_symbol="C", edge_center_symbol="O", primitive=True):
    """
    Read cgd format and return topology as ase.Atoms object.
    """
    
    with io.open(filename, "r") as f:
        # Neglect "CRYSTAL" and "END"
        lines = f.readlines()[1:-1]
    lines = [line for line in lines if not line.startswith("#")]

    # Get topology name.
    name = lines[0].split()[1]
    # Get spacegroup.
    spacegroup = lines[1].split()[1]

    # Get cell paremeters and expand cell lengths by 10.
    cellpar = np.array(lines[2].split()[1:], dtype=np.float32)

    # Parse node information.
    node_positions = []
    coordination_numbers = []
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "NODE":
            continue

        coordination_number = int(tokens[2])
        pos = [float(r) for r in tokens[3:]]
        node_positions.append(pos)
        coordination_numbers.append(coordination_number)

    node_positions = np.array(node_positions)
    #coordination_numbers = np.array(coordination_numbers)

    # Parse edge information.
    edge_center_positions = []
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "EDGE":
            continue

        pos_i = np.array([float(r) for r in tokens[1:4]])
        pos_j = np.array([float(r) for r in tokens[4:]])

        edge_center_pos = 0.5 * (pos_i+pos_j)
        edge_center_positions.append(edge_center_pos)

    # New feature. Read EDGE_CENTER.
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "EDGE_CENTER":
            continue

        edge_center_pos = np.array([float(r) for r in tokens[1:]])
        edge_center_positions.append(edge_center_pos)

    edge_center_positions = np.array(edge_center_positions)

    # Carbon for nodes, oxygen for edges.
    n_nodes = node_positions.shape[0]
    n_edges = edge_center_positions.shape[0]
    species = np.concatenate([
        np.full(shape=n_nodes, fill_value=node_symbol),
        np.full(shape=n_edges, fill_value=edge_center_symbol),
    ])

    coords = np.concatenate([node_positions, edge_center_positions], axis=0)

    # Pymatget can handle : indicator in spacegroup.
    # Mark symmetrically equivalent sites.
    node_types = [i for i, _ in enumerate(node_positions)]
    edge_types = [-(i+1) for i, _ in enumerate(edge_center_positions)]
    site_properties = {
        "type": node_types + edge_types,
        "cn": coordination_numbers + [2 for _ in edge_center_positions],
    }

    # I don't know why pymatgen can't parse this spacegroup.
    if spacegroup == "Cmca":
        spacegroup = "Cmce"

    if primitive:
        structure = mg.Structure.from_spacegroup(
                        sg=spacegroup,
                        lattice=mg.Lattice.from_parameters(*cellpar),
                        species=species,
                        coords=coords,
                        site_properties=site_properties,
                    ).get_primitive_structure()
    else:
        structure = mg.Structure.from_spacegroup(
                        sg=spacegroup,
                        lattice=mg.Lattice.from_parameters(*cellpar),
                        species=species,
                        coords=coords,
                        site_properties=site_properties,
                    )
    return structure

# mof
def viz_mof_sub(mof_obj):
    table_dict = {}
    # https://github.com/Bowserinator/Periodic-Table-JSON
    with io.open("PeriodicTableJSON.json", "rb") as f:
        table_dict = json.load(f)
    element_df = pd.DataFrame(table_dict["elements"])
    element_df["cpk-hex"] = element_df["cpk-hex"].fillna("0000ff")
    symbol2color = dict(zip(element_df["symbol"], element_df["cpk-hex"]))
    symbol2color["X"] = "0000ff"
    idx2symbol = dict(zip(element_df["number"], element_df["symbol"]))
    idx2symbol[0] = "X"

    symbol = [idx2symbol[x] for x in mof_obj.atoms.arrays["numbers"]]
    node_x = mof_obj.atoms.arrays["positions"].T[0]
    node_y = mof_obj.atoms.arrays["positions"].T[1]
    node_z = mof_obj.atoms.arrays["positions"].T[2]

    bond_scatter = []
    if hasattr(mof_obj, "bonds"):
        for bond in mof_obj.bonds:
            if np.sqrt((node_x[bond[0]]-node_x[bond[1]])*(node_x[bond[0]]-node_x[bond[1]]) + \
                       (node_y[bond[0]]-node_y[bond[1]])*(node_y[bond[0]]-node_y[bond[1]]) + \
                       (node_z[bond[0]]-node_z[bond[1]])*(node_z[bond[0]]-node_z[bond[1]])) < 5:
                bond_scatter.append(go.Scatter3d(x=[node_x[bond[0]], node_x[bond[1]]], 
                                                 y=[node_y[bond[0]], node_y[bond[1]]], 
                                                 z=[node_z[bond[0]], node_z[bond[1]]],
                                                 mode="lines",
                                                 line=dict(
                                                     color='rgba(220, 200, 200, 0.5)',
                                                     width=5,
                                                 ),
                                                 showlegend=False,
                                                 hoverinfo='skip')
                )
    node_x_dict = {}
    node_y_dict = {}
    node_z_dict = {}
    symbol_dict = {}
    color_dict = {}
    for i in range(0, len(symbol)):
        if symbol[i] in node_x_dict.keys():
            node_x_dict[symbol[i]].append(node_x[i])
            node_y_dict[symbol[i]].append(node_y[i])
            node_z_dict[symbol[i]].append(node_z[i])
            symbol_dict[symbol[i]].append(symbol[i])
            color_dict[symbol[i]].append("#"+symbol2color[symbol[i]])
        else:
            node_x_dict[symbol[i]] = [node_x[i]]
            node_y_dict[symbol[i]] = [node_y[i]]
            node_z_dict[symbol[i]] = [node_z[i]]
            symbol_dict[symbol[i]] = [symbol[i]]
            color_dict[symbol[i]] = ["#"+symbol2color[symbol[i]]]
    data=[go.Scatter3d(x=node_x_dict[k], 
                       y=node_y_dict[k], 
                       z=node_z_dict[k], 
                       text=symbol_dict[k],
                       mode='markers',
                       showlegend=True,
                       name=k,
                       marker=dict(
                           color=color_dict[k],
                           size=20,
                           opacity=1.0,
                       )) for k in symbol_dict.keys()] + \
                       bond_scatter
    xyzmin = min([min(node_x), min(node_y), min(node_z)])
    xyzmax = max([max(node_x), max(node_y), max(node_z)])
    DeltaX = xyzmax - xyzmin
    padding_xyz = DeltaX * 0.05
    fig = go.Figure(data=data)
    annotation_list = []
    fig.update_layout(
        scene = dict(
            annotations=annotation_list,
            xaxis = dict(nticks=10, range=[xyzmin-padding_xyz,xyzmax+padding_xyz],
                         backgroundcolor="rgba(80, 70, 70, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                        ),
            yaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 80, 70, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                        ),
            zaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 70, 80, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
        ),
        width=1200,
        height=1200,
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=True)
    fig.update_layout(scene_aspectmode='cube', 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        font_color="rgba(150,150,150,1)",
        title_font_color="rgba(150,150,150,1)",
        legend_title_font_color="rgba(150,150,150,1)",
    )
    #fig.show()
    return fig



# mof topo
def viz_topo(topo_obj, tol=1e-4):
    df = pd.DataFrame(topo_obj.cart_coords, columns=["x", "y", "z"])
    element2nodelinker = {"C": "node", "O": "linker"}
    df["nl"] = [element2nodelinker[x.symbol] for x in topo_obj.species]

    cell = topo_obj.lattice.matrix
    df_list = []

    critical_points = [np.array([0, 0, 0]),
                       cell[0], 
                       cell[1], 
                       cell[2], 
                       cell[0]+cell[1], 
                       cell[1]+cell[2], 
                       cell[0]+cell[2], 
                       cell[0]+cell[1]+cell[2]]

    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            for iz in [-1, 0, 1]:
                disp = (ix*cell[0])+(iy*cell[1])+(iz*cell[2])
                _df = df.copy(deep=True)
                _df.loc[:, ["x", "y", "z"]] = _df.loc[:, ["x", "y", "z"]] + disp
                df_list.append(_df)
    df = pd.concat(df_list, axis=0).copy(deep=True).drop_duplicates().reset_index(drop=True)


    hull = Delaunay(critical_points)

    isInHull = hull.find_simplex(df.loc[:, ["x", "y", "z"]].values)>=-tol
    df = df[isInHull]
    node_df = df[df["nl"]=="node"].copy(deep=True).reset_index(drop=True)
    node_x = node_df["x"]
    node_y = node_df["y"]
    node_z = node_df["z"]

    linker_df = df[df["nl"]=="linker"].copy(deep=True).reset_index(drop=True)


    x1 = pd.DataFrame(np.tile(np.tile(node_df[["x", "y", "z"]].values.astype(float), (len(node_df), 1)), (len(linker_df), 1)))
    x2 = pd.DataFrame(np.tile(np.repeat(node_df[["x", "y", "z"]].values.astype(float), len(node_df), axis=0), (len(linker_df), 1)))
    ct = pd.DataFrame(np.repeat(linker_df[["x", "y", "z"]].values.astype(float), len(node_df)*len(node_df), axis=0))

    _df = pd.concat([x1, ct, x2], axis=1)
    _df.columns = ["p1x", "p1y", "p1z", "pcx", "pcy", "pcz", "p2x", "p2y", "p2z"]
    _df = _df.astype({"p1x": float, "p1y": float, "p1z": float, 
                      "pcx": float, "pcy": float, "pcz": float, 
                      "p2x": float, "p2y": float, "p2z": float})

    _df["v1x"] = None
    _df["v2x"] = None
    _df["v1y"] = None
    _df["v2y"] = None
    _df["v1z"] = None
    _df["v2z"] = None
    _df["_v1x"] = None
    _df["_v2x"] = None
    _df["_v1y"] = None
    _df["_v2y"] = None
    _df["_v1z"] = None
    _df["_v2z"] = None
    _df["dpx"] = None
    _df["dpy"] = None
    _df["dpz"] = None

    _df.loc[:, ["v1x", "v1y", "v1z"]] = _df.loc[:, ["p1x", "p1y", "p1z"]].values.astype(float) - _df.loc[:, ["pcx", "pcy", "pcz"]].values.astype(float)
    _df.loc[:, ["v2x", "v2y", "v2z"]] = _df.loc[:, ["p2x", "p2y", "p2z"]].values.astype(float) - _df.loc[:, ["pcx", "pcy", "pcz"]].values.astype(float)
    _df.loc[:, ["dpx", "dpy", "dpz"]] = _df.loc[:, ["p1x", "p1y", "p1z"]].values.astype(float) - _df.loc[:, ["p2x", "p2y", "p2z"]].values.astype(float)
    _df["dp2"] = np.sum(_df.loc[:, ["dpx", "dpy", "dpz"]].values.astype(float) * _df.loc[:, ["dpx", "dpy", "dpz"]].values.astype(float), axis=1)
    _df = _df[_df["dp2"] < 2]
    _df.loc[:, ["_v1x", "_v1y", "_v1z"]] = _df.loc[:, ["v1x", "v1y", "v1z"]].values.astype(float) / np.tile(np.linalg.norm(_df.loc[:, ["v1x", "v1y", "v1z"]].values.astype(float), axis=1), (3, 1)).T
    _df.loc[:, ["_v2x", "_v2y", "_v2z"]] = _df.loc[:, ["v2x", "v2y", "v2z"]].values.astype(float) / np.tile(np.linalg.norm(_df.loc[:, ["v2x", "v2y", "v2z"]].values.astype(float), axis=1), (3, 1)).T
    _df["cos_theta"] = np.sum(_df.loc[:, ["_v1x", "_v1y", "_v1z"]].values.astype(float) * _df.loc[:, ["_v2x", "_v2y", "_v2z"]].values.astype(float), axis=1).astype(float)
    
    bond_df = _df[np.abs(_df["cos_theta"]+1) < tol].copy(deep=True).drop_duplicates().reset_index(drop=True)


    bond_scatter = []
    for bond_idx in bond_df.index:
        if bond_idx == bond_df.index.min():
            showlegend=True
        else:
            showlegend=False
        bond_scatter.append(go.Scatter3d(x=[bond_df.at[bond_idx, "p1x"], bond_df.at[bond_idx, "pcx"], bond_df.at[bond_idx, "p2x"]], 
                                         y=[bond_df.at[bond_idx, "p1y"], bond_df.at[bond_idx, "pcy"], bond_df.at[bond_idx, "p2y"]], 
                                         z=[bond_df.at[bond_idx, "p1z"], bond_df.at[bond_idx, "pcz"], bond_df.at[bond_idx, "p2z"]],
                                         mode="lines+markers",
                                         line=dict(
                                             color='rgba(0, 220, 220, 0.8)',
                                             width=20,
                                         ),
                                         marker=dict(
                                             size=20,
                                             color='rgba(0, 220, 220, 0.1)',
                                             opacity=1.0,
                                         ),
                                         showlegend=showlegend,
                                         text=[None, "Linker", None],
                                         name="Linker",)
        )

    # for orig_bond_id in linker_df.index:
    #     bond_scatter.append(go.Scatter3d(x=[linker_df.at[orig_bond_id, "x"]], 
    #                                      y=[linker_df.at[orig_bond_id, "y"]], 
    #                                      z=[linker_df.at[orig_bond_id, "z"]], 
    #                                      mode="markers",
    #                                      marker=dict(
    #                                          size=40,
    #                                          color='rgba(0, 220, 220, 0.1)',
    #                                          opacity=0.1,
    #                                      ),
    #                                      showlegend=showlegend,
    #                                      text="Linker",
    #                                      name="Linker",)
    #     )

    A = cell[0]
    B = cell[1]
    C = cell[2]
    unit_cell = [[np.array([0, 0, 0]), A], [np.array([0, 0, 0]), B], [np.array([0, 0, 0]), C], 
                 [A, A+B], [A, A+C], [B, B+A], 
                 [B, B+C], [C, C+A], [C, C+B], 
                 [A+B, A+B+C], [B+C, A+B+C], [A+C, A+B+C]]
    unit_cell_color = ["rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)"]
    
    # XYZ arrows
    axis_length = 0.5
    V_diag = 0.25 * (A + B + C)
    x_start = -V_diag[0]
    y_start = -V_diag[1]
    z_start = -V_diag[2]
    x_end1 = (axis_length * A - V_diag)[0]
    y_end1 = (axis_length * A - V_diag)[1]
    z_end1 = (axis_length * A - V_diag)[2]
    x_end2 = (axis_length * B - V_diag)[0]
    y_end2 = (axis_length * B - V_diag)[1]
    z_end2 = (axis_length * B - V_diag)[2]
    x_end3 = (axis_length * C - V_diag)[0]
    y_end3 = (axis_length * C - V_diag)[1]
    z_end3 = (axis_length * C - V_diag)[2]
    arrows = [
        go.Scatter3d(x=[x_start, x_end1], 
                     y=[y_start, y_end1], 
                     z=[z_start, z_end1],
                     mode="lines+text",
                     text=["", "A"],
                     line=dict(
                         color='rgba(255, 0, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(255, 0, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end2], 
                     y=[y_start, y_end2], 
                     z=[z_start, z_end2],
                     mode="lines+text",
                     text=["", "B"],
                     line=dict(
                         color='rgba(0, 255, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 255, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end3], 
                     y=[y_start, y_end3], 
                     z=[z_start, z_end3],
                     mode="lines+text",
                     text=["", "C"],
                     line=dict(
                         color='rgba(0, 0, 255, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 0, 255, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
    ]


    data=arrows + [go.Scatter3d(x=node_x, 
                       y=node_y, 
                       z=node_z, 
                       text="Node",
                       name="Node",
                       mode='markers',
                       showlegend=True,
                       marker=dict(
                           size=40,
                           opacity=1.0,
                       ))] + bond_scatter + [go.Scatter3d(x=np.array(unit_cell[i]).T[0], 
                                                          y=np.array(unit_cell[i]).T[1], 
                                                          z=np.array(unit_cell[i]).T[2],
                                                          mode="lines",
                                                          line=dict(
                                                              color=unit_cell_color[i],
                                                              width=2,
                                                          ),
                                                          showlegend=False,
                                                          hoverinfo='skip') for i in range(0, len(unit_cell))]


    xyzmin = min([min(node_x), min(node_y), min(node_z), np.array(unit_cell).min(), (0-V_diag).min(), 
        bond_df.loc[:, ["p1x", "p1y", "p1z", "p2x", "p2y", "p2z", "pcx", "pcy", "pcz"]].values.min(), 
        linker_df.loc[:, ["x", "y", "z"]].values.min()])
    xyzmax = max([max(node_x), max(node_y), max(node_z), np.array(unit_cell).max(), (0-V_diag).max(), 
        bond_df.loc[:, ["p1x", "p1y", "p1z", "p2x", "p2y", "p2z", "pcx", "pcy", "pcz"]].values.max(), 
        linker_df.loc[:, ["x", "y", "z"]].values.max()])
    DeltaX = xyzmax - xyzmin
    padding_xyz = DeltaX * 0.1
    fig = go.Figure(data=data)
    annotation_list = []
    fig.update_layout(
        scene = dict(
            annotations=annotation_list,
            xaxis = dict(nticks=10, range=[xyzmin-padding_xyz,xyzmax+padding_xyz],
                         backgroundcolor="rgba(80, 70, 70, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                        ),
            yaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 80, 70, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                        ),
            zaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 70, 80, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
        ),
        width=1200,
        height=1200,
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=True)
    fig.update_layout(scene_aspectmode='cube', 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        font_color="rgba(150,150,150,1)",
        title_font_color="rgba(150,150,150,1)",
        legend_title_font_color="rgba(150,150,150,1)",
    )
    #fig.show()
    return fig
    #return bond_df, linker_df


# mof_obj
def viz_mof_obj(mof_obj):
    table_dict = {}
    # https://github.com/Bowserinator/Periodic-Table-JSON
    with io.open("PeriodicTableJSON.json", "rb") as f:
        table_dict = json.load(f)
    element_df = pd.DataFrame(table_dict["elements"])
    element_df["cpk-hex"] = element_df["cpk-hex"].fillna("0000ff")
    symbol2color = dict(zip(element_df["symbol"], element_df["cpk-hex"]))
    symbol2color["X"] = "0000ff"
    idx2symbol = dict(zip(element_df["number"], element_df["symbol"]))
    idx2symbol[0] = "X"
    
    symbol = [idx2symbol[x] for x in mof_obj.atoms.arrays["numbers"]]
    node_x = mof_obj.atoms.arrays["positions"].T[0]
    node_y = mof_obj.atoms.arrays["positions"].T[1]
    node_z = mof_obj.atoms.arrays["positions"].T[2]
    
    bond_scatter = []
    for bond in mof_obj.bonds:
        if np.sqrt((node_x[bond[0]]-node_x[bond[1]])*(node_x[bond[0]]-node_x[bond[1]]) + \
                   (node_y[bond[0]]-node_y[bond[1]])*(node_y[bond[0]]-node_y[bond[1]]) + \
                   (node_z[bond[0]]-node_z[bond[1]])*(node_z[bond[0]]-node_z[bond[1]])) < 5:
            bond_scatter.append(
                go.Scatter3d(
                    x=[node_x[bond[0]], node_x[bond[1]]], 
                    y=[node_y[bond[0]], node_y[bond[1]]], 
                    z=[node_z[bond[0]], node_z[bond[1]]],
                    mode="lines",
                    line=dict(
                        color='rgba(220, 200, 200, 0.5)',
                        width=5,
                    ),
                    showlegend=False,
                    hoverinfo='skip')
            )

    A = mof_obj.atoms.cell[0]
    B = mof_obj.atoms.cell[1]
    C = mof_obj.atoms.cell[2]
    unit_cell = [[np.array([0, 0, 0]), A], [np.array([0, 0, 0]), B], [np.array([0, 0, 0]), C], 
                 [A, A+B], [A, A+C], [B, B+A], 
                 [B, B+C], [C, C+A], [C, C+B], 
                 [A+B, A+B+C], [B+C, A+B+C], [A+C, A+B+C]]
    unit_cell_color = ["rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)"]
    
    # XYZ arrows
    axis_length = 0.5
    V_diag = 0.25 * (A + B + C)
    x_start = -V_diag[0]
    y_start = -V_diag[1]
    z_start = -V_diag[2]
    x_end1 = (axis_length * A - V_diag)[0]
    y_end1 = (axis_length * A - V_diag)[1]
    z_end1 = (axis_length * A - V_diag)[2]
    x_end2 = (axis_length * B - V_diag)[0]
    y_end2 = (axis_length * B - V_diag)[1]
    z_end2 = (axis_length * B - V_diag)[2]
    x_end3 = (axis_length * C - V_diag)[0]
    y_end3 = (axis_length * C - V_diag)[1]
    z_end3 = (axis_length * C - V_diag)[2]
    arrows = [
        go.Scatter3d(x=[x_start, x_end1], 
                     y=[y_start, y_end1], 
                     z=[z_start, z_end1],
                     mode="lines+text",
                     text=["", "A"],
                     line=dict(
                         color='rgba(255, 0, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(255, 0, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end2], 
                     y=[y_start, y_end2], 
                     z=[z_start, z_end2],
                     mode="lines+text",
                     text=["", "B"],
                     line=dict(
                         color='rgba(0, 255, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 255, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end3], 
                     y=[y_start, y_end3], 
                     z=[z_start, z_end3],
                     mode="lines+text",
                     text=["", "C"],
                     line=dict(
                         color='rgba(0, 0, 255, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 0, 255, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
    ]

    node_x_dict = {}
    node_y_dict = {}
    node_z_dict = {}
    symbol_dict = {}
    color_dict = {}
    for i in range(0, len(symbol)):
        if symbol[i] in node_x_dict.keys():
            node_x_dict[symbol[i]].append(node_x[i])
            node_y_dict[symbol[i]].append(node_y[i])
            node_z_dict[symbol[i]].append(node_z[i])
            symbol_dict[symbol[i]].append(symbol[i])
            color_dict[symbol[i]].append("#"+symbol2color[symbol[i]])
        else:
            node_x_dict[symbol[i]] = [node_x[i]]
            node_y_dict[symbol[i]] = [node_y[i]]
            node_z_dict[symbol[i]] = [node_z[i]]
            symbol_dict[symbol[i]] = [symbol[i]]
            color_dict[symbol[i]] = ["#"+symbol2color[symbol[i]]]
    data=arrows + [go.Scatter3d(x=node_x_dict[k], 
                       y=node_y_dict[k], 
                       z=node_z_dict[k], 
                       text=symbol_dict[k],
                       mode='markers',
                       showlegend=True,
                       name=k,
                       marker=dict(
                           color=color_dict[k],
                           size=20,
                           opacity=1.0,
                       )) for k in symbol_dict.keys()] + \
                       bond_scatter + [go.Scatter3d(x=np.array(unit_cell[i]).T[0], 
                                                    y=np.array(unit_cell[i]).T[1], 
                                                    z=np.array(unit_cell[i]).T[2],
                                                    mode="lines",
                                                    line=dict(
                                                        color=unit_cell_color[i],
                                                        width=2,
                                                    ),
                                                    showlegend=False,
                                                    hoverinfo='skip') for i in range(0, len(unit_cell))]
    
    xyzmin = min([min(node_x), min(node_y), min(node_z), np.array(unit_cell).min(), (0-V_diag).min()])
    xyzmax = max([max(node_x), max(node_y), max(node_z), np.array(unit_cell).max(), (0-V_diag).max()])
    DeltaX = xyzmax - xyzmin
    padding_xyz = DeltaX * 0.05
    fig = go.Figure(data=data)
    annotation_list = []
    fig.update_layout(
        scene = dict(
            annotations=annotation_list,
            xaxis = dict(nticks=10, range=[xyzmin-padding_xyz,xyzmax+padding_xyz],
                         backgroundcolor="rgba(80, 70, 70, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                        ),
            yaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 80, 70, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                        ),
            zaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 70, 80, 0.5)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
        ),
        width=1200,
        height=1200,
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=True)
    fig.update_layout(scene_aspectmode='cube', 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        font_color="rgba(150,150,150,1)",
        title_font_color="rgba(150,150,150,1)",
        legend_title_font_color="rgba(150,150,150,1)",
    )
    #fig.show()
    return fig


# mof_cif
def viz_mof_cif_v1(cif_name, fract_disp=np.array([0, 0, 0]), showbackground=False, tol=0.01, bond_dist_sqr_threshold=6):
    table_dict = {}
    # https://github.com/Bowserinator/Periodic-Table-JSON
    with io.open("PeriodicTableJSON.json", "rb") as f:
        table_dict = json.load(f)
    element_df = pd.DataFrame(table_dict["elements"])
    element_df["cpk-hex"] = element_df["cpk-hex"].fillna("0000ff")
    symbol2color = dict(zip(element_df["symbol"], element_df["cpk-hex"]))
    symbol2color["X"] = "0000ff"
    idx2symbol = dict(zip(element_df["number"], element_df["symbol"]))
    idx2symbol[0] = "X"

    _cif_name = cif_name
    #cif_str = None
    with io.open(cif_name, "r", newline="\n") as cif:
        cif_str = cif.read().replace("(", "").replace(")", "")
    cif_list = cif_str.split("loop_")
    for cif_section in cif_list:
        if "_symmetry_equiv_pos_as_xyz" in cif_section:
            if len(cif_section.split("_cell_")[0].split("_symmetry_equiv_pos_as_xyz")[1].strip().split("\n")) > 1:
                cif_name = cif_name.replace(".cif", "_P1.cif")
                CifWriter(mg.Structure.from_str(cif_str, fmt="cif"), refine_struct=False).write_file(cif_name)


    M = lat_param2vec(read_cif_lattice_param(cif_name))
    A, B, C = M
    atom_df = read_cif_xyz(cif_name)
    bond_df = read_cif_bond(cif_name)
    #bond_df = None
    Natoms = len(atom_df)
    atom_df.loc[:, ["_atom_site_fract_x", 
                    "_atom_site_fract_y", 
                    "_atom_site_fract_z"]] = (atom_df.loc[:, ["_atom_site_fract_x", 
                                                              "_atom_site_fract_y", 
                                                              "_atom_site_fract_z"]].values + fract_disp) % 1# - np.array([0.5, 0.5, 0.5])) % 1
    atom_df["x"] = None
    atom_df["y"] = None
    atom_df["z"] = None
    atom_df["_atom_site_label_prefix"] = atom_df["_atom_site_label"].apply(lambda x: re.findall(r"[A-Z]*[a-z]*", x)[0])
    atom_df["_atom_site_label_postfix"] = atom_df["_atom_site_label"].apply(lambda x: re.findall(r"[0-9]+", x)[0])

    if type(bond_df) != type(None):
        bond_df["_geom_bond_atom_site_label_1_prefix"] = bond_df["_geom_bond_atom_site_label_1"].apply(lambda x: re.findall(r"[A-Z]*[a-z]*", x)[0])
        bond_df["_geom_bond_atom_site_label_1_postfix"] = bond_df["_geom_bond_atom_site_label_1"].apply(lambda x: re.findall(r"[0-9]+", x)[0])
        bond_df["_geom_bond_atom_site_label_2_prefix"] = bond_df["_geom_bond_atom_site_label_2"].apply(lambda x: re.findall(r"[A-Z]*[a-z]*", x)[0])
        bond_df["_geom_bond_atom_site_label_2_postfix"] = bond_df["_geom_bond_atom_site_label_2"].apply(lambda x: re.findall(r"[0-9]+", x)[0])
        bond_df_atom_idx_offset1 = np.zeros(len(bond_df), dtype=int)
        bond_df_atom_idx_offset2 = np.zeros(len(bond_df), dtype=int)
        bond_df_list = [bond_df.copy(deep=True)]
        
    error_atom_index = atom_df[atom_df["_atom_site_label_prefix"]!=atom_df["_atom_site_type_symbol"]].index
    atom_df.loc[error_atom_index, "_atom_site_label"] = atom_df.loc[error_atom_index, "_atom_site_type_symbol"].astype(str) + atom_df.loc[error_atom_index, "_atom_site_label_postfix"].astype(str)

    unique_elements = sorted(atom_df["_atom_site_type_symbol"].unique().tolist())
    natoms_per_element = {}
    atom_idx_offset = np.zeros(Natoms, dtype=int)
    for element in unique_elements:
        selected_index = atom_df[atom_df["_atom_site_type_symbol"]==element].index
        natoms_per_element[element] = len(selected_index)
        atom_idx_offset[selected_index] = natoms_per_element[element]
        if type(bond_df) != type(None):
            bond_df_selected_index = bond_df[bond_df["_geom_bond_atom_site_label_1_prefix"]==element].index
            bond_df_atom_idx_offset1[bond_df_selected_index] = natoms_per_element[element]
            bond_df_selected_index = bond_df[bond_df["_geom_bond_atom_site_label_2_prefix"]==element].index
            bond_df_atom_idx_offset2[bond_df_selected_index] = natoms_per_element[element]
        atom_df.loc[selected_index, "_atom_site_label_postfix"] = [x for x in range(1, len(selected_index)+1)]
        atom_df.loc[selected_index, "_atom_site_label"] = element + atom_df.loc[selected_index, "_atom_site_label_postfix"].astype(str)


    atom_df["original_label"] = atom_df["_atom_site_label"].to_list()
    atom_df["periodic_image"] = "[0, 0, 0]"
    atom_df_list = [atom_df.copy(deep=True)]

    displacement_vectors = [[0,0,1], [0,0,-1],  
                            [0,1,0], [0,-1,0], 
                            [1,0,0], [-1,0,0], 
                            [-1,-1,0], [-1,1,0], [1,-1,0], [1,1,0], 
                            [-1,0,-1], [-1,0,1], [1,0,-1], [1,0,1], 
                            [0,-1,-1], [0,-1,1], [0,1,-1], [0,1,1], 
                            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [1,-1,-1], 
                            [-1,1,1], [1,-1,1], [1,1,-1], [1,1,1]]


    for disp_i in range(0, len(displacement_vectors)):
        disp = displacement_vectors[disp_i]
        if type(bond_df) != type(None):
            _bond_df = bond_df.copy(deep=True)
            _bond_df["_geom_bond_atom_site_label_1_postfix"] = ((disp_i + 1) * bond_df_atom_idx_offset1 + \
                                                                _bond_df["_geom_bond_atom_site_label_1_postfix"].astype(int)).astype(str)
            _bond_df["_geom_bond_atom_site_label_1"] = _bond_df["_geom_bond_atom_site_label_1_prefix"] + _bond_df["_geom_bond_atom_site_label_1_postfix"]
            _bond_df["_geom_bond_atom_site_label_2_postfix"] = ((disp_i + 1) * bond_df_atom_idx_offset2 + \
                                                                _bond_df["_geom_bond_atom_site_label_2_postfix"].astype(int)).astype(str)
            _bond_df["_geom_bond_atom_site_label_2"] = _bond_df["_geom_bond_atom_site_label_2_prefix"] + _bond_df["_geom_bond_atom_site_label_2_postfix"]
            bond_df_list.append(_bond_df)

        _atom_df = atom_df.copy(deep=True)
        _atom_df.loc[:, "periodic_image"] = str(disp)
        _atom_df.loc[:, ["_atom_site_fract_x", 
                         "_atom_site_fract_y", 
                         "_atom_site_fract_z"]] = _atom_df.loc[:, ["_atom_site_fract_x", 
                                                                   "_atom_site_fract_y", 
                                                                   "_atom_site_fract_z"]].values + np.array(disp)
        _atom_df["_atom_site_label_postfix"] = ((disp_i + 1) * atom_idx_offset + _atom_df["_atom_site_label_postfix"].astype(int)).astype(str)
        atom_df["original_label"] = _atom_df["_atom_site_label"].to_list()
        _atom_df["_atom_site_label"] = _atom_df["_atom_site_type_symbol"] + _atom_df["_atom_site_label_postfix"]
        
        atom_df_list.append(_atom_df)

    atom_df = pd.concat(atom_df_list, axis=0).reset_index(drop=True)
    if type(bond_df) != type(None):
        bond_df = pd.concat(bond_df_list, axis=0).reset_index(drop=True)

    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["_atom_site_fract_x", 
                                                      "_atom_site_fract_y", 
                                                      "_atom_site_fract_z"]].values @ M
    unit_cell_expansion_rate = 0.09
    critical_points = [unit_cell_expansion_rate*(-A-B-C),
                       unit_cell_expansion_rate*(A-B-C)+A, 
                       unit_cell_expansion_rate*(-A+B-C)+B, 
                       unit_cell_expansion_rate*(-A-B+C)+C, 
                       unit_cell_expansion_rate*(A+B-C)+A+B, 
                       unit_cell_expansion_rate*(-A+B+C)+B+C, 
                       unit_cell_expansion_rate*(A-B+C)+A+C, 
                       unit_cell_expansion_rate*(A+B+C)+A+B+C]
    hull = Delaunay(critical_points)
    tol=0.01

    isInHull = hull.find_simplex(atom_df.loc[:, ["x", "y", "z"]].values)>=-tol
    atom_df = atom_df[isInHull].copy(deep=True)
    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["_atom_site_fract_x", 
                                                      "_atom_site_fract_y", 
                                                      "_atom_site_fract_z"]].values @ M

    bond_scatter = []
    if type(bond_df) != type(None):
        bond_df["atom1"] = None
        bond_df["atom2"] = None
        bond_df["atom1"] = bond_df["_geom_bond_atom_site_label_1"].map(dict(zip(atom_df["_atom_site_label"], atom_df.index)))
        bond_df["atom2"] = bond_df["_geom_bond_atom_site_label_2"].map(dict(zip(atom_df["_atom_site_label"], atom_df.index)))
        bond_df = bond_df.dropna()
        bond_df["atom1"] = bond_df["atom1"].astype(int)
        bond_df["atom2"] = bond_df["atom2"].astype(int)
        bond_df["dist_sqr"] = [np.sum((atom_df.loc[bond_df.at[i, "atom1"].astype(int), ["x", "y", "z"]].values - \
                                       atom_df.loc[bond_df.at[i, "atom2"].astype(int), ["x", "y", "z"]].values) ** 2) for i in bond_df.index]

        for i in bond_df.index:
            if bond_df.at[i, "dist_sqr"] < bond_dist_sqr_threshold:
                bond_scatter.append(
                    go.Scatter3d(
                        x=[atom_df.at[bond_df.at[i, "atom1"], "x"], atom_df.at[bond_df.at[i, "atom2"], "x"]], 
                        y=[atom_df.at[bond_df.at[i, "atom1"], "y"], atom_df.at[bond_df.at[i, "atom2"], "y"]], 
                        z=[atom_df.at[bond_df.at[i, "atom1"], "z"], atom_df.at[bond_df.at[i, "atom2"], "z"]],
                        mode="lines",
                        line=dict(
                            color='rgba(220, 200, 200, 0.5)',
                            width=5,
                        ),
                        showlegend=False,
                        hoverinfo='skip')
                )
            else:
                orig_label1 = atom_df.at[bond_df.at[i, "atom1"], "original_label"]
                affiliated_atoms1 = atom_df[(atom_df["original_label"]==orig_label1)&(atom_df["_atom_site_label"]!=atom_df.at[bond_df.at[i, "atom1"], "_atom_site_label"])].copy(deep=True).reset_index(drop=True)
                affiliated_atoms1["dist_sqr"] = None
                affiliated_atoms1["dist_sqr"] = np.sum((affiliated_atoms1.loc[:, ["x", "y", "z"]].values - atom_df.loc[bond_df.at[i, "atom2"], ["x", "y", "z"]].values) ** 2, axis=1)
                if affiliated_atoms1["dist_sqr"].min() < bond_dist_sqr_threshold:
                    bonding_atom_idx = affiliated_atoms1[affiliated_atoms1["dist_sqr"]==affiliated_atoms1["dist_sqr"].min()].index[0]
                    bond_scatter.append(
                        go.Scatter3d(
                            x=[atom_df.at[bond_df.at[i, "atom2"], "x"], affiliated_atoms1.at[bonding_atom_idx, "x"]], 
                            y=[atom_df.at[bond_df.at[i, "atom2"], "y"], affiliated_atoms1.at[bonding_atom_idx, "y"]], 
                            z=[atom_df.at[bond_df.at[i, "atom2"], "z"], affiliated_atoms1.at[bonding_atom_idx, "z"]],
                            mode="lines",
                            line=dict(
                                color='rgba(100, 100, 255, 0.5)',
                                width=5,
                            ),
                            showlegend=False,
                            hoverinfo='skip')
                    )
                
                orig_label2 = atom_df.at[bond_df.at[i, "atom2"], "original_label"]
                affiliated_atoms2 = atom_df[(atom_df["original_label"]==orig_label2)&(atom_df["_atom_site_label"]!=atom_df.at[bond_df.at[i, "atom2"], "_atom_site_label"])].copy(deep=True).reset_index(drop=True)
                affiliated_atoms2["dist_sqr"] = None
                affiliated_atoms2["dist_sqr"] = np.sum((affiliated_atoms2.loc[:, ["x", "y", "z"]].values - atom_df.loc[bond_df.at[i, "atom1"], ["x", "y", "z"]].values) ** 2, axis=1)
                if affiliated_atoms2["dist_sqr"].min() < bond_dist_sqr_threshold:
                    bonding_atom_idx = affiliated_atoms2[affiliated_atoms2["dist_sqr"]==affiliated_atoms2["dist_sqr"].min()].index[0]
                    bond_scatter.append(
                        go.Scatter3d(
                            x=[atom_df.at[bond_df.at[i, "atom1"], "x"], affiliated_atoms2.at[bonding_atom_idx, "x"]], 
                            y=[atom_df.at[bond_df.at[i, "atom1"], "y"], affiliated_atoms2.at[bonding_atom_idx, "y"]], 
                            z=[atom_df.at[bond_df.at[i, "atom1"], "z"], affiliated_atoms2.at[bonding_atom_idx, "z"]],
                            mode="lines",
                            line=dict(
                                color='rgba(100, 100, 255, 0.5)',
                                width=5,
                            ),
                            showlegend=False,
                            hoverinfo='skip')
                    )


    unit_cell = [[np.array([0, 0, 0]), A], [np.array([0, 0, 0]), B], [np.array([0, 0, 0]), C], 
                 [A, A+B], [A, A+C], [B, B+A], 
                 [B, B+C], [C, C+A], [C, C+B], 
                 [A+B, A+B+C], [B+C, A+B+C], [A+C, A+B+C]]
    unit_cell_color = ["rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)"]

    # XYZ arrows
    axis_length = 0.5
    V_diag = 0.25 * (A + B + C)
    x_start = -V_diag[0]
    y_start = -V_diag[1]
    z_start = -V_diag[2]
    x_end1 = (axis_length * A - V_diag)[0]
    y_end1 = (axis_length * A - V_diag)[1]
    z_end1 = (axis_length * A - V_diag)[2]
    x_end2 = (axis_length * B - V_diag)[0]
    y_end2 = (axis_length * B - V_diag)[1]
    z_end2 = (axis_length * B - V_diag)[2]
    x_end3 = (axis_length * C - V_diag)[0]
    y_end3 = (axis_length * C - V_diag)[1]
    z_end3 = (axis_length * C - V_diag)[2]
    arrows = [
        go.Scatter3d(x=[x_start, x_end1], 
                     y=[y_start, y_end1], 
                     z=[z_start, z_end1],
                     mode="lines+text",
                     text=["", "A"],
                     line=dict(
                         color='rgba(255, 0, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(255, 0, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end2], 
                     y=[y_start, y_end2], 
                     z=[z_start, z_end2],
                     mode="lines+text",
                     text=["", "B"],
                     line=dict(
                         color='rgba(0, 255, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 255, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end3], 
                     y=[y_start, y_end3], 
                     z=[z_start, z_end3],
                     mode="lines+text",
                     text=["", "C"],
                     line=dict(
                         color='rgba(0, 0, 255, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 0, 255, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
    ]
    atom_df["color"] = "#" + atom_df["_atom_site_type_symbol"].map(symbol2color)
    data=[go.Scatter3d(x=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "x"], 
                       y=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "y"], 
                       z=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "z"], 
                       text=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "_atom_site_type_symbol"] + "<br>" + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "_atom_site_label"] + "<br>" + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "periodic_image"] + "<br>was " + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "original_label"],
                       mode='markers',
                       showlegend=True,
                       name=k,
                       marker=dict(
                           color=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "color"],
                           size=10,
                           opacity=1.0,
                       )) for k in unique_elements] + \
                       bond_scatter + [go.Scatter3d(x=np.array(unit_cell[i]).T[0], 
                                                    y=np.array(unit_cell[i]).T[1], 
                                                    z=np.array(unit_cell[i]).T[2],
                                                    mode="lines",
                                                    line=dict(
                                                        color=unit_cell_color[i],
                                                        width=2,
                                                    ),
                                                    showlegend=False,
                                                    hoverinfo='skip') for i in range(0, len(unit_cell))] + arrows

    xyzmin = min([atom_df["x"].min(), atom_df["y"].min(), atom_df["z"].min(), np.array(unit_cell).min(), (0-V_diag).min()])
    xyzmax = max([atom_df["x"].max(), atom_df["y"].max(), atom_df["z"].max(), np.array(unit_cell).max(), (0-V_diag).max()])
    DeltaX = xyzmax - xyzmin
    padding_xyz = DeltaX * 0.05
    fig = go.Figure(data=data)
    annotation_list = []
    fig.update_layout(
        scene = dict(
            annotations=annotation_list,
            xaxis = dict(nticks=10, range=[xyzmin-padding_xyz,xyzmax+padding_xyz],
                         backgroundcolor="rgba(80, 70, 70, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
            yaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 80, 70, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
            zaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 70, 80, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
        ),
        width=1200,
        height=1200,
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=True)
    fig.update_layout(scene_aspectmode='cube', 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        font_color="rgba(150,150,150,1)",
        title_font_color="rgba(150,150,150,1)",
        legend_title_font_color="rgba(150,150,150,1)",
    )
    #fig.show()
    if _cif_name != cif_name:
        os.remove(cif_name)
    return fig
    #return


def viz_mof_cif_v2(cif_name, fract_disp=np.array([0, 0, 0]), showbackground=False, tol=0.01, bond_dist_sqr_threshold=6):
    table_dict = {}
    # https://github.com/Bowserinator/Periodic-Table-JSON
    with io.open("PeriodicTableJSON.json", "rb") as f:
        table_dict = json.load(f)
    element_df = pd.DataFrame(table_dict["elements"])
    element_df["cpk-hex"] = element_df["cpk-hex"].fillna("0000ff")
    symbol2color = dict(zip(element_df["symbol"], element_df["cpk-hex"]))
    symbol2color["X"] = "0000ff"
    idx2symbol = dict(zip(element_df["number"], element_df["symbol"]))
    idx2symbol[0] = "X"

    _cif_name = cif_name
    #cif_str = None
    with io.open(cif_name, "r", newline="\n") as cif:
        cif_str = cif.read().replace("(", "").replace(")", "")
    cif_list = cif_str.split("loop_")
    for cif_section in cif_list:
        if "_symmetry_equiv_pos_as_xyz" in cif_section:
            if len(cif_section.split("_cell_")[0].split("_symmetry_equiv_pos_as_xyz")[1].strip().split("\n")) > 1:
                cif_name = cif_name.replace(".cif", "_P1.cif")
                CifWriter(mg.Structure.from_str(cif_str, fmt="cif"), refine_struct=False).write_file(cif_name)


    M = lat_param2vec(read_cif_lattice_param(cif_name))
    A, B, C = M
    atom_df = read_cif_xyz(cif_name)
    bond_df = read_cif_bond(cif_name)

    #bond_df = None
    Natoms = len(atom_df)
    atom_df.loc[:, ["_atom_site_fract_x", 
                    "_atom_site_fract_y", 
                    "_atom_site_fract_z"]] = (atom_df.loc[:, ["_atom_site_fract_x", 
                                                              "_atom_site_fract_y", 
                                                              "_atom_site_fract_z"]].values + fract_disp) % 1# - np.array([0.5, 0.5, 0.5])) % 1
    atom_df["x"] = None
    atom_df["y"] = None
    atom_df["z"] = None
    atom_df["_atom_site_label_prefix"] = atom_df["_atom_site_label"].apply(lambda x: re.findall(r"[A-Z]*[a-z]*", x)[0])
    atom_df["_atom_site_label_postfix"] = atom_df["_atom_site_label"].apply(lambda x: re.findall(r"[0-9]+", x)[0])


    error_atom_index = atom_df[atom_df["_atom_site_label_prefix"]!=atom_df["_atom_site_type_symbol"]].index
    atom_df.loc[error_atom_index, "_atom_site_label"] = atom_df.loc[error_atom_index, "_atom_site_type_symbol"].astype(str) + \
                                                        atom_df.loc[error_atom_index, "_atom_site_label_postfix"].astype(str)

    unique_elements = sorted(atom_df["_atom_site_type_symbol"].unique().tolist())
    natoms_per_element = {}
    atom_idx_offset = np.zeros(Natoms, dtype=int)
    atom_df.loc[:, "_atom_site_label_save"] = atom_df.loc[:, "_atom_site_label"]
    for element in unique_elements:
        selected_index = atom_df[atom_df["_atom_site_type_symbol"]==element].index
        natoms_per_element[element] = len(selected_index)
        atom_idx_offset[selected_index] = natoms_per_element[element]

        atom_df.loc[selected_index, "_atom_site_label_postfix"] = [x for x in range(1, len(selected_index)+1)]
        atom_df.loc[selected_index, "_atom_site_label"] = element + atom_df.loc[selected_index, "_atom_site_label_postfix"].astype(str)
    # label_map_dict = dict(zip(atom_df["_atom_site_label_save"].to_list() + atom_df["_atom_site_label"].to_list(), 
    #                           atom_df["_atom_site_label"].to_list() + atom_df["_atom_site_label"].to_list()))
    label_map_dict = dict(zip(atom_df["_atom_site_label_save"].to_list(), 
                          atom_df["_atom_site_label"].to_list()))
    for old_label_idx in range(0, len(atom_df["_atom_site_label"])):
        if atom_df.at[old_label_idx, "_atom_site_label"] not in label_map_dict:
            label_map_dict[atom_df.at[old_label_idx, "_atom_site_label"]] = atom_df.at[old_label_idx, "_atom_site_label"]
    if type(bond_df) != type(None):
        bond_df["_geom_bond_atom_site_label_1_save"] = bond_df["_geom_bond_atom_site_label_1"]
        bond_df["_geom_bond_atom_site_label_1"] = bond_df["_geom_bond_atom_site_label_1"].map(label_map_dict)
        bond_df["_geom_bond_atom_site_label_2_save"] = bond_df["_geom_bond_atom_site_label_2"]
        bond_df["_geom_bond_atom_site_label_2"] = bond_df["_geom_bond_atom_site_label_2"].map(label_map_dict)
        bond_df["_geom_bond_atom_site_label_1_prefix"] = bond_df["_geom_bond_atom_site_label_1"].apply(lambda x: re.findall(r"[A-Z]*[a-z]*", x)[0])
        bond_df["_geom_bond_atom_site_label_1_postfix"] = bond_df["_geom_bond_atom_site_label_1"].apply(lambda x: re.findall(r"[0-9]+", x)[0])
        bond_df["_geom_bond_atom_site_label_2_prefix"] = bond_df["_geom_bond_atom_site_label_2"].apply(lambda x: re.findall(r"[A-Z]*[a-z]*", x)[0])
        bond_df["_geom_bond_atom_site_label_2_postfix"] = bond_df["_geom_bond_atom_site_label_2"].apply(lambda x: re.findall(r"[0-9]+", x)[0])
        bond_df_atom_idx_offset1 = np.zeros(len(bond_df), dtype=int)
        bond_df_atom_idx_offset2 = np.zeros(len(bond_df), dtype=int)
        for element in unique_elements:
            bond_df_selected_index = bond_df[bond_df["_geom_bond_atom_site_label_1_prefix"]==element].index
            bond_df_atom_idx_offset1[bond_df_selected_index] = natoms_per_element[element]
            bond_df_selected_index = bond_df[bond_df["_geom_bond_atom_site_label_2_prefix"]==element].index
            bond_df_atom_idx_offset2[bond_df_selected_index] = natoms_per_element[element]
        bond_df_list = [bond_df.copy(deep=True)]

    atom_df["original_label"] = atom_df["_atom_site_label"].to_list()
    atom_df["periodic_image"] = "[0, 0, 0]"
    atom_df_list = [atom_df.copy(deep=True)]

    displacement_vectors = [[0,0,1], [0,0,-1],  
                            [0,1,0], [0,-1,0], 
                            [1,0,0], [-1,0,0], 
                            [-1,-1,0], [-1,1,0], [1,-1,0], [1,1,0], 
                            [-1,0,-1], [-1,0,1], [1,0,-1], [1,0,1], 
                            [0,-1,-1], [0,-1,1], [0,1,-1], [0,1,1], 
                            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [1,-1,-1], 
                            [-1,1,1], [1,-1,1], [1,1,-1], [1,1,1]]

    for disp_i in range(0, len(displacement_vectors)):
        disp = displacement_vectors[disp_i]
        if type(bond_df) != type(None):
            _bond_df = bond_df.copy(deep=True)
            _bond_df["_geom_bond_atom_site_label_1_postfix"] = ((disp_i + 1) * bond_df_atom_idx_offset1 + \
                                                                _bond_df["_geom_bond_atom_site_label_1_postfix"].astype(int)).astype(str)
            _bond_df["_geom_bond_atom_site_label_1"] = _bond_df["_geom_bond_atom_site_label_1_prefix"] + _bond_df["_geom_bond_atom_site_label_1_postfix"]
            _bond_df["_geom_bond_atom_site_label_2_postfix"] = ((disp_i + 1) * bond_df_atom_idx_offset2 + \
                                                                _bond_df["_geom_bond_atom_site_label_2_postfix"].astype(int)).astype(str)
            _bond_df["_geom_bond_atom_site_label_2"] = _bond_df["_geom_bond_atom_site_label_2_prefix"] + _bond_df["_geom_bond_atom_site_label_2_postfix"]
            bond_df_list.append(_bond_df)

        _atom_df = atom_df.copy(deep=True)
        _atom_df.loc[:, "periodic_image"] = str(disp)
        _atom_df.loc[:, ["_atom_site_fract_x", 
                         "_atom_site_fract_y", 
                         "_atom_site_fract_z"]] = _atom_df.loc[:, ["_atom_site_fract_x", 
                                                                   "_atom_site_fract_y", 
                                                                   "_atom_site_fract_z"]].values + np.array(disp)
        _atom_df["_atom_site_label_postfix"] = ((disp_i + 1) * atom_idx_offset + _atom_df["_atom_site_label_postfix"].astype(int)).astype(str)
        _atom_df["original_label"] = _atom_df["_atom_site_label"].to_list()
        _atom_df["_atom_site_label"] = _atom_df["_atom_site_type_symbol"] + _atom_df["_atom_site_label_postfix"]

        atom_df_list.append(_atom_df)

    atom_df = pd.concat(atom_df_list, axis=0).reset_index(drop=True)
    if type(bond_df) != type(None):
        bond_df = pd.concat(bond_df_list, axis=0).reset_index(drop=True)

    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["_atom_site_fract_x", 
                                                      "_atom_site_fract_y", 
                                                      "_atom_site_fract_z"]].values @ M
    unit_cell_expansion_offset = 2 # Angstrom
    critical_points = [np.array([-1, -1, -1])*unit_cell_expansion_offset,
                       np.array([1, -1, -1])*unit_cell_expansion_offset + A, 
                       np.array([-1, 1, -1])*unit_cell_expansion_offset + B, 
                       np.array([-1, -1, 1])*unit_cell_expansion_offset + C,
                       np.array([1, 1, -1])*unit_cell_expansion_offset + A + B, 
                       np.array([-1, 1, 1])*unit_cell_expansion_offset + B + C, 
                       np.array([1, -1, 1])*unit_cell_expansion_offset + A + C, 
                       np.array([1, 1, 1])*unit_cell_expansion_offset + A + B + C]
    hull = Delaunay(critical_points)
    #tol=0.01

    isInHull = hull.find_simplex(atom_df.loc[:, ["x", "y", "z"]].values)>=-tol
    atom_df = atom_df[isInHull].copy(deep=True)
    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["_atom_site_fract_x", 
                                                      "_atom_site_fract_y", 
                                                      "_atom_site_fract_z"]].values @ M

    bond_scatter = []
    
    if type(bond_df) != type(None):
        bond_df["atom1"] = None
        bond_df["atom2"] = None
        bond_df["atom1"] = bond_df["_geom_bond_atom_site_label_1"].map(dict(zip(atom_df["_atom_site_label"], atom_df.index)))
        bond_df["atom2"] = bond_df["_geom_bond_atom_site_label_2"].map(dict(zip(atom_df["_atom_site_label"], atom_df.index)))
        bond_df = bond_df.dropna()
        bond_df["atom1"] = bond_df["atom1"].astype(int)
        bond_df["atom2"] = bond_df["atom2"].astype(int)
        bond_df["dist_sqr"] = [np.sum((atom_df.loc[bond_df.at[i, "atom1"].astype(int), ["x", "y", "z"]].values - \
                                       atom_df.loc[bond_df.at[i, "atom2"].astype(int), ["x", "y", "z"]].values) ** 2) for i in bond_df.index]

        for i in bond_df.index:
            orig_label1 = atom_df.at[bond_df.at[i, "atom1"], "original_label"]
            affiliated_atoms1 = atom_df[atom_df["original_label"]==orig_label1].copy(deep=True).reset_index(drop=True)
            orig_label2 = atom_df.at[bond_df.at[i, "atom2"], "original_label"]
            affiliated_atoms2 = atom_df[atom_df["original_label"]==orig_label2].copy(deep=True).reset_index(drop=True)
            possible_bonds = pd.concat([pd.DataFrame(np.repeat(affiliated_atoms1.values, len(affiliated_atoms2), axis=0), columns=affiliated_atoms1.columns + "_1"), 
                                        pd.DataFrame(np.tile(affiliated_atoms2.values, (len(affiliated_atoms1), 1)), columns=affiliated_atoms2.columns + "_2")], axis=1)
            possible_bonds["dist_sqr"] = np.sum((possible_bonds.loc[:, ["x_1", "y_1", "z_1"]].values - possible_bonds.loc[:, ["x_2", "y_2", "z_2"]].values) ** 2, axis=1)
            possible_bonds = possible_bonds[possible_bonds["dist_sqr"]<bond_dist_sqr_threshold]
            for bond_idx in possible_bonds.index:
                bond_scatter.append(
                    go.Scatter3d(
                        x=[possible_bonds.at[bond_idx, "x_1"], possible_bonds.at[bond_idx, "x_2"]], 
                        y=[possible_bonds.at[bond_idx, "y_1"], possible_bonds.at[bond_idx, "y_2"]], 
                        z=[possible_bonds.at[bond_idx, "z_1"], possible_bonds.at[bond_idx, "z_2"]],
                        mode="lines",
                        line=dict(
                            color='rgba(100, 100, 255, 0.5)',
                            width=5,
                        ),
                        showlegend=False,
                        hoverinfo='skip')
                )

    unit_cell = [[np.array([0, 0, 0]), A], [np.array([0, 0, 0]), B], [np.array([0, 0, 0]), C], 
                 [A, A+B], [A, A+C], [B, B+A], 
                 [B, B+C], [C, C+A], [C, C+B], 
                 [A+B, A+B+C], [B+C, A+B+C], [A+C, A+B+C]]
    unit_cell_color = ["rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)"]

    # XYZ arrows
    axis_length = 0.5
    V_diag = 0.25 * (A + B + C)
    x_start = -V_diag[0]
    y_start = -V_diag[1]
    z_start = -V_diag[2]
    x_end1 = (axis_length * A - V_diag)[0]
    y_end1 = (axis_length * A - V_diag)[1]
    z_end1 = (axis_length * A - V_diag)[2]
    x_end2 = (axis_length * B - V_diag)[0]
    y_end2 = (axis_length * B - V_diag)[1]
    z_end2 = (axis_length * B - V_diag)[2]
    x_end3 = (axis_length * C - V_diag)[0]
    y_end3 = (axis_length * C - V_diag)[1]
    z_end3 = (axis_length * C - V_diag)[2]
    arrows = [
        go.Scatter3d(x=[x_start, x_end1], 
                     y=[y_start, y_end1], 
                     z=[z_start, z_end1],
                     mode="lines+text",
                     text=["", "A"],
                     line=dict(
                         color='rgba(255, 0, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(255, 0, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end2], 
                     y=[y_start, y_end2], 
                     z=[z_start, z_end2],
                     mode="lines+text",
                     text=["", "B"],
                     line=dict(
                         color='rgba(0, 255, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 255, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end3], 
                     y=[y_start, y_end3], 
                     z=[z_start, z_end3],
                     mode="lines+text",
                     text=["", "C"],
                     line=dict(
                         color='rgba(0, 0, 255, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 0, 255, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
    ]
    atom_df["color"] = "#" + atom_df["_atom_site_type_symbol"].map(symbol2color)
    data=[go.Scatter3d(x=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "x"], 
                       y=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "y"], 
                       z=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "z"], 
                       text=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "_atom_site_type_symbol"] + "<br>" + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "_atom_site_label"] + "<br>" + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "periodic_image"] + "<br>was " + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "original_label"],
                       mode='markers',
                       showlegend=True,
                       name=k,
                       marker=dict(
                           color=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "color"],
                           size=10,
                           opacity=1.0,
                       )) for k in unique_elements] + \
                       bond_scatter + [go.Scatter3d(x=np.array(unit_cell[i]).T[0], 
                                                    y=np.array(unit_cell[i]).T[1], 
                                                    z=np.array(unit_cell[i]).T[2],
                                                    mode="lines",
                                                    line=dict(
                                                        color=unit_cell_color[i],
                                                        width=2,
                                                    ),
                                                    showlegend=False,
                                                    hoverinfo='skip') for i in range(0, len(unit_cell))] + arrows

    xyzmin = min([atom_df["x"].min(), atom_df["y"].min(), atom_df["z"].min(), np.array(unit_cell).min(), (0-V_diag).min()])
    xyzmax = max([atom_df["x"].max(), atom_df["y"].max(), atom_df["z"].max(), np.array(unit_cell).max(), (0-V_diag).max()])
    DeltaX = xyzmax - xyzmin
    padding_xyz = DeltaX * 0.05
    fig = go.Figure(data=data)
    annotation_list = []
    fig.update_layout(
        scene = dict(
            annotations=annotation_list,
            xaxis = dict(nticks=10, range=[xyzmin-padding_xyz,xyzmax+padding_xyz],
                         backgroundcolor="rgba(80, 70, 70, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
            yaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 80, 70, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
            zaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 70, 80, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
        ),
        width=1200,
        height=1200,
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=True)
    fig.update_layout(scene_aspectmode='cube', 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        font_color="rgba(150,150,150,1)",
        title_font_color="rgba(150,150,150,1)",
        legend_title_font_color="rgba(150,150,150,1)",
    )

    if _cif_name != cif_name:
        os.remove(cif_name)

    return fig


def viz_mof_cif(cif_name, 
                fract_disp=np.array([0, 0, 0]), 
                color_override=None, # list-like plotly color strings with the length of the number of atoms
                proximity_bonding=True, 
                paper_bgcolor='rgba(20, 20, 50, 0.8)',
                showbackground=False, 
                tol=0.01, 
                bond_dist_sqr_threshold = 10, 
                unit_cell_expansion_offset = 3):
    element_df = pd.read_csv("atom_visualization.csv")
    symbol2is_metal = dict(zip(element_df["symbol"], element_df["is_metal"]))
    symbol2is_metal["X"] = False
    symbol2actual_radius = dict(zip(element_df["symbol"], element_df["atom_radii"]))
    symbol2actual_radius["X"] = 1.2
    symbol2plot_radius = dict(zip(element_df["symbol"], element_df["plotly_radii"]))
    symbol2plot_radius["X"] = 12
    symbol2color = dict(zip(element_df["symbol"], element_df["cpk-hex"]))
    symbol2color["X"] = "0000ff"
    idx2symbol = dict(zip(element_df["number"], element_df["symbol"]))
    idx2symbol[0] = "X"

    _cif_name = cif_name
    cif_str = None
    with io.open(cif_name, "r", newline="\n") as cif:
        cif_str = cif.read().replace("(", "").replace(")", "")
    # cif_list = cif_str.split("loop_")
    # for cif_section in cif_list:
    #     if "_symmetry_equiv_pos_as_xyz" in cif_section:
    #         if len(cif_section.split("_cell_")[0].split("_symmetry_equiv_pos_as_xyz")[1].strip().split("\n")) > 1:
    cif_name = cif_name.replace(".cif", "_P1.cif")
    CifWriter(mg.Structure.from_str(cif_str, fmt="cif"), refine_struct=False).write_file(cif_name)

    M = lat_param2vec(read_cif_lattice_param(cif_name))
    A, B, C = M
    atom_df = read_cif_xyz(cif_name)
    bond_df = read_cif_bond(cif_name)

    Natoms = len(atom_df)
    atom_df.loc[:, ["_atom_site_fract_x", 
                    "_atom_site_fract_y", 
                    "_atom_site_fract_z"]] = (atom_df.loc[:, ["_atom_site_fract_x", 
                                                              "_atom_site_fract_y", 
                                                              "_atom_site_fract_z"]].values + fract_disp) % 1
    atom_df["x"] = None
    atom_df["y"] = None
    atom_df["z"] = None
    atom_df["original_label"] = atom_df["_atom_site_label"].to_list()

    unique_elements = sorted(atom_df["_atom_site_type_symbol"].unique().tolist())
    if type(color_override) != type(None):
        if len(color_override) == len(atom_df):
            atom_df["color"] = color_override
        else:
            raise Exception("""color_override length does not match the number of atoms in cif file! 
Exiting!
""")
    else:
        atom_df["color"] = "#" + atom_df["_atom_site_type_symbol"].map(symbol2color)
    atom_df["periodic_image"] = "[0, 0, 0]"
    atom_df_list = [atom_df.copy(deep=True)]
    if type(bond_df) != type(None) and proximity_bonding == False:
        bond_df_list = [bond_df.copy(deep=True)]

    displacement_vectors = [[0,0,1], [0,0,-1],  
                            [0,1,0], [0,-1,0], 
                            [1,0,0], [-1,0,0], 
                            [-1,-1,0], [-1,1,0], [1,-1,0], [1,1,0], 
                            [-1,0,-1], [-1,0,1], [1,0,-1], [1,0,1], 
                            [0,-1,-1], [0,-1,1], [0,1,-1], [0,1,1], 
                            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [1,-1,-1], 
                            [-1,1,1], [1,-1,1], [1,1,-1], [1,1,1]]

    for disp_i in range(0, len(displacement_vectors)):
        disp = displacement_vectors[disp_i]
        _atom_df = atom_df.copy(deep=True)
        _atom_df.loc[:, "periodic_image"] = str(disp)
        _atom_df.loc[:, ["_atom_site_fract_x", 
                         "_atom_site_fract_y", 
                         "_atom_site_fract_z"]] = _atom_df.loc[:, ["_atom_site_fract_x", 
                                                                   "_atom_site_fract_y", 
                                                                   "_atom_site_fract_z"]].values + np.array(disp)
        atom_df["_atom_site_label"] = _atom_df["original_label"] + " " + _atom_df["periodic_image"]
        atom_df_list.append(_atom_df)

    atom_df = pd.concat(atom_df_list, axis=0).reset_index(drop=True)
    if type(bond_df) != type(None) and proximity_bonding == False:
        bond_df = pd.concat(bond_df_list, axis=0).reset_index(drop=True)

    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["_atom_site_fract_x", 
                                                      "_atom_site_fract_y", 
                                                      "_atom_site_fract_z"]].values @ M

    critical_points = [np.array([-1, -1, -1])*unit_cell_expansion_offset,
                       np.array([1, -1, -1])*unit_cell_expansion_offset + A, 
                       np.array([-1, 1, -1])*unit_cell_expansion_offset + B, 
                       np.array([-1, -1, 1])*unit_cell_expansion_offset + C,
                       np.array([1, 1, -1])*unit_cell_expansion_offset + A + B, 
                       np.array([-1, 1, 1])*unit_cell_expansion_offset + B + C, 
                       np.array([1, -1, 1])*unit_cell_expansion_offset + A + C, 
                       np.array([1, 1, 1])*unit_cell_expansion_offset + A + B + C]
    hull = Delaunay(critical_points)

    isInHull = hull.find_simplex(atom_df.loc[:, ["x", "y", "z"]].values)>=-tol
    atom_df = atom_df[isInHull].copy(deep=True).reset_index(drop=True)
    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["_atom_site_fract_x", 
                                                      "_atom_site_fract_y", 
                                                      "_atom_site_fract_z"]].values @ M
    #atom_df["idx"] = atom_df.index
    if proximity_bonding:
        dist_sqr = np.sum((np.repeat(atom_df.loc[:, ["x", "y", "z"]].values, len(atom_df), axis=0) - np.tile(atom_df.loc[:, ["x", "y", "z"]].values, (len(atom_df), 1)))**2, axis=1)
        #print("distance sqaured array size:", dist_sqr.size * dist_sqr.itemsize / 1024 / 1024, "MB")
        idx1 = np.tile(np.array([x for x in range(0, len(atom_df))]), (len(atom_df), 1)).flatten()
        idx2 = np.repeat(np.array([x for x in range(0, len(atom_df))]), len(atom_df), axis=0)
        bond_df = pd.DataFrame(np.array([idx1, idx2, dist_sqr]).T, columns = ["atom1_idx", "atom2_idx", "dist_sqr"])
        bond_df = bond_df[(bond_df["dist_sqr"]<50) & (bond_df["dist_sqr"]>0.1)].reset_index(drop=True)
        bond_df = bond_df.astype({"atom1_idx": int, "atom2_idx": int, "dist_sqr": float})
        atom_idx_min = np.array(list(zip(bond_df["atom1_idx"], bond_df["atom2_idx"]))).min(axis=1)
        atom_idx_max = np.array(list(zip(bond_df["atom1_idx"], bond_df["atom2_idx"]))).max(axis=1)
        bond_df["atom_idx_pair"] = pd.Series(atom_idx_min.astype(str)) + "-" + pd.Series(atom_idx_max.astype(str))
        bond_df = bond_df.drop_duplicates(subset=["atom_idx_pair"])
        bond_df["atom1_element"] = bond_df["atom1_idx"].map(dict(zip(atom_df.index, atom_df["_atom_site_type_symbol"])), na_action="ignore")
        bond_df["atom2_element"] = bond_df["atom2_idx"].map(dict(zip(atom_df.index, atom_df["_atom_site_type_symbol"])), na_action="ignore")
        bond_df["dist_sqr_threshold"] = (1.1 * (bond_df["atom1_element"].map(symbol2actual_radius) + bond_df["atom2_element"].map(symbol2actual_radius)))**2
        carbon_oxygen_bonds_idx = bond_df[((bond_df["atom1_element"]=="C")&(bond_df["atom2_element"]=="O")) | \
                                         ((bond_df["atom2_element"]=="C")&(bond_df["atom1_element"]=="O"))].index
        bond_df.loc[carbon_oxygen_bonds_idx, "dist_sqr_threshold"] = 1.9 ** 2
        atom1_is_metal = pd.Series(bond_df["atom1_element"].map(symbol2is_metal))
        atom2_is_metal = pd.Series(bond_df["atom2_element"].map(symbol2is_metal))
        carbon_metal_bonds_idx = bond_df[(atom1_is_metal&(bond_df["atom2_element"]=="C")) | \
                                         (atom2_is_metal&(bond_df["atom1_element"]=="C"))].index
        bond_df.loc[carbon_metal_bonds_idx, "dist_sqr_threshold"] = 1.5
        hydrogen_metal_bonds_idx = bond_df[(atom1_is_metal&(bond_df["atom2_element"]=="H")) | \
                                         (atom2_is_metal&(bond_df["atom1_element"]=="H"))].index
        bond_df.loc[hydrogen_metal_bonds_idx, "dist_sqr_threshold"] = 1.
        metal_metal_bonds_idx = bond_df[atom1_is_metal&atom2_is_metal].index
        bond_df.loc[metal_metal_bonds_idx, "dist_sqr_threshold"] = 1.

        #print("bond dataframe size:", bond_df.memory_usage(deep=True).sum() / 1024 / 1024, "MB")
        bond_df = bond_df[(bond_df["dist_sqr"]<bond_df["dist_sqr_threshold"]*1.2) & (bond_df["dist_sqr"]>0.1)].reset_index(drop=True)
        bond_df["x1"] = bond_df["atom1_idx"].map(dict(zip(atom_df.index, atom_df["x"])), na_action="ignore")
        bond_df["y1"] = bond_df["atom1_idx"].map(dict(zip(atom_df.index, atom_df["y"])), na_action="ignore")
        bond_df["z1"] = bond_df["atom1_idx"].map(dict(zip(atom_df.index, atom_df["z"])), na_action="ignore")
        bond_df["x2"] = bond_df["atom2_idx"].map(dict(zip(atom_df.index, atom_df["x"])), na_action="ignore")
        bond_df["y2"] = bond_df["atom2_idx"].map(dict(zip(atom_df.index, atom_df["y"])), na_action="ignore")
        bond_df["z2"] = bond_df["atom2_idx"].map(dict(zip(atom_df.index, atom_df["z"])), na_action="ignore")
        #print("bond dataframe size after trimming:", bond_df.memory_usage(deep=True).sum() / 1024 / 1024, "MB")
        connected_atoms_idx = list(set(list(set(bond_df["atom1_idx"])) + list(set(bond_df["atom2_idx"]))))
        bond_scatter = [go.Scatter3d(
            x=[bond_df.at[bond_idx, "x1"], bond_df.at[bond_idx, "x2"]], 
            y=[bond_df.at[bond_idx, "y1"], bond_df.at[bond_idx, "y2"]], 
            z=[bond_df.at[bond_idx, "z1"], bond_df.at[bond_idx, "z2"]],
            mode="lines",
            line=dict(
                color='rgba(100, 100, 255, 0.5)',
                width=5,
            ),
            showlegend=False,
            hoverinfo='skip') for bond_idx in bond_df.index]
    else:
        bond_scatter = []
        if type(bond_df) != type(None):
            bond_df["atom1"] = None
            bond_df["atom2"] = None
            bond_df["atom1"] = bond_df["_geom_bond_atom_site_label_1"].map(dict(zip(atom_df["_atom_site_label"], atom_df.index)))
            bond_df["atom2"] = bond_df["_geom_bond_atom_site_label_2"].map(dict(zip(atom_df["_atom_site_label"], atom_df.index)))
            bond_df = bond_df.dropna()
            bond_df["atom1"] = bond_df["atom1"].astype(int)
            bond_df["atom2"] = bond_df["atom2"].astype(int)
            bond_df["dist_sqr"] = [np.sum((atom_df.loc[bond_df.at[i, "atom1"].astype(int), ["x", "y", "z"]].values - \
                                           atom_df.loc[bond_df.at[i, "atom2"].astype(int), ["x", "y", "z"]].values) ** 2) for i in bond_df.index]

            connected_atoms_idx = []
            for i in bond_df.index:
                orig_label1 = atom_df.at[bond_df.at[i, "atom1"], "original_label"]
                affiliated_atoms1 = atom_df[atom_df["original_label"]==orig_label1]#.copy(deep=True).reset_index(drop=True)
                orig_label2 = atom_df.at[bond_df.at[i, "atom2"], "original_label"]
                affiliated_atoms2 = atom_df[atom_df["original_label"]==orig_label2]#.copy(deep=True).reset_index(drop=True)
                possible_bonds = pd.concat([pd.DataFrame(np.repeat(affiliated_atoms1.values, len(affiliated_atoms2), axis=0), columns=affiliated_atoms1.columns + "_1"), 
                                            pd.DataFrame(np.tile(affiliated_atoms2.values, (len(affiliated_atoms1), 1)), columns=affiliated_atoms2.columns + "_2")], axis=1)
                possible_bonds["idx1"] = np.repeat(np.array(affiliated_atoms1.index).astype(int), len(affiliated_atoms2), axis=0).flatten()
                possible_bonds["idx2"] = np.tile(np.array(affiliated_atoms2.index).astype(int), (len(affiliated_atoms1), 1)).flatten()
                possible_bonds["dist_sqr"] = np.sum((possible_bonds.loc[:, ["x_1", "y_1", "z_1"]].values - possible_bonds.loc[:, ["x_2", "y_2", "z_2"]].values) ** 2, axis=1)
                possible_bonds = possible_bonds[possible_bonds["dist_sqr"]<bond_dist_sqr_threshold]

                for bond_idx in possible_bonds.index:
                    if possible_bonds.at[bond_idx, "dist_sqr"] < bond_dist_sqr_threshold:
                        if possible_bonds.at[bond_idx, "idx1"] not in connected_atoms_idx:
                            connected_atoms_idx.append(possible_bonds.at[bond_idx, "idx1"])
                        if possible_bonds.at[bond_idx, "idx2"] not in connected_atoms_idx:
                            connected_atoms_idx.append(possible_bonds.at[bond_idx, "idx2"])
                        connected_atoms_idx = connected_atoms_idx
                        bond_scatter.append(                        
                            go.Scatter3d(
                                x=[possible_bonds.at[bond_idx, "x_1"], possible_bonds.at[bond_idx, "x_2"]], 
                                y=[possible_bonds.at[bond_idx, "y_1"], possible_bonds.at[bond_idx, "y_2"]], 
                                z=[possible_bonds.at[bond_idx, "z_1"], possible_bonds.at[bond_idx, "z_2"]],
                                mode="lines",
                                line=dict(
                                    color='rgba(100, 100, 255, 0.5)',
                                    width=5,
                                ),
                                showlegend=False,
                                hoverinfo='skip')
                        )

    
    atom_df = atom_df.loc[atom_df.index.isin(connected_atoms_idx), :]

    unit_cell = [[np.array([0, 0, 0]), A], [np.array([0, 0, 0]), B], [np.array([0, 0, 0]), C], 
                 [A, A+B], [A, A+C], [B, B+A], 
                 [B, B+C], [C, C+A], [C, C+B], 
                 [A+B, A+B+C], [B+C, A+B+C], [A+C, A+B+C]]
    unit_cell_color = ["rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", 
                       "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)", "rgba(255, 255, 0, 1)"]

    # XYZ arrows
    axis_length = 0.5
    V_diag = 0.25 * (A + B + C)
    x_start = -V_diag[0]
    y_start = -V_diag[1]
    z_start = -V_diag[2]
    x_end1 = (axis_length * A - V_diag)[0]
    y_end1 = (axis_length * A - V_diag)[1]
    z_end1 = (axis_length * A - V_diag)[2]
    x_end2 = (axis_length * B - V_diag)[0]
    y_end2 = (axis_length * B - V_diag)[1]
    z_end2 = (axis_length * B - V_diag)[2]
    x_end3 = (axis_length * C - V_diag)[0]
    y_end3 = (axis_length * C - V_diag)[1]
    z_end3 = (axis_length * C - V_diag)[2]
    arrows = [
        go.Scatter3d(x=[x_start, x_end1], 
                     y=[y_start, y_end1], 
                     z=[z_start, z_end1],
                     mode="lines+text",
                     text=["", "A"],
                     line=dict(
                         color='rgba(255, 0, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(255, 0, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end2], 
                     y=[y_start, y_end2], 
                     z=[z_start, z_end2],
                     mode="lines+text",
                     text=["", "B"],
                     line=dict(
                         color='rgba(0, 255, 0, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 255, 0, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
        go.Scatter3d(x=[x_start, x_end3], 
                     y=[y_start, y_end3], 
                     z=[z_start, z_end3],
                     mode="lines+text",
                     text=["", "C"],
                     line=dict(
                         color='rgba(0, 0, 255, 1)',
                         width=5,
                     ),
                     textfont=dict(
                         color='rgba(0, 0, 255, 1)',
                         size=20,
                     ),
                     showlegend=False,
                     hoverinfo='skip',
        ),
    ]


    data=[go.Scatter3d(x=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "x"], 
                       y=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "y"], 
                       z=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "z"], 
                       text=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "_atom_site_type_symbol"] + "<br>" + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "_atom_site_label"] + "<br>" + \
                            atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "periodic_image"],
                       mode='markers',
                       showlegend=True,
                       name=k,
                       marker=dict(
                           color=atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "color"],
                           size=1.5*atom_df[atom_df["_atom_site_type_symbol"]==k].loc[:, "_atom_site_type_symbol"].map(symbol2plot_radius),
                           opacity=1.0,
                       )) for k in unique_elements] + \
                       bond_scatter + [go.Scatter3d(x=np.array(unit_cell[i]).T[0], 
                                                    y=np.array(unit_cell[i]).T[1], 
                                                    z=np.array(unit_cell[i]).T[2],
                                                    mode="lines",
                                                    line=dict(
                                                        color=unit_cell_color[i],
                                                        width=2,
                                                    ),
                                                    showlegend=False,
                                                    hoverinfo='skip') for i in range(0, len(unit_cell))] + arrows

    xyzmin = min([atom_df["x"].min(), atom_df["y"].min(), atom_df["z"].min(), np.array(unit_cell).min(), (0-V_diag).min()])
    xyzmax = max([atom_df["x"].max(), atom_df["y"].max(), atom_df["z"].max(), np.array(unit_cell).max(), (0-V_diag).max()])
    DeltaX = xyzmax - xyzmin
    padding_xyz = DeltaX * 0.05
    fig = go.Figure(data=data)
    annotation_list = []
    fig.update_layout(
        scene = dict(
            annotations=annotation_list,
            xaxis = dict(nticks=10, range=[xyzmin-padding_xyz,xyzmax+padding_xyz],
                         backgroundcolor="rgba(80, 70, 70, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
            yaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 80, 70, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
            zaxis = dict(nticks=10, range=[xyzmin-padding_xyz, xyzmax+padding_xyz],
                         backgroundcolor="rgba(70, 70, 80, 0.5)",
                         gridcolor="white",
                         showbackground=showbackground,
                         showgrid=False, 
                         zeroline=False,
                         showticklabels=False,
                         visible=False,
                         #zerolinecolor="white",
                        ),
        ),
        width=1200,
        height=1200,
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=True,
        template="plotly_white",
    )
    fig.update_layout(
        scene_aspectmode='cube', 
        title=dict(
            text=os.path.basename(cif_name),
            font_size=30,
            font_color="rgba(200,200,200,1)",
            yanchor="top",
            y=0.98,
            xanchor="center",
            x=0.5,
        ),
        font_color="rgba(200,200,200,1)",
        legend_title_font_color="rgba(200,200,200,1)",
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_layout(
        legend=dict(
            font=dict(
                size = 20,
            ),
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=0.01,
        )
    )
    if _cif_name != cif_name:
        os.remove(cif_name)
    return fig