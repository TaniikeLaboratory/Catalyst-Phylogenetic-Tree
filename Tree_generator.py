import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree, TreeStyle, TextFace, NodeStyle
from PyQt5 import QtGui
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Setting for non-GUI usage
import matplotlib.pyplot as plt

# ==================================================
#  Reading configuration parameters from init.txt
# ==================================================
# Use the same directory as the code file as the base path
init_file = "input/init.txt"
init_params = {}
with open(init_file, "r") as f:
    current_key = None
    current_value = []
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            if current_key:
                init_params[current_key] = "\n".join(current_value).strip()
                current_key = None
                current_value = []
            continue
        if line in ["File_name", "Alpha", "Color", "Filter"]:
            if current_key:
                init_params[current_key] = "\n".join(current_value).strip()
            current_key = line
            current_value = []
        else:
            current_value.append(line)
    if current_key:
        init_params[current_key] = "\n".join(current_value).strip()

# Obtaining configuration parameters
csv_file_name = init_params.get("File_name")
alpha = float(init_params.get("Alpha", 0.5))  # Default value changed to 0.5
color = init_params.get("Color")
filter_condition = init_params.get("Filter")

print(alpha)

# Other user settings
desired_dpi = 300
clip_value = 3000

# ==================================================
# 1) Data loading & preprocessing
# ==================================================
df = pd.read_csv(csv_file_name)
df_pt = pd.read_csv('input/distance_matrix_elements.csv', index_col=0)
elements = df_pt.index.tolist()

def compute_name_and_groups(row):
    # Use .loc or .iloc to avoid FutureWarning
    valid_elements = [e for e in elements if e in row.index]
    elem_series = row[valid_elements]
    present = elem_series[elem_series > 0]
    if present.empty:
        return "NoElements", [], []

    max_val = present.max()
    threshold = 0.5 * max_val - 1

    s_elements = []
    m_elements = []
    for e in valid_elements:
        val = row[e]
        if val > 0:
            if val >= threshold:
                s_elements.append(e)
            else:
                m_elements.append(e)

    m_elements_sorted = sorted(m_elements, key=lambda x: elements.index(x))
    s_elements_sorted = sorted(s_elements, key=lambda x: elements.index(x))
    if m_elements_sorted:
        new_name = '-'.join(m_elements_sorted) + '/' + '-'.join(s_elements_sorted)
    else:
        new_name = '-'.join(s_elements_sorted)
    return new_name, m_elements_sorted, s_elements_sorted

# Create NewName, M_group, and S_group
df[["Name", "M_group", "S_group"]] = df.apply(lambda row: pd.Series(compute_name_and_groups(row)), axis=1)
df.set_index("Name", inplace=True)
df.index.name = None

# ==================================================
# 2) Creating DataFrame for aggregation
# ==================================================
exclude_cols = ["M_group", "S_group"]
non_element_cols = [c for c in df.columns if c not in exclude_cols]

name_counter = Counter(df.index.tolist())
df_for_agg = df[non_element_cols].copy()
df_for_agg["count"] = [name_counter[nm] for nm in df.index]

# ==================================================
# 3) Aggregation of statistics
# ==================================================
def mode(x):
    m = x.mode(dropna=True)
    return m.iloc[0] if len(m) > 0 else np.nan

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def total(x):
    return x.iloc[0]

aggregations = {}
for col in df_for_agg.columns:
    if col == "count":
        aggregations[col] = [total]
    elif is_numeric_dtype(df_for_agg[col]):
        aggregations[col] = ['min', 'max', 'median', 'mean', mode, q25, q75]
    else:
        aggregations[col] = [mode]

df_agg = df_for_agg.groupby(level=0).agg(aggregations)

agg_name_map = {
    total: 'total',
    'max': 'max',
    'median': 'med',
    'mean': 'avg',
    mode: 'mode',
    q25: 'q25',
    q75: 'q75'
}

def get_agg_name(func):
    if func in agg_name_map:
        return agg_name_map[func]
    elif isinstance(func, str):
        return func
    else:
        return func.__name__

df_agg.columns = [f"{col[0]}_{get_agg_name(col[1])}" for col in df_agg.columns]

# --- Rename columns for filtering if applicable ---
if "count_total" in df_agg.columns:
    df_agg = df_agg.rename(columns={"count_total": "Frequency of appearances"})
if "year_min" in df_agg.columns:
    df_agg = df_agg.rename(columns={"year_min": "First appearance year"})

df_agg.to_csv("output/statistics.csv")

# ==================================================
# 4) Calculation of the distance matrix
# ==================================================
valid_names = df_agg.index
df_for_distance = df.loc[valid_names, ["M_group", "S_group"]].copy()
df_for_distance = df_for_distance[~df_for_distance.index.duplicated(keep='first')]
Name_unique = df_for_distance.index.tolist()
M_groups = df_for_distance["M_group"].tolist()
S_groups = df_for_distance["S_group"].tolist()
Comp = list(zip(M_groups, S_groups))
N = len(Comp)

# Use the inter-element distance matrix (df_pt) read earlier
element_distance_matrix = df_pt

def compare(a, b):
    if not a and not b:
        return 0
    distance_a = element_distance_matrix.loc[a, b].min(axis=1).sum() if a else 0
    distance_b = element_distance_matrix.loc[b, a].min(axis=1).sum() if b else 0
    return distance_a + distance_b

def compute_distance_row(i):
    distances = np.zeros(N)
    M_i, S_i = Comp[i]
    M_i_local = S_i if len(M_i) == 0 else M_i
    for j in range(i, N):
        M_j, S_j = Comp[j]
        M_j_local = S_j if len(M_j) == 0 else M_j
        d_m = compare(M_i_local, M_j_local)
        d_s = compare(S_i, S_j)
        distances[j] = alpha * d_m + (1 - alpha) * d_s
    return i, distances

D = np.zeros((N, N))
results = Parallel(n_jobs=-1)(delayed(compute_distance_row)(i) for i in tqdm(range(N)))
for i, distances in results:
    D[i, i:] = distances[i:]
    D[i:, i] = distances[i:]
if D.max() != D.min():
    D = (D - D.min()) / (D.max() - D.min())
D = np.round(D, 5)
df_distance = pd.DataFrame(D, index=Name_unique, columns=Name_unique)
df_distance.to_csv("output/distance_matrix.csv")

# ==================================================
# 5) Parameters for drawing the phylogenetic tree
# ==================================================
tree_img = "output/tree1.png"
final_output_file = "output/tree2.png"

# ==================================================
# 6) Apply filter and sorting
# ==================================================
if filter_condition is not None:
    filter_expr = filter_condition.replace(" or ", " | ").replace(" and ", " & ")
    # Replace column names with spaces by enclosing them in backticks
    for col in df_agg.columns:
        if " " in col and col in filter_expr:
            filter_expr = filter_expr.replace(col, f"`{col}`")
    df_agg = df_agg.query(filter_expr)

remaining_catalysts = df_agg.index
df_distance = df_distance.loc[remaining_catalysts, remaining_catalysts]

print(len(df_distance))

if color not in df_agg.columns:
    raise ValueError(f"Specified column '{color}' does not exist: {df_agg.columns.tolist()}")
df_order = df_agg.sort_values(by=color)
key = df_agg[color]
label = color
cmap_obj = plt.cm.GnBu

# ==================================================
# 7) Color mapping function and value clipping
# ==================================================
def color_dict(z, scale="linear", alpha=1, cmap_obj=None):
    if cmap_obj is None:
        cmap_obj = plt.cm.jet
    color_mapping = {}
    z_sorted = sorted(z)
    min_val, max_val = z_sorted[0], z_sorted[-1]
    range_val = max_val - min_val
    beta = min(alpha * 1.5, 1)
    alpha_int = int(alpha * 255)
    beta_int = int(beta * 255)
    if scale == "linear":
        for val in z_sorted:
            norm = 0.5 if range_val == 0 else (val - min_val) / range_val
            rgb = cmap_obj(norm)
            rgb_255 = [int(c * 255) for c in rgb[:3]]
            rgb_255_origin = rgb_255.copy() + [beta_int]
            color_origin = QtGui.QColor(*tuple(rgb_255_origin))
            hex_color_origin = color_origin.name(QtGui.QColor.HexRgb)
            rgb_255.append(alpha_int)
            color_final = QtGui.QColor(*tuple(rgb_255))
            hex_color = color_final.name(QtGui.QColor.HexRgb)
            color_mapping[str(val)] = [hex_color, hex_color_origin]
    else:
        pass
    return color_mapping

unique_values = key.unique().astype(int)
min_val, max_val = int(unique_values.min()), int(unique_values.max())
max_val = min(max_val, clip_value)
clipped_values = list(range(min_val, max_val + 1))
color_mapping = color_dict(clipped_values, scale="linear", alpha=0.4, cmap_obj=cmap_obj)

# ==================================================
# 8) Construction and drawing of the phylogenetic tree using the Neighbor-Joining method
# ==================================================
names = df_distance.index.tolist()
dm = DistanceMatrix(df_distance.values, names)
tree_newick = nj(dm, result_constructor=str)
t = Tree(tree_newick, format=1)
name_order = df_order.index.tolist()

# Set up each leaf node (branch coloring is removed; default style is applied)
for name in name_order:
    leaf_nodes = t.search_nodes(name=name)
    if not leaf_nodes:
        continue
    leaf_node = leaf_nodes[0]
    tk = 5  # Fixed branch line width
    default_style = NodeStyle()
    default_style["hz_line_width"] = tk
    default_style["vt_line_width"] = tk
    default_style["size"] = 0

    current_node = leaf_node
    while current_node:
        current_node.set_style(default_style)
        current_node = current_node.up

    val = key.loc[name]
    val_clipped = min(int(val), clip_value)
    midpoint = (max(clipped_values) + min(clipped_values)) / 2

    # Remove any dependency on 'Flag_min_total'; apply default text style.
    name_face = TextFace(name, fgcolor="white" if val_clipped > midpoint else "black", fsize=50, bold=True)
    if (colors := color_mapping.get(str(val_clipped))) is not None:
        name_face.background.color = colors[0]
    leaf_node.add_face(name_face, column=1, position='aligned')

circular_style = TreeStyle()
circular_style.show_leaf_name = False
circular_style.mode = "c"
circular_style.draw_guiding_lines = True
circular_style.guiding_lines_type = 1
circular_style.guiding_lines_color = "Silver"
circular_style.show_scale = False

for node in t.traverse():
    if node.dist == 0 and not node.is_root():
        node.dist = 0.01

width_px = int(8 * desired_dpi)
height_px = int(8 * desired_dpi)
t.render(tree_img, w=width_px, h=height_px, units="px", tree_style=circular_style)

# ==================================================
# 9) Create color bar and combine images
# ==================================================
def create_colorbar(cmap_obj, values_range, filename, scale='linear',
                    orientation='vertical', height_in_pixels=1000):
    norm = plt.Normalize(vmin=min(values_range), vmax=max(values_range))
    fig_height = height_in_pixels / desired_dpi
    fig, ax = plt.subplots(figsize=(1, fig_height))
    fig.subplots_adjust(left=0.5, right=0.7, bottom=0.05, top=0.95)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation=orientation)
    cbar.set_label(label, rotation=270, labelpad=15)
    fig.savefig(filename, dpi=desired_dpi, bbox_inches='tight')
    plt.close(fig)

def combine_images(tree_image_filename, colorbar_image_filename, final_image_filename):
    from PIL import Image
    tree_image = Image.open(tree_image_filename)
    colorbar_image = Image.open(colorbar_image_filename)
    new_colorbar_width = int((colorbar_image.width / colorbar_image.height) * tree_image.height)
    colorbar_image = colorbar_image.resize((new_colorbar_width, tree_image.height), Image.LANCZOS)
    total_width = tree_image.width + colorbar_image.width
    new_im = Image.new('RGB', (total_width, tree_image.height), (255, 255, 255))
    new_im.paste(tree_image, (0, 0))
    new_im.paste(colorbar_image, (tree_image.width, 0))
    new_im.save(final_image_filename)

tree_image = Image.open(tree_img)
colorbar_img = "output/colorbar.png"
create_colorbar(cmap_obj, clipped_values, colorbar_img, orientation='vertical', height_in_pixels=tree_image.height)
combine_images(tree_img, colorbar_img, final_output_file)
