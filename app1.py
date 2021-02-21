from openml.datasets.functions import get_dataset
from gtda.plotting import plot_point_cloud
from gtda.pipeline import Pipeline
from scipy.integrate import odeint
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gtda.time_series as ts
import gtda.diagrams as diag
import plotly_express as px
import gtda.homology as hl
import gtda.graphs as gr
import plotly.io as pio 
import streamlit as st
import pandas as pd 
import numpy as np

# HEADER

st.title("CHAOS AND TOPOLOGICAL DATA ANALYSIS")
st.title("")

# CSS

def css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

css("styles.css")

# SIDEBAR CONTENT

# LORENZ ATTRACTOR

    # PARAMS AND INITIAL CONDITIONS

sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05

    # TIME STEPS

tmax, n = 100, 10000

def Lorenz(X, t, sigma, beta, rho):

    u, v, w = X

    up = - sigma * (u - v)

    vp = rho * u - v - u * w

    wp = - beta * w + u * v

    return up, vp, wp

    # NUMERICAL INTEGRATIONS

t = np.linspace(0, tmax, n) # time
f = odeint(Lorenz, (u0, v0, w0), t, args = (sigma, beta, rho)) # numerical values for x,y,z states
x, y, z = f.T

    # PLOT

lor = px.scatter_3d(x = x, y = y, z = z, template = 'plotly_dark', width = 920, height = 700)
lor.update_traces(marker = dict(size = 4,
                              color = '#FFFFFF',
                              line = dict(width=2, color = '#000000')),
                  selector = dict(mode = 'markers'))
lor.update_xaxes(showgrid = False)
lor.update_yaxes(showgrid = False)
lor.update_xaxes(showticklabels = False)
lor.update_yaxes(showticklabels = False)

    # TEXT

st.subheader("3-DIMENSIONAL LORENZ SYSTEM: ")
st.markdown("")
st.latex("\Large{\dot{x} = \sigma(y - x)}")
st.latex("\Large{\dot{y} = rx - y - xz}")
st.latex("\Large{\dot{z} = xy - bz}")
st.subheader("")

st.write(lor)
st.subheader("Figure 1. Lorenz Attractor in 3-Dimensional Space")

    # LORENZ SIGNAL TO POINT CLOUD

# st.title("")
# st.subheader("POINT CLOUD REALIZATION")

# lorpc = get_dataset(42182).get_data(dataset_format = 'array')[0]
# pcplt = plot_point_cloud(lorpc, plotly_params = dict(template = 'plotly_dark'))

# st.write(pcplt)
# st.subheader("Figure 2. Lorenz System Point Cloud")

    # LORENZ SIGNAL

lorpc = get_dataset(42182).get_data(dataset_format = 'array')[0] # FOUR COLUMNS, I.E, X, Y, Z, RHO
Z = lorpc[:, 2] # Z
RHO = lorpc[:, 3] # RHO

sig = px.line(template = 'plotly_dark')
sig.add_scatter(y = Z, name = 'Z')
sig.add_scatter(y = RHO, name = 'RHO')

st.title("")
st.subheader("LORENZ SYSTEM SIGNAL")
st.title("")

st.write(sig)
st.subheader("Figure 2. Lorenz Signal in 2-Dimensional Space")
st.subheader("")
if st.checkbox("Show Lorenz System Data", False):
    st.write(lorpc)

    # LORENZ SIGNAL RESAMPLING

st.title("")
periodSampler = ts.Resampler(period = 10)
X_resamp, y_resamp = periodSampler.fit_transform_resample(Z, RHO)

sigii = px.line(template = 'plotly_dark')
sigii.add_scatter(y = X_resamp.flatten(), name = 'X_sampled')
sigii.add_scatter(y = y_resamp, name = 'y_sampled')

st.write(sigii)
st.subheader("Figure 3. Resampled Lorenz Signal in 2-Dimensional Space")

# TAKEN'S EMBEDDING

    # SEE YOUTUBE: STATE-SPACE RECONSTRUCTION FROM TIME-SERIES DATA

st.title("")
st.subheader("TIME-DELAY EMBEDDING VIA TAKEN'S EMBEDDING")

st.subheader("")
st.subheader("Given a time-series $\Large{X(t)}$, we can extract a sequence of vectors of the form: ")
st.subheader("$\Large{X_i = [X(t_i), \; X(t_i + 2*tau,\; . \; . \; .,\; X(t_i + M*tau))]}$")
st.subheader("The difference between each timestep $t_{i+1} \: and \: t_i$, is called the stride")
st.markdown("")
st.subheader("This is the loose formalization of Taken's Embedding")

    # SIGNAL EMBEDDING OPTIMAL PARAMETERS

max_time_delay = 3
max_embedding_dimension = 10
optimal_time_delay, optimal_embedding_dimension = ts.takens_embedding_optimal_parameters(
    X_resamp, max_time_delay, max_embedding_dimension, stride = 1
)

st.subheader("")
st.subheader("OPTIMAL EMBEDDING PARAMETERS")
st.subheader("$\large{\hat{T} = 3}$")
st.subheader("$\large{\hat{d} = 10}$")

    # SLIDING WINDOW AND RESAMPLED DATA FIT

slide = ts.SlidingWindow(size = 42, stride = 5)
X_win, y_win = slide.fit_transform_resample(X_resamp, y_resamp)

    # TAKEN'S EMBEDDING (MULTIVARIATE)

taken = ts.TakensEmbedding(time_delay = 3,
                           dimension = 10,
                           stride = 10)
X_embed = taken.fit_transform(X_win, y_win)

st.subheader("")
st.write(X_embed.shape)
if st.checkbox("Show Embedding", False):
    st.write(X_embed)

    # PLOTTING SUBSET OF THE DIMESNIONS

        # X EMBEDDED SHAPE (42, 2 10)

st.subheader("")
# tak = px.scatter_3d(x = X_embed[:, 0],
#                     y = X_embed[:, 1],
#                     z = X_embed[:, 2],
#                     template = 'plotly_dark',
#                     labels = {'x': 'Taken 0',
#                               'y': 'Taken 1',
#                               'z': 'Taken 2'})

# st.write(tak)
# st.subheader("Figure 4. 3-Dimensional Subset of Taken Embedded Lorenz Signal")

    # GIOTTO-TDA EMBED PLOT W/ WINDOW SLIDER

# window = st.slider(label = 'Select Time-Delay Window', min_value = 1, max_value = 42, value = 21)
# taksubset = taken.plot(X_embed, sample = window)
# taksubset.update_layout(template = 'plotly_dark')

# st.write(taksubset)
# st.subheader("Figure 4. 3-Dimensional Subset of Taken Embedded Lorenz Signal")
# st.subheader("")

# hdim = (0, 1, 2)
# per = hl.WeakAlphaPersistence(homology_dimensions = hdim)
# X_perisitence = per.fit_transform(X_embed)

    # PERSISTENCE DIAGRAM

# perdiag = per.plot(X_perisitence, sample = window)
# perdiag.update_layout(template = 'plotly_dark')

# st.write(perdiag)
# st.subheader("Figure 5. Persistence Diagram for Selected Sliding Window")

# WE CAN ALSO USE PRINCIPAL COMPONENT ANALYSIS (PCA) ON THE EMBEDDED DATA
# https://giotto-ai.github.io/gtda-docs/latest/notebooks/gravitational_waves_detection.html#useful-references

# FROM GIOTTO-TDA ARTICLE 

point_cloud_ = get_dataset(42182).get_data(dataset_format='array')[0]
# st.write(plot_point_cloud(point_cloud_))

X_ = point_cloud_[:, 2]
y_ = point_cloud_[:, 3]

periodicSampler = ts.Resampler(period = 10)

X_sampled_, y_sampled_ = periodicSampler.fit_transform_resample(X_, y_)

max_time_delay = 3
max_embedding_dimension = 10
stride = 1
optimal_time_delay, optimal_embedding_dimension = ts.takens_embedding_optimal_parameters(
    X_sampled_, max_time_delay, max_embedding_dimension, stride = stride
    )

    # CONSOLE PRINTS

print(f"Optimal embedding time delay based on mutual information: {optimal_time_delay}")
print(f"Optimal embedding dimension based on false nearest neighbors: {optimal_embedding_dimension}")

window_size = 41
window_stride = 5
SW = ts.SlidingWindow(size = window_size, stride = window_stride)

X_windows_, y_windows_ = SW.fit_transform_resample(X_sampled_, y_sampled_)

TE = ts.TakensEmbedding(time_delay = optimal_time_delay, dimension = optimal_embedding_dimension, stride = stride)
X_embedded_ = TE.fit_transform(X_windows_)

teplot0 = TE.plot(X_embedded_, sample = 3)
teplot0.update_layout(template = 'plotly_dark')

st.subheader("TAKEN'S EMBEDDING")
st.subheader("")
st.write(teplot0)
st.subheader("Figure 4. 3-Dimensional Subset of Taken Embedded Lorenz Signal")
st.subheader("")

    # INTERPRETATION

st.subheader("From Figure 4, we can see that the 3-Dimensional embedding exhibits some quasi-periodicity that we see from the original 3-Dimensional attractor.")

    # PERSISTENCE DIAGRAM(S)

st.title("")
st.subheader("PERSISTENCE DIAGRAM")
st.subheader("")

    # THREE 

homology_dimensions = (0, 1, 2)
WA = hl.WeakAlphaPersistence(homology_dimensions = homology_dimensions)

X_diagrams_ = WA.fit_transform(X_embedded_)

per0 = WA.plot(X_diagrams_, sample = 3)
per0.update_layout(template = 'plotly_dark')

st.write(per0)
st.subheader("Figure 5. Persistence Diagram for $\large{H_3}$")
st.subheader("")

    # FIVE

# homology_dimensions_0 = (0, 1, 2, 3, 4)
# WA_0 = hl.WeakAlphaPersistence(homology_dimensions = homology_dimensions_0)

# X_diagrams_0 = WA.fit_transform(X_embedded_)

# per1 = WA_0.plot(X_diagrams_, sample = 5)
# per1.update_layout(template = 'plotly_dark')

# st.write(per1)
# st.subheader("Figure 6. Persistence Diagram for $\large{H_5}$")

    # PERSISTENCE DIAGRAMS

st.subheader("DIAGRAM FILTERING AND RESCALING")

diagramScaler = diag.Scaler()

X_scaled = diagramScaler.fit_transform(X_diagrams_)

resplot = diagramScaler.plot(X_scaled, sample = 3)
resplot.update_layout(template = 'plotly_dark')

st.subheader("")
st.write(resplot)
st.subheader("Figure 6. Rescaled Persistence Diagram for $\large{H_3}$")
st.subheader("")

diagramFiltering = diag.Filtering(epsilon = 0.1, homology_dimensions = (1, 2))

X_filtered = diagramFiltering.fit_transform(X_scaled)

filplot = diagramFiltering.plot(X_filtered, sample = 3)
filplot.update_layout(template = 'plotly_dark')

st.subheader("")
st.write(filplot)
st.subheader("Figure 7. Filtered Persistence Diagram for $\large{H_3}$")
st.subheader("")

st.title("")
st.subheader("PERSISTENCE ENTROPY")
st.subheader("")

PE = diag.PersistenceEntropy()
X_persistence_entropy = PE.fit_transform(X_scaled)

entropyplot = px.line(title = 'Persistence Entropies, indexed by Sliding Window Number')

for dim in range(X_persistence_entropy.shape[1]):

    entropyplot.add_scatter(y = X_persistence_entropy[:, dim], name = f"Persistent Entropy in Homology Dimension: {dim}")

entropyplot.update_layout(template = 'plotly_dark')

st.subheader("")
st.write(entropyplot)
st.subheader("Figure 8. Persistence Entropy Diagram for Homological Dimensions")
st.subheader("")

st.title("")
st.subheader("BETTI CURVES")
st.subheader("")

BC = diag.BettiCurve()

X_betti_curves = BC.fit_transform(X_scaled)

bettiplot = BC.plot(X_betti_curves, sample = 3)
bettiplot.update_layout(template = 'plotly_dark')

st.subheader("")
st.write(bettiplot)
st.subheader("Figure 9. Betti Curves")
st.subheader("")

# DISTANCE METRICS