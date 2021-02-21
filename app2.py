from sklearn.model_selection import train_test_split
from openml.datasets.functions import get_dataset
from gtda.homology import CubicalPersistence
from gtda.plotting import plot_point_cloud
from sklearn.datasets import fetch_openml
from gtda.images import RadialFiltration
from gtda.plotting import plot_heatmap
from gtda.pipeline import Pipeline
from scipy.integrate import odeint
import plotly.graph_objects as go
from gtda.images import Binarizer
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

st.title("HANDWRITTEN DIGIT CLASSIFICATION AND TOPOLOGICAL DATA ANALYSIS")
st.title("")

# st.markdown("What will be apparent here is that the topological holes present in the 1st dimension for a given digit selections persistence diagram would accurately reflect that of the handwritten digits.")

# CSS

def css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

css("styles.css")

# SIDEBAR CONTENT

# MNIST DATASET

X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)

# TRAINING AND TESTING SUBSETS

trains, tests = 100, 20
X = X.reshape((-1, 28, 28))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = trains, test_size = tests, stratify = y, random_state = 42
)

# IMAGE FUNCTION

# st.sidebar.selectbox('Select Digit', options = ('0','1','2','3','4','5','6','7','8','9','10'))

# Bianizer

im_idx = np.flatnonzero(y_train == str(st.sidebar.selectbox('Select Digit', options = ('0','1','2','3','4','5','6','7','8','9','10'))))[0]

im = X_train[im_idx][None, :, :]

binarizer = Binarizer(threshold = 0.4)

im_binarized = binarizer.fit_transform(im)

binplot = binarizer.plot(im_binarized)

binplot.update_layout(template = 'plotly_dark')

# Radial Filtration

radial_filtration = RadialFiltration(center=np.array([20, 6]))

im_filtration = radial_filtration.fit_transform(im_binarized)

radplot = radial_filtration.plot(im_filtration, colorscale = "jet")

radplot.update_layout(template = 'plotly_dark')

# PLOTS

st.subheader("")
st.write(binplot)
st.subheader("Figure 1. Binarized Plot")
st.subheader("")

st.subheader("")
st.write(radplot)
st.subheader("Figure 2. Radial Filtration Plot")
st.subheader("")

# CUBICAL SIMPLICIAL COMPLEXES

cubical_persistence = CubicalPersistence(n_jobs = -1)
im_cubical = cubical_persistence.fit_transform(im_filtration)

cubplot = cubical_persistence.plot(im_cubical)
cubplot.update_layout(template = 'plotly_dark')

st.subheader("")
st.write(cubplot)
st.subheader("Figure 3. Cubical Simplicial Complex Persistence Diagram")
st.subheader("")