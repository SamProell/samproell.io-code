import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams["axes.prop_cycle"] = \
    "cycler('color', ['078C7E','d81159','fbb13c','73d2de','8f2d56'])"

st.set_page_config(page_title="Ad-hoc data filters", page_icon="📈")

@st.cache
def read_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def create_slicer(df, col):
    """Create a data slicer depending on column data type (multiselct/slider).
    """
    if col in df.select_dtypes(include="number"):
        if col in df.select_dtypes(include="integer"):
            valmin, valmax = int(df[col].min()), int(df[col].max())
        else:
            valmin, valmax = float(df[col].min()), float(df[col].max())
        return st.slider(col, valmin, valmax, (valmin, valmax))
    elif col in df.select_dtypes(["object"]):
        options = df[col].dropna().unique()
        return st.multiselect(col, options, default=options)
    return None

def apply_slicers(df, filters):
    """Filter dataset according to slicer selections.
    """
    for col, selection in filters.items():
        if col in df.select_dtypes("number"):
            df = df.query(f"{selection[0]} <= `{col}` <= {selection[1]}")
        if col in df.select_dtypes(exclude=["number"]) and len(selection) > 0:
            # empty selections are ignored, as they would lead to an empty set
            df = df.query(f"{col} in @selection")
    return df

def plot_regression(df, x, y, hue, regression=True):
    """Create (colored) scatter plot with optional regression line.
    """
    fig = plt.figure()
    palette = "mako" if hue in df.select_dtypes("number") else None
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.75, palette=palette)

    if regression:
        sns.regplot(data=df, x=x, y=y, scatter=False, line_kws=dict(color=".3"))

    return fig

"""# Quick analysis with ad-hoc filters

This Streamlit app showcases how data filters can be created and used for fast
data exploration.
You may chose any columns in the dataset to create slicers that subset the
data based on user input. The type of slicer is chosen based on the data type.
For numeric columns, data can be filtered with a range slider.
For all other data types, a multiselect box is created.

In addition, for any subset that cannot be obtained through simple slicers,
you may specify an arbitrary query which will be interpreted with
`pd.DataFrame.query`.
"""

datafile = st.sidebar.file_uploader("Upload dataset", ["csv"])
if datafile is None:
    st.info("""Upload a dataset (.csv) in the sidebar to get started.""")
    st.stop()

data = read_data(datafile).copy()

filter_cols = st.sidebar.multiselect("Filter columns", data.columns)

filters = {}
with st.sidebar.expander("Filters", expanded=True):
    for col in filter_cols:
        selection = create_slicer(data, col)
        if selection is not None:  # skip unrecognized column types
            filters[col] = selection

    query = st.text_area("Custom query") or "tuple()"

# apply slicers and custom query
data = apply_slicers(data, filters)
data = data.query(query, engine="python")

# identify numeric columns to use as X/Y in regression plot.
numeric_cols = data.select_dtypes("number").columns
if len(numeric_cols) < 1:
    st.warning("No numeric columns found for plotting.")
    st.stop()

leftcol, rightcol = st.columns([2, 1])
with rightcol:  # plot setup selectors on the right
    xcol = st.selectbox("X variable", numeric_cols)
    ycol = st.selectbox("Y variable", numeric_cols, index=len(numeric_cols)-1)

    # hue column is optional - the "None" string is replaced by actual None
    huecol = st.selectbox("Color by", ["None"] + data.columns.tolist())
    if huecol == "None":
        huecol = None
with leftcol:  # plot to the left
    fig = plot_regression(data, xcol, ycol, hue=huecol, regression=True)
    st.pyplot(fig)