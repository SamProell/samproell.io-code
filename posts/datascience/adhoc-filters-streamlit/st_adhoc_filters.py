import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Ad-hoc data filters", page_icon="📈")
sns.set_style("darkgrid")


@st.cache
def read_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def create_slicer(df, col):
    """Create a data filter (slider/multiselect) and return user selection.

    Filters are created for 'number', 'date' and 'object'/'category' types.
    Returns `None` otherwise.
    """
    if col in df.select_dtypes(include=["number", "datetime"]):
        if col in df.select_dtypes(include="integer"):
            valmin, valmax = int(df[col].min()), int(df[col].max())
        elif col in df.select_dtypes(include="datetime"):
            valmin, valmax = df[col].min().date(), df[col].max().date()
        else:
            valmin, valmax = float(df[col].min()), float(df[col].max())
        return st.slider(col, valmin, valmax, (valmin, valmax))
    elif col in df.select_dtypes(["object", "category"]):
        options = df[col].dropna().unique()
        return st.multiselect(col, options, default=options)

    return None


def apply_slicer(df, col, selection):
    """Filter dataset according to slicer selections.
    """
    if col in df.select_dtypes(include=["number", "datetime"]):
        low, high = selection
        df = df.query(f"@low <= `{col}` <= @high")
    if col in df.select_dtypes(exclude=["number", "datetime"]):
        df = df.query(f"`{col}` in @selection")
    return df


def plot_regression(df, x, y, hue, regression=True):
    """Create (colored) scatter plot with optional regression line.
    """
    fig = plt.figure()
    palette = "mako" if hue in df.select_dtypes("number") else None
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.75, palette=palette)

    if regression:
        sns.regplot(data=df, x=x, y=y, scatter=False, line_kws={"color": ".3"})

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

*The code behind this app is discussed in detail on
[samproell.io](https://www.samproell.io) (post coming soon!).*
"""

# ==================================================================== #
#                               INPUTS                                 #
# ==================================================================== #
datafile = st.sidebar.file_uploader("Upload dataset", ["csv"])
if datafile is None:
    st.info("""Upload a dataset (.csv) in the sidebar to get started.""")
    st.stop()

data = read_data(datafile).copy()

# ==================================================================== #
#                          FILTER DEFINITION                           #
# ==================================================================== #
# get list of columns used for ad-hoc filters
filter_cols = st.sidebar.multiselect("Filter columns", data.columns)

# create filters for each selected column
filters = {}
with st.sidebar.expander("Filters", expanded=True):
    for col in filter_cols:
        selection = create_slicer(data, col)
        if selection is not None:  # skip unrecognized column types
            filters[col] = selection

    query = st.text_area("Custom query") or "tuple()"

# ==================================================================== #
#                         FILTER APPLICATION                           #
# ==================================================================== #
# apply the slicers and custom query
for col, selection in filters.items():
    data = apply_slicer(data, col, selection)
data = data.query(query, engine="python")

# ==================================================================== #
#                         FINAL CONFIGURATION                          #
# ==================================================================== #
# identify numeric columns to use as X/Y in regression plot
numeric_vars = data.select_dtypes("number").columns
if len(numeric_vars) < 1:
    st.warning("No numeric columns found for plotting.")
    st.stop()

leftcol, rightcol = st.columns([2, 1])
with rightcol:  # plot setup selectors on the right
    xvar = st.selectbox("X variable", numeric_vars)
    yvar = st.selectbox("Y variable", numeric_vars, index=len(numeric_vars)-1)

    # hue column is optional - the "None" string is replaced by actual None
    huevar = st.selectbox("Color by", ["None"] + data.columns.tolist())
    if huevar == "None":
        huevar = None

# ==================================================================== #
#                            VISUALIZATION                             #
# ==================================================================== #
with leftcol:  # plot to the left
    fig = plot_regression(data, xvar, yvar, hue=huevar, regression=True)
    st.pyplot(fig)
