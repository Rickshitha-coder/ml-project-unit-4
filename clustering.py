import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒")

st.title("🛒 Market Basket Product Clustering")

st.write("Upload a market basket dataset to group products using K-Means clustering.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    k = st.slider("Select Number of Clusters",2,10,3)

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(df.T)

    product_names = df.columns

    result = pd.DataFrame({
        "Product":product_names,
        "Cluster":clusters
    })

    st.subheader("Product Clusters")
    st.write(result)

    fig, ax = plt.subplots()

    ax.scatter(range(len(product_names)), clusters)

    ax.set_xticks(range(len(product_names)))
    ax.set_xticklabels(product_names, rotation=90)

    ax.set_ylabel("Cluster")
    ax.set_title("Product Grouping")

    st.pyplot(fig)

else:
    st.info("Upload dataset to start clustering.")
