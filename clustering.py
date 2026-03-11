import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")

st.title("🛒 Market Basket Product Clustering")
st.write("Upload a market basket dataset to group products using K-Means clustering.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    k = st.slider("Select Number of Clusters",2,10,3)

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(df.T)

    product_names = df.columns

    result = pd.DataFrame({
        "Product":product_names,
        "Cluster":clusters
    })

    st.subheader("🧺 Product Clusters Table")
    st.dataframe(result)

    # Horizontal visualization
    st.subheader("📊 Product Cluster Visualization")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.scatter(product_names, clusters)

    ax.set_xlabel("Products")
    ax.set_ylabel("Cluster Number")
    ax.set_title("Product Clustering using K-Means")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    # Cluster summary
    st.subheader("📦 Cluster Groups")

    for i in range(k):
        items = result[result["Cluster"] == i]["Product"].tolist()
        if items:
            st.success(f"Cluster {i} → {', '.join(items)}")

else:
    st.info("Upload dataset to start clustering.")
