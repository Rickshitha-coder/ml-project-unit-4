import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Market Basket AI Dashboard", page_icon="🛒", layout="wide")

st.title("🛒 Market Basket Product Clustering Dashboard")
st.markdown("Analyze **product buying patterns** using K-Means clustering and visualize them with PCA.")

# Sidebar
st.sidebar.header("⚙️ Model Settings")

uploaded_file = st.file_uploader("📂 Upload Market Basket Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", df.shape[0])

    with col2:
        st.metric("Total Products", df.shape[1])

    with col3:
        st.metric("Dataset Size", f"{df.shape[0]*df.shape[1]} values")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Cluster selection
    k = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)

    # KMeans clustering
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(df.T)

    product_names = df.columns

    result = pd.DataFrame({
        "Product": product_names,
        "Cluster": clusters
    })

    st.subheader("🧺 Product Cluster Table")
    st.dataframe(result)

    # Cluster summary
    st.subheader("📦 Cluster Groups")

    for i in range(k):
        items = result[result["Cluster"] == i]["Product"].tolist()

        if len(items) > 0:
            st.success(f"Cluster {i}: {', '.join(items)}")

    # PCA Visualization
    st.subheader("📉 PCA Cluster Visualization")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df.T)

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(reduced[:,0], reduced[:,1], c=clusters)

    for i, product in enumerate(product_names):
        ax.text(reduced[i,0], reduced[i,1], product)

    ax.set_title("Product Clusters using PCA")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    st.pyplot(fig)

    # Download button
    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Clustered Dataset",
        data=csv,
        file_name="product_clusters.csv",
        mime="text/csv"
    )

else:
    st.info("⬆ Upload a dataset to begin analysis.")
