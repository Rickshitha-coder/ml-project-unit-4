import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")

# Title
st.title("🛒 Market Basket Product Clustering")
st.markdown("Group products that customers frequently buy together using **K-Means Clustering**.")

# Sidebar
st.sidebar.header("⚙️ Clustering Settings")

uploaded_file = st.file_uploader("📂 Upload Market Basket CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    col1, col2 = st.columns(2)

    # Dataset preview
    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

    # Dataset info
    with col2:
        st.subheader("📈 Dataset Info")
        st.metric("Transactions", df.shape[0])
        st.metric("Products", df.shape[1])

    # Cluster slider
    k = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)

    # Apply KMeans
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(df.T)

    product_names = df.columns

    result = pd.DataFrame({
        "Product": product_names,
        "Cluster": clusters
    })

    st.subheader("🧺 Product Clusters")
    st.dataframe(result)

    # Download clustered dataset
    csv = result.to_csv(index=False).encode("utf-8")
    
    st.download_button(
        label="⬇ Download Clustered Dataset",
        data=csv,
        file_name="product_clusters.csv",
        mime="text/csv"
    )
    # Cluster summary
    st.subheader("📦 Cluster Groups")

    for i in range(k):
        products = result[result["Cluster"] == i]["Product"].tolist()
        st.write(f"**Cluster {i} →** {', '.join(products)}")

    # Visualization
    st.subheader("📉 Product Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.scatter(range(len(product_names)), clusters)

    ax.set_xticks(range(len(product_names)))
    ax.set_xticklabels(product_names, rotation=45)

    ax.set_ylabel("Cluster")
    ax.set_xlabel("Products")
    ax.set_title("Product Grouping using K-Means")

    st.pyplot(fig)

else:
    st.info("⬆️ Upload a dataset to start clustering.")
