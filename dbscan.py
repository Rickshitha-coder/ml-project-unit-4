import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="Ride Sharing Clustering", page_icon="🚕")

st.title("🚕 Ride Sharing Pickup Location Clustering")
st.write("Upload taxi pickup location dataset to detect demand clusters.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:

        X = df[['pickup_latitude','pickup_longitude']]

        eps = st.slider("DBSCAN eps value",0.001,0.05,0.01)
        min_samples = st.slider("Minimum Samples",5,50,10)

        if st.button("Run Clustering"):

            dbscan = DBSCAN(eps=eps,min_samples=min_samples)

            clusters = dbscan.fit_predict(X)

            df["cluster"] = clusters

            st.success("Clustering Completed")

            fig, ax = plt.subplots()

            scatter = ax.scatter(
                df["pickup_longitude"],
                df["pickup_latitude"],
                c=df["cluster"],
                cmap="rainbow",
                s=5
            )

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Pickup Location Clusters")

            st.pyplot(fig)

            st.subheader("Clustered Data")
            st.dataframe(df.head())

    else:
        st.error("Dataset must contain pickup_latitude and pickup_longitude columns")