import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Customer Segmentation Dashboard", layout="wide")

st.title("🚀 AI-Powered Customer Segmentation Dashboard")

# Upload
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload your dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ Data Loaded Successfully")

# Preview
with st.expander("📊 View Raw Data"):
    st.dataframe(df.head())

# Sidebar
st.sidebar.header("⚙️ Settings")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

features = st.sidebar.multiselect(
    "Select Features for Clustering",
    numeric_cols,
    default=numeric_cols[1:4]
)

scale = st.sidebar.checkbox("Apply Scaling", True)
k = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)

if len(features) < 2:
    st.error("Select at least 2 features")
    st.stop()

X = df[features]

# Scaling
if scale:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📉 Elbow Method", "📈 Visualization", "📥 Download"])

# ---------------- DASHBOARD ----------------
with tab1:
    st.subheader("📊 Business Dashboard")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_scaled)

    cluster_summary = df.groupby("Cluster")[features].mean()

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Clusters Created", k)
    col3.metric("Avg Income", int(df[features[0]].mean()))

    st.subheader("📊 Cluster Summary")
    st.dataframe(cluster_summary)

    # ---------------- AI INSIGHTS ----------------
    st.subheader("🤖 AI Insights")

    insights = []

    for i in cluster_summary.index:
        avg_val = cluster_summary.loc[i][features[0]]

        if avg_val > cluster_summary[features[0]].mean():
            insights.append(f"Cluster {i} contains high-value customers. Focus premium marketing.")
        else:
            insights.append(f"Cluster {i} contains lower-value customers. Use discounts or engagement strategies.")

    for ins in insights:
        st.write("•", ins)

# ---------------- ELBOW ----------------
with tab2:
    st.subheader("📉 Elbow Method")

    inertia = []
    K_range = range(1, 11)

    for k_val in K_range:
        model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertia.append(model.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")

    st.pyplot(fig)

# ---------------- VISUALIZATION ----------------
with tab3:
    st.subheader("📈 Customer Segments")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df[features[0]], df[features[1]], c=df["Cluster"])

    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])

    st.pyplot(fig2)

    # 3D
    if len(features) >= 3:
        from mpl_toolkits.mplot3d import Axes3D

        st.subheader("🌐 3D View")
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection='3d')

        ax3.scatter(
            df[features[0]],
            df[features[1]],
            df[features[2]],
            c=df["Cluster"]
        )

        st.pyplot(fig3)

    # ---------------- MAP ----------------
    if "Latitude" in df.columns and "Longitude" in df.columns:
        st.subheader("🌍 Geographic Segmentation")
        st.map(df[["Latitude", "Longitude"]])

# ---------------- DOWNLOAD ----------------
with tab4:
    st.subheader("📥 Download Results")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Clustered Data",
        csv,
        "segmented_customers.csv",
        "text/csv"
    )

st.success("🎉 Analysis Completed Successfully!")