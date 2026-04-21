import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Review Clustering App", layout="wide")

st.title("🛒 E-commerce Review Clustering (Unsupervised ML)")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload Reviews CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload your dataset")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Check column
if "Review_Text" not in df.columns:
    st.error("CSV must contain 'Review_Text' column")
    st.stop()

# Sidebar settings
st.sidebar.header("⚙️ Settings")

k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
max_features = st.sidebar.slider("Max TF-IDF Features", 100, 2000, 500)

# -----------------------------
# TEXT PROCESSING
# -----------------------------
st.subheader("🧠 Text Vectorization (TF-IDF)")

vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
X = vectorizer.fit_transform(df["Review_Text"])

st.write("TF-IDF Matrix Shape:", X.shape)

# -----------------------------
# ELBOW METHOD
# -----------------------------
st.subheader("📉 Elbow Method")

inertia = []
K_range = range(1, 11)

for k_val in K_range:
    model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(K_range, inertia, marker='o')
ax1.set_xlabel("K")
ax1.set_ylabel("Inertia")

st.pyplot(fig1)

# -----------------------------
# CLUSTERING
# -----------------------------
st.subheader("📌 Applying K-Means")

model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

st.dataframe(df.head())

# -----------------------------
# TOP KEYWORDS PER CLUSTER
# -----------------------------
st.subheader("🔑 Top Keywords per Cluster")

terms = vectorizer.get_feature_names_out()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

for i in range(k):
    st.write(f"Cluster {i}:")
    keywords = [terms[ind] for ind in order_centroids[i, :10]]
    st.write(", ".join(keywords))

# -----------------------------
# BASIC SENTIMENT INSIGHT
# -----------------------------
st.subheader("📊 Sentiment Insight (using Rating)")

if "Rating" in df.columns:
    sentiment = df.groupby("Cluster")["Rating"].mean()
    st.bar_chart(sentiment)
else:
    st.info("No Rating column found")

# -----------------------------
# CLUSTER DISTRIBUTION
# -----------------------------
st.subheader("👥 Cluster Distribution")
st.bar_chart(df["Cluster"].value_counts())

# -----------------------------
# DOWNLOAD RESULTS
# -----------------------------
st.subheader("⬇️ Download Clustered Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Results",
    csv,
    "clustered_reviews.csv",
    "text/csv"
)

st.success("🎉 Review Clustering Completed!")