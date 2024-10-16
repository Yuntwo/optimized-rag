# _*_ coding: utf-8 _*_
__author__ = 'yuntwo'
__date__ = '2024/10/12 17:59:28'

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from vector_store import EmbeddingProxy  # Adjust this to your actual import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import seaborn as sns



def get_overview(db):
    # get record count
    record_count = db._collection.count()
    print(f"Total number of records in the Vector DB: {record_count}")

    # 1536 dimensions for each embedding
    records = db._collection.get(limit=2, include=['embeddings', 'documents', 'metadatas'])
    print(records)


def get_document_by_id(db, document_id):
    records = db.get(ids=[document_id], include=['documents', 'metadatas'])

    if records['documents']:
        print(records['documents'][0])
        print(records['metadatas'][0])


def plot_distance_matrix(db, num_samples=500):
    """
    Calculate and visualize the pairwise distance matrix for high-dimensional embeddings.

    Args:
    - embeddings: A numpy array of shape (n_samples, n_features), which are the high-dimensional vectors.
    """
    # Extract embeddings and metadata
    records = db._collection.get(limit=num_samples, include=['embeddings'])

    embeddings = records['embeddings']

    if not embeddings:
        print("No embeddings found.")
        return
    # Ensure the embeddings are in numpy array format
    embeddings = np.array(embeddings)

    # Calculate the pairwise distance matrix using Euclidean distance
    dist_matrix = squareform(pdist(embeddings, metric='euclidean'))

    # Plot the heatmap of the distance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap='coolwarm')
    plt.title('Pairwise Distance Matrix')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.savefig("data/visualization.png", format='png', dpi=300)
    print(f"3D visualization saved as 'visualization.png'.")


def explore_embedding_manifold(db, method='tsne', num_samples=500):
    """
    Extract embeddings from the Chroma DB and visualize using t-SNE or PCA.

    Args:
    - db: Chroma vector DB instance.
    - method: The method to use for dimensionality reduction ('tsne' or 'pca').
    - num_samples: Number of samples to visualize (default is 500).
    """
    # Extract embeddings and metadata
    records = db._collection.get(limit=num_samples, include=['embeddings', 'documents', 'metadatas'])

    embeddings = records['embeddings']

    if not embeddings:
        print("No embeddings found.")
        return

    embeddings = np.array(embeddings)

    # Apply dimensionality reduction (t-SNE or PCA)
    if method == 'tsne':
        tsne = TSNE(n_components=3, perplexity=30, n_iter=300)  # 3D t-SNE
        reduced_embeddings = tsne.fit_transform(embeddings)
        title = 't-SNE Embedding Manifold (3D)'
    elif method == 'pca':
        pca = PCA(n_components=3)  # 3D PCA
        reduced_embeddings = pca.fit_transform(embeddings)
        title = 'PCA Embedding Manifold (3D)'
    else:
        print("Invalid method. Use 'tsne' or 'pca'.")
        return

    # Visualize the reduced embeddings in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], s=10, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    # Save the plot to a file
    plt.savefig("data/visualization.png", format='png', dpi=300)
    print(f"3D visualization saved as 'visualization.png'.")


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    collection_name = "chroma"
    # Initialize embedding proxy and Chroma DB
    proxy_embeddings = EmbeddingProxy(embeddings)
    db = Chroma(collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", collection_name))
    # get_overview(db)
    # get_document_by_id(db, "CS5344")
    # explore_embedding_manifold(db, method='tsne', num_samples=db._collection.count())
    explore_embedding_manifold(db, method='pca', num_samples=db._collection.count())
    # plot_distance_matrix(db, num_samples=db._collection.count())
    # plot_distance_matrix(db, num_samples=db._collection.count() / 4)

if __name__ == "__main__":
    main()
