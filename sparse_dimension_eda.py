import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

from local_loader import load_and_chunk_json


def analyze_and_visualize_text_vectors_tfidf(texts, n_components=3, max_eigenvalues=140):
    """
    使用TF-IDF分析文本向量的特征值并可视化，随后按指定的主成分数降维并可视化。

    参数:
    - texts (list of str): 包含文本内容的列表
    - n_components (int): PCA的降维目标
    - max_eigenvalues (int): 只展示前 max_eigenvalues 个特征值

    返回:
    - None: 显示PCA的特征值柱状图和降维后的散点图
    """
    # 使用TfidfVectorizer将文本转换为向量表示
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(texts).toarray()

    # 计算所有主成分的特征值并绘制特征值柱状图
    pca_full = PCA()
    pca_full.fit(text_vectors)
    explained_variances = pca_full.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, min(max_eigenvalues, len(explained_variances)) + 1),
            explained_variances[:max_eigenvalues], color='blue')
    plt.title("Eigenvalues from TF-IDF (Ordered)")
    plt.xlabel("Index of Eigenvalue")
    plt.ylabel("Eigenvalue Magnitude")
    plt.savefig("visualization/sparse/visualization_tfidf_eigenvalue_bar.png", format='png', dpi=300)
    print("TF-IDF eigenvalue bar chart saved as 'visualization_tfidf_eigenvalue_bar.png'.")

    # 使用指定的主成分数进行降维
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(text_vectors)

    # 绘制降维后的散点图
    plt.figure(figsize=(10, 7))
    if n_components == 2:
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)
        plt.title("PCA Visualization of Text Vectors (TF-IDF) - 2D")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], alpha=0.5)
        plt.title("PCA Visualization of Text Vectors (TF-IDF) - 3D")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
    plt.savefig("visualization/sparse/visualization_tfidf_dimension.png", format='png', dpi=300)
    print("TF-IDF dimension-reduced visualization saved as 'visualization_tfidf_dimension.png'.")


def analyze_and_visualize_text_vectors_bm25(texts, n_components=3, max_eigenvalues=140):
    """
    使用BM25分析文本向量的特征值并可视化，随后按指定的主成分数降维并可视化。

    参数:
    - texts (list of str): 包含文本内容的列表
    - n_components (int): PCA的降维目标
    - max_eigenvalues (int): 只展示前 max_eigenvalues 个特征值

    返回:
    - None: 显示PCA的特征值柱状图和降维后的散点图
    """
    # 使用BM25将文本转换为向量表示
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    text_vectors = np.array([bm25.get_scores(query) for query in tokenized_texts])

    # 计算所有主成分的特征值并绘制特征值柱状图
    pca_full = PCA()
    pca_full.fit(text_vectors)
    explained_variances = pca_full.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, min(max_eigenvalues, len(explained_variances)) + 1),
            explained_variances[:max_eigenvalues], color='blue')
    plt.title("Eigenvalues from BM25 (Ordered)")
    plt.xlabel("Index of Eigenvalue")
    plt.ylabel("Eigenvalue Magnitude")
    plt.savefig("visualization/sparse/visualization_bm25_eigenvalue_bar.png", format='png', dpi=300)
    print("BM25 eigenvalue bar chart saved as 'visualization_bm25_eigenvalue_bar.png'.")

    # 使用指定的主成分数进行降维
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(text_vectors)

    # 绘制降维后的散点图
    plt.figure(figsize=(10, 7))
    if n_components == 2:
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)
        plt.title("PCA Visualization of Text Vectors (BM25) - 2D")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], alpha=0.5)
        plt.title("PCA Visualization of Text Vectors (BM25) - 3D")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
    plt.savefig("visualization/sparse/visualization_bm25_dimension.png", format='png', dpi=300)
    print("BM25 dimension-reduced visualization saved as 'visualization_bm25_dimension.png'.")


def main():
    # 使用示例
    mods = load_and_chunk_json('data/mods22_23.json')
    texts = [t.page_content for t in mods]  # 你的文本列表

    # 分别使用TF-IDF和BM25分析和可视化
    # analyze_and_visualize_text_vectors_tfidf(texts)
    analyze_and_visualize_text_vectors_bm25(texts)


if __name__ == "__main__":
    main()
