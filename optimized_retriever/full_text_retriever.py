# _*_ coding: utf-8 _*_
__author__ = 'yuntwo'
__date__ = '2024/11/04 16:41:40'

from typing import Any, Dict, Iterable, List, Optional
import yake
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever
from sklearn.decomposition import PCA, TruncatedSVD
from pydantic import Field


class KeywordTFIDFRetriever(TFIDFRetriever):

    def __init__(self, vectorizer=None, docs=None, tfidf_array=None, **kwargs):
        # 调用父类的构造函数并传递所有参数
        super(TFIDFRetriever, self).__init__(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array, **kwargs)

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks=None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        # 提取关键字并增强查询
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            top=10,
            dedupLim=0.9  # 去重程度，可根据需要调整
        )
        # 提取关键字的字符串部分
        keywords = [keyword for keyword, _ in kw_extractor.extract_keywords(query)]
        print("Extracted Keywords:", keywords)

        enhanced_query = " ".join([query] + keywords)

        print("Enhanced Query:", enhanced_query)

        # 创建 TFIDFRetriever 实例并调用其 get_relevant_documents
        base_retriever = TFIDFRetriever(vectorizer=self.vectorizer, docs=self.docs, tfidf_array=self.tfidf_array)
        return base_retriever.get_relevant_documents(
            enhanced_query,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            **kwargs
        )


class TruncatedSVDTFIDFRetriever(TFIDFRetriever):

    n_components: int = Field(default=10)

    truncatedSVD: Optional[TruncatedSVD] = None

    # 实例化时的操作与父类一致，只是额外进行降维加上降维参数
    def __init__(self, vectorizer=None, docs=None, tfidf_array=None, n_components=10, **kwargs):
        # 调用父类构造函数
        super().__init__(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array, **kwargs)
        self.n_components = n_components
        self.truncatedSVD = TruncatedSVD(n_components=self.n_components)
        # 应用 PCA 降维并覆盖 tfidf_array
        # Original 37242 Sparse dimension
        self.tfidf_array = self.truncatedSVD.fit_transform(self.tfidf_array)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        tfidf_params: Optional[Dict[str, Any]] = None,
        n_components: int = 10,
        **kwargs: Any,
    ) -> "TruncatedSVDTFIDFRetriever":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`."
            )

        tfidf_params = tfidf_params or {}
        vectorizer = TfidfVectorizer(**tfidf_params)
        tfidf_array = vectorizer.fit_transform(texts)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

        instance = cls(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array, n_components=n_components, **kwargs)
        return instance

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks=None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        # 使用 vectorizer 生成查询的 TF-IDF 向量并应用 PCA
        query_vector = self.truncatedSVD.transform(self.vectorizer.transform([query]).toarray())
        # 计算查询与降维后的 TF-IDF 向量的相似度
        scores = self._calculate_similarity(query_vector)
        top_indices = scores.argsort()[::-1]
        return [self.docs[i] for i in top_indices]

    def _calculate_similarity(self, query_vector):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(query_vector, self.tfidf_array).flatten()
