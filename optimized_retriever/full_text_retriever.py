# _*_ coding: utf-8 _*_
__author__ = 'yuntwo'
__date__ = '2024/11/04 16:41:40'

from typing import List, Optional, Dict, Any
from langchain_community.retrievers import TFIDFRetriever
import yake
from langchain_core.documents import Document


class CustomTFIDFRetriever(TFIDFRetriever):

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

