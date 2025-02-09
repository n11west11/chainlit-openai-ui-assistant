"""
This function searches a page and returns a part of the page that most closely matches the search term.
"""

import chainlit as cl
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores.faiss import FAISS


underlying_embeddings = OpenAIEmbeddings()

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)


async def search_page_tool(search_term):
    """
    Searches a page and returns a part of the page that most closely matches the search term.
    """
    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=2000, chunk_overlap=50
    )
    search_page = cl.user_session.get("page")
    search_page_html = await search_page.content()
    html_docs = html_splitter.create_documents([search_page_html])

    us_vector_store = FAISS.from_documents(html_docs, cached_embedder)

    search_results = us_vector_store.similarity_search(search_term, k=1)
    return search_results[0].page_content


def open_ai_representation() -> dict:
    """
    Returns a dictionary representation of the function that can be used in an OpenAI assistant.
    """
    return {
        "name": "search_page_tool",
        "description": "Searches a page and returns a part of the page that most closely matches the search term.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": ["search_term"],
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The term to search for within the page",
                },
            },
            "additionalProperties": False,
        },
    }


if __name__ == "__main__":
    from langchain_community.document_loaders import AsyncHtmlLoader

    URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    SEARCH_TERM = "Python"
    loader = AsyncHtmlLoader([URL])
    html = loader.load()[0].page_content
    result = search_page_tool(SEARCH_TERM)
    print(result)
