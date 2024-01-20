from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings

class CharBertEmbeddings(Embeddings):
    def __init__(self, sentenceTransformerModel):
        """Initialize CharBertEmbeddings."""
        self.client = sentenceTransformerModel
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a SentenceTransformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a SentenceTransformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    #async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #    """Asynchronous Embed search docs."""
    #    return await run_in_executor(None, self.embed_documents, texts)


    #async def aembed_query(self, text: str) -> List[float]:
    #    """Asynchronous Embed query text."""
    #    return await run_in_executor(None, self.embed_query, text)