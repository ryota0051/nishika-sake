import faiss


class SimilaritySearcher:
    def __init__(self, embeddings):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, queries, k=20):
        assert (
            queries.shape[1] == self.dimension
        ), "Query dimensions should match embeddings dimension."
        faiss.normalize_L2(queries)
        D, index = self.index.search(queries, k)
        return D, index
