# LLM-style-scientific-paper-recommender-1
A semantic search and recommendation engine for scientific papers.
Users can type either keywords or a natural-language description (“talk to an LLM” style). The system retrieves and ranks the most relevant papers using a hybrid approach: lexical search (BM25) + dense vector embeddings, and then generates grounded explanations via RAG with citations to the matched passages.
