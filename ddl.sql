CREATE SCHEMA paper_chunks;

SET search_path TO paper_chunks, public;

CREATE EXTENSION vector SCHEMA paper_chunks;

CREATE TABLE paper_chunks.paper_chunks (
  id           bigserial PRIMARY KEY,
  paper_id     text NOT NULL,
  bucket       text NOT NULL,
  section_index int NOT NULL,
  chunk_index   int NOT NULL,
  path_string  text,
  title        text,
  chunk_text   text NOT NULL,
  embedding    vector(768) NOT NULL,

  CONSTRAINT uq_paper_chunks UNIQUE (paper_id, section_index, chunk_index)
);


CREATE INDEX paper_chunks_embedding_hnsw
ON paper_chunks
USING hnsw (embedding vector_cosine_ops);