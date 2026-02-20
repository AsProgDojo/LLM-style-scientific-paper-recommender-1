from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pgvector.psycopg import register_vector
from dotenv import load_dotenv, find_dotenv
from psycopg.rows import dict_row
import psycopg
import argparse
import os



MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)


sql = """
WITH candidates AS (
  SELECT
    c.paper_id,
    c.id AS chunk_id,
    c.bucket,
    c.section_index,
    c.chunk_index,
    c.path_string,
    c.title,
    c.chunk_text,
    (c.embedding <=> %s::vector)       AS dist,
    (1 - (c.embedding <=> %s::vector)) AS sim
  FROM paper_chunks c
  ORDER BY c.embedding <=> %s::vector
  LIMIT %s::int
),
paper_scores AS (
  SELECT paper_id, MAX(sim) AS paper_score
  FROM candidates
  GROUP BY paper_id
),
top_papers AS (
  SELECT
    paper_id,
    paper_score,
    ROW_NUMBER() OVER (ORDER BY paper_score DESC, paper_id) AS paper_rank
  FROM paper_scores
  ORDER BY paper_score DESC, paper_id
  LIMIT %s::int
),
evidence AS (
  SELECT
    c.*,
    ROW_NUMBER() OVER (PARTITION BY c.paper_id ORDER BY c.sim DESC) AS ev_rank
  FROM candidates c
)
SELECT
  tp.paper_rank,
  tp.paper_id,
  tp.paper_score,
  jsonb_agg(
    jsonb_build_object(
      'chunk_id', e.chunk_id,
      'bucket', e.bucket,
      'section_index', e.section_index,
      'chunk_index', e.chunk_index,
      'path_string', e.path_string,
      'title', e.title,
      'sim', e.sim,
      'text', e.chunk_text
    )
    ORDER BY e.sim DESC
  ) FILTER (WHERE e.ev_rank <= %s::int) AS evidence
FROM top_papers tp
LEFT JOIN evidence e ON e.paper_id = tp.paper_id
GROUP BY tp.paper_rank, tp.paper_id, tp.paper_score
ORDER BY tp.paper_rank;

"""


def embed_query(query : str):
    query_embedding = model.encode(query, convert_to_tensor=False)
    return query_embedding


if __name__ == '__main__':

    load_dotenv(find_dotenv())

    ap = argparse.ArgumentParser()
    ap.add_argument("--top-chunks", default=3, help="Enter the number of top chunks to sum by and use that score for ranking")
    ap.add_argument("--top-papers", default=5, help="Enter the number of papers to return")
    ap.add_argument("--evidence", default=3, help="Enter the number of evidence chunks to return with their respective top-ranked papers")
    args = ap.parse_args()

    top_chunks = args.top_chunks
    top_papers = args.top_papers
    evidence = args.evidence

    db_config = {
        'dbname': os.getenv("DB_NAME"),
        'user': os.getenv("DB_USER"),
        'password': os.getenv("DB_PASSWORD"),
        'host': os.getenv("DB_HOST"),
        'port': os.getenv("DB_PORT"),
    }

    print("For terminating the program type in \'out\'")
    query = input("Enter query:")

    with psycopg.connect(**db_config, options="-c search_path=paper_chunks,public") as conn:
        print("Connected to PostgreSQL database")
        register_vector(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            while query != 'out':
                query_embedding = embed_query(query)

                cur.execute(sql, (query_embedding, query_embedding, query_embedding, top_chunks, top_papers, evidence))
                rows = cur.fetchall()
                paper_id = rows[0]["paper_id"]
                paper_rank = rows[0]["paper_rank"]
                paper_score = rows[0]["paper_score"]
                print(f"Best paper has id: {paper_id}, rank: {paper_rank}, score: {paper_score}")

                query = input("Enter query:")

    print("Terminating program...")