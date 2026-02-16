from sentence_transformers import SentenceTransformer
from typing import List
from transformers import AutoTokenizer
from pgvector.psycopg import register_vector
from dotenv import load_dotenv, find_dotenv
import psycopg
import jsonlines
import numpy as np
import os


MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)


def get_chunk_embedding(text_chunks : List[str]) -> np.ndarray:
    """
    Return chunk embedding/s of passed text chunk/s
    """
    embeddings = model.encode(text_chunks, convert_to_tensor=False)
    return embeddings

def get_chunks(section_text : str, header : str, window_wp : int = 348, overlap_wp : int = 70):
    """
    Return list of strings where each string represents chunk of text which is of maximum length of 350 BERT WordPieces
    """
    header_wp = 0
    header_prefix = ''
    if header:
        header_wp = len(tokenizer(header, add_special_tokens=False)['input_ids'])
        if header_wp > window_wp:
            raise ValueError(f"Header is too long. Header {header_wp} WP > Window {window_wp} WP")
        header_prefix = header.strip() + '\n\n'

    body_window_wp = window_wp - header_wp
    step_wp = max(1, body_window_wp - overlap_wp)

    text_tokens = tokenizer(section_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = text_tokens['offset_mapping']
    n = len(offsets)

    chunks = []
    start_token = 0
    while start_token < n:
        end_token = min(start_token + body_window_wp, n)

        sentence_begin = offsets[start_token][0]
        sentence_end = offsets[end_token - 1][1]

        body_part = section_text[sentence_begin:sentence_end].strip()
        chunk_text = header_prefix + body_part if header_prefix else body_part
        chunks.append(chunk_text)

        if end_token == n:
            break
        start_token += step_wp

    return chunks

def insert_data(paper_id : str, bucket : str, section_index : int, chunk_index : int, path_string : str, title : str, chunk_text : str, embedding : np.ndarray) -> None:
    """
    Insert text chunk into PostgreSQL database
    """

    print(os.getenv('DB_PASSWORD'))
    print(os.getenv('DB_HOST'))
    db_config = {
        'dbname' : os.getenv("DB_NAME"),
        'user' : os.getenv("DB_USER"),
        'password' : os.getenv("DB_PASSWORD"),
        'host' : os.getenv("DB_HOST"),
        'port' : os.getenv("DB_PORT"),
    }

    try:
        with psycopg.connect(**db_config, options="-c search_path=paper_chunks,public") as conn:
            print("Connected to PostgreSQL database")

            register_vector(conn) # enables the use of the PostgreSQL vector data type within the Python database connection
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO paper_chunks
                        (paper_id, bucket, section_index, chunk_index, path_string, title, chunk_text, embedding)
                    VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (paper_id, section_index, chunk_index)
                    DO UPDATE SET
                            bucket      = EXCLUDED.bucket,
                            path_string = EXCLUDED.path_string,
                            title       = EXCLUDED.title,
                            chunk_text  = EXCLUDED.chunk_text,
                            embedding   = EXCLUDED.embedding
                    """,
                    (paper_id, bucket, section_index, chunk_index, path_string, title, chunk_text, embedding)
                )
        print("Inserted a row!")
    except Exception as e:
        print(f"Error connecting to the database: {e}")

def process_jsonl_file(object_file : str):
    """
    Process all the JSON objects inside object_file which represents JSONL
    """
    with jsonlines.open(object_file) as reader:
        for json_object in reader:
            paper_id = json_object['paper_id']
            title = json_object['title']

            # Inserting data for abstract
            abstract = json_object["abstract"]
            bucket = 'abstract'
            section_index = -1
            text_chunks = get_chunks(abstract, json_object["title"])
            for i, text_chunk in enumerate(text_chunks):
                embedding = get_chunk_embedding(text_chunk)
                chunk_index = i
                insert_data(paper_id, bucket, section_index, chunk_index, '', title, text_chunk, embedding)
                print("Paper ID:", paper_id, "Section Index: ", section_index,"-- Chunk index", chunk_index)

            # Inserting data for sections
            sections = json_object["sections"]
            for section in sections:
                section_index = section["section_index"]
                path_string = section["path_string"]
                title = section["title"]
                bucket = section["bucket"]
                section_text = section["text"]
                text_chunks = get_chunks(section_text, path_string)
                for i, chunk_text in enumerate(text_chunks):
                    embedding = get_chunk_embedding(chunk_text)
                    chunk_index = i
                    insert_data(paper_id, bucket, section_index, chunk_index, path_string, title, chunk_text, embedding)
                    print("Paper ID:", paper_id, "Section Index: ", section_index, "-- Chunk index", chunk_index)


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    process_jsonl_file('./output/oa_comm/txt/all/corpus.jsonl')