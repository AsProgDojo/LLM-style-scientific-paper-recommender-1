from pathlib import Path
from typing import List

import jsonlines
import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)


def get_chunk_embedding(text_chunks : List[str]) -> np.ndarray:
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


def process_jsonl_file(object_file : str):
    with jsonlines.open(object_file) as reader:
        for json_object in reader:
            abstract = json_object["abstract"]
            text_chunks = get_chunks(abstract, json_object["title"])
            chunk_embedding = get_chunk_embedding(text_chunks)
            print(chunk_embedding)

if __name__ == '__main__':
    process_jsonl_file('./output/oa_comm/txt/all/corpus.jsonl')