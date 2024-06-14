import logging
import os
import faiss
import shutil
import sys
import json
import time
from tqdm import tqdm
import collections
import copy
import argparse
from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from llama_index.core import StorageContext, Settings, Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
import sys

sys.path.append('.')
from tasks.common import trace_langfuse
from tasks.kilt_utils import normalize_answer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_nodes(max_documents):
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["nba-datalake"]["wiki-documents"]
    num_docs = db.count_documents({})
    nodes = []
    for page in tqdm(db.find(), total=num_docs):
        if max_documents and len(nodes) > max_documents:
            break
        wikipedia_id = page["wikipedia_id"]
        wikipedia_title = page["wikipedia_title"]
        page_category = page["categories"]
        for paragraph_id, paragraph in enumerate(page["text"]):
            if paragraph_id == 0:  # the first paragraph is the title
                continue
            if paragraph.strip() == "":
                continue
            if paragraph.startswith("Section::::") or paragraph.startswith("BULLET::::"):
                continue
            node = TextNode(
                text=paragraph,
                metadata={
                    "wikipedia_id": wikipedia_id,
                    "wikipedia_title": wikipedia_title,
                    "categories": page_category,
                    "paragraph_id": paragraph_id,
                },
                excluded_llm_metadata_keys=["wikipedia_id", "wikipedia_title", "categories", "paragraph_id"],
                excluded_embed_metadata_keys=["wikipedia_id", "wikipedia_title", "categories", "paragraph_id"],
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
            nodes.append(node)
    return nodes


def create_index(persist_dir, index_type="default", max_documents=None):
    os.makedirs(persist_dir, exist_ok=True)

    nodes = get_nodes(max_documents)

    if index_type == "default":
        created_index = VectorStoreIndex(nodes, show_progress=True)
        created_index.storage_context.persist(persist_dir=persist_dir)
    elif index_type == "faiss":
        d = 1536
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        created_index = VectorStoreIndex(
            nodes, storage_context=storage_context, show_progress=True
        )
        created_index.storage_context.persist(persist_dir=persist_dir)
    elif index_type == "duckdb":
        raise NotImplementedError()


def get_chunk_index(
        llm, emb_model, index_type="default", max_documents=None,
):
    Settings.llm = OpenAI(temperature=0, model=llm)
    if emb_model.startswith("text-embedding"):
        Settings.embed_model = OpenAIEmbedding(model=emb_model, embed_batch_size=1000)
    else:
        Settings.embed_model = HuggingFaceEmbedding(emb_model)
    # Settings.chunk_size = chunk_size

    persist_dir = os.path.join("indices", 'paragraph_' + index_type + '_' + emb_model.replace('/', '--'))

    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        t0 = time.time()
        create_index(persist_dir, index_type, max_documents)
        print(f"Index created in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    if index_type == "default":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    elif "faiss" == index_type:
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    elif "duckdb" == index_type:
        vector_store = DuckDBVectorStore.from_local(os.path.join(persist_dir, "nba.duckdb"))
        index = VectorStoreIndex.from_vector_store(vector_store)

    print(f"Index loaded in {time.time() - t0:.2f} seconds")

    return index


@trace_langfuse(name="paragraph_discovery")
def get_responses(index, dataset, chunk_top_k) -> List[dict]:
    if index is None:  # BM25
        nodes = get_nodes(max_documents=None)
    else:
        nodes = list(index.docstore.docs.values())
    id2node = {node.id_: node for node in nodes}
    title2nodes = collections.defaultdict(list)
    for node in nodes:
        title2nodes[node.metadata['wikipedia_title']].append(node)

    all_response = []
    for d in dataset:
        titles = list(set(paragraph['wikipedia_title'] for paragraph in d['provenance_doc']['paragraphs']))
        if index is None:  # BM25
            retriever = BM25Retriever.from_defaults(
                nodes=[node for title in titles for node in title2nodes[title]],
                similarity_top_k=chunk_top_k,
            )
        else:
            retriever = VectorIndexRetriever(
                index,
                similarity_top_k=chunk_top_k,
                node_ids=[node.id_ for title in titles for node in title2nodes[title]],
                callback_manager=index._callback_manager,
                object_map=index._object_map,
            )
        engine = RetrieverQueryEngine.from_args(
            retriever,
            llm=Settings.llm,
            verbose=True,
        )

        response = engine.query(d["question"])
        d = copy.deepcopy(d)
        d['model_response'] = str(response)
        d['model_provenance'] = {'paragraphs': []}
        for node_id in response.metadata.keys():
            node = id2node[node_id]
            d['model_provenance']['paragraphs'].append({
                'wikipedia_title': node.metadata['wikipedia_title'],
                'paragraph_id': node.metadata['paragraph_id'],
                'text': node.text,
            })
        all_response.append(d)
    return all_response


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def r_precision(retrieved: list[str], relevant: list[str]) -> float:
    return precision_at_k(retrieved, relevant, len(relevant)) if relevant else 0.0


def evaluate(all_response: List[dict]) -> dict:
    res = {
        'metrics': {},
        'responses': []
    }
    ks = [1, 2, 3, 5, 10, 20]
    # max_k = max(ks)
    # assert max_k <= all(max_k <= len(d['model_provenance']['paragraphs']) for d in all_response)

    for d in all_response:
        d = copy.deepcopy(d)

        # Compute accuracy
        d['metric_accuracy'] = float(normalize_answer(d["answer"]) in normalize_answer(d["model_response"]))

        # Compute retrieval metrics
        retrieved = [(d1['wikipedia_title'], d1['paragraph_id']) for d1 in d['model_provenance']['paragraphs']]
        relevant = list(set((d1['wikipedia_title'], d1['paragraph_id']) for d1 in d['provenance_doc']['paragraphs']))
        d['metric_r_precision'] = r_precision(retrieved, relevant)
        for k in ks:
            d[f'metric_precision@{k}'] = precision_at_k(retrieved, relevant, k)
        for k in ks:
            d[f'metric_recall@{k}'] = recall_at_k(retrieved, relevant, k)

        res['responses'].append(d)

    metrics = ['accuracy', 'r_precision'] + [f'precision@{k}' for k in ks] + [f'recall@{k}' for k in ks]
    for metric in metrics:
        res['metrics'][metric] = sum(d[f'metric_{metric}'] for d in res['responses']) / len(res['responses'])
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="graph", choices=["graph", "doc", "table", "all"])
    parser.add_argument('--inputs', default=["benchmark/q_doc.json"], nargs="+")
    parser.add_argument('--output_dir', default='outputs/test_paragraph_discovery/')
    parser.add_argument('--overwrite', action="store_true")

    # parameters for mode=doc
    parser.add_argument('--llm', default="gpt-3.5-turbo")
    parser.add_argument('--chunk_top_k', default=20, type=int)
    # parser.add_argument('--chunk_size', default=512, type=int)
    parser.add_argument('--index_type', default="default", choices=["default", "faiss", "duckdb"])
    parser.add_argument('--retriever', default="BAAI/bge-base-en-v1.5")
    args = parser.parse_args()
    print(args)
    print()

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    response_output_path = os.path.join(args.output_dir, "responses.json")
    if not os.path.exists(response_output_path):
        if args.retriever.lower() == 'bm25':
            index = None
        else:
            # Get query engine
            index = get_chunk_index(
                llm=args.llm,
                emb_model=args.retriever,
                index_type=args.index_type,
                max_documents=None
            )
        # Load dataset
        dataset = []
        for path in args.inputs:
            with open(path) as f:
                dataset += json.load(f)

        # Run queries
        all_response = get_responses(index, dataset, chunk_top_k=args.chunk_top_k)
        with open(response_output_path, "w") as f:
            json.dump(all_response, f, indent=2)
        print(f'Responses saved to {response_output_path}')

    with open(response_output_path) as f:
        all_response = json.load(f)
    print(f'Loaded {len(all_response)} responses from {response_output_path}')

    # Evaluate and save metrics
    result = evaluate(all_response)
    for k, v in result['metrics'].items():
        print(f"{k}: {v:.4f}")
    result_output_path = os.path.join(args.output_dir, "result.json")
    with open(result_output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f'Results saved to {result_output_path}')


if __name__ == "__main__":
    main()
