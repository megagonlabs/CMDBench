import logging
import os
import faiss
import shutil
import sys
import json
import time
from tqdm import tqdm
import pandas as pd
import copy
import argparse
from sql_metadata import Parser
from typing import Any, Dict, List, Optional
import psycopg2
from pymongo import MongoClient
from llama_index.core import StorageContext, Settings, Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI
import os
import openai
import pprint
from sqlalchemy import create_engine

from llama_index.core import SQLDatabase, Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
import sys

sys.path.append('.')
from tasks.common import trace_langfuse
from tasks.kilt_utils import normalize_answer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_abstract(paragraphs: List[str]) -> str:
    res = []
    for p in paragraphs:  # paragraphs[0] is the title
        if p.startswith('Section::::') or p.startswith('BULLET::::'):
            break
        res.append(p)
    return '\n'.join(res).strip()


def get_table_index(emb_model, index_type="default"):
    assert index_type == "default"
    if emb_model.startswith('text-embedding'):
        Settings.embed_model = OpenAIEmbedding(model=emb_model, embed_batch_size=1000)
    else:
        Settings.embed_model = HuggingFaceEmbedding(emb_model)
    persist_dir = os.path.join("indices", 'table_' + index_type + '_' + emb_model.replace('/', '--'))

    user = os.environ.get('PGUSER')
    db = 'nba'
    conn_str = f'postgresql+psycopg://{user}:{password}@localhost/{db}'
    schema = 'nba_wikisql'
    engine = create_engine(conn_str)
    sql_database = SQLDatabase(engine, schema=schema)
    table_node_mapping = SQLTableNodeMapping(sql_database)

    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        t0 = time.time()
        # connect the db
        connection = psycopg2.connect(f"host=localhost dbname=nba port=5432 user={user}")
        cursor = connection.cursor()

        # select query for table meta data
        cursor.execute("SELECT id, page_title, section_title, caption FROM metadata.nba_context")
        query_result = cursor.fetchall()

        # create df_meta
        df_meta = pd.DataFrame(query_result, columns=['id', 'page_title', 'section_title', 'caption'])
        df_meta = df_meta.set_index('id')
        df_meta.loc[['1-11734041-2']].to_json(orient='records').strip('[,]')

        # Execute the SQL query to fetch the whole table list
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema ='nba_wikisql' ORDER BY table_name;")
        id_list = cursor.fetchall()
        id_list = [id[0][2:].replace('_', '-') for id in id_list]  # change table name to 1-10015132-1

        cursor.close()

        context_str_dict = {id: df_meta.loc[[id]].to_json(orient='records').strip('[,]') for id in id_list}
        print(f'Fetched {len(context_str_dict)} table contexts')

        table_schema_objs = [
            SQLTableSchema(table_name='t_' + id.replace('-', '_'), context_str=context_str_dict[id])
            for id in id_list
        ]
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
            show_progress=True,
        )

        # persist to disk (no path provided will persist to the default path ./storage)
        obj_index.persist(persist_dir)
        print(f"Index created in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    index = ObjectIndex.from_persist_dir(persist_dir, table_node_mapping)
    print(f"Index loaded in {time.time() - t0:.2f} seconds")
    return index, engine, sql_database


def get_table_query_engine(
        llm, retriever, table_top_k,
):
    Settings.llm = OpenAI(temperature=0, model=llm)
    if retriever.lower() == "bm25":
        raise NotImplementedError()
    else:
        obj_index, engine, sql_database = get_table_index(emb_model=retriever)

        object_retriever = obj_index.as_retriever(similarity_top_k=table_top_k)
        query_engine = SQLTableRetrieverQueryEngine(
            sql_database, object_retriever, verbose=False
        )
    return query_engine, object_retriever


@trace_langfuse(name="table_discovery")
def get_responses(engine, table_retriever, dataset) -> List[dict]:
    all_response = []
    for d in dataset:
        d = copy.deepcopy(d)
        response = engine.query(d["question"])
        retrieved_tables = table_retriever.retrieve(d["question"])
        table_names = [table.table_name for table in retrieved_tables]
        d['model_response'] = str(response)
        d['model_provenance'] = {
            'tables': {
                'retrieved': table_names,
                'sql': response.metadata['sql_query'].replace("\n", " "),
                'sql_columns': Parser(response.metadata['sql_query']).columns,
                'sql_tables': Parser(response.metadata['sql_query']).tables,
            }
        }
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
    # assert max_k <= all(max_k <= len(d['model_provenance']['spans']) for d in all_response)

    for d in all_response:
        d = copy.deepcopy(d)

        # Compute accuracy
        d['metric_accuracy'] = float(normalize_answer(d["answer"]) in normalize_answer(d["model_response"]))

        # Compute retrieval metrics
        retrieved = [s[4:] for s in d['model_provenance']['tables']['retrieved']]  # "t_1_24856332_4" -> "24856332-4"
        relevant = [d['provenance_table']['table'][4:]]
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
    parser.add_argument('--inputs', default=["benchmark/q_table.json"], nargs="+")
    parser.add_argument('--output_dir', default='outputs/test_table_discovery/')
    parser.add_argument('--overwrite', action="store_true")

    # parameters for mode=doc
    parser.add_argument('--llm', default="gpt-3.5-turbo")
    parser.add_argument('--table_top_k', default=20, type=int)
    parser.add_argument('--retriever', default="BAAI/bge-base-en-v1.5")
    args = parser.parse_args()
    print(args)
    print()

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    response_output_path = os.path.join(args.output_dir, "responses.json")
    if not os.path.exists(response_output_path):
        # Get query engine
        engine, table_retriever = get_table_query_engine(
            llm=args.llm,
            retriever=args.retriever,
            table_top_k=args.table_top_k,
        )
        # Load dataset
        dataset = []
        for path in args.inputs:
            with open(path) as f:
                dataset += json.load(f)
        # Run queries
        all_response = get_responses(engine, table_retriever, dataset)
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
