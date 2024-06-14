import logging
import os
import faiss
import shutil
import sys
import json
import time
from tqdm import tqdm
import copy
import argparse
from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import StorageContext, Settings, Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
import sys

sys.path.append('.')
from tasks.common import trace_langfuse
from tasks.kilt_utils import normalize_answer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class OracleRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
            self,
            q2nodes: Dict[str, List[TextNode]],
    ) -> None:
        """Init params."""
        super().__init__()

        self._q2nodes = copy.deepcopy(q2nodes)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        query_str = query_bundle.query_str
        nodes = self._q2nodes[query_str]
        return [NodeWithScore(node=node, score=1.0) for node in nodes]


def get_q2nodes(data):
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["nba-datalake"]["wiki-documents"]
    q2nodes = {}
    for d in tqdm(data):
        q2nodes[d['question']] = []
        provenance = list(set((p['wikipedia_title'], p['paragraph_id']) for p in d['provenance_doc']['paragraphs']))
        for title, para_id in provenance:
            doc = db.find_one({"wikipedia_title": title})
            text = doc['text'][para_id]
            node = TextNode(
                text=text,
                metadata={
                    "wikipedia_id": doc['wikipedia_id'],
                    "wikipedia_title": title,
                    "categories": doc['categories'],
                    "paragraph_id": para_id,
                },
                excluded_llm_metadata_keys=["wikipedia_id", "categories", "paragraph_id"],
                excluded_embed_metadata_keys=["wikipedia_id", "categories", "paragraph_id"],
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
            q2nodes[d['question']].append(node)
    return q2nodes


def get_doc_query_engine(
        llm, data
):
    Settings.llm = OpenAI(temperature=0, model=llm)
    q2nodes = get_q2nodes(data)
    retriever = OracleRetriever(q2nodes=q2nodes)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
    )
    return query_engine


@trace_langfuse(name="doc_discovery")
def get_responses(engine, dataset) -> List[dict]:
    all_response = []
    for d in dataset:
        response = engine.query(d["question"])
        d = copy.deepcopy(d)
        d['model_response'] = str(response)
        d['model_provenance'] = {
            'paragraphs': [{
                'wikipedia_title': d1['wikipedia_title'],
                'paragraph_id': d1['paragraph_id'],
            } for d1 in response.metadata.values()]
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

    r_precision_by_category = {
        'single': [],
        'multi': [],
    }
    for d in res['responses']:
        relevant = list(set(span['wikipedia_title'] for span in d['provenance_doc']['paragraphs']))
        if len(relevant) == 1:
            r_precision_by_category['single'].append(d['metric_r_precision'])
        else:
            r_precision_by_category['multi'].append(d['metric_r_precision'])
    res['metrics']['r_precision_single'] = sum(r_precision_by_category['single']) / len(
        r_precision_by_category['single'])
    res['metrics']['r_precision_multi'] = sum(r_precision_by_category['multi']) / len(r_precision_by_category['multi'])
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="graph", choices=["graph", "doc", "table", "all"])
    parser.add_argument('--inputs', default=["benchmark/test_qa/q_doc.json"], nargs="+")
    parser.add_argument('--output_dir', default='outputs/test_answer_gen_doc/')
    parser.add_argument('--overwrite', action="store_true")

    parser.add_argument('--llm', default="gpt-3.5-turbo")
    args = parser.parse_args()
    print(args)
    print()

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    response_output_path = os.path.join(args.output_dir, "responses.json")
    if not os.path.exists(response_output_path):
        # Load dataset
        dataset = []
        for path in args.inputs:
            with open(path) as f:
                dataset += json.load(f)

        # Get query engine
        engine = get_doc_query_engine(
            llm=args.llm,
            data=dataset
        )

        # Run queries
        all_response = get_responses(engine, dataset)
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
