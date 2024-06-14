from llama_index.core import StorageContext, Settings, QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai_like import OpenAILike
import logging
import os
import json
from neo4j import GraphDatabase
import copy
import shutil
import argparse
import warnings
from typing import Any, Dict, List, Optional, Union, Sequence
import re
import sys

sys.path.append('.')
from tasks.common import trace_langfuse
from tasks.kilt_utils import normalize_answer

logger = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

GRAPH_QUERY_SYNTHESIS_TMPL = """
Given a question and the schema of a Neo4j knowledge graph, create a Cypher query that retrieve either the answer or information that can be used to infer the answer.

Here are the requirements:
- The Cypher query should be compatible with the graph schema. In the graph schema you will be provided with the node properties for each type of node in the graph, the relationship properties for each type of relationship, as well as all unique relationship schemas.
- The Cypher query should NOT use the path syntax (e.g., `MATCH path = ()-[*1..2]-() RETURN path`).
- Always do case-insensitive matching in the Cypher query.
- All nodes and relationships in the MATCH clause should be assigned a variable even if the variable is not used in the RETURN clause (e.g. use `[r:playsFor]` instead of `[:playsFor]`).
- Output the Cyper query directly without ```. Do not generate explanation or other additional output.

Question: {query_str}

Graph Schema: {schema}
""".strip()
GRAPH_QUERY_SYNTHESIS_PROMPT = PromptTemplate(GRAPH_QUERY_SYNTHESIS_TMPL)


class KnowledgeGraphQueryEngineWithProvenance(KnowledgeGraphQueryEngine):
    def generate_query(self, query_str: str) -> str:
        """Generate a Graph Store Query from a query bundle."""
        # Get the query engine query string
        graph_store_query: str = self._llm.predict(
            self._graph_query_synthesis_prompt,
            query_str=query_str,
            schema=self._graph_schema,
        )
        match = re.search(r'```(cypher)?[\s]*(.*?)[\s]*```', graph_store_query, flags=re.DOTALL)
        if match:
            graph_store_query = match.group(2)

        return graph_store_query

    def _get_evidence_subgraph(self, cypher):
        graph_store = self.graph_store

        node_ids = set()
        edge_ids = set()
        with graph_store._driver.session(database=graph_store._database) as session:
            cypher = re.sub(r"\bRETURN\b.*$", "RETURN *", cypher, flags=re.IGNORECASE)
            result = session.run(cypher)
            result = list(result)
            for record in result:
                for key, value in record.items():
                    if hasattr(value, "id"):
                        if hasattr(value, "start_node"):
                            node_ids.add(value.start_node.id)
                            node_ids.add(value.end_node.id)
                            edge_ids.add(value.id)
                        else:
                            node_ids.add(value.id)
        return {"cypher": cypher, "node_ids": list(node_ids), "edge_ids": list(edge_ids)}

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        try:
            nodes = super()._retrieve(query_bundle)
            cypher = nodes[0].metadata['graph_store_query']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subgraph = self._get_evidence_subgraph(cypher)
            nodes[0].metadata['node_ids'] = subgraph['node_ids']
            nodes[0].metadata['edge_ids'] = subgraph['edge_ids']
            nodes[0].node.excluded_llm_metadata_keys = list(nodes[0].metadata.keys())
        except Exception as e:
            logger.error(f"Error executing query")
            nodes = []
        return nodes


def get_graph_query_engine(llm, verbose=False):
    if llm.startswith("gpt"):
        Settings.llm = OpenAI(temperature=0, model=llm)
    else:
        Settings.llm = OpenAILike(
            model=llm,
            is_chat_model=True,
            api_base="http://localhost:9600/v1",
            context_window=3072,
            max_new_tokens=1024,
            temperature=0
        )
        Settings.context_window = 3072
        Settings.num_output = 1024

    graph_store = Neo4jGraphStore(
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD"),
        url="bolt://localhost:7687",
        database="neo4j",
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    query_engine = KnowledgeGraphQueryEngineWithProvenance(
        storage_context=storage_context,
        verbose=verbose,
        graph_query_synthesis_prompt=GRAPH_QUERY_SYNTHESIS_PROMPT,
    )
    return query_engine


@trace_langfuse(name="nl2cypher")
def get_all_responses(engine, dataset) -> List[dict]:
    all_response = []
    for d in dataset:
        d = copy.deepcopy(d)
        response = engine.query(d["question"])
        d['model_response'] = str(response)
        if response.metadata is None or len(response.metadata) == 0:
            d['model_provenance'] = {
                'graph': {
                    'cypher': None,
                    'graph_str': None,
                    'node_ids': [],
                    'edge_ids': [],
                }
            }
        else:
            node_metatdata = list(response.metadata.values())[0]
            d['model_provenance'] = {
                'graph': {
                    'cypher': node_metatdata['graph_store_query'],
                    'graph_str': str(node_metatdata['graph_store_response']),
                    'node_ids': node_metatdata['node_ids'],
                    'edge_ids': node_metatdata['edge_ids'],
                }
            }
        all_response.append(d)
    return all_response


def get_ground_truth_subgraphs(all_response):
    cyphers = []
    for d in all_response:
        cyphers.append(d['provenance_graph']['cypher'])
    driver = GraphDatabase.driver(
        'bolt://localhost:7687',
        auth=(os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD'])
    )
    res = []
    with driver.session() as session:
        for cypher in cyphers:
            node_ids = set()
            edge_ids = set()
            cypher = re.sub(r"\bRETURN\b.*$", "RETURN *", cypher, flags=re.IGNORECASE)
            result = session.run(cypher)
            result = list(result)
            for record in result:
                for key, value in record.items():
                    if hasattr(value, "id"):
                        if hasattr(value, "start_node"):
                            node_ids.add(value.start_node.id)
                            node_ids.add(value.end_node.id)
                            edge_ids.add(value.id)
                        else:
                            node_ids.add(value.id)
            res.append({"cypher": cypher, "node_ids": list(node_ids), "edge_ids": list(edge_ids)})
    assert len(res) == len(all_response), f"len(res)={len(res)}, len(all_response)={len(all_response)}"
    return res


def evaluate(all_response: List[dict]) -> dict:
    res = {
        'metrics': {
            'accuracy': None,
            'node_p': None,
            'node_r': None,
            'node_f1': None,
            'edge_p': None,
            'edge_r': None,
            'edge_f1': None,
        },
        'responses': []
    }

    ground_truth_subgraphs = get_ground_truth_subgraphs(all_response)

    for i, d in enumerate(all_response):
        d = copy.deepcopy(d)

        # Compute accuracy
        d['metric_accuracy'] = float(normalize_answer(d["answer"]) in normalize_answer(d["model_response"]))

        # Compute retrieval metrics
        nodes_pred = set(d['model_provenance']['graph']['node_ids'])
        nodes_true = set(ground_truth_subgraphs[i]['node_ids'])
        assert len(nodes_true) > 0
        d['metric_node_p'] = len(nodes_pred & nodes_true) / len(nodes_pred) if nodes_pred else 0.
        d['metric_node_r'] = len(nodes_pred & nodes_true) / len(nodes_true) if nodes_true else 0.
        edges_pred = set(d['model_provenance']['graph']['edge_ids'])
        edges_true = set(ground_truth_subgraphs[i]['edge_ids'])
        d['metric_edge_p'] = len(edges_pred & edges_true) / len(edges_pred) if edges_pred else 0.
        d['metric_edge_r'] = len(edges_pred & edges_true) / len(edges_true) if edges_true else 0.
        res['responses'].append(d)

    metrics = ['accuracy', 'node_p', 'node_r', 'edge_p', 'edge_r']
    for metric in metrics:
        res['metrics'][metric] = sum(d[f'metric_{metric}'] for d in res['responses']) / len(res['responses'])
    res['metrics']['node_f1'] = 2 * res['metrics']['node_p'] * res['metrics']['node_r'] / (
            res['metrics']['node_p'] + res['metrics']['node_r']) if res['metrics']['node_p'] > 0 else 0.0
    res['metrics']['edge_f1'] = 2 * res['metrics']['edge_p'] * res['metrics']['edge_r'] / (
            res['metrics']['edge_p'] + res['metrics']['edge_r']) if res['metrics']['edge_p'] > 0 else 0.0

    # compute per-category metrics
    node_p_by_category = {
        'point': [],
        'path': [],
        'sub-graph': [],
    }
    node_r_by_category = {
        'point': [],
        'path': [],
        'sub-graph': [],
    }
    for d in res['responses']:
        category = d['provenance_graph']['type']
        node_p_by_category[category].append(d['metric_node_p'])
        node_r_by_category[category].append(d['metric_node_r'])
    for category in ('point', 'path', 'sub-graph'):
        res['metrics'][f'node_p_{category}'] = sum(node_p_by_category[category]) / len(node_p_by_category[category])
        res['metrics'][f'node_r_{category}'] = sum(node_r_by_category[category]) / len(node_r_by_category[category])
        res['metrics'][f'node_f1_{category}'] = 2 * res['metrics'][f'node_p_{category}'] * res['metrics'][f'node_r_{category}'] / (
                res['metrics'][f'node_p_{category}'] + res['metrics'][f'node_r_{category}']) if res['metrics'][f'node_p_{category}'] > 0 else 0.0

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', default=None)
    parser.add_argument('--inputs', default=["benchmark/q_graph.json"], nargs="+")
    parser.add_argument('--output_dir', default='outputs/test_nl2cypher/')
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--print_graph_schema', action="store_true")

    # parameters for graph rag
    parser.add_argument('--llm', default="gpt-4-turbo-preview")
    args = parser.parse_args()
    print(args)
    print()

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Get query engine
    engine = get_graph_query_engine(llm=args.llm, verbose=(args.question is not None))

    if args.print_graph_schema:
        graph_schema = engine.graph_store.get_schema()
        print(graph_schema)
        return

    if args.question:
        engine.query(args.question)
        return

    # Load dataset
    dataset = []

    for path in args.inputs:
        with open(path) as f:
            dataset += json.load(f)

    # Run queries
    response_output_path = os.path.join(args.output_dir, "responses.json")
    if not os.path.exists(response_output_path):
        all_response = get_all_responses(engine, dataset)
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
