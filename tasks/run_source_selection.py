import logging
import os
import shutil
import sys
import json
import copy
import random
import argparse
from typing import Any, Dict, List, Optional
from llama_index.core.prompts.base import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, Settings, QueryBundle
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if os.environ.get('LANGFUSE_SECRET_KEY') is not None:
    langfuse_callback_handler = LlamaIndexCallbackHandler(
        public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
        secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
        host=os.environ.get('LANGFUSE_HOST')
    )
    Settings.callback_manager = CallbackManager([langfuse_callback_handler])
else:
    langfuse_callback_handler = None
    print('Warning: LANGFUSE_SECRET_KEY not set. Skipping Langfuse callback handler.')

SOURCE_SELECTION_TMPL = """
Given a question and the descriptions of three databases, select the list of databases that can answer the question.
- The output should be a JSON list of strings, where each string is one of "graph", "doc" or "table".
- If the question cannot be answered by any of the databases, the output should be an empty list.

### Question: {query_str}

### Database descriptions
graph:
{graph_desc}

doc:
{doc_desc}

table:
{table_desc}
""".strip()
SOURCE_SELECTION_PROMPT = PromptTemplate(SOURCE_SELECTION_TMPL)


def get_responses(dataset, modality_summary, baseline=None) -> List[dict]:
    if baseline is not None:
        random.seed(0)
        all_response = []
        for d in dataset:
            d = copy.deepcopy(d)
            if baseline == 'select_random':
                d["model_provenance"] = {"sources": random.sample(["graph", "doc", "table"], 1)}
            elif baseline == 'select_all':
                d["model_provenance"] = {"sources": ["graph", "doc", "table"]}
            elif baseline == 'select_doc':
                d["model_provenance"] = {"sources": ["doc"]}
            elif baseline == 'select_graph':
                d["model_provenance"] = {"sources": ["graph"]}
            elif baseline == 'select_table':
                d["model_provenance"] = {"sources": ["table"]}
            else:
                raise ValueError(f"Invalid baseline: {baseline}")
            all_response.append(d)
        return all_response

    llm = OpenAI(model="gpt-4-turbo-preview", temperature=0)
    all_response = []
    for d in dataset:
        d = copy.deepcopy(d)
        question = d["question"]
        resp = llm.predict(
            SOURCE_SELECTION_PROMPT,
            query_str=question,
            graph_desc=modality_summary["graph"],
            doc_desc=modality_summary["doc"],
            table_desc=modality_summary["table"]
        )
        try:
            resp = resp.replace("```json", "").replace("```", "").strip()
            resp = json.loads(resp)
        except:
            print('Error parsing response for question:', question)
            resp = None
        d["model_provenance"] = {"sources": resp}
        all_response.append(d)
    return all_response


def evaluate(all_response: List[dict]) -> dict:
    res = {
        'metrics': {},
        'responses': []
    }
    labels = ['graph', 'doc', 'table']
    y_true = []
    y_pred = []
    for d in all_response:
        d = copy.deepcopy(d)
        y_true.append([int(l in d['provenance_sources']) for l in labels])
        y_pred.append([int(l in d['model_provenance']['sources']) for l in labels])
        res['responses'].append(d)

    res['metrics']['accuracy'] = accuracy_score(y_true, y_pred)
    res['metrics']['macro_p'] = precision_score(y_true, y_pred, average='macro')
    res['metrics']['macro_r'] = recall_score(y_true, y_pred, average='macro')
    res['metrics']['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    per_class_p = precision_score(y_true, y_pred, average=None)
    per_class_r = recall_score(y_true, y_pred, average=None)
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    for i, label in enumerate(labels):
        res['metrics'][f'{label}_p'] = per_class_p[i]
        res['metrics'][f'{label}_r'] = per_class_r[i]
        res['metrics'][f'{label}_f1'] = per_class_f1[i]
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default=["benchmark/test_qa/q_source.json"], nargs="+")
    parser.add_argument('--output_dir', default='outputs/test_source_selection/')
    parser.add_argument('--overwrite', action="store_true")

    parser.add_argument('--modality_summary', default="tasks/modality_summary_basic.json")
    parser.add_argument('--llm', default="gpt-4-turbo-preview")
    parser.add_argument('--baseline', choices=['select_random', 'select_all', 'select_doc', 'select_graph', 'select_table'], default=None)
    args = parser.parse_args()
    print(args)
    print()

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    response_output_path = os.path.join(args.output_dir, "responses.json")
    if not os.path.exists(response_output_path):
        # Load modality summary
        with open(args.modality_summary) as f:
            modality_summary = json.load(f)
        # Load dataset
        dataset = []
        for path in args.inputs:
            with open(path) as f:
                dataset += json.load(f)
        # Run queries
        all_response = get_responses(dataset, modality_summary, baseline=args.baseline)
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
