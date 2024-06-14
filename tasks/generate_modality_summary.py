import argparse
from pymongo import MongoClient
from typing import List
import numpy as np
import os
import json
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def get_centroid_titles(titles: List[str], model: str, n_clusters: int = 20):
    model = SentenceTransformer(model)
    embeddings = model.encode(titles, show_progress_bar=True)
    embeddings = np.array(embeddings)  # (n_samples, n_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    centroids = kmeans.cluster_centers_  # (n_clusters, n_features)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    centroids = centroids / np.linalg.norm(centroids, axis=1)[:, None]
    cosine_sim = (embeddings @ centroids.T)  # (n_samples, n_clusters)
    most_similar_ids = np.argmax(cosine_sim, axis=0)
    return [titles[i] for i in most_similar_ids]


def get_abstract(paragraphs: List[str]) -> str:
    res = []
    for p in paragraphs[1:]:  # paragraphs[0] is the title
        if p.startswith('Section::::') or p.startswith('BULLET::::'):
            break
        res.append(p)
    return '\n'.join(res).strip()


def get_doc_prompt(emb_model: str, n_doc_examples: int = 50, with_categories=False, with_abstract=False):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["nba-datalake"]["wiki-documents"]
    docs = list(db.find({}, {'wikipedia_title': 1, '_id': 0, 'categories': 1, 'text': 1}))
    text = []
    for d in docs:
        s = f'Title: {d["wikipedia_title"]}'
        if with_categories:
            s += f'  Categories: {d["categories"]}'
        if with_abstract:
            s += f'  Abstract: {get_abstract(d["text"])}'
        text.append(s)

    representative_titles = get_centroid_titles(text, emb_model, n_clusters=n_doc_examples)
    prompt = 'Write a brief description for a document database. The description should be a single paragraph. Summarize the key information covered by this database.\n'
    prompt += 'Here are the titles of some documents sampled from the database:\n'
    for i, title in enumerate(representative_titles):
        prompt += f'- {title}\n'
    return prompt


def get_table_prompt(emb_model: str, n_table_examples: int = 50, no_header=False):
    client = MongoClient("mongodb://localhost:27017/")
    tables = []
    for split in ["train", "test", "dev"]:
        db = client["nba-datalake"][f"wiki-tables_{split}"]
        tables += list(db.find({}))
    text = []
    for table in tables:
        if no_header:
            text.append(f'Table title: "{table["page_title"]} - {table["section_title"]}"')
        else:
            text.append(
                f'Table title: "{table["page_title"]} - {table["section_title"]}"  Table Columns: {table["header"]}')

    representative_titles = get_centroid_titles(text, emb_model, n_clusters=n_table_examples)
    prompt = 'Write a brief description for a table database. The description should be a single paragraph. Summarize the key information covered by this database.\n'
    prompt += 'Here are the titles and schemas of some tables sampled from the database:\n'
    for i, title in enumerate(representative_titles):
        prompt += f'- {title}\n'
    return prompt


def get_graph_prompt(relationship_only=False):
    graph_store = Neo4jGraphStore(
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD"),
        url="bolt://localhost:7687",
        database="neo4j",
    )
    schema = graph_store.get_schema()
    if relationship_only:
        schema = schema.split('The relationships are the following:')[-1].strip()
    prompt = 'Write a brief description for a graph database. The description should be a single paragraph. Summarize the key information covered by this database.\n'
    prompt += f'Here is the schema of the graph database:\n'
    prompt += f'{schema}'
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_doc_examples', default=50, type=int)
    parser.add_argument('--n_table_examples', default=50, type=int)
    parser.add_argument('--emb_model', default="paraphrase-MiniLM-L6-v2")
    parser.add_argument('--llm_model', default="gpt-4-turbo-preview")
    parser.add_argument('--output_path', default="tasks/modality_summary_1.json")
    args = parser.parse_args()
    print(args)
    print()

    # graph_prompt = get_graph_prompt()
    # doc_prompt = get_doc_prompt(args.emb_model, args.n_doc_examples)
    # table_prompt = get_table_prompt(args.emb_model, args.n_table_examples)
    prompts = {
        'graph_no_properties': get_graph_prompt(relationship_only=True),
        'graph_with_properties': get_graph_prompt(relationship_only=False),
        'doc_title': get_doc_prompt(args.emb_model, args.n_doc_examples, with_categories=False, with_abstract=False),
        'doc_title_categories': get_doc_prompt(args.emb_model, args.n_doc_examples, with_categories=True,
                                               with_abstract=False),
        'doc_title_abstract': get_doc_prompt(args.emb_model, args.n_doc_examples, with_categories=False,
                                             with_abstract=True),
        'table_title': get_table_prompt(args.emb_model, args.n_table_examples, no_header=True),
        'table_title_header': get_table_prompt(args.emb_model, args.n_table_examples, no_header=False),
    }

    llm = ChatOpenAI(model=args.llm_model, temperature=0.)

    chain = llm | StrOutputParser()
    batch_response = chain.batch(list(prompts.values()))
    output = {k: v for k, v in zip(prompts.keys(), batch_response)}
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Output saved to {args.output_path}')


if __name__ == "__main__":
    main()
