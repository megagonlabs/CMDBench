# CMDBench
The public repos of the CMDBench paper


## Serving the NBA Datalake

Run the following commands to start the CMDBench datalake.

```bash
git clone https://github.com/rit-git/cmdbench.git
cd cmdbench/docker/compose
docker compose up
```

 The following databases will be exposed at various ports:
- Neo4j: `localhost:7687` (bolt) and `localhost:7474` (http)
- MongoDB: `localhost:27017`
- PostgreSQL: `localhost:5432`

When you are done, you can stop with `Ctrl+C` and then run:
```bash
docker compose down
```

### Notes:
- You can find the database credentials in [docker/compose/.env](docker/compose/.env)
- If there are port conflicts, you can change the ports in [docker/compose/.env](docker/compose/.env)



The core task we are focusing on is coars-to-fine "Data Discovery." The coarse grained data discovery methods
essentially discover the source (e.g., mongodb, postgresdb, neo4jdb) containing the response to a query. The finer the
granularity the finer the result of the discover method. For example, a fine-grained discovery for text, table, and
graphs is finding candidate document(s), table(s), and sub-graph(s), respectively. The finest granularity is the exact
source, e.g., (sentence, span) in document, (row, column) in table, or (path, node) in graph.

## Running Experiments

### Depedencies

Create a virtual environment:

```bash
conda create -n cmdbench python=3.11
conda activate cmdbench
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Setup openai api key in .bashrc:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

We implement experiment pipelines for each data discovery tasks. Each pipeline takes the data JSON file as input and
outputs two JSON files:

1. `responses.json` contains the outputs of the data discovery models.
2. `result.json` additionally includes the evaluation metrics.

Here are a full list of implemented pipelines:

- [tasks/run_source_selection.py](run_source_selection.py) runs the source selection task.
    - Note: run [tasks/generate_modality_summary.py](generate_modality_summary.py) to generate the modality summary.
- [tasks/run_nl2cypher.py](run_nl2cypher.py) runs the subgraph discovery task.
- [tasks/run_doc_discovery.py](run_doc_discovery.py) runs the document discovery task.
- [tasks/run_table_discovery.py](run_table_discovery.py) runs the table discovery task.
- [tasks/run_paragraph_discovery.py](run_paragraph_discovery.py) runs the paragraph discovery task.
- [tasks/run_answer_gen_graph.py](run_answer_gen_graph.py) runs the answer generation task for graph questions.
- [tasks/run_answer_gen_doc.py](run_answer_gen_doc.py) runs the answer generation task for document questions.

Here is a sample command to run the source selection task:

```bash
python tasks/run_source_selection.py --inputs benchmark/q_source.json \
  --modality_summary tasks/modality_summary_basic.json \
  --llm gpt-3.5-turbo-0125 \
  --output_dir outputs/exp0/
```


## CMDBench Dataset

[benchmark/q_graph.json](./benchmark/q_graph.json) contains the graph questions with annotated graph provenance. Here is a sample:

```json
{
  "qid": "216f99dc-2826-4db8-92c3-f5066c8cf528",
  "question": "When did Dwyane Wade leave the Miami Heat?",
  "answer": "2016",
  "provenance_graph": {
    "cypher": "MATCH (p:Player {name: 'Dwyane Wade'})-[r:playsFor]->(t:Team {name: 'Miami Heat'}) RETURN r.end_time AS leave_time",
    "node_ids": [
      5231,
      1503
    ],
    "edge_ids": [
      11865
    ],
    "type": "path"
  }
}
```

[benchmark/q_doc.json](./benchmark/q_doc.json) contains the document questions. Here is a sample:

```json
{
  "qid": "aab8025d-7414-42cc-bc4a-86bbb1cc2903",
  "question": "Named for it's proximity to the local NBA team, what is the name of the WNBA team in Phoenix?",
  "answer": "Mercury",
  "provenance_doc": {
    "paragraphs": [
      {
        "wikipedia_title": "Women's National Basketball Association",
        "paragraph_id": 2,
        "text": "Five WNBA teams have direct NBA counterparts and play in the same arena: the Atlanta Dream, Indiana Fever, Los Angeles Sparks, Minnesota Lynx, and Phoenix Mercury. The Chicago Sky, Connecticut Sun, Dallas Wings, Las Vegas Aces, New York Liberty, Seattle Storm, and Washington Mystics do not share an arena with a direct NBA counterpart, although four of the seven (the Sky, the Wings, the Liberty, and the Mystics) share a market with an NBA counterpart, and the Storm shared an arena and market with an NBA team at the time of its founding. The Sky, the Sun, the Wings, the Aces, the Sparks, and the Storm are all independently owned.\n"
      }
    ]
  }
}
```

[benchmark/q_source.json](./benchmark/q_source.json) contains the source selection questions. Here is a sample:

```json
{
  "question": "When did Dwyane Wade leave the Miami Heat?",
  "answer": "2016",
  "provenance_sources": [
    "doc",
    "graph"
  ]
}
```

[benchmark/q_table.json](./benchmark/q_table.json) contains the table questions. Here is a sample:

```json
  {
  "qid": "8",
  "question": "What is the average number of points scored by the Seattle Storm players in the 2005 season?",
  "answer": "208",
  "provenance_table": {
    "table": "t_1_24915964_4",
    "column": "['points']"
  }
}
```


## Helpful Resources

1. [Core concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)
2. [Documents and customizing documents](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html)
3. [Indexing](https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing.html)
4. [Storage](https://docs.llamaindex.ai/en/stable/understanding/storing/storing.html)
5. [Query engine](https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html)
6. [Metadata]()

### Other resources

Following resources were not used for implementation but are worth checking out:

1. [Nodes](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_nodes.html)
2. [Metdata extraction](https://docs.llamaindex.ai/en/stable/module_guides/indexing/metadata_extraction.html)
3. [Metadata enhanced extraction](https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC.html)
4. [Advanced retrieval](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes.html#metadata-references-summaries-generated-questions-referring-to-a-bigger-chunk)

### Graph querying

- [Neo4j Graph Store](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/Neo4jKGIndexDemo.html)
    - It assumes all entities have the same
      label [reference](https://github.com/run-llama/llama_index/blob/46f1e13b63d6735a6fb86fabbf8d6104552b2f49/llama-index-integrations/graph_stores/llama-index-graph-stores-neo4j/llama_index/graph_stores/neo4j/base.py#L47)
    - It only queries entities by the property `id`
- [Knowledge Graph RAG Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine.html)
- [Knowledge Graph Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine.html)

- [Graph Database Cypher Loader](https://github.com/run-llama/llama-hub/tree/main/llama_hub/graphdb_cypher)

