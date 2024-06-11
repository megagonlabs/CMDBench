import streamlit as st
import time
from llama_index.core.tools import QueryEngineTool
from llama_index.core import StorageContext, Settings, QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import StorageContext, Settings, Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, QueryType
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.utils import print_text
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import SQLDatabase, Settings, VectorStoreIndex
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.base.base_selector import (
    BaseSelector,
    MultiSelection,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.indices.struct_store.sql_retriever import SQLRetriever, NLSQLRetriever, DefaultSQLParser
from sqlalchemy import create_engine
import pandas as pd
from pymongo import MongoClient
from sql_metadata import Parser
import json
from pydantic import BaseModel
import os
import re
import logging
import warnings
from typing import List, Any, Dict, Tuple, Optional, Sequence
import sys

sys.path.append('.')
from toolbox.langfuse_utils import trace_langfuse

from toolbox.visualization_utils import graph2graphviz

logger = logging.getLogger(__name__)


def patch_DefaultSQLParser():
    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""
        if response.startswith('SELECT'):
            return response.strip()
        sqls = re.findall(r'SQLQuery:(.*?)SQLResult:', response, flags=re.DOTALL)
        if len(sqls) > 0 and sqls[0].strip().startswith('SELECT'):
            return sqls[0].strip()
        sqls = re.findall(r'```(?:sql)?\n(.*?)\n```', response, flags=re.DOTALL)
        if len(sqls) > 0 and sqls[0].strip().startswith('SELECT'):
            return sqls[0].strip()
        return ""

    DefaultSQLParser.parse_response_to_sql = parse_response_to_sql


def patch_SQLRetriever():
    def retrieve_with_metadata(
            self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        raw_response_str, metadata = self._sql_database.run_sql(query_bundle.query_str)
        if self._return_raw:
            return [NodeWithScore(node=TextNode(
                text=raw_response_str,
                metadata={
                    "sql_query": query_bundle.query_str,
                    "result": metadata["result"]
                }
            ))], metadata
        else:
            # return formatted
            results = metadata["result"]
            col_keys = metadata["col_keys"]
            return self._format_node_results(results, col_keys), metadata

    SQLRetriever.retrieve_with_metadata = retrieve_with_metadata


def patch_NLSQLRetriever():
    def retrieve_with_metadata(
            self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        table_desc_str = self._get_table_context(query_bundle)

        logger.info(f"> Table desc str: {table_desc_str}")
        if self._verbose:
            print(f"> Table desc str: {table_desc_str}")

        response_str = self._llm.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        # assume that it's a valid SQL query
        logger.debug(f"> Predicted SQL query: {sql_query_str}")
        if self._verbose:
            print(f"> Predicted SQL query: {sql_query_str}")

        retrieved_tables = re.findall(r"Table '(.*?)' has columns", table_desc_str)

        if self._sql_only:
            sql_only_node = TextNode(text=f"{sql_query_str}")
            retrieved_nodes = [NodeWithScore(node=sql_only_node)]
            metadata = {"result": sql_query_str}
        else:
            try:
                retrieved_nodes, metadata = self._sql_retriever.retrieve_with_metadata(
                    sql_query_str
                )
                retrieved_nodes[0].metadata["retrieved_table_names"] = retrieved_tables
            except BaseException as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    err_node = TextNode(text=f"Error: {e!s}", metadata={"sql_query": sql_query_str,
                                                                        "retrieved_table_names": retrieved_tables})
                    retrieved_nodes = [NodeWithScore(node=err_node)]
                    metadata = {}
                else:
                    raise

        return retrieved_nodes, {"sql_query": sql_query_str, **metadata}

    NLSQLRetriever.retrieve_with_metadata = retrieve_with_metadata


patch_DefaultSQLParser()
patch_SQLRetriever()
patch_NLSQLRetriever()


class ProvenanceGraph(BaseModel):
    reason: Optional[str] = None
    cypher: str
    result_json: Any
    node2name: Dict[Any, str]
    edge2label: Dict[Tuple[Any, Any], str]


class Table(BaseModel):
    table_id: str
    page_title: str
    section_title: str
    columns: List[str]
    rows: List[List[Any]]


class ProvenanceTable(BaseModel):
    reason: Optional[str] = None
    sql_query: str
    result_str: Optional[str]
    retrieved_tables: List[Table]
    sql_tables: List[Table]


class Chunk(BaseModel):
    title: str
    text: str


class ProvenanceDoc(BaseModel):
    reason: Optional[str] = None
    chunks: List[Chunk]


class DiscoveryResults(BaseModel):
    final_response: str
    provenance_graph: Optional[ProvenanceGraph]
    provenance_doc: Optional[ProvenanceDoc]
    provenance_table: Optional[ProvenanceTable]


class AllSelector(BaseSelector):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_defaults(cls) -> "AllSelector":
        return cls()

    def _select(
            self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        res = SelectorResult(
            selections=[SingleSelection(index=i, reason='') for i, _ in enumerate(choices)])
        return res

    async def _aselect(
            self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        return self._select(choices, query)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        # TODO: no accessible prompts for a base pydantic program
        return {}

    def _update_prompts(self, prompts) -> None:
        """Update prompts."""


GRAPH_QUERY_SYNTHESIS_TMPL = """
Given a question and the schema of a Neo4j knowledge graph, create a Cypher query that retrieve either the answer or information that can be used to infer the answer.

Here are the requirements:
- The Cypher query should be compatible with the graph schema. In the graph schema you will be provided with the node properties for each type of node in the graph, the relationship properties for each type of relationship, as well as all unique relationship schemas.
- The Cypher query should NOT use the path syntax (e.g., `MATCH path = ()-[*1..2]-() RETURN path`).
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
        node2name = {}
        edge2label = {}
        with graph_store._driver.session(database=graph_store._database) as session:
            i = 0
            while '-[:' in cypher:
                cypher = cypher.replace(f'-[:', f'-[rr{i}:', 1)
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
                            edge2label[(value.start_node.id, value.end_node.id)] = value.type
                        else:
                            node_ids.add(value.id)
                            node2name[value.id] = value['name']
        return {
            "cypher": cypher, "node_ids": list(node_ids), "edge_ids": list(edge_ids),
            "node2name": node2name, "edge2label": edge2label
        }

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Get nodes for response."""
        graph_store_query = self.generate_query(query_bundle.query_str)
        if self._verbose:
            print_text(f"Graph Store Query:\n{graph_store_query}\n", color="yellow")
        logger.debug(f"Graph Store Query:\n{graph_store_query}")

        try:
            with self.callback_manager.event(
                    CBEventType.RETRIEVE,
                    payload={EventPayload.QUERY_STR: graph_store_query},
            ) as retrieve_event:
                # Get the graph store response
                graph_store_response = self.graph_store.query(query=graph_store_query)
                if self._verbose:
                    print_text(
                        f"Graph Store Response:\n{graph_store_response}\n",
                        color="yellow",
                    )
                logger.debug(f"Graph Store Response:\n{graph_store_response}")

                retrieve_event.on_end(payload={EventPayload.RESPONSE: graph_store_response})

            retrieved_graph_context: Sequence = self._graph_response_answer_prompt.format(
                query_str=query_bundle.query_str,
                kg_query_str=graph_store_query,
                kg_response_str=graph_store_response,
            )
            node = NodeWithScore(
                node=TextNode(
                    text=retrieved_graph_context,
                    score=1.0,
                    metadata={
                        "query_str": query_bundle.query_str,
                        "graph_store_query": graph_store_query,
                        "graph_store_response": graph_store_response,
                        "graph_schema": self._graph_schema,
                    },
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subgraph = self._get_evidence_subgraph(graph_store_query)
            node.metadata['node_ids'] = subgraph['node_ids']
            node.metadata['edge_ids'] = subgraph['edge_ids']
            node.metadata['node2name'] = subgraph['node2name']
            node.metadata['edge2label'] = subgraph['edge2label']
            node.node.excluded_llm_metadata_keys = list(node.metadata.keys())
        except Exception:
            node = NodeWithScore(
                node=TextNode(
                    text=f"Error executing query: {graph_store_query}",
                    metadata={'graph_store_query': graph_store_query}
                )
            )
        return [node]


# @st.cache_resource()
def get_graph_query_engine(verbose=False):
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


@st.cache_resource()
def get_doc_query_engine(
        emb_model, doc_top_k, index_type="default"
):
    persist_dir = os.path.join("data/toolbox/indices", 'doc_default_' + emb_model.replace('/', '--'))
    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        raise ValueError(f"Index not found in {persist_dir}")

    t0 = time.time()

    if index_type == "default":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    elif "faiss" == index_type:
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        raise ValueError(f"Index type {index_type} not supported")

    query_engine = index.as_query_engine(
        similarity_top_k=doc_top_k,
        verbose=True,
    )
    # st.write(f"Index loaded in {time.time() - t0:.2f} seconds")
    return query_engine


@st.cache_resource()
def get_table_query_engine(
        emb_model, table_top_k, index_type="default"
):
    persist_dir = os.path.join("data/toolbox/indices", 'table_' + index_type + '_' + emb_model.replace('/', '--'))
    user = 'yanlin'
    password = '444Castro'
    db = 'lake'
    conn_str = f'postgresql+psycopg://{user}:{password}@localhost/{db}'
    schema = 'wikisql'
    engine = create_engine(conn_str)
    sql_database = SQLDatabase(engine, schema=schema)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    obj_index = ObjectIndex.from_persist_dir(persist_dir, table_node_mapping)
    object_retriever = obj_index.as_retriever(similarity_top_k=table_top_k)
    query_engine = SQLTableRetrieverQueryEngine(
        sql_database, object_retriever, verbose=False
    )
    return query_engine, object_retriever


def get_tables(table_names: List[str]) -> List[Table]:
    table_ids = [t[2:].replace('_', '-') for t in table_names]
    mongo_client = MongoClient("mongodb://localhost:27018/")
    tables = []
    for collection in ('wiki-tables_train', 'wiki-tables_dev', 'wiki-tables_test'):
        db = mongo_client['nba-datalake'][collection]
        docs = db.find({'id': {'$in': table_ids}})
        for doc in docs:
            table = Table(
                table_id=doc['id'],
                page_title=doc['page_title'],
                section_title=doc['section_title'],
                columns=doc['header'],
                rows=doc['rows']
            )
            tables.append(table)
    return tables


@st.cache_data()
def load_json(path):
    with open(path) as f:
        return json.load(f)


@trace_langfuse(name='data_discovery_demo')
def run_data_discovery(question: str, llm: str, emb_model: str, doc_top_k: int,
                       table_top_k: int, summary_type: str, selection_mode: str,
                       debug: bool = False) -> DiscoveryResults:
    if selection_mode == 'PydanticMultiSelector':
        selector = PydanticMultiSelector.from_defaults(max_outputs=3)
    elif selection_mode == 'PydanticSingleSelector':
        selector = PydanticSingleSelector.from_defaults()
    elif selection_mode == 'Select all':
        selector = AllSelector.from_defaults()
    else:
        raise ValueError(f"Selection mode {selection_mode} not supported")

    if summary_type == 'Human':
        summary = HUMAN_SUMMARY
    elif summary_type == 'Complex':
        summary = load_json('data/benchmark/modality_summary_complex.json')
    elif summary_type == 'Basic':
        summary = load_json('data/benchmark/modality_summary_basic.json')
    else:
        raise ValueError(f"Summary type {summary_type} not supported")

    graph_engine = get_graph_query_engine()
    graph_engine_tool = QueryEngineTool.from_defaults(
        query_engine=graph_engine,
        description=summary['graph']
    )

    with st.spinner("Loading document index..."):
        doc_engine = get_doc_query_engine(emb_model, doc_top_k)
    doc_engine_tool = QueryEngineTool.from_defaults(
        query_engine=doc_engine,
        description=summary['doc']
    )

    with st.spinner("Loading table index..."):
        table_engine, table_retriever = get_table_query_engine(emb_model, table_top_k)
    table_engine_tool = QueryEngineTool.from_defaults(
        query_engine=table_engine,
        description=summary['table']
    )

    tools = {
        'graph': graph_engine_tool,
        'doc': doc_engine_tool,
        'table': table_engine_tool
    }

    # st.write(selector.select(["place", "person"], 'person'))

    engine = RouterQueryEngine(
        selector=selector,
        query_engine_tools=list(tools.values()),
    )
    resp = engine.query(question)
    if debug:
        st.write(resp)
        st.write(resp.source_nodes)
        st.write(resp.metadata)
        st.write(resp.get_formatted_sources(100000))

    provenance_graph = None
    provenance_table = None
    provenance_doc = None
    selection_results = []
    for node in resp.source_nodes:
        if 'graph_store_query' in node.metadata:
            provenance_graph = ProvenanceGraph(
                cypher=node.metadata['graph_store_query'],
                result_json=json.dumps(
                    node.metadata['graph_store_response']) if 'graph_store_response' in node.metadata else None,
                node2name=node.metadata['node2name'] if 'node2name' in node.metadata else {},
                edge2label=node.metadata['edge2label'] if 'edge2label' in node.metadata else {}
            )
        elif 'wikipedia_title' in node.metadata:
            if provenance_doc is None:
                provenance_doc = ProvenanceDoc(chunks=[])
            provenance_doc.chunks.append(Chunk(
                title=node.metadata['wikipedia_title'],
                text=node.text
            ))
        elif 'sql_query' in node.metadata:

            provenance_table = ProvenanceTable(
                sql_query=node.metadata['sql_query'],
                result_str=str(node.metadata['result']) if 'result' in node.metadata else None,
                sql_tables=get_tables(Parser(node.metadata['sql_query']).tables) if 'result' in node.metadata else [],
                retrieved_tables=get_tables(node.metadata['retrieved_table_names'])
            )
    for sel in resp.metadata['selector_result'].selections:
        source = list(tools.keys())[sel.index]
        if source == 'graph' and provenance_graph is not None:
            provenance_graph.reason = sel.reason
        elif source == 'doc' and provenance_doc is not None:
            provenance_doc.reason = sel.reason
        elif source == 'table' and provenance_table is not None:
            provenance_table.reason = sel.reason

    # response = engine.query(d["question"])
    #     retrieved_tables = table_retriever.retrieve(d["question"])
    #     table_names = [table.table_name for table in retrieved_tables]
    #     d['model_response'] = str(response)
    #     d['model_provenance'] = {
    #         'tables': {
    #             'retrieved': table_names,
    #             'sql': response.metadata['sql_query'].replace("\n", " "),
    #             'sql_columns': Parser(response.metadata['sql_query']).columns,
    #             'sql_tables': Parser(response.metadata['sql_query']).tables,
    #         }
    #     }
    return DiscoveryResults(
        final_response=str(resp),
        selection_results=selection_results,
        provenance_graph=provenance_graph,
        provenance_doc=provenance_doc,
        provenance_table=provenance_table
    )


def format_doc(title, doc: str):
    if doc.startswith(f'{title}\n'):
        doc = re.sub(rf'^{title}\n', '', doc)
    enwiki_url = f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}'
    doc = f'###### [doc] [{title}]({enwiki_url})\n{doc}'
    doc = re.sub(r'^Section::::(.*?)\.$',
                 lambda m: '#' * min(6 + m.group(1).count('.'), 6) + f' {m.group(1).split(".")[-1].lstrip(":")}', doc,
                 flags=re.MULTILINE)
    doc = doc.replace('$', '\$')
    return doc


def display_provenance(result: DiscoveryResults):
    col_graph, col_doc, col_table = st.columns(3)

    with col_graph:
        if not result.provenance_graph:
            st.warning(f'No graph provenance found', icon="⚠️")
        else:
            prov_g = result.provenance_graph
            with st.container(border=True):
                st.write(f'###### [source] datalake.nba.graph')
                st.caption(f'🤖 Reason for selection: {prov_g.reason or "NaN"}')

            cypher = prov_g.cypher
            st.caption("Cypher Query")
            st.code(cypher, language='cypher')
            if prov_g.result_json is not None:
                tab1, tab2 = st.tabs(['Raw Neo4j response', 'Provenance Graph'])
                with tab1:
                    s = json.dumps(json.loads(prov_g.result_json), indent=2)
                    if len(s) >= 500:
                        st.warning(f'Only displaying truncated response', icon="⚠️")
                        s = s[:500] + '\n...'
                    st.code(s, language='json')
                with tab2:
                    if len(prov_g.node2name) > 50:
                        st.warning(f'Cannot display graph with more than 50 nodes', icon="⚠️")
                    else:
                        g = graph2graphviz(
                            node2name=prov_g.node2name,
                            edge2label=prov_g.edge2label,
                            id2color={}
                        )
                        with st.container(border=True):
                            st.graphviz_chart(g)
            else:
                st.error(f"Error executing Cypher query", icon="❌")

    with col_doc:
        if not result.provenance_doc:
            st.warning(f'No document provenance found', icon="⚠️")
        else:
            prov_doc = result.provenance_doc
            with st.container(border=True):
                st.write(f'###### [source] datalake.nba.doc')
                st.caption(f'🤖 Reason for selection: {prov_doc.reason or "NaN"}')

            st.caption(f"Retrieved `{len(prov_doc.chunks)}` docs")
            chunks = prov_doc.chunks
            if len(chunks) >= 10:
                st.warning(f'Only displaying the first 10 documents', icon="⚠️")
                chunks = chunks[:10]
            for i, chunk in enumerate(chunks):
                with st.container(height=300 if len(chunk.text) > 600 else None, border=True):
                    # url = f'https://en.wikipedia.org/wiki/{chunk.title.replace(" ", "_")}'
                    # st.write(f'[{chunk.title}]({url})')
                    st.write(format_doc(chunk.title, chunk.text))

    with col_table:
        if not result.provenance_table:
            st.warning(f'No table provenance found', icon="⚠️")
        else:
            prov_t = result.provenance_table
            with st.container(border=True):
                st.write(f'###### [source] datalake.nba.table')
                st.caption(f'🤖 Reason for selection: {prov_t.reason or "NaN"}')

            # Display retrieved tables
            st.caption(f'Retrieved `{len(prov_t.retrieved_tables)}` tables')
            with st.container(border=True):
                for table in prov_t.retrieved_tables:
                    enwiki_url = f'https://en.wikipedia.org/wiki/{table.page_title.replace(" ", "_")}#{table.section_title.replace(" ", "_")}'
                    st.write(f'###### [table] [{table.page_title} - {table.section_title}]({enwiki_url})')

            # Display SQL query
            if prov_t.sql_query == "":
                st.error(f"No SQL query found", icon="❌")
            else:
                st.caption("SQL Query")
                st.code(prov_t.sql_query, language='sql')

                # Display SQL response
                if prov_t.result_str is None:
                    st.error(f"Error executing SQL query", icon="❌")
                else:
                    tab1, tab2 = st.tabs(['Raw SQL response', 'Provenance Tables'])
                    with tab1:
                        st.code(prov_t.result_str, language='json')

                    # Display tables referenced in the SQL query
                    with tab2:
                        tables = prov_t.sql_tables
                        if len(tables) >= 10:
                            st.warning(f'Only displaying the first 10 tables', icon="⚠️")
                            tables = tables[:10]
                        for table in tables:
                            df = pd.DataFrame(table.rows, columns=table.columns)
                            enwiki_url = f'https://en.wikipedia.org/wiki/{table.page_title.replace(" ", "_")}'
                            st.write(f'###### [table] [{table.page_title} - {table.section_title}]({enwiki_url})')
                            st.dataframe(df, hide_index=True)


HUMAN_SUMMARY = {
    "graph": "This graph database schema outlines a comprehensive structure designed to capture and represent the intricate relationships within the world of sports, focusing on players, coaches, teams, positions, venues, divisions, and awards. Nodes in the database represent entities such as Players, Coaches, Teams, Positions, Venues, Divisions, and Awards. Players and coaches have properties name, enwiki_title, country_of_citizenship, place_of_birth, native_language, sex_or_gender, date_of_birth, height, mass, work_period_start, work_period_end, nickname, handedness, schools_attended. Teams have properties name, enwiki_title, inception, owners. Relationships between these entities are meticulously defined to reflect real-world interactions: players play for teams, occupy specific positions, are drafted by teams, can be the same individuals as coaches, and receive awards. Teams, on the other hand, have home venues, are coached by individuals, and belong to divisions. This knowledge graph provides a detailed schema for relationships within the context of sports, focusing on players, teams, and their interactions. It outlines the playsFor relationship, where players are linked to teams they play for, complete with properties like start and end times, player numbers, and positions. The draftedBy relationship tracks which team drafted a player and when. playsPosition links players to their specific roles or positions in the sport, while receivesAward connects players with awards they've received, noting the time of the award. Additionally, teams are linked to their coaches through the coachedBy relation and to their home venues in the hasHomeVenue relation, again with start and end times. The graph also includes an isSameAs relationship, identifying players who are also coaches, and a partOfDivision relation that connects teams with the sports divisions they belong to.",
    "doc": "This document database provides detailed information on various aspects of basketball, focusing primarily on professional players, teams, and seasons within the National Basketball Association (NBA), as well as touching on the Women's National Basketball Association (WNBA) and international basketball careers. It includes profiles of retired and active players, detailing their careers, achievements, and transitions post-retirement. The database covers team histories and season summaries, highlighting significant achievements, playoff performances, and records. It also features information on basketball arenas, notable games, draft picks, and coaching careers. Additionally, the database includes entries on basketball-related media and awards, offering insights into the broader cultural and historical impact of the sport. Through a mix of individual player narratives, team stories, and broader basketball themes, this database serves as a comprehensive resource for basketball history and statistics.",
    "table": "This database provides a comprehensive collection of data related to various basketball teams and seasons, primarily focusing on the NBA but also including other leagues and basketball-related activities. It encompasses game logs for multiple teams across different seasons, detailing each game's date, teams involved, score, high points, high rebounds, high assists, location attendance, and the team's record after the game. Additionally, it includes all-time rosters for several teams, listing players along with their nationality, position, tenure with the team, and educational or club background. It also covers the player statistics for various seasons."
}


@st.cache_data()
def load_questions():
    questions = []
    q2src = {}
    for src in ['graph', 'doc', 'table']:
        with open(f'data/benchmark/q_{src}.json') as f:
            q = json.load(f)
            questions.extend([d['question'] for d in q])
            for d in q:
                q2src[d['question']] = src
    return questions, q2src


def main():
    st.set_page_config(
        page_title='Data Discovery',
        page_icon='👾',
        layout='wide',
    )

    st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 4rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        llm = st.selectbox("LLM", ["gpt-4o", "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"])
        emb_model = st.selectbox("Embedding Model", ["BAAI/bge-base-en-v1.5"])
        doc_top_k = st.slider("Document Top K", 1, 10, 3)
        table_top_k = st.slider("Table Top K", 1, 10, 3)

        summary_type = st.radio("Source Summary", ["Human", "Complex", "Basic"])

        selection_mode = st.radio("Selection Mode", ["Select all", "PydanticMultiSelector", "PydanticSingleSelector"])

        st.divider()

        questions, q2src = load_questions()
        questions = ["What is the height of LeBron James?"] + questions

        example = st.selectbox(
            "Example questions",
            ["Select an example"] + questions,
            format_func=lambda x: f'[{q2src[x]}] {x}' if x in q2src else x
        )
        st.text_area('', "" if example == "Select an example" else example, label_visibility='collapsed', height=100)

        st.divider()

        debug_mode = st.toggle("Debug Mode", False)

    Settings.llm = OpenAI(temperature=0, model=llm)
    if emb_model.startswith('text-embedding'):
        Settings.embed_model = OpenAIEmbedding(model=emb_model, embed_batch_size=1000)
    else:
        Settings.embed_model = HuggingFaceEmbedding(emb_model)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message(message["role"]).write(message["content"])
        elif message['role'] == 'assistant':
            st.chat_message(message["role"]).write(message["content"].final_response)
            _, body = st.columns([0.001, 0.999], gap="large")
            with body:
                with st.expander('View provenance'):
                    display_provenance(message["content"])
            # st.caption('')
            st.divider()

    question = st.chat_input('Enter a question')
    if question is None:
        st.stop()
    st.chat_message('user').write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    result = run_data_discovery(question, llm, emb_model, doc_top_k, table_top_k, summary_type, selection_mode,
                                debug=debug_mode)
    st.chat_message('assistant').write(result.final_response)
    _, body = st.columns([0.001, 0.999], gap="large")
    with body:
        with st.expander('View provenance'):
            display_provenance(result)
    st.session_state.messages.append({"role": "assistant", "content": result})


if __name__ == '__main__':
    main()
