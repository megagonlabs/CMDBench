import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import time
from llama_index.core.tools import QueryEngineTool, RetrieverTool
from llama_index.core import StorageContext, Settings, QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.prompts.base import PromptTemplate, PromptType
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.schema import NodeWithScore, TextNode, MetadataMode
from llama_index.core import StorageContext, Settings, Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, QueryType
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.utils import print_text
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core import SQLDatabase, Settings, VectorStoreIndex
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.base.base_selector import (
    BaseSelector,
    MultiSelection,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.base.base_retriever import BaseRetriever
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
from typing import List, Any, Dict, Tuple, Optional, Sequence, Union
import sys

sys.path.append('.')

from visualization_utils import graph2graphviz

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
                retrieved_nodes[0].node.metadata["table_description"] = table_desc_str
                retrieved_nodes[0].node.metadata["retrieved_table_names"] = retrieved_tables
                retrieved_nodes[0].node.excluded_llm_metadata_keys.append("retrieved_table_names")
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


def patch_RouterRetriever():
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as query_event:
            result = self._selector.select(self._metadatas, query_bundle)
            self.selector_result_ = result

            if len(result.inds) > 1:
                retrieved_results = {}
                for i, engine_ind in enumerate(result.inds):
                    logger.info(
                        f"Selecting retriever {engine_ind}: " f"{result.reasons[i]}."
                    )
                    selected_retriever = self._retrievers[engine_ind]
                    cur_results = selected_retriever.retrieve(query_bundle)
                    retrieved_results.update({n.node.node_id: n for n in cur_results})
            else:
                try:
                    selected_retriever = self._retrievers[result.ind]
                    logger.info(f"Selecting retriever {result.ind}: {result.reason}.")
                except ValueError as e:
                    raise ValueError("Failed to select retriever") from e

                cur_results = selected_retriever.retrieve(query_bundle)
                retrieved_results = {n.node.node_id: n for n in cur_results}

            query_event.on_end(payload={EventPayload.NODES: retrieved_results.values()})

        return list(retrieved_results.values())

    RouterRetriever._retrieve = _retrieve


patch_DefaultSQLParser()
patch_SQLRetriever()
patch_NLSQLRetriever()
patch_RouterRetriever()


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
    cited_provenance: List[Union[ProvenanceGraph, Chunk, ProvenanceTable]]


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
DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT = PromptTemplate(GRAPH_QUERY_SYNTHESIS_TMPL)

DEFAULT_KG_RESPONSE_ANSWER_PROMPT_TMPL = """
The original question is given below.
This question has been translated into a Graph Database query.
Both the Graph query and the response are given below.
Given the Graph Query response, synthesise a response to the original question.

Original question: {query_str}
Graph query: {kg_query_str}
Graph response: {kg_response_str}
Response:
"""

DEFAULT_KG_RESPONSE_ANSWER_PROMPT = PromptTemplate(
    DEFAULT_KG_RESPONSE_ANSWER_PROMPT_TMPL,
    prompt_type=PromptType.QUESTION_ANSWER,
)


class Nl2CypherRetriever(BaseRetriever):
    def __init__(
            self,
            llm: Optional[LLM] = None,
            storage_context: Optional[StorageContext] = None,
            graph_query_synthesis_prompt: Optional[PromptTemplate] = None,
            graph_response_answer_prompt: Optional[PromptTemplate] = None,
    ):
        super().__init__()
        # Ensure that we have a graph store
        assert storage_context is not None, "Must provide a storage context."
        assert (
                storage_context.graph_store is not None
        ), "Must provide a graph store in the storage context."
        self._storage_context = storage_context
        self.graph_store = storage_context.graph_store
        self._llm = llm or llm_from_settings_or_context(Settings, None)
        self._graph_schema = self.graph_store.get_schema()

        self._graph_query_synthesis_prompt = graph_query_synthesis_prompt or DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT

        self._graph_response_answer_prompt = (
                graph_response_answer_prompt or DEFAULT_KG_RESPONSE_ANSWER_PROMPT
        )

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
                i += 1
            cypher = re.sub(r"\bRETURN\b.*$", "RETURN *", cypher, flags=re.IGNORECASE | re.DOTALL)
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


CITATION_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "Consider all sources when you decide the answer, some source might not have the desired information but other might have. "
    "Cite all sources that support your answer, but do not cite sources that are not relevant to the answer. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "Consider all sources when you decide the answer, some source might not have the desired information but other might have. "
    "Cite all sources that support your answer, but do not cite sources that are not relevant to the answer. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. "
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!"
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)


# @st.cache_resource()
def get_graph_retriever():
    graph_store = Neo4jGraphStore(
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD"),
        url="bolt://neo4j:7687",
        database="neo4j",
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    return Nl2CypherRetriever(storage_context=storage_context)


@st.cache_resource()
def get_doc_index(emb_model):
    persist_dir = os.path.join("data/indices", 'doc_default_' + emb_model.replace('/', '--'))
    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        raise ValueError(f"Index not found in {persist_dir}")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index


@st.cache_resource()
def get_table_query_engine(
        emb_model, table_top_k
):
    persist_dir = os.path.join("data/indices", 'table_default_' + emb_model.replace('/', '--'))
    user = os.environ.get("PGUSER")
    db = 'nba'
    # conn_str = f'postgresql+psycopg://{user}:{password}@postgres/{db}'
    conn_str = f'postgresql+psycopg://{user}@postgres/{db}'
    schema = 'nba_wikisql'
    engine = create_engine(conn_str)
    sql_database = SQLDatabase(engine, schema=schema)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    obj_index = ObjectIndex.from_persist_dir(persist_dir, table_node_mapping)
    object_retriever = obj_index.as_retriever(similarity_top_k=table_top_k)
    query_engine = SQLTableRetrieverQueryEngine(
        sql_database, object_retriever, verbose=False
    )
    return query_engine


def get_tables(table_names: List[str]) -> List[Table]:
    table_ids = [t[2:].replace('_', '-') for t in table_names]
    mongo_client = MongoClient("mongodb://mongo:27017/")
    tables = []
    db = mongo_client['nba-datalake']['wiki-tables']
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


def get_router_query_engine(emb_model, doc_top_k, table_top_k, selector, summary):
    graph_retriever = get_graph_retriever()
    graph_engine = RetrieverQueryEngine(retriever=graph_retriever)
    graph_engine_tool = QueryEngineTool.from_defaults(
        query_engine=graph_engine,
        description=summary['graph']
    )

    doc_index = get_doc_index(emb_model)
    doc_engine = doc_index.as_query_engine(similarity_top_k=doc_top_k)
    doc_engine_tool = QueryEngineTool.from_defaults(
        query_engine=doc_engine,
        description=summary['doc']
    )

    table_engine = get_table_query_engine(emb_model, table_top_k)
    table_engine_tool = QueryEngineTool.from_defaults(
        query_engine=table_engine,
        description=summary['table']
    )

    tools = {
        'graph': graph_engine_tool,
        'doc': doc_engine_tool,
        'table': table_engine_tool
    }

    engine = RouterQueryEngine(
        selector=selector,
        query_engine_tools=list(tools.values()),
    )
    return engine


class CitationNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        new_nodes: List[NodeWithScore] = []
        for node in nodes:
            text = node.node.get_content(metadata_mode=MetadataMode.LLM)
            text = f"Source {len(new_nodes) + 1}:\n{text}\n"
            new_node = NodeWithScore(
                node=TextNode.parse_obj(node.node), score=node.score
            )
            new_node.node.text = text
            new_nodes.append(new_node)
        return new_nodes


def get_custom_query_engine(emb_model, doc_top_k, table_top_k, selector, summary):
    graph_retriever_tool = RetrieverTool.from_defaults(
        retriever=get_graph_retriever(),
        description=summary['graph']
    )
    doc_retriever_tool = RetrieverTool.from_defaults(
        retriever=get_doc_index(emb_model).as_retriever(similarity_top_k=doc_top_k),
        description=summary['doc']
    )
    table_retriever_tool = RetrieverTool.from_defaults(
        retriever=get_table_query_engine(emb_model, table_top_k)._sql_retriever,
        description=summary['table']
    )
    retriever = RouterRetriever(
        selector=selector,
        retriever_tools=[
            graph_retriever_tool,
            doc_retriever_tool,
            table_retriever_tool
        ],
    )
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        streaming=True,
        text_qa_template=CITATION_QA_TEMPLATE,
        refine_template=CITATION_REFINE_TEMPLATE,
        node_postprocessors=[CitationNodePostprocessor()]
    )


class StreamingOutput:
    def __init__(self, response_gen):
        self.response_gen = response_gen
        self.final_response = ''
        self.buffer = ''
        self.citation_mapping = {}

    def _yield(self):
        yield self.buffer
        self.final_response += self.buffer
        self.buffer = ''

    def __iter__(self):
        for text in self.response_gen:
            self.buffer += text
            # case 1: no brackets
            if '[' not in self.buffer:
                yield from self._yield()
            # case 2: all brackets are closed
            elif len(re.findall(r'\[($|\d+)', self.buffer)) == len(re.findall(r'\[([0-9]+)\]', self.buffer)):
                citations = re.findall(r'\[([0-9]+)\]', self.buffer)
                for c in citations:
                    if int(c) not in self.citation_mapping:
                        self.citation_mapping[int(c)] = len(self.citation_mapping) + 1
                self.buffer = re.sub(r'\[([0-9]+)\]', lambda m: f'[{self.citation_mapping[int(m.group(1))]}]',
                                     self.buffer)
                yield from self._yield()
            # case 3: brackets are not closed
            else:
                continue  # wait for more text


def run_data_discovery(question: str, llm: str, emb_model: str, doc_top_k: int,
                       table_top_k: int, summary_type: str, selection_mode: str,
                       debug: bool = False, query_engine_type: str = 'custom',
                       temperature: float = 0, avatar=None) -> DiscoveryResults:
    _, body = st.columns([0.001, 0.999], gap="large")
    with body:
        with st.spinner('Running data discovery...'):
            Settings.llm = OpenAI(temperature=temperature, model=llm)
            if emb_model.startswith('text-embedding'):
                Settings.embed_model = OpenAIEmbedding(model=emb_model, embed_batch_size=1000)
            else:
                Settings.embed_model = HuggingFaceEmbedding(emb_model)

            if selection_mode == 'MultiSelector':
                selector = PydanticMultiSelector.from_defaults(max_outputs=3)
            elif selection_mode == 'SingleSelector':
                selector = PydanticSingleSelector.from_defaults()
            elif selection_mode == 'Select all':
                selector = AllSelector.from_defaults()
            else:
                raise ValueError(f"Selection mode {selection_mode} not supported")

            if summary_type == 'Human':
                summary = HUMAN_SUMMARY
            elif summary_type == 'Complex':
                summary = load_json('data/cmdbench/modality_summary_complex.json')
            elif summary_type == 'Basic':
                summary = load_json('data/cmdbench/modality_summary_basic.json')
            else:
                raise ValueError(f"Summary type {summary_type} not supported")

            if query_engine_type == 'RouterQueryEngine':
                engine = get_router_query_engine(emb_model, doc_top_k, table_top_k, selector, summary)
            elif query_engine_type == 'Custom':
                engine = get_custom_query_engine(emb_model, doc_top_k, table_top_k, selector, summary)
            else:
                raise ValueError(f"Query engine type {query_engine_type} not supported")

            resp = engine.query(question)
    streaming_output = StreamingOutput(resp.response_gen)
    with st.chat_message('assistant', avatar=avatar):
        st.write_stream(streaming_output)

    final_resp = streaming_output.final_response

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
                text=re.sub(r'Source \d+:', '', node.text).strip()
            ))
        elif 'sql_query' in node.metadata:

            provenance_table = ProvenanceTable(
                sql_query=node.metadata['sql_query'],
                result_str=str(node.metadata['result']) if 'result' in node.metadata else None,
                sql_tables=get_tables(Parser(node.metadata['sql_query']).tables) if 'result' in node.metadata else [],
                retrieved_tables=get_tables(node.metadata['retrieved_table_names'])
            )

    selector_result = None
    if hasattr(engine, '_retriever') and hasattr(engine.retriever, 'selector_result_'):
        selector_result = engine.retriever.selector_result_
    elif 'selector_result' in resp.metadata:
        selector_result = resp.metadata['selector_result']
    if selector_result is not None:
        for sel in selector_result.selections:
            source = ['graph', 'doc', 'table'][sel.index]
            if source == 'graph' and provenance_graph is not None:
                provenance_graph.reason = sel.reason
            elif source == 'doc' and provenance_doc is not None:
                provenance_doc.reason = sel.reason
            elif source == 'table' and provenance_table is not None:
                provenance_table.reason = sel.reason

    sources = []
    if provenance_graph:
        sources.append(provenance_graph)
    if provenance_doc:
        for chunk in provenance_doc.chunks:
            sources.append(chunk)
    if provenance_table:
        sources.append(provenance_table)
    cited_provenance = [sources[c - 1] for c in streaming_output.citation_mapping]

    return DiscoveryResults(
        final_response=final_resp,
        selection_results=selection_results,
        provenance_graph=provenance_graph,
        provenance_doc=provenance_doc,
        provenance_table=provenance_table,
        cited_provenance=cited_provenance
    )


def format_doc(title, doc: str, remove_title: bool = True):
    if remove_title:
        if doc.startswith(f'{title}\n'):
            doc = re.sub(rf'^{title}\n', '', doc)
    doc = re.sub(r'^Section::::(.*?)\.$',
                 lambda m: '#' * min(6 + m.group(1).count('.'), 6) + f' {m.group(1).split(".")[-1].lstrip(":")}', doc,
                 flags=re.MULTILINE)
    doc = doc.replace('$', '\$')
    return doc


def truncate_text(text: str, max_len: int = 36):
    if len(text) > max_len:
        return text[:max_len - 3] + '...'
    return text


def display_cited_provenance(result: DiscoveryResults):
    cols = []
    for idx, src in enumerate(result.cited_provenance, 1):
        if not cols:
            cols = st.columns(4)
        col = cols.pop(0)

        if len(result.cited_provenance) == 1 and hasattr(result.cited_provenance[0], 'cypher'):
            height = None
        elif not any(hasattr(src, 'sql_query') for src in result.cited_provenance):
            height = 300
        else:
            height = 350

        with col:
            if hasattr(src, 'cypher'):
                prov_g = src
                with st.container(border=True, height=height):
                    st.write(f'###### [{idx}] NBA Knowledge Graph')
                    if prov_g.result_json is not None:
                        if len(prov_g.node2name) > 50:
                            st.warning(f'Graph too large to display', icon="⚠️")
                        else:
                            if len(prov_g.node2name) == 0:
                                st.warning(f'Empty graph', icon="⚠️")
                            else:
                                g = graph2graphviz(
                                    node2name=prov_g.node2name,
                                    edge2label=prov_g.edge2label,
                                    id2color={}
                                )
                                with st.container(border=False, height=None if len(prov_g.node2name) <= 3 else 200):
                                    with stylable_container(
                                            key="graph_",
                                            css_styles="""
                                            {
                                                background-color: #1a1b24;
                                                border-radius: 8px;
                                                padding: 1.5em;
                                            }
                                            """,
                                    ):
                                        st.graphviz_chart(g)
                        s = json.dumps(json.loads(prov_g.result_json))
                        if len(s) >= 500:
                            s = s[:500] + '...'
                        st.code(s, language='json')
                    with st.expander('View Cypher'):
                        st.code(prov_g.cypher, language='cypher')


            elif hasattr(src, 'text'):
                with st.container(border=True, height=height):
                    text = format_doc(src.title, src.text, remove_title=True)
                    enwiki_url = f'https://en.wikipedia.org/wiki/{src.title.replace(" ", "_")}'
                    title = truncate_text(src.title)
                    text = f'###### [{idx}] [{title}]({enwiki_url})\n{text}'
                    st.write(text)
            elif hasattr(src, 'sql_query'):
                prov_t = src
                with st.container(border=True, height=height):
                    if prov_t.result_str is not None:
                        for i, table in enumerate(prov_t.sql_tables):
                            enwiki_url = f'https://en.wikipedia.org/wiki/{table.page_title.replace(" ", "_")}#{table.section_title.replace(" ", "_")}'
                            title = truncate_text(f'{table.page_title} - {table.section_title}')
                            if i == 0:
                                st.write(f'###### [{idx}] [{title}]({enwiki_url})')
                            else:
                                st.write(f'###### [{title}]({enwiki_url})')
                            df = pd.DataFrame(table.rows, columns=table.columns)
                            st.dataframe(df, hide_index=True, height=230)
                        st.code(prov_t.result_str, language='json')
                    else:
                        st.warning(f'No result found', icon="⚠️")
                    with st.expander('View SQL'):
                        st.code(prov_t.sql_query, language='sql')


def display_full_provenance(result: DiscoveryResults):
    col_graph, col_doc, col_table = st.columns(3)

    with col_graph:
        if not result.provenance_graph:
            st.warning(f'Graph source not selected', icon="⚠️")
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
                    elif len(prov_g.node2name) == 0:
                        st.warning(f'Empty graph', icon="⚠️")
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
            st.warning(f'Document source not selected', icon="⚠️")
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
                    text = format_doc(chunk.title, chunk.text, remove_title=True)
                    enwiki_url = f'https://en.wikipedia.org/wiki/{chunk.title.replace(" ", "_")}'
                    text = f'###### [doc] [{chunk.title}]({enwiki_url})\n{text}'
                    st.write(text)

    with col_table:
        if not result.provenance_table:
            st.warning(f'Table source not selected', icon="⚠️")
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
        with open(f'data/cmdbench/q_{src}.json') as f:
            q = json.load(f)
            questions.extend([d['question'] for d in q])
            for d in q:
                q2src[d['question']] = src
    return questions, q2src


def add_spinner(fn):
    def fn_with_spinner(*args, **kwargs):
        with st.spinner():
            return fn(*args, **kwargs)

    return fn_with_spinner


@st.cache_data()
def load_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


# Nl2CypherRetriever.generate_query = add_spinner(Nl2CypherRetriever.generate_query)
# RetrieverQueryEngine.query = add_spinner(RetrieverQueryEngine.query)


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

    # st.markdown(" <style> div[class^='st-emotion-cache-16txtl3 '] { padding-top: 2rem; } </style> ",
    #             unsafe_allow_html=True)

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        llm = st.selectbox("LLM", ["gpt-4o", "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"])
        temperature = st.select_slider("Temperature", options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        c1, c2 = st.columns(2)
        with c1:
            doc_top_k = st.slider("Doc Top K", 1, 10, 3)
        with c2:
            table_top_k = st.slider("Table Top K", 1, 10, 3)

        c1, c2 = st.columns(2)
        with c1:
            summary_type = st.radio("Source Summary", ["Human", "Complex", "Basic"])
        with c2:
            selection_mode = st.radio("Selection Mode", ["Select all", "MultiSelector", "SingleSelector"])

        st.divider()
        questions, q2src = load_questions()
        questions = ["What is the height of LeBron James?"] + questions
        q2src["What is the height of LeBron James?"] = "graph"

        key_word = st.text_input("Search sample question by keyword", "")
        sources = st.multiselect("Filter by sources", ["graph", "doc", "table"], ["graph", "doc", "table"],
                                 label_visibility='collapsed')
        questions = [q for q in questions if (key_word.lower() in q.lower() or key_word == "")
                     and q2src.get(q) in sources]
        st.dataframe(pd.DataFrame({f'{len(questions)} matches': questions}),
                     hide_index=True, height=220)

        st.divider()
        emb_model = st.selectbox("Embedding Model", ["BAAI/bge-base-en-v1.5"])
        query_engine_type = st.radio("Query Engine", ["RouterQueryEngine", "Custom"], index=1)

        st.divider()
        debug_mode = st.toggle("Debug Mode", False)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    svg = load_file('assets/r2d2.svg')
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <style>
        .centered-content-icon {
            display: flex;
            flex-direction: column;  /* Stack items vertically */
            justify-content: center;
            align-items: center;
            # height: 65vh;  /* 65% of the viewport height */
            padding-top: 15vh;  /* 30% from the top */
        }
        # .text-style {
        #     font-size: 32px;  /* Larger text size */
        #     color: rgba(255, 255, 255, 0.2);  /* White text at 40% opacity */
        #     font-family: 'Press Start 2P', monospace;  /* Pixel-like, monospace font */
        # }
        .main-icon {
            fill: #FFFFFF;
            opacity: 0.3;
            height: 250px;
        }
        .block-container div[data-testid="column"]:nth-of-type(5)
            {
                # border:1px solid red;
                text-align: end;
            } 
    
        .block-container div[data-testid="column"]:nth-of-type(6)
            {
                # border:1px solid blue;
            } 
        
        button[kind="secondary"] {
            border: 1px solid #444444;  
            color: #aaaaaa; 
            height: 42px;
            width: 440px;
            text-align: left;
            display: inline-block;
            border-radius: 3px;
            background-color: transparent; 
            padding: 0px 12px;
            opacity: 0.7;
            # cursor: pointer;
            # padding: 8px 16px;
            # text-decoration: none;
            # font-size: 16px;
            # margin: 4px 2px;
        }
        button[kind="secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid #444444; 
            color: #aaaaaa; 
        }
        button[kind="secondary"]:focus {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid #444444; 
            color: #aaaaaa; 
        }
        </style>
    """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .centered-content-icon { display: none; }
            .block-container div[data-testid="column"]:nth-of-type(5) { display: none; }
            .block-container div[data-testid="column"]:nth-of-type(6) { display: none; }
            .stChatInput {
                position: fixed;
                z-index: 1000;
                bottom: 50px;
            }
            </style>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="centered-content-icon">{svg}', unsafe_allow_html=True)

    # if len(st.session_state.messages) == 0:
    #     _, col, _ = st.columns([0.05, 0.92, 0.03])
    #     with col:
    #         question = st.chat_input(' Ask anything about NBA', key='initial_')
    # else:
    #     question = st.chat_input(' Ask anything about NBA')
    # st.write('')
    from streamlit import _bottom
    _bottom.container(height=50, border=False)

    _, col, _ = st.columns([0.05, 0.92, 0.03])
    with col:
        question = st.chat_input(' Ask anything about NBA')

    _, _, _, _, col1, col2, _, _, _, _ = st.columns([0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.01, 0.01, 0.01, 0.07])
    with col1:
        st.write('')
        # example_anthony = st.button('⛹️ &nbsp;Introduce Anthony Davis')
        example_lebron_stephen = st.button('⛹️ &nbsp;Compare LeBron James to Stephen Curry')
        example_big_ticker = st.button('🎫 &nbsp;Which NBA player was nicknamed "The Big Ticket"?')
    with col2:
        st.write('')
        example_mvp = st.button('🏆 &nbsp;Who has won the most NBA Most Valuable Player Awards?')
        example_kevin = st.button('📅 &nbsp;When did Kevin Garnett leave the Boston Celtics?')

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message(message["role"]).write(message["content"])
        elif message['role'] == 'assistant':
            st.chat_message(message["role"], avatar='assets/m2d2-icon.png').write(message["content"].final_response)
            _, body = st.columns([0.001, 0.999], gap="large")
            with body:
                display_cited_provenance(message["content"])
                with st.expander('View full provenance'):
                    display_full_provenance(message["content"])
            # st.caption('')
            st.divider()

    if not any([question, example_lebron_stephen, example_big_ticker, example_mvp,
                example_kevin]):
        st.stop()

    if not openai_api_key:
        if len(st.session_state.messages) == 0:
            _bottom.error("Please enter your OpenAI API Key in the sidebar.")
        else:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        st.stop()

    if question:
        question = question.strip()
    elif example_lebron_stephen:
        question = "Compare LeBron James to Stephen Curry"
    elif example_kevin:
        question = "When did Kevin Garnett leave the Boston Celtics?"
    elif example_big_ticker:
        question = "Which NBA player was nicknamed 'The Big Ticket'?"
    elif example_mvp:
        question = "Who has won the most NBA Most Valuable Player Awards?"

    st.markdown("""
            <style>
            .centered-content-icon { display: none; }
            .block-container div[data-testid="column"]:nth-of-type(5) { display: none; }
            .block-container div[data-testid="column"]:nth-of-type(6) { display: none; }
            .stChatInput {
                position: fixed;
                z-index: 1000;
                bottom: 50px;
            }
            </style>
        """, unsafe_allow_html=True)

    # if len(st.session_state.messages) == 0:
    #     st.chat_input(' Ask anything about NBA')

    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message('user').write(question)

    result = run_data_discovery(question, llm, emb_model, doc_top_k, table_top_k, summary_type, selection_mode,
                                debug=debug_mode, query_engine_type=query_engine_type, temperature=temperature,
                                avatar='assets/m2d2-icon.png')
    st.session_state.messages.append({"role": "assistant", "content": result})
    _, body = st.columns([0.001, 0.999], gap="large")
    with body:
        display_cited_provenance(result)
        with st.expander('View full provenance'):
            display_full_provenance(result)


if __name__ == '__main__':
    main()
