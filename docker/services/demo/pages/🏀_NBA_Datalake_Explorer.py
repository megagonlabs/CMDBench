import streamlit as st
import argparse
from typing import Iterable, List
from pymongo import MongoClient
import random
import json
import pandas as pd
import tempfile
import neo4j
from neo4j import GraphDatabase
import sys

sys.path.append('.')
import dataclasses
import requests
import graphviz
import os
import re
from streamlit.logger import get_logger

logger = get_logger(__name__)


def _split_to_multilines(s, maxlen=20):
    tokens = s.split()
    res = []
    curr_line_tokens = []
    curr_line_len = -1
    for t in tokens:
        if curr_line_len >= maxlen:
            res.append(' '.join(curr_line_tokens))
            curr_line_tokens = []
            curr_line_len = -1
        curr_line_tokens.append(t)
        curr_line_len += len(t) + 1
    if len(curr_line_tokens):
        res.append(' '.join(curr_line_tokens))
    return '\n'.join(res)


def neo4j2graphviz(result, id2color, dpi=64, transparent=True):
    random.seed(0)

    id2node = {}
    edges = {}

    for record in result:
        for relationship in record['path'].relationships:
            start_id = str(relationship.start_node.id)
            end_id = str(relationship.end_node.id)
            edges[(start_id, end_id)] = relationship.type
        for node in record['path'].nodes:
            id2node[str(node.id)] = node

    color_profile = {
        'orange': {
            'fill': '#ffcc99',
            'edge': '#ff6600',
        },
        'blue': {
            'fill': '#99ccff',
            'edge': '#0066ff',
        },
        'green': {
            'fill': '#99ff99',
            'edge': '#009900',
        },
        'red': {
            'fill': '#ff9999',
            'edge': '#cc0000',
        },
        'yellow': {
            'fill': '#ffff99',
            'edge': '#ffcc00',
        },
        'purple': {
            'fill': '#cc99ff',
            'edge': '#6600cc',
        },
        'gray': {
            'fill': '#eeeeee',
            'edge': '#666666',
        },
    }

    g = graphviz.Digraph('G')
    g.attr(label='', dpi=str(dpi),
           rankdir='LR', outputorder='edgesfirst', splines='splines',
           compound='true', fontname='Sans Not-Rotated', fontsize='16',
           labelloc='t', labeljust='l', newrank='true',
           bgcolor='transparent' if transparent else '')

    edge_attrs = dict(arrowsize='0.5', penwidth='1.5', arrowhead='dot',
                      color=color_profile['gray']['edge'], style='dashed',
                      fontname='Sans Not-Rotated', fontsize='8', fontcolor=color_profile['gray']['edge'],
                      tailport='_', headport='_')
    for (sv, tv), edge_type in edges.items():
        g.edge(sv, tv, label=edge_type, **edge_attrs)

    node_attrs = dict(shape='rect', height='0.3', margin='0.22,0.055',
                      fontsize='8', fontname='Sans Not-Rotated', style='rounded,filled,bold')
    for node_id in id2node:
        label = id2node[node_id]['name']
        label = _split_to_multilines(label, maxlen=30)
        color = id2color.get(node_id, 'gray')
        color_attrs = dict(
            fillcolor=color_profile[color]['fill'],
            fontcolor=color_profile[color]['edge'],
            color=color_profile[color]['edge']
        )
        g.node(node_id, label=label, **node_attrs, **color_attrs)

    # st.write(g.source)

    return g, len(id2node)


# @st.cache_resource()
def init_mongodb_conn(mongodb_url: str):
    return MongoClient(mongodb_url)


@st.cache_data()
def get_neo4j_entity_names(_driver: GraphDatabase):
    with _driver.session() as session:
        result = session.run('MATCH (n) RETURN n')
        return [record['n']['name'] for record in result]


def format_doc(text: list[str]):
    # return '\n'.join(text)
    doc = '\n'.join(text)
    doc = '### ' + doc
    doc = re.sub(r'^Section::::(.*?)\.$',
                 lambda m: '#' * (3 + m.group(1).count('.')) + f' {m.group(1).split(".")[-1].lstrip(":")}', doc,
                 flags=re.MULTILINE)
    doc = doc.replace('$', '\$')
    return doc


def get_documents(_mongodb_client: MongoClient, db_name: str, entity_name: str, doc_search_mode: str) -> List[dict]:
    db = _mongodb_client[db_name]['wiki-documents']
    if doc_search_mode == 'Title only':
        query = {'wikipedia_title': {'$regex': entity_name, '$options': 'i'}}
    elif doc_search_mode == 'Full text':
        query = {
            '$or': [
                {'wikipedia_title': {'$regex': entity_name, '$options': 'i'}},
                {'text': {'$elemMatch': {'$regex': entity_name, '$options': 'i'}}}
            ]
        }
    else:
        raise ValueError(f'Invalid doc_search_mode: {doc_search_mode}')
    docs = list(db.find(query))
    docs = sorted(docs, key=lambda doc: len(doc['wikipedia_title']))
    return [{
        'title': doc['wikipedia_title'],
        'enwiki_url': f'https://en.wikipedia.org/wiki/{doc["wikipedia_title"].replace(" ", "_")}',
        'text': format_doc(doc['text'])
    } for doc in docs]


def table_to_markdown(headers: List[str], table: List[List[str]]) -> str:
    markdown = '|'.join(headers) + '\n'
    markdown += '|'.join(['---' for _ in headers]) + '\n'
    for row in table:
        row = [str(cell) for cell in row]
        markdown += '|'.join(row) + '\n'
    return markdown


def get_tables(_mongodb_client: MongoClient, db_name: str,
               entity_name: str,
               table_search_mode: str = 'Full table') -> List[dict]:
    if db_name == 'nba-datalake-v2':
        collection_names = ['wiki-tables']
    elif db_name == 'nba-datalake':
        # collection_names = [f'wiki-tables_{split}' for split in ('train', 'dev', 'test')]
        collection_names = ['wiki-tables']
    else:
        raise ValueError(f'Invalid db_name: {db_name}')

    if table_search_mode == 'Full table':
        query = {
            '$or': [
                {'page_title': {'$regex': entity_name, '$options': 'i'}},
                {'heading_path': {'$elemMatch': {'$regex': entity_name, '$options': 'i'}}},
                {'rows': {'$elemMatch': {'$elemMatch': {'$regex': entity_name, '$options': 'i'}}}}
            ]
        }
    elif table_search_mode == 'Title only':
        query = {'page_title': {'$regex': entity_name, '$options': 'i'}}
    else:
        raise ValueError(f'Invalid table_search_mode: {table_search_mode}')

    res = []
    for collection in collection_names:
        db = _mongodb_client[db_name][collection]
        tables = list(db.find(query))

        for table in tables:
            df = pd.DataFrame(table['rows'], columns=table['header'])
            if db_name == 'nba-datalake-v2':
                heading = ' / '.join([table['page_title']] + table['heading_path'])
            elif db_name == 'nba-datalake':
                split = collection.split('_')[-1]
                heading = f'{table["page_title"]} / {table["section_title"]} (wikisql_split: {split})'
            res.append({
                'enwiki_url': f'https://en.wikipedia.org/wiki/{table["page_title"].replace(" ", "_")}',
                'heading': heading,
                'markdown': table_to_markdown(table['header'], table['rows']),
                'df': df
            })
    return res


def get_graph(_neo4j_driver: GraphDatabase, entity: str, graph_search_mode: str = 'Subject only'):
    with _neo4j_driver.session() as session:
        nodes_resp = session.run(
            'MATCH (n) WHERE toLower(n.name) CONTAINS toLower($entity) RETURN n',
            entity=entity
        )
        color_map = {}
        nodes = []
        for record in nodes_resp:
            node = record['n']
            color_map[str(node.id)] = 'green'
            node = dict(node._properties)
            nodes.append(node)

        if graph_search_mode == 'Subject only':
            cypher = 'MATCH path=(n)-[*1]->(m) WHERE toLower(n.name) CONTAINS toLower($entity) RETURN path'
        elif graph_search_mode == 'All triples':
            cypher = 'MATCH path=(n)-[*1]-(m) WHERE toLower(n.name) CONTAINS toLower($entity) RETURN path'
        else:
            raise ValueError(f'Invalid graph_search_mode: {graph_search_mode}')

        subgraph = session.run(
            cypher,  # 'MATCH (n {name: $entity})-[r]->(m) RETURN n, r, m',
            entity=entity
        )
        g, num_nodes_in_g = neo4j2graphviz(subgraph, color_map)
        return nodes, g, num_nodes_in_g


def datalake_browser(args):
    mongodb_client = init_mongodb_conn(args.mongodb_url)
    neo4j_driver = GraphDatabase.driver(
        args.neo4j_url,
        auth=(os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD'))
    )
    neo4j_driver.verify_connectivity()

    with st.sidebar:
        entity_names = get_neo4j_entity_names(neo4j_driver)
        col1, col2 = st.columns([1, 1])
        with col1:
            graph_search_mode = st.radio('Graph search mode', ['Subj. only', 'All triples'], index=0)

        with col2:
            doc_search_mode = st.radio('Doc search mode', ['Title only', 'Full text'], index=0)

        table_search_mode = st.radio('Table search mode', ['Title only', 'Full table'], index=1)

        if graph_search_mode == 'Subj. only':
            graph_search_mode = 'Subject only'

        db_name = st.selectbox('Select database', ['nba-datalake',])

        examples = [
            '1963–64 Boston Celtics season',
            'player stats',
            'LeBron James',
            'Los Angeles Lakers',
            'John Long',
            'Stephen Curry',
        ]
        example_queries = examples + [entity for entity in entity_names if entity not in examples]
        entity = st.selectbox('Example queries', example_queries)

        query = st.text_input('Query', entity)

        search = st.button('Search')

    if not search:
        st.stop()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        with st.spinner('Querying Neo4j...'):
            nodes, g, num_nodes_in_g = get_graph(neo4j_driver, query, graph_search_mode)

        st.write(f"`nodes: {len(nodes)}`")

        if num_nodes_in_g >= 50:
            st.warning(f'Cannot display the graph with {num_nodes_in_g} nodes', icon="⚠️")
            with tempfile.NamedTemporaryFile(suffix='.gv') as tmp1, \
                    tempfile.NamedTemporaryFile(suffix='.pdf') as tmp2:
                g.render(filename=tmp1.name, outfile=tmp2.name, format='pdf')
                st.download_button('📥 Download graph (.pdf)', data=tmp2.read(), file_name='graph.pdf')
        elif num_nodes_in_g > 0:
            with st.expander(f'[1-hop subgraph] "{query}"', expanded=True):
                with st.container(height=300 if num_nodes_in_g >= 8 else None, border=False):
                    st.graphviz_chart(g.source)

        if len(nodes) == 0:
            st.warning(f'"{query}" not found in the graph', icon="⚠️")
        else:
            for i, node in enumerate(nodes):
                title = re.sub(query, f':green[\g<0>]', node['name'], flags=re.IGNORECASE)
                with st.expander(f'[node] {title}', expanded=i == 0):
                    st.write('https://www.wikidata.org/wiki/' + node['wikidata_id'])
                    # with st.container(height=200 if len(node) >= 8 else None, border=False):
                    df = pd.DataFrame(list(node.items()), columns=['property', 'value'])
                    st.dataframe(df, hide_index=True, use_container_width=True)

    with col2:
        with st.spinner('Querying MongoDB...'):
            docs = get_documents(mongodb_client, db_name, query, doc_search_mode)
        st.write(f"`docs: {len(docs)}`")
        if len(docs) >= 50:
            st.warning(f'Only displaying the first 50 documents', icon="⚠️")
            docs = docs[:50]
        for i, doc in enumerate(docs):
            title = re.sub(query, f':green[\g<0>]', doc['title'], flags=re.IGNORECASE)
            with st.expander('[doc] ' + title, expanded=i == 0):
                # st.text_area(doc['enwiki_url'], doc['text'], height=400)
                st.write(doc['enwiki_url'])
                with st.container(height=600 if len(doc['text']) > 1000 else None, border=False):
                    text = doc['text']
                    text = re.sub(query, f':green[\g<0>]', text, flags=re.IGNORECASE)
                    st.write(text)
        if len(docs) == 0:
            st.warning(f'"{query}" not found in the documents', icon="⚠️")

    with col3:
        with st.spinner(f'Querying MongoDB...'):
            tables = get_tables(mongodb_client, db_name, query, table_search_mode)
        st.write(f"`tables: {len(tables)}`")
        if len(tables) >= 50:
            st.warning(f'Only displaying the first 50 tables', icon="⚠️")
            tables = tables[:50]
        for i, table in enumerate(tables):
            heading = re.sub(query, f':green[\g<0>]', table['heading'], flags=re.IGNORECASE)
            with st.expander(f'[table] {heading}',
                             expanded=i == 0):
                st.write(table['enwiki_url'])
                df = table['df'].style.applymap(
                    # dark theme green: #3dd56d
                    lambda x: 'color: #158237; font-weight: bold;' if query.lower() in str(x).lower() else '')
                st.dataframe(df, hide_index=True, use_container_width=True)
        if len(tables) == 0:
            st.warning(f'"{query}" not found in the tables', icon="⚠️")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb_url', default='mongodb://mongo:27017')
    parser.add_argument('--neo4j_url', default='bolt://neo4j:7687')
    args = parser.parse_args()
    logger.info(args)
    logger.info('')

    st.set_page_config(
        page_title='NBA Datalake',
        page_icon='🏀',
        layout='wide',
    )
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)
    # st.title('🎈 NBA datalake')

    # datalake_browser, = st.tabs(['datalake_browser'])

    datalake_browser(args)


if __name__ == '__main__':
    main()
