from neo4j import GraphDatabase
import os
from sqlalchemy import create_engine, text
from pymongo import MongoClient


def test_neo4j():
    driver = GraphDatabase.driver(
        'bolt://localhost:7687',
        auth=(os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD'])
    )
    with driver.session() as session:
        resp = session.run("MATCH (p:Player {name: 'LeBron James'}) RETURN p.name")
        resp = [record for record in resp]
        assert resp[0]['p.name'] == 'LeBron James'
    print('Neo4j connectivity test passed.')


def test_mongo():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['nba-datalake']['wiki-documents']
    docs = db.find({'wikipedia_title': 'LeBron James'})
    docs = [doc for doc in docs]
    assert len(docs) == 1 and docs[0]['wikipedia_title'] == 'LeBron James'
    print('MongoDB connectivity test passed.')


def test_postgres_select_nba_wikisql():
    user = os.environ['PGUSER']
    db = 'nba'
    host = 'localhost'
    conn_str = f'postgresql+psycopg://{user}@{host}/{db}'
    engine = create_engine(conn_str)
    with engine.connect() as connection:
        sql = "SELECT player FROM nba_wikisql.t_1_10015132_1 LIMIT 3"
        result = connection.execute(text(sql))
        # print()
        for row in result:
            # print("Player:", row.player)
            assert isinstance(row.player, str)
            assert len(row.player) > 0
    print('Postgres connectivity test passed.')


if __name__ == "__main__":
    test_neo4j()
    test_mongo()
    test_postgres_select_nba_wikisql()
