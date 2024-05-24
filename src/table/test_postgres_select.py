import os
from sqlalchemy import create_engine, text

def test_postgres_select_nba_wikisql():
    user = os.getenv('PGUSER')
    host = os.getenv('PGHOST')
    db = os.getenv('PGDATABASE')
    conn_str = f'postgresql+psycopg://{user}@{host}/{db}'
    engine = create_engine(conn_str)
    with engine.connect() as connection:
        sql = "SELECT player FROM nba_wikisql.t_1_10015132_1 LIMIT 3"
        result = connection.execute(text(sql))
        for row in result:
            print("Player:", row.player)
            assert isinstance(row.player, str)
            assert len(row.player) > 0

if __name__ == "__main__":
    test_postgres_select_nba_wikisql()