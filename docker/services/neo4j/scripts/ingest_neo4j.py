from neo4j import GraphDatabase
import argparse
import os
import time
import re
from tqdm import trange, tqdm
import json
import pandas as pd


def convert_columns_to_json(df):
    """
    Attempts to convert each column in the DataFrame from a JSON string to a Python object.
    It only converts a column if all rows in that column can be parsed as JSON without error.

    :param df: pandas DataFrame with potential JSON strings in its columns.
    :return: DataFrame with columns converted where applicable.
    """
    for col in df.columns:
        all_rows_are_json = True
        for value in df[col]:
            if pd.isna(value):
                continue

            if not isinstance(value, str) or value[0] not in ('{', '['):
                # If any value in the column is not a string, skip JSON parsing check for this column
                all_rows_are_json = False
                break
            try:
                json.loads(value)
            except json.JSONDecodeError:
                # If any row in the column cannot be parsed as JSON, skip this column
                all_rows_are_json = False
                break

        if all_rows_are_json:
            print(f"Converting column {col} to JSON")
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else x)  # Convert the entire column to JSON

    return df


class BulkNeo4jImporterPerLabelFormat:
    def __init__(self, uri, user, password, database, batch_size=100, overwrite=False):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        while True:
            try:
                self.driver.verify_connectivity()
                break
            except Exception as e:
                print("Neo4j is not ready yet. Waiting for 1 second...")
                time.sleep(1)
        self.database = database
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.label2id_column = {}

    def close(self):
        self.driver.close()

    def delete_all_data(self):
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

    def import_nodes(self, file_path):
        print(f"Importing nodes from {file_path}")
        df = pd.read_csv(file_path, dtype=str)
        df = convert_columns_to_json(df)
        label = os.path.basename(file_path).replace('.csv', '')
        assert df.columns[0].endswith(f':ID({label})')
        id_column = df.columns[0].replace(f':ID({label})', '')
        self.label2id_column[label] = id_column
        df.rename(columns={df.columns[0]: id_column}, inplace=True)

        records = df.to_dict('records')

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(records), self.batch_size)):
                batch = records[i:i + self.batch_size]
                batch = [{
                    'properties': {k: v for k, v in record.items() if isinstance(v, (list, dict)) or pd.notna(v)}
                } for record in batch]
                session.execute_write(self._create_nodes_batch, batch, label)

    @staticmethod
    def _create_nodes_batch(tx, batch, label):
        query = """
        UNWIND $batch AS row
        CREATE (n:%s)
        SET n += row.properties
        RETURN count(n)
        """ % label
        tx.run(query, batch=batch)

    def import_relations(self, file_path):
        print(f"Importing relations from {file_path}")

        relation_label = os.path.basename(file_path).replace('.csv', '').replace('relation_', '')

        df = pd.read_csv(file_path, dtype=str)
        df = convert_columns_to_json(df)

        subj_label = re.match(r':START_ID\((.*?)\)', df.columns[0]).group(1)
        obj_label = re.match(r':END_ID\((.*?)\)', df.columns[1]).group(1)

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(df), self.batch_size)):
                batch = df[i:i + self.batch_size]
                batch = [{
                    'start_id': start_id,
                    'end_id': end_id,
                    'properties': {k: v for k, v in zip(df.columns[2:], properties) if isinstance(v, (list, dict)) or pd.notna(v)}
                } for _, (start_id, end_id, *properties) in batch.iterrows()]
                session.execute_write(self._create_relations_batch, batch, relation_label,
                                      subj_label, obj_label)

    def _create_relations_batch(self, tx, batch, relation_label, subj_label, obj_label):
        query = f"""
        UNWIND $batch AS row
        MATCH (startNode:%s), (endNode:%s)
        WHERE startNode.%s = row.start_id AND endNode.%s = row.end_id
        CREATE (startNode)-[r:%s]->(endNode)
        SET r += row.properties
        RETURN count(*) AS rel_count
        """ % (subj_label, obj_label,
               self.label2id_column[subj_label], self.label2id_column[obj_label],
               relation_label)
        tx.run(query, batch=batch)

    def delete_all_indexes(self):
        with self.driver.session() as session:
            # Retrieve all indexes (this will include both node and relationship indexes)
            indexes = session.run("SHOW INDEXES")
            for index in indexes:
                # Extract the index name - adjust the field key if necessary based on your Neo4j version
                index_name = index["name"]
                # Construct the Cypher command to drop the index by its name
                drop_command = f"DROP INDEX {index_name}"
                # Execute the command to drop the index
                session.run(drop_command)
                print(f"Dropped index: {index_name}")

    def create_index(self):
        with self.driver.session(database=self.database) as session:
            for label, id_column in self.label2id_column.items():
                index_query = f"CREATE INDEX FOR (n:{label}) ON (n.{id_column})"
                session.run(index_query)
                print(f"Index created for {label}.{id_column}")

    def ingest(self, input_dir):
        # check if the database is empty
        with self.driver.session(database=self.database) as session:
            result = session.run('MATCH (n) RETURN count(n) as count')
            num_nodes = result.single()["count"]
        if num_nodes > 0 and not self.overwrite:
            raise ValueError(
                f"Database {self.database} is not empty. Use --overwrite to delete all data before import.")

        self.delete_all_indexes()
        self.delete_all_data()
        files = os.listdir(input_dir)
        for f in files:
            if f.endswith('.csv') and not f.startswith('relation_'):
                self.import_nodes(os.path.join(input_dir, f))
        self.create_index()
        for f in files:
            if f.startswith('relation_') and f.endswith('.csv'):
                self.import_relations(os.path.join(input_dir, f))
        self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', default='data/neo4j/nba_v3', help='Directory containing CSV files')
    parser.add_argument('--uri', default='bolt://localhost:7687', help='URI for Neo4j database')
    parser.add_argument('--database', default='neo4j', help='Name of Neo4j database')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size for import')
    parser.add_argument('--overwrite', action='store_true', help='Delete all data before import')
    args = parser.parse_args()
    print(args)
    print()

    if os.path.exists(os.path.join(args.csv_dir, 'entities.csv')):
        raise NotImplementedError()
    else:
        import_class = BulkNeo4jImporterPerLabelFormat

    user, password = os.environ.get('NEO4J_AUTH').split('/')

    importer = import_class(
        uri=args.uri,
        user=user,
        password=password,
        database=args.database,
        batch_size=args.batch_size,
        overwrite=args.overwrite
    )
    importer.ingest(args.csv_dir)
    print("Data imported successfully.")


if __name__ == "__main__":
    main()
