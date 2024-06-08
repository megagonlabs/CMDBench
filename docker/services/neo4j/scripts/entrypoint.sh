#!/bin/bash
set -e

# Delegate to the original entrypoint to start Neo4j
tini -g -- /startup/docker-entrypoint.sh "$@" &

# Check that Neo4j has started
echo "Waiting for Neo4j to start..."
while ! nc -z localhost 7474; do
  sleep 5
done
echo "Neo4j has started."

# Execute your Python script
echo "Running Python script..."
python3 repo/scripts/ingest_neo4j.py --csv_dir repo/nba_v3/

# Keep the container running (if needed)f
tail -f /dev/null
