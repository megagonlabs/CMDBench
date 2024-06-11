#!/bin/bash
set -e

# Wait for Neo4j to start listening on its HTTP port
until curl --silent --fail "http://localhost:7474"; do
  >&2 echo "Neo4j is unavailable - sleeping"
  sleep 1
done
