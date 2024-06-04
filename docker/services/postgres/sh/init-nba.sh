#!/bin/bash
set -e

psql -a -v ON_ERROR_STOP=1 -v db="$PGDATABASE_NBA" <<-EOSQL
	CREATE DATABASE :db;
EOSQL

psql -qb -v ON_ERROR_STOP=1 --d "$PGDATABASE_NBA" -f /init/nba/metadata.sql
psql -qb -v ON_ERROR_STOP=1 --d "$PGDATABASE_NBA" -f /init/nba/nba_wikisql.sql