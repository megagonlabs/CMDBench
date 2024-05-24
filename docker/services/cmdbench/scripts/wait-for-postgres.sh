#!/bin/bash
set -e
cmd="$@"

until psql -h "$PGHOST" -d "$PGDATABASE" -U "$PGUSER" -c '\l'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 5
done

>&2 echo "Postgres is up - executing command"
exec $cmd