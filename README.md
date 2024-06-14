# CMDBench
The public repos of the CMDBench paper

## Usage

Run the following commands to start the CMDBench datalake. The following databases will be exposed at various ports:
- Neo4j: `localhost:7687`
- MongoDB: `localhost:27017`
- PostgreSQL: `localhost:5432`

```bash
git clone https://github.com/rit-git/cmdbench.git
cd cmdbench/docker/compose
docker compose up
```

When you are done, you can stop with `Ctrl+C` and then run:
```bash
docker compose down
```

### Notes:
- You can find the database credentials in [docker/compose/.env](docker/compose/.env)
- If there are port conflicts, you can change the ports in [docker/compose/.env](docker/compose/.env)
