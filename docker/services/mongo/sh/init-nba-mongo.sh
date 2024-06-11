mongoimport --host localhost --port 27017 --db nba-datalake --collection wiki-documents --type json --file /init/nba-datalake.wiki-documents.json --jsonArray
mongoimport --host localhost --port 27017 --db nba-datalake --collection wiki-tables --type json --file /init/nba-datalake.wiki-tables.json --jsonArray
