# CMDBench
The public repos of the CMDBench paper

## a few useful docker commands
```
$ docker compose -f docker/compose/compose.yaml up --build # rebuild and start all services
$ docker compose -f docker/compose/compose.yaml down # stop and remove
```

## start the dockers and get access to the database
```
$ docker compose -f docker/compose/compose.yaml run --rm cmdbench bash
[+] Creating 1/0
 ✔ Container compose-postgres-1  Created
[+] Running 1/1
 ✔ Container compose-postgres-1  Started
root@79ea19764de4:/cmdbench# psql -U megagon nba
psql (16.3 (Ubuntu 16.3-1.pgdg24.04+1))
Type "help" for help.

nba=#\dn
         List of schemas
    Name     |       Owner       
-------------+-------------------
 metadata    | megagon
 nba_wikisql | megagon
 public      | pg_database_owner
(3 rows)

nba=# \d metadata.*
              Table "metadata.nba_context"
    Column     | Type  | Collation | Nullable | Default 
---------------+-------+-----------+----------+---------
 id            | text  |           |          | 
 page_title    | text  |           |          | 
 section_title | text  |           |          | 
 caption       | text  |           |          | 
 doc           | jsonb |           |          | 

nba=# SELECT id, page_title, section_title, caption FROM metadata.nba_context LIMIT 3;
      id      |   page_title   | section_title |   caption   
--------------+----------------+---------------+-------------
 2-12093318-3 | 1947 BAA draft | Draft         | Draft
 2-12093691-3 | 1948 BAA draft | Draft         | Draft
 2-12093691-4 | 1948 BAA draft | Other picks   | Other picks
(3 rows)

nba=# 
\q
root@79ea19764de4:/cmdbench# pytest -s
=========================== test session starts ===========================
platform linux -- Python 3.12.3, pytest-8.2.1, pluggy-1.5.0
rootdir: /cmdbench
collected 1 item                                                          

src/table/test_postgres_select.py 
Player: Quincy Acy
Player: Hassan Adams
Player: Alexis Ajinça
.

============================ 1 passed in 0.38s ============================


```
