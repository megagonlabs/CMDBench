# CMDBench
The public repos of the CMDBench paper

## start the dockers and gte access to the database
```
(base) aaron@ip-10-0-158-213:~/code/cmdbench$ docker compose run --rm cmdbench
[+] Creating 1/0
 ✔ Container cmdbench-postgres-1  Running                             0.0s 
root@79ea19764de4:/cmdbench# ls
Dockerfile  README.md     main.py           sql
LICENSE     compose.yaml  requirements.txt  wait-for-postgres.sh
root@79ea19764de4:/cmdbench# psql -U megagon cmdbench
psql (16.3 (Ubuntu 16.3-1.pgdg24.04+1))
Type "help" for help.

cmdbench=#\dn
         List of schemas
    Name     |       Owner       
-------------+-------------------
 metadata    | megagon
 nba_wikisql | megagon
 public      | pg_database_owner
(3 rows)

cmdbench=# \d metadata.*
              Table "metadata.nba_context"
    Column     | Type  | Collation | Nullable | Default 
---------------+-------+-----------+----------+---------
 id            | text  |           |          | 
 page_title    | text  |           |          | 
 section_title | text  |           |          | 
 caption       | text  |           |          | 
 doc           | jsonb |           |          | 

cmdbench=# SELECT id, page_title, section_title, caption FROM metadata.nba_context LIMIT 3;
      id      |   page_title   | section_title |   caption   
--------------+----------------+---------------+-------------
 2-12093318-3 | 1947 BAA draft | Draft         | Draft
 2-12093691-3 | 1948 BAA draft | Draft         | Draft
 2-12093691-4 | 1948 BAA draft | Other picks   | Other picks
(3 rows)

cmdbench=# 
\q
root@79ea19764de4:/cmdbench# python main.py 
Player: Quincy Acy
Player: Hassan Adams
Player: Alexis Ajinça
```
