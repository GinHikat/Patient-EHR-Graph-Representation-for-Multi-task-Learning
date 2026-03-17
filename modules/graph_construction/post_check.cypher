// delete only 1 label nodes, this will ensure deduplication and non bursting pattern
MATCH (n)
WHERE size(labels(n)) = 1
WITH n LIMIT 1000
DETACH DELETE n;

// Only id is nonsense
MATCH (p:Test)
WHERE p.name IS NULL
DETACH DELETE p

//ensure bi-directional edge
MATCH (d1:Drug:DB:Test)-[r:INTERACTS_WITH]->(d2:Drug:DB:Test)
WITH DISTINCT d1, d2, r
DELETE r
MERGE (d1)-[:INTERACTS_WITH]-(d2);

//deduplicate edges
MATCH (a)-[r:INTERACTS_WITH]->(b)
WITH a, b, collect(r) AS rels
WHERE size(rels) > 1
FOREACH (r IN tail(rels) | DELETE r);