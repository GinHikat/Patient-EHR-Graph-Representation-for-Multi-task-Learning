from neo4j import GraphDatabase
from core.config import settings

driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_AUTH))

def get_db_driver():
    return driver

def close_db_driver():
    driver.close()
