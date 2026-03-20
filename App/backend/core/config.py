import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_AUTH: str = os.getenv("NEO4J_AUTH", "")

settings = Settings()
