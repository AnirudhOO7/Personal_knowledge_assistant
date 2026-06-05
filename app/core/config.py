from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    #embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    #Chunk size
    chunk_size: int = 800
    chunk_overlap: int = 200

    #vector database
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "personal_rag"

    #retrieval
    top_k: int = 5

    #LLM
    anthropic_api_key: str = "" 
    llm_model: str = "claude-haiku-4-5"
    llm_temperature: float = 0
    llm_max_tokens: int = 1024

    #path
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
