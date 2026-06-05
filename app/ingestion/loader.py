from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

def load_file(file_path: str) -> list:
    """Load a file and return list of LangChain Documents with metadata."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    logger.info(f"Loading: {path.name}")

    if path.suffix.lower() == ".pdf":
        docs = PyMuPDFLoader(str(path)).load()
    elif path.suffix.lower() == ".md":
        docs = UnstructuredMarkdownLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    logger.info(f"Loaded {len(docs)} pages/sections from {path.name}")
    return docs


def load_directory(dir_path: str) -> list:
    """Load all supported files from a directory."""
    path = Path(dir_path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning(f"No supported files found in {dir_path}")
        return []

    logger.info(f"Found {len(files)} files in {dir_path}")

    all_docs = []
    for file in files:
        try:
            all_docs.extend(load_file(str(file)))
        except Exception as e:
            logger.error(f"Failed to load {file.name}: {e}")
            continue

    return all_docs