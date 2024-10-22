import logging
import os
import gc
import uuid
import hashlib
import json
from datetime import datetime
from typing import List, Tuple, Generator, Optional, Dict, Any, Set
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
import chromadb
from chromadb.config import Settings
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 500
    chunk_overlap: int = 20
    batch_size: int = 100
    max_threads: int = mp.cpu_count()
    cache_dir: str = ".cache"
    processed_docs_file: str = "processed_documents.json"
    debug_mode: bool = False

class DocumentTracker:
    """Tracks processed documents to prevent re-processing."""
    
    def __init__(self, processed_docs_file: str):
        self.processed_docs_file = processed_docs_file
        self.processed_docs: Dict[str, Dict[str, Any]] = self._load_processed_docs()

    def _load_processed_docs(self) -> Dict[str, Dict[str, Any]]:
        """Load processed documents from tracking file."""
        if os.path.exists(self.processed_docs_file):
            try:
                with open(self.processed_docs_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error reading {self.processed_docs_file}, starting fresh")
                return {}
        return {}

    def _save_processed_docs(self):
        """Save processed documents to tracking file."""
        with open(self.processed_docs_file, 'w') as f:
            json.dump(self.processed_docs, f, indent=2)

    def get_document_hash(self, file_path: str) -> str:
        """Calculate document hash using content and metadata."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            file_stat = os.stat(file_path)
            metadata = f"{file_stat.st_size}{file_stat.st_mtime}"
            return hashlib.sha256(content + metadata.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            raise

    def is_document_processed(self, file_path: str) -> bool:
        """Check if document has been processed."""
        doc_hash = self.get_document_hash(file_path)
        return doc_hash in self.processed_docs

    def mark_document_processed(self, file_path: str, metadata: Dict[str, Any]):
        """Mark document as processed with metadata."""
        doc_hash = self.get_document_hash(file_path)
        self.processed_docs[doc_hash] = {
            'file_path': file_path,
            'processed_date': datetime.now().isoformat(),
            'metadata': metadata
        }
        self._save_processed_docs()

    def get_unprocessed_documents(self, file_paths: List[str]) -> List[str]:
        """Filter out already processed documents."""
        return [fp for fp in file_paths if not self.is_document_processed(fp)]

class DocumentProcessor:
    """Handles document processing and text extraction."""
    
    SUPPORTED_FORMATS = {'.pdf', '.txt'}

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.setup_cache()
        
    def setup_cache(self):
        """Set up cache directory."""
        if not os.path.exists(self.config.cache_dir):
            os.makedirs(self.config.cache_dir)
            
    def validate_file(self, file_path: str) -> bool:
        """Validate if file exists and is supported."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return True

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by removing special characters and normalizing."""
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
        return ' '.join(tokens)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with error handling."""
        try:
            logger.info(f"Extracting text from {file_path}")
            text = extract_text(file_path)
            return self.preprocess_text(text)
        except PDFSyntaxError as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {str(e)}")
            raise

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using LangChain's splitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return splitter.split_text(text)

    def process_document(self, file_path: str) -> Tuple[Generator[List[str], None, None], int]:
        """Process document and return chunks with memory monitoring."""
        self.validate_file(file_path)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        text = self.extract_text_from_pdf(file_path)
        chunks = self.split_text(text)
        total_chunks = len(chunks)

        del text
        gc.collect()

        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after processing: {current_memory:.2f} MB")

        def chunk_generator():
            for i in range(0, total_chunks, self.config.batch_size):
                yield chunks[i:i + self.config.batch_size]

        return chunk_generator(), total_chunks

class ChromaDBManager:
    """Manages ChromaDB operations."""

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()

    def _initialize_client(self) -> chromadb.Client:
        """Initialize ChromaDB client with optional persistence."""
        settings = Settings(
            persist_directory=self.persist_directory if self.persist_directory else None,
            anonymized_telemetry=False
        )
        return chromadb.Client(settings)

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(name=self.collection_name)

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to collection with metadata."""
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadata if metadata else [{} for _ in range(len(documents))]
            )
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to collection: {str(e)}")
            raise

    def search(self, 
              query: str, 
              n_results: int = 5, 
              filter_dict: Optional[Dict[str, Any]] = None,
              search_type: str = "similarity") -> Dict[str, Any]:
        """Search documents with multiple options."""
        try:
            return self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict,
                include=['distances', 'metadatas', 'documents']
            )
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

class SemanticSearchPipeline:
    """Main pipeline for semantic search operations."""

    def __init__(self, 
                 config: ProcessingConfig,
                 collection_name: str,
                 persist_directory: Optional[str] = None):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.db_manager = ChromaDBManager(collection_name, persist_directory)
        self.doc_tracker = DocumentTracker(config.processed_docs_file)

    def process_and_index_document(self, file_path: str) -> bool:
        """Process and index a single document with tracking."""
        try:
            if self.doc_tracker.is_document_processed(file_path):
                logger.info(f"Skipping already processed document: {file_path}")
                return False

            start_time = datetime.now()
            logger.info(f"Starting processing of {file_path}")

            chunk_generator, total_chunks = self.document_processor.process_document(file_path)

            with tqdm(total=total_chunks, desc=f"Indexing {Path(file_path).name}", unit="chunk") as pbar:
                for batch in chunk_generator:
                    metadata = [{
                        'source': file_path,
                        'timestamp': datetime.now().isoformat(),
                        'chunk_index': i
                    } for i in range(len(batch))]
                    
                    self.db_manager.add_documents(batch, metadata)
                    pbar.update(len(batch))

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")

            # Mark document as processed
            self.doc_tracker.mark_document_processed(
                file_path,
                {
                    'processing_time': processing_time,
                    'total_chunks': total_chunks,
                    'chunk_size': self.config.chunk_size
                }
            )
            return True

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False

    def process_multiple_documents(self, file_paths: List[str], skip_errors: bool = True) -> Dict[str, bool]:
        """Process multiple documents with parallel processing."""
        results: Dict[str, bool] = {}
        unprocessed_docs = self.doc_tracker.get_unprocessed_documents(file_paths)
        
        if not unprocessed_docs:
            logger.info("No new documents to process")
            return results

        logger.info(f"Processing {len(unprocessed_docs)} documents")
        
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            future_to_path = {
                executor.submit(self.process_and_index_document, file_path): file_path 
                for file_path in unprocessed_docs
            }
            
            for future in tqdm(future_to_path, desc="Processing documents", unit="doc"):
                file_path = future_to_path[future]
                try:
                    results[file_path] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    results[file_path] = False
                    if not skip_errors:
                        raise

        successful = sum(1 for v in results.values() if v)
        logger.info(f"Successfully processed {successful}/{len(unprocessed_docs)} documents")
        return results

    def search_documents(self, 
                        query: str, 
                        n_results: int = 5, 
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search documents with optional filters."""
        return self.db_manager.search(query, n_results, filters)

# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = ProcessingConfig(
        chunk_size=500,
        chunk_overlap=20,
        batch_size=100,
        debug_mode=True,
        processed_docs_file="processed_docs.json"
    )

    # Initialize pipeline
    pipeline = SemanticSearchPipeline(
        config=config,
        collection_name="books",
        persist_directory="./chroma_persist"
    )

    # Process multiple documents
    document_paths = [
        "document1.pdf",
        "document2.pdf",
        "document3.pdf"
    ]

    try:
        # Process all documents
        results = pipeline.process_multiple_documents(document_paths)
        
        # Print processing results
        for doc_path, success in results.items():
            print(f"Document {doc_path}: {'Successfully processed' if success else 'Failed'}")

        # Perform a search across all documents
        search_results = pipeline.search_documents(
            query="bottleneck",
            n_results=5,
            filters=None  # No filters to search across all documents
        )
        
        # Process results
        if search_results['documents'][0]:
            for i, (doc, distance, metadata) in enumerate(zip(
                search_results['documents'][0],
                search_results['distances'][0],
                search_results['metadatas'][0]
            )):
                print(f"\nResult {i+1} (Distance: {distance:.4f}):")
                print(f"Source: {metadata['source']}")
                print(f"Timestamp: {metadata['timestamp']}")
                print(doc[:200] + "..." if len(doc) > 200 else doc)
                
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")