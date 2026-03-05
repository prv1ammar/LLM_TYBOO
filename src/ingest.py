"""
ingest.py — Document Ingestion Pipeline
=========================================
PURPOSE:
  Processes documents (PDF, TXT, Markdown) and stores them in the Qdrant
  vector database so the RAG system can search them.

WHAT HAPPENS TO EACH DOCUMENT:
  1. File is read and text is extracted (pypdf for PDFs, direct read for text)
  2. Text is split into overlapping chunks of ~1000 characters
     (overlap of 100 chars ensures context isn't lost at chunk boundaries)
  3. Each chunk is checked against a SHA-256 registry to avoid duplicates
  4. New chunks are embedded with BGE-M3 and stored in Qdrant

TWO-LEVEL DUPLICATE DETECTION:
  Level 1 — File level:
    If a file with the same name AND same content hash is found in the registry,
    the entire file is skipped. Re-running ingest.py never re-processes the same file.

  Level 2 — Chunk level:
    Even if a file is new, individual chunks are checked by content hash.
    This catches cases where the same text appears in multiple files.

  The registry is stored in .ingest_registry.json (configurable via INGEST_REGISTRY_PATH).

HOW TO RUN:
  # Basic: ingest all files in a directory
  python ingest.py --dir documents --collection knowledge_base

  # Check what's already been ingested
  python ingest.py --stats

  # Reset the registry (force re-ingestion of all files on next run)
  python ingest.py --clear-registry

  # Use a different collection name
  python ingest.py --dir contracts --collection legal_docs

SUPPORTED FILE FORMATS:
  .pdf  — Text extracted from all pages using pypdf
  .txt  — Read directly as UTF-8
  .md   — Read directly as UTF-8 (Markdown treated as plain text)

CHUNK SETTINGS:
  chunk_size    = 1000 characters
  chunk_overlap = 100 characters
  These are set in DataIngestor.__init__() and can be adjusted for your use case.
  Smaller chunks = more precise retrieval but more Qdrant points.
  Larger chunks  = more context per result but less precise matching.
"""

import os
import glob
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

# Structured log format with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Path where the duplicate registry JSON file is stored
# Set INGEST_REGISTRY_PATH in .env to change this
REGISTRY_PATH = os.getenv("INGEST_REGISTRY_PATH", ".ingest_registry.json")


class DuplicateRegistry:
    """
    Persists a record of every file and chunk that has been ingested.
    Prevents the same content from being stored in Qdrant twice.

    The registry is a JSON file with this structure:
      {
        "files": {
          "contract.pdf": {
            "hash": "sha256...",
            "path": "/app/documents/contract.pdf",
            "chunk_count": 12,
            "ingested_at": "2025-01-01T00:00:00"
          }
        },
        "chunks": ["sha256hash1", "sha256hash2", ...],
        "stats": {"total_ingested": 120, "total_skipped": 5}
      }
    """

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = registry_path
        self.registry = self._load()

    def _load(self) -> Dict:
        """Load registry from disk, or return empty registry if file doesn't exist yet."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # First run — initialize empty registry
        return {
            "files": {},
            "chunks": [],   # Stored as list in JSON, converted to set in memory
            "stats": {"total_ingested": 0, "total_skipped": 0}
        }

    def _save(self):
        """Persist current registry state to disk."""
        data = {
            "files": self.registry["files"],
            "chunks": list(self.registry["chunks"]),  # Convert set to list for JSON
            "stats": self.registry["stats"]
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _ensure_set(self):
        """
        Convert the chunks list to a set for O(1) lookup performance.
        Called before any operation that reads or writes to chunks.
        JSON doesn't support sets, so we store as list and convert on load.
        """
        if isinstance(self.registry.get("chunks"), list):
            self.registry["chunks"] = set(self.registry["chunks"])

    @staticmethod
    def hash_file(file_path: str) -> str:
        """
        Compute SHA-256 hash of a file's binary content.
        Used to detect if a file has changed since last ingestion.
        Reads in 64KB blocks to handle large files without loading all into RAM.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        """
        Compute SHA-256 hash of a text string.
        Used to detect duplicate chunks across different files.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_file_known(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a file has already been ingested.

        Returns:
            (True, hash)  — file is identical to what's already ingested → skip it
            (False, hash) — file is new or has changed → process it

        How it works:
          - If filename is not in registry → definitely new → process
          - If filename IS in registry AND hash matches → unchanged → skip
          - If filename IS in registry BUT hash differs → file was modified → re-process
        """
        self._ensure_set()
        current_hash = self.hash_file(file_path)
        filename = os.path.basename(file_path)

        if filename in self.registry["files"]:
            stored_hash = self.registry["files"][filename]["hash"]
            if stored_hash == current_hash:
                return True, current_hash   # Exact duplicate
            else:
                return False, current_hash  # Modified file — re-ingest
        return False, current_hash          # New file

    def is_chunk_known(self, chunk_text: str) -> bool:
        """
        Check if an identical chunk already exists in the knowledge base.
        Uses the chunk's SHA-256 hash for comparison.
        """
        self._ensure_set()
        return self.hash_text(chunk_text) in self.registry["chunks"]

    def register_file(self, file_path: str, file_hash: str, chunk_count: int):
        """
        Record a successfully ingested file in the registry.
        Call this after all chunks from the file have been stored in Qdrant.
        """
        self._ensure_set()
        filename = os.path.basename(file_path)
        self.registry["files"][filename] = {
            "hash": file_hash,
            "path": file_path,
            "chunk_count": chunk_count,
            "ingested_at": datetime.utcnow().isoformat()
        }
        self._save()

    def register_chunk(self, chunk_text: str):
        """
        Record a chunk's hash to prevent it from being stored again in the future.
        Call this for each new chunk after it has been sent to Qdrant.
        """
        self._ensure_set()
        self.registry["chunks"].add(self.hash_text(chunk_text))
        # Note: we don't save here for performance — saved in bulk via register_file

    def update_stats(self, ingested: int = 0, skipped: int = 0):
        """Update the running totals and persist to disk."""
        self._ensure_set()
        self.registry["stats"]["total_ingested"] += ingested
        self.registry["stats"]["total_skipped"] += skipped
        self._save()

    def get_stats(self) -> Dict:
        """Return current registry statistics."""
        self._ensure_set()
        return {
            **self.registry["stats"],
            "known_files": len(self.registry["files"]),
            "known_chunks": len(self.registry["chunks"]),
        }

    def clear(self):
        """
        Reset the entire registry.
        After clearing, all files will be re-ingested on the next run.
        Does NOT delete anything from Qdrant — only clears the tracking file.
        """
        self.registry = {
            "files": {},
            "chunks": set(),
            "stats": {"total_ingested": 0, "total_skipped": 0}
        }
        self._save()
        logger.info("Registry cleared — all files will be re-ingested on next run.")


class DataIngestor:
    """
    High-level pipeline that reads files, splits them into chunks,
    deduplicates, and stores new content in Qdrant.
    """

    def __init__(self, collection_name: str = "knowledge_base"):
        """
        Args:
            collection_name: Which Qdrant collection to store documents in.
                             The collection is created automatically if it doesn't exist.
        """
        self.vector_store = VectorStore(collection_name=collection_name)
        self.registry = DuplicateRegistry()

        # RecursiveCharacterTextSplitter splits text hierarchically:
        # First tries to split on "\n\n" (paragraphs), then "\n" (lines),
        # then " " (words), then "" (characters) as a last resort.
        # This preserves semantic units better than hard character splits.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,     # Maximum characters per chunk
            chunk_overlap=100,   # Overlap between consecutive chunks (preserves context)
            separators=["\n\n", "\n", " ", ""]
        )

    def _extract_pdf_text(self, file_path: str) -> str:
        """
        Extract all text from a PDF file by reading each page.
        Pages are joined with newlines to preserve document structure.
        """
        reader = PdfReader(file_path)
        return "\n".join(
            page.extract_text()
            for page in reader.pages
            if page.extract_text()  # Skip blank pages
        )

    def _extract_text_file(self, file_path: str) -> str:
        """Read a plain text file (TXT or MD) as UTF-8."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _process_file(self, file_path: str) -> Tuple[List[Dict], int, int]:
        """
        Process a single file through the full ingestion pipeline.

        Steps:
          1. Check registry — skip if file is already ingested (Level 1 dedup)
          2. Extract text based on file extension
          3. Split text into chunks
          4. For each chunk, check if it already exists (Level 2 dedup)
          5. Register new chunks in the registry

        Returns:
            (new_chunks, total_chunks, skipped_chunks)
            new_chunks:     list of {"text": ..., "metadata": ...} dicts ready for Qdrant
            total_chunks:   how many chunks the file was split into
            skipped_chunks: how many chunks were duplicates
        """
        filename = os.path.basename(file_path)

        # Level 1: Check if entire file is a duplicate
        is_known, file_hash = self.registry.is_file_known(file_path)
        if is_known:
            logger.warning(f"  Skipping (exact duplicate): {filename}")
            return [], 0, 0

        # Extract text based on file type
        if file_path.endswith(".pdf"):
            content = self._extract_pdf_text(file_path)
        else:
            content = self._extract_text_file(file_path)

        if not content.strip():
            logger.warning(f"  Empty file, nothing to ingest: {filename}")
            return [], 0, 0

        # Split into chunks
        raw_chunks = self.text_splitter.split_text(content)
        total_chunks = len(raw_chunks)
        new_chunks = []
        skipped_chunks = 0

        # Level 2: Check each chunk for duplicates
        for i, chunk_text in enumerate(raw_chunks):
            if self.registry.is_chunk_known(chunk_text):
                skipped_chunks += 1
                logger.debug(f"    Chunk {i} is duplicate — skipping")
                continue

            # New chunk — add to the list for Qdrant storage
            new_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "path": file_path,
                    "chunk_index": i,
                    "file_hash": file_hash,
                    "ingested_at": datetime.utcnow().isoformat()
                }
            })

        # Register the file and its chunks in the registry
        if new_chunks:
            self.registry.register_file(file_path, file_hash, len(new_chunks))
            for chunk_doc in new_chunks:
                self.registry.register_chunk(chunk_doc["text"])

        logger.info(
            f"  {filename} — {len(new_chunks)}/{total_chunks} new chunks "
            f"({skipped_chunks} duplicates skipped)"
        )
        return new_chunks, total_chunks, skipped_chunks

    def ingest_directory(self, directory_path: str) -> Dict:
        """
        Ingest all supported files in a directory.
        Processes PDF, TXT, and MD files recursively (flat directory only).

        Args:
            directory_path: Path to the directory containing documents.

        Returns:
            Report dict with: files_found, files_processed, files_skipped,
            chunks_new, chunks_skipped, errors, registry_stats
        """
        logger.info(f"Scanning directory: {directory_path}")

        # Collect all supported files
        files = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            files.extend(glob.glob(os.path.join(directory_path, ext)))

        if not files:
            logger.warning("No supported files found (PDF, TXT, MD)")
            return {"status": "empty", "files_found": 0}

        logger.info(f"Found {len(files)} file(s). Starting ingestion...")

        report = {
            "files_found": len(files),
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_new": 0,
            "chunks_skipped": 0,
            "errors": [],
        }

        all_new_chunks = []

        for file_path in files:
            filename = os.path.basename(file_path)
            logger.info(f"Processing: {filename}")
            try:
                new_chunks, total, skipped = self._process_file(file_path)
                if total == 0 and not new_chunks:
                    report["files_skipped"] += 1
                    continue
                all_new_chunks.extend(new_chunks)
                report["files_processed"] += 1
                report["chunks_new"] += len(new_chunks)
                report["chunks_skipped"] += skipped
            except Exception as e:
                logger.error(f"  Error processing {filename}: {str(e)}")
                report["errors"].append({"file": filename, "error": str(e)})

        # Send all new chunks to Qdrant in one batch (more efficient than per-file)
        if all_new_chunks:
            logger.info(f"Storing {len(all_new_chunks)} new chunks in Qdrant...")
            self.vector_store.add_documents(all_new_chunks)
            self.registry.update_stats(ingested=len(all_new_chunks))
            logger.info("Ingestion complete.")
        else:
            logger.info("No new chunks to store — all content already in knowledge base.")
            self.registry.update_stats(skipped=report["chunks_skipped"])

        report["registry_stats"] = self.registry.get_stats()
        logger.info(f"Report: {json.dumps(report, indent=2, ensure_ascii=False)}")
        return report

    def ingest_single_file(self, file_path: str) -> Dict:
        """
        Ingest a single file.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            Dict with: file, status, chunks_new, chunks_skipped, registry_stats

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        logger.info(f"Ingesting single file: {filename}")

        new_chunks, total, skipped = self._process_file(file_path)

        if new_chunks:
            self.vector_store.add_documents(new_chunks)
            self.registry.update_stats(ingested=len(new_chunks), skipped=skipped)

        return {
            "file": filename,
            "status": "skipped" if (not new_chunks and total == 0) else "processed",
            "chunks_new": len(new_chunks),
            "chunks_skipped": skipped,
            "registry_stats": self.registry.get_stats(),
        }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest documents into the LLM_TYBOO knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --dir documents --collection knowledge_base
  python ingest.py --dir contracts  --collection legal_docs
  python ingest.py --stats
  python ingest.py --clear-registry
        """
    )
    parser.add_argument("--dir", default="documents",
                        help="Directory containing files to ingest (default: documents)")
    parser.add_argument("--collection", default="knowledge_base",
                        help="Qdrant collection name (default: knowledge_base)")
    parser.add_argument("--clear-registry", action="store_true",
                        help="Reset the duplicate registry (forces re-ingestion of all files)")
    parser.add_argument("--stats", action="store_true",
                        help="Show registry statistics and exit")
    args = parser.parse_args()

    ingestor = DataIngestor(collection_name=args.collection)

    if args.clear_registry:
        ingestor.registry.clear()

    if args.stats:
        print(json.dumps(ingestor.registry.get_stats(), indent=2))
    elif not os.path.exists(args.dir):
        os.makedirs(args.dir)
        logger.info(f"Created '{args.dir}' directory. Add your documents there and re-run.")
    else:
        ingestor.ingest_directory(args.dir)
