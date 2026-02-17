import os
import glob
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

class DataIngestor:
    """Enterprise Data Ingestion Pipeline"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.vector_store = VectorStore(collection_name=collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_pdf(self, file_path: str) -> str:
        """Load text from a PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def load_text(self, file_path: str) -> str:
        """Load text from a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def ingest_directory(self, directory_path: str):
        """Ingest all supported files from a directory"""
        print(f"📂 Scanning directory: {directory_path}")
        
        # Supported extensions
        extensions = ['*.pdf', '*.txt', '*.md']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory_path, ext)))
            
        if not files:
            print("⚠️ No supported files found.")
            return

        print(f"📄 Found {len(files)} files. Starting ingestion...")
        
        all_chunks = []
        for file_path in files:
            file_name = os.path.basename(file_path)
            print(f"  ⏳ Processing: {file_name}")
            
            try:
                if file_path.endswith('.pdf'):
                    content = self.load_pdf(file_path)
                else:
                    content = self.load_text(file_path)
                
                # Split content into chunks
                chunks = self.text_splitter.split_text(content)
                
                # Create detailed metadata
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": file_name,
                            "path": file_path,
                            "chunk_index": i
                        }
                    })
            except Exception as e:
                print(f"  ❌ Error processing {file_name}: {str(e)}")

        if all_chunks:
            print(f"🚀 Upserting {len(all_chunks)} chunks to Qdrant...")
            self.vector_store.add_documents(all_chunks)
            print("✅ Ingestion complete.")
        else:
            print("⚠️ No content extracted from files.")

if __name__ == "__main__":
    # Example: Ingest data from the 'documents' folder
    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--dir", default="documents", help="Directory containing documents")
    parser.add_argument("--collection", default="knowledge_base", help="Qdrant collection name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
        print(f"📁 Created '{args.dir}' directory. Place your PDFs/TXTs/MDs there and run again.")
    else:
        ingestor = DataIngestor(collection_name=args.collection)
        ingestor.ingest_directory(args.dir)
