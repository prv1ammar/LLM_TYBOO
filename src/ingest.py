"""
Enterprise Data Ingestion Pipeline
Avec détection et gestion des doublons via hashing SHA-256 / ou chunks
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

# Logging structuré
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Chemin du registre local des fichiers déjà ingérés
REGISTRY_PATH = os.getenv("INGEST_REGISTRY_PATH", ".ingest_registry.json")


class DuplicateRegistry:
    """
    Registre local des documents déjà ingérés.
    Stocke les hashes SHA-256 des fichiers et des chunks
    pour éviter les doublons à deux niveaux :
    - Niveau fichier : même fichier re-soumis
    - Niveau chunk   : même contenu textuel dans un fichier différent
    """

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = registry_path
        self.registry: Dict = self._load()

    def _load(self) -> Dict:
        """Charge le registre depuis le disque"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"files": {}, "chunks": set(), "stats": {"total_ingested": 0, "total_skipped": 0}}

    def _save(self):
        """Persiste le registre sur le disque"""
        # Les sets ne sont pas sérialisables en JSON
        data = {
            "files": self.registry["files"],
            "chunks": list(self.registry["chunks"]),
            "stats": self.registry["stats"]
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _after_load_fix(self):
        """Convertit la liste chunks en set après chargement JSON"""
        if isinstance(self.registry.get("chunks"), list):
            self.registry["chunks"] = set(self.registry["chunks"])

    @staticmethod
    def hash_file(file_path: str) -> str:
        """Calcule le hash SHA-256 d'un fichier"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        """Calcule le hash SHA-256 d'un texte"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_file_known(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si un fichier a déjà été ingéré.
        Retourne (déjà_connu, hash_actuel)
        """
        self._after_load_fix()
        current_hash = self.hash_file(file_path)
        file_name = os.path.basename(file_path)

        if file_name in self.registry["files"]:
            stored_hash = self.registry["files"][file_name]["hash"]
            if stored_hash == current_hash:
                return True, current_hash  # Fichier identique → doublon
            else:
                return False, current_hash  # Même nom, contenu modifié → mise à jour
        return False, current_hash

    def is_chunk_known(self, chunk_text: str) -> bool:
        """Vérifie si un chunk identique existe déjà dans la KB"""
        self._after_load_fix()
        chunk_hash = self.hash_text(chunk_text)
        return chunk_hash in self.registry["chunks"]

    def register_file(self, file_path: str, file_hash: str, chunk_count: int):
        """Enregistre un fichier dans le registre"""
        self._after_load_fix()
        file_name = os.path.basename(file_path)
        self.registry["files"][file_name] = {
            "hash": file_hash,
            "path": file_path,
            "chunk_count": chunk_count,
            "ingested_at": datetime.utcnow().isoformat()
        }
        self._save()

    def register_chunk(self, chunk_text: str):
        """Enregistre le hash d'un chunk"""
        self._after_load_fix()
        self.registry["chunks"].add(self.hash_text(chunk_text))

    def update_stats(self, ingested: int = 0, skipped: int = 0):
        """Met à jour les statistiques globales"""
        self._after_load_fix()
        self.registry["stats"]["total_ingested"] += ingested
        self.registry["stats"]["total_skipped"] += skipped
        self._save()

    def get_stats(self) -> Dict:
        """Retourne les statistiques du registre"""
        self._after_load_fix()
        return {
            **self.registry["stats"],
            "known_files": len(self.registry["files"]),
            "known_chunks": len(self.registry["chunks"])
        }

    def clear(self):
        """Réinitialise le registre (utile pour les tests)"""
        self.registry = {"files": {}, "chunks": set(), "stats": {"total_ingested": 0, "total_skipped": 0}}
        self._save()
        logger.info("🗑️  Registre réinitialisé.")


class DataIngestor:
    """
    Enterprise Data Ingestion Pipeline
    Avec détection des doublons à deux niveaux (fichier + chunk)
    """

    def __init__(self, collection_name: str = "knowledge_base"):
        self.vector_store = VectorStore(collection_name=collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        self.registry = DuplicateRegistry()

    def load_pdf(self, file_path: str) -> str:
        """Extrait le texte d'un fichier PDF"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def load_text(self, file_path: str) -> str:
        """Charge le texte d'un fichier texte"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _process_file(self, file_path: str) -> Tuple[List[Dict], int, int]:
        """
        Traite un fichier et retourne les chunks nouveaux (non-doublons).

        Returns:
            (chunks_nouveaux, nb_total_chunks, nb_chunks_ignores)
        """
        file_name = os.path.basename(file_path)

        # ── Niveau 1 : vérification au niveau fichier ──────────────────────────
        is_known, file_hash = self.registry.is_file_known(file_path)

        if is_known:
            logger.warning(f"  ⏭️  IGNORÉ (doublon exact) : {file_name}")
            return [], 0, 0

        # Chargement du contenu
        if file_path.endswith(".pdf"):
            content = self.load_pdf(file_path)
        else:
            content = self.load_text(file_path)

        if not content.strip():
            logger.warning(f"  ⚠️  Fichier vide : {file_name}")
            return [], 0, 0

        # Découpage en chunks
        raw_chunks = self.text_splitter.split_text(content)
        total_chunks = len(raw_chunks)
        new_chunks = []
        skipped_chunks = 0

        # ── Niveau 2 : vérification au niveau chunk ────────────────────────────
        for i, chunk in enumerate(raw_chunks):
            if self.registry.is_chunk_known(chunk):
                skipped_chunks += 1
                logger.debug(f"    ↩️  Chunk {i} identique dans la KB — ignoré")
                continue

            new_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": file_name,
                    "path": file_path,
                    "chunk_index": i,
                    "file_hash": file_hash,
                    "ingested_at": datetime.utcnow().isoformat()
                }
            })

        if new_chunks:
            # Enregistrement du fichier dans le registre
            self.registry.register_file(file_path, file_hash, len(new_chunks))
            # Enregistrement des hashes de chunks
            for chunk_doc in new_chunks:
                self.registry.register_chunk(chunk_doc["text"])

        logger.info(
            f"  ✅ {file_name} → {len(new_chunks)}/{total_chunks} chunks nouveaux "
            f"({skipped_chunks} doublons ignorés)"
        )
        return new_chunks, total_chunks, skipped_chunks

    def ingest_directory(self, directory_path: str) -> Dict:
        """
        Ingère tous les fichiers supportés d'un répertoire.
        Retourne un rapport d'ingestion.
        """
        logger.info(f"📂 Scan du répertoire : {directory_path}")

        extensions = ["*.pdf", "*.txt", "*.md"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory_path, ext)))

        if not files:
            logger.warning("⚠️  Aucun fichier supporté trouvé.")
            return {"status": "empty", "files_found": 0}

        logger.info(f"📄 {len(files)} fichier(s) trouvé(s). Démarrage de l'ingestion...")

        report = {
            "files_found": len(files),
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_new": 0,
            "chunks_skipped": 0,
            "errors": []
        }

        all_new_chunks = []

        for file_path in files:
            file_name = os.path.basename(file_path)
            logger.info(f"  ⏳ Traitement : {file_name}")

            try:
                new_chunks, total, skipped = self._process_file(file_path)

                if total == 0 and not new_chunks:
                    # Fichier ignoré (doublon exact)
                    report["files_skipped"] += 1
                    continue

                all_new_chunks.extend(new_chunks)
                report["files_processed"] += 1
                report["chunks_new"] += len(new_chunks)
                report["chunks_skipped"] += skipped

            except Exception as e:
                logger.error(f"  ❌ Erreur sur {file_name} : {str(e)}")
                report["errors"].append({"file": file_name, "error": str(e)})

        # Ingestion dans Qdrant des chunks nouveaux seulement
        if all_new_chunks:
            logger.info(f"🚀 Envoi de {len(all_new_chunks)} nouveaux chunks vers Qdrant...")
            self.vector_store.add_documents(all_new_chunks)
            self.registry.update_stats(ingested=len(all_new_chunks))
            logger.info("✅ Ingestion terminée.")
        else:
            logger.info("ℹ️  Aucun nouveau chunk à ingérer (tout est déjà en base).")
            self.registry.update_stats(skipped=report["chunks_skipped"])

        # Rapport final
        report["registry_stats"] = self.registry.get_stats()
        logger.info(f"\n📊 Rapport d'ingestion : {json.dumps(report, indent=2, ensure_ascii=False)}")
        return report

    def ingest_single_file(self, file_path: str) -> Dict:
        """Ingère un seul fichier avec gestion des doublons"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        file_name = os.path.basename(file_path)
        logger.info(f"📄 Ingestion du fichier : {file_name}")

        try:
            new_chunks, total, skipped = self._process_file(file_path)

            if new_chunks:
                self.vector_store.add_documents(new_chunks)
                self.registry.update_stats(ingested=len(new_chunks), skipped=skipped)

            return {
                "file": file_name,
                "status": "skipped" if not new_chunks and total == 0 else "processed",
                "chunks_new": len(new_chunks),
                "chunks_skipped": skipped,
                "registry_stats": self.registry.get_stats()
            }
        except Exception as e:
            logger.error(f"❌ Erreur : {str(e)}")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant (with duplicate detection)")
    parser.add_argument("--dir", default="documents", help="Répertoire contenant les documents")
    parser.add_argument("--collection", default="knowledge_base", help="Nom de la collection Qdrant")
    parser.add_argument("--clear-registry", action="store_true", help="Réinitialise le registre des doublons")
    parser.add_argument("--stats", action="store_true", help="Affiche les statistiques du registre")

    args = parser.parse_args()

    ingestor = DataIngestor(collection_name=args.collection)

    if args.clear_registry:
        ingestor.registry.clear()

    if args.stats:
        print(json.dumps(ingestor.registry.get_stats(), indent=2))
    elif not os.path.exists(args.dir):
        os.makedirs(args.dir)
        logger.info(f"📁 Répertoire '{args.dir}' créé. Placez vos fichiers et relancez.")
    else:
        ingestor.ingest_directory(args.dir)
