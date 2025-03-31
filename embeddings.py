import os
import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.config import DataType
import weaviate.classes.config as wc
import pypdf
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import uuid

load_dotenv()

weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080/v1")
pdf_path = os.getenv("POSTGRES_MANUAL_PDF", "docs/postgres_manual.pdf")
chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
embedding_model_name = os.getenv("EMBEDDING_MODEL", "microsoft/codebert-base")
embedding_class_name = os.getenv("EMBEDDING_CLASS", "PostgresManualChunk")


def stream_chunks_from_pdf(pdf_path, chunk_size=1000, overlap=200):
    """
    Lee el PDF de forma secuencial (página a página) y genera chunks de texto con un solapamiento.
    Esto evita cargar el documento completo en memoria.
    """
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        current_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                current_text += page_text + "\n"
                while len(current_text) >= chunk_size:
                    chunk = current_text[:chunk_size]
                    yield chunk
                    current_text = current_text[chunk_size - overlap:]
        if current_text.strip():
            yield current_text.strip()


def create_weaviate_schema(client):
    """
    Crea una colección en Weaviate para almacenar los chunks del manual de Postgres.
    """
    client.collections.create(
        name=embedding_class_name,
        properties=[
            {"name": "chunk_id", "data_type": DataType.UUID},
            {"name": "text", "data_type": DataType.TEXT}
        ],
        vectorizer_config=[
            Configure.NamedVectors.text2vec_transformers(
                name="text_vector",
                source_properties=["text"]
            )
        ],
        description="Fragment of text extracted from the official PostgreSQL manual."

    )
    print(f"Collection {embedding_class_name} created in Weaviate.")


def clear_weaviate_class(client, class_name=embedding_class_name):
    print(f"Reseting Collection {embedding_class_name}.")
    client.collections.delete(embedding_class_name)


def cosine_similarity(vec1, vec2):
    """
    Calcula la similitud coseno entre dos vectores.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def process_and_upload_chunk(chunk, idx, client, model, ref_embedding, threshold=0.5):
    """
    Evalúa la relevancia del chunk comparando su embedding con un embedding de referencia.
    Si la similitud es mayor o igual al umbral, sube el chunk a Weaviate.
    """
    chunk_embedding = model.encode(chunk).tolist()
    sim = cosine_similarity(chunk_embedding, ref_embedding)
    
    if sim >= threshold:
        data_object = {
            "chunk_id": str(uuid.uuid4()),
            "text": chunk
        }

        knowledge_base.data.insert(data_object)

        print(f"Chunk {idx} uploaded (similarity: {sim:.2f}).")
        print(f"Inserted chunk content: {chunk}")
    else:
        print(f"Chunk {idx} discarded due to low relevance (similarity: {sim:.2f}).")



def refine_chunk(chunk, min_chunk_size=500, max_chunk_size=1500):
    """
    Mejora la relevancia del chunk evitando cortar oraciones a la mitad.
    Se tokeniza el chunk en oraciones y se reagrupan oraciones completas
    hasta alcanzar un tamaño adecuado.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    refined_chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_chunk_size:
                refined_chunks.append(current_chunk.strip())
            else:
                if refined_chunks:
                    refined_chunks[-1] += " " + current_chunk.strip()
                else:
                    refined_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk.strip():
        if refined_chunks and len(refined_chunks[-1]) + len(current_chunk.strip()) <= max_chunk_size:
            refined_chunks[-1] += " " + current_chunk.strip()
        else:
            refined_chunks.append(current_chunk.strip())
    return refined_chunks


def improved_chunks_from_pdf(pdf_path, chunk_size=1000, overlap=200):
    """
    Utiliza stream_chunks_from_pdf para obtener los chunks originales y luego
    los refina para evitar cortar oraciones a la mitad, mejorando así la relevancia semántica.
    """
    for chunk in stream_chunks_from_pdf(pdf_path, chunk_size, overlap):
        refined_chunks = refine_chunk(chunk)
        for refined in refined_chunks:
            yield refined



def main():
    try: 
        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host="localhost",
                http_port="8080",
                http_secure=False,
                grpc_host="localhost",
                grpc_port="50051",
                grpc_secure=False,
            ),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=2, query=45, insert=120),
            ),
        )

        client.connect()

        global knowledge_base
        knowledge_base = client.collections.get(embedding_class_name)
        
        clear_weaviate_class(client, class_name=embedding_class_name)

        create_weaviate_schema(client)
        
        if not os.path.exists(pdf_path):
            print(f"File {pdf_path} not found.")
            return
        
        print("Loading embedding model...")
        model = SentenceTransformer(embedding_model_name)
        
        reference_text = (
            "This document serves as an advanced technical guide for PostgreSQL, emphasizing robust SQL query optimization techniques, "
            "in-depth execution plan analysis, and comprehensive indexing strategies—including B-tree, hash, GiST, and partial indexes. "
            "It covers sophisticated transaction control, high-availability architectures, replication and failover mechanisms, as well as "
            "detailed performance tuning for memory, disk I/O, and concurrency management. Designed for experienced SQL professionals and "
            "DBAs, it provides best practices for ensuring data integrity, security compliance, and efficient resource utilization."
        )
        ref_embedding = model.encode(reference_text).tolist()

        
        print("Processing and uploading relevant chunks from the PDF...")
        chunk_generator = improved_chunks_from_pdf(pdf_path, chunk_size=chunk_size, overlap=chunk_overlap)
        
        for idx, chunk in enumerate(chunk_generator):
            process_and_upload_chunk(chunk, idx, client, model, ref_embedding, threshold=similarity_threshold)
        
        print("Process completed.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
