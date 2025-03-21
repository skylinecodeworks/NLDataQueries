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

# Cargar variables de entorno desde el archivo .env
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
                # Generar chunks mientras el buffer sea mayor al tamaño deseado
                while len(current_text) >= chunk_size:
                    chunk = current_text[:chunk_size]
                    yield chunk
                    # Mantener el solapamiento
                    current_text = current_text[chunk_size - overlap:]
        # Generar el último chunk si queda texto
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
    # Note: you can also delete all collections in the Weaviate instance with:
    # client.collections.delete_all()


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
    # Generar el embedding del chunk
    chunk_embedding = model.encode(chunk).tolist()
    sim = cosine_similarity(chunk_embedding, ref_embedding)
    
    if sim >= threshold:
        # Crear el objeto de datos y subir a Weaviate
        data_object = {
            "chunk_id": str(uuid.uuid4()),
            "text": chunk
        }

        knowledge_base.data.insert(data_object)

        print(f"Chunk {idx} uploaded (similarity: {sim:.2f}).")
        # ----> Mejora: Mostrar en el log el contenido del chunk insertado
        print(f"Inserted chunk content: {chunk}")
    else:
        print(f"Chunk {idx} discarded due to low relevance (similarity: {sim:.2f}).")


# ===== NUEVAS FUNCIONES PARA MEJORAR LA RELEVANCIA DE LOS CHUNKS =====

def refine_chunk(chunk, min_chunk_size=500, max_chunk_size=1500):
    """
    Mejora la relevancia del chunk evitando cortar oraciones a la mitad.
    Se tokeniza el chunk en oraciones y se reagrupan oraciones completas
    hasta alcanzar un tamaño adecuado.
    """
    import re
    # Dividir el chunk en oraciones utilizando una expresión regular
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
                # Si el chunk es muy corto, se concatena con el anterior (si existe)
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


# =====================================================================

def main():
    try: 
        # Inicializar el cliente de Weaviate con la nueva API v4
        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host="localhost",
                http_port="8080",
                http_secure=False,
                grpc_host="localhost",
                grpc_port="50051",
                grpc_secure=False,
            ),
            # auth_client_secret=weaviate.auth.AuthApiKey("secr3tk3y"),
            # additional_headers={
            #     "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")
            # },
            additional_config=AdditionalConfig(
                timeout=Timeout(init=2, query=45, insert=120),  # Values in seconds
            ),
        )

        client.connect()

        global knowledge_base
        knowledge_base = client.collections.get(embedding_class_name)
        
        # Borrar el contenido actual de la clase para asegurar un estado limpio
        clear_weaviate_class(client, class_name=embedding_class_name)

        # Crear el esquema en Weaviate si aún no existe
        create_weaviate_schema(client)
        
        # Verificar existencia del archivo PDF
        if not os.path.exists(pdf_path):
            print(f"File {pdf_path} not found.")
            return
        
        # Cargar el modelo de embeddings
        print("Loading embedding model...")
        model = SentenceTransformer(embedding_model_name)
        
        # Definir un reference_text mejorado, con lenguaje técnico y dirigido a SQL experts y DBAs.
        reference_text = (
            "This document serves as an advanced technical guide for PostgreSQL, emphasizing robust SQL query optimization techniques, "
            "in-depth execution plan analysis, and comprehensive indexing strategies—including B-tree, hash, GiST, and partial indexes. "
            "It covers sophisticated transaction control, high-availability architectures, replication and failover mechanisms, as well as "
            "detailed performance tuning for memory, disk I/O, and concurrency management. Designed for experienced SQL professionals and "
            "DBAs, it provides best practices for ensuring data integrity, security compliance, and efficient resource utilization."
        )
        ref_embedding = model.encode(reference_text).tolist()

        
        # Procesar el PDF de forma secuencial y evaluar cada chunk
        print("Processing and uploading relevant chunks from the PDF...")
        # --- ORIGINAL: chunk_generator = stream_chunks_from_pdf(pdf_path, chunk_size=chunk_size, overlap=chunk_overlap)
        # ----> Mejora: Reemplazar el generador de chunks por la versión mejorada que mantiene oraciones completas
        chunk_generator = improved_chunks_from_pdf(pdf_path, chunk_size=chunk_size, overlap=chunk_overlap)
        
        for idx, chunk in enumerate(chunk_generator):
            process_and_upload_chunk(chunk, idx, client, model, ref_embedding, threshold=similarity_threshold)
        
        print("Process completed.")
    finally:
        # Asegurar que el cliente se cierra correctamente
        client.close()


if __name__ == "__main__":
    main()
