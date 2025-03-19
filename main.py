from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData
import uvicorn
from dotenv import load_dotenv
import os
import subprocess
import asyncio

# Cargar variables de entorno desde el fichero .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
SCHEMA = os.getenv("SCHEMA", "public")

# Crear la conexión a la base de datos y la instancia de MetaData
engine = create_engine(DATABASE_URL)
metadata = MetaData()


app = FastAPI(title="NLDataQueries: Consulta en Lenguaje Natural a SQL")

# Variable global para almacenar el esquema actualizado
current_schema = {}

# Modelos de datos para la API
class QueryRequest(BaseModel):
    natural_language_query: str

class QueryResponse(BaseModel):
    sql_query: str
    result: list

def get_db_schema() -> dict:
    """
    Realiza la introspección del esquema actual de la base de datos utilizando el esquema definido.
    """
    metadata.reflect(engine, schema=SCHEMA)
    schema_info = {}
    for table_name, table in metadata.tables.items():
        schema_info[table_name] = [col.name for col in table.columns]
    return schema_info

async def update_schema_periodically(interval: int = 60):
    """
    Tarea en segundo plano que actualiza el esquema de la base de datos cada 'interval' segundos.
    """
    global current_schema
    while True:
        try:
            current_schema = get_db_schema()
            print("Esquema actualizado:", current_schema)
        except Exception as e:
            print("Error al actualizar el esquema:", e)
        await asyncio.sleep(interval)

def load_example_queries(file_path: str) -> list:
    """
    Carga las consultas de ejemplo desde el fichero especificado.
    Se espera que el fichero contenga consultas separadas por punto y coma.
    """
    examples = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            # Dividir las consultas y eliminar espacios en blanco
            queries = [q.strip() for q in content.split(";") if q.strip()]
            examples.extend(queries)
    except Exception as e:
        print(f"Error al cargar el archivo de consultas de ejemplo: {e}")
    return examples

def generate_sql_query(natural_language_query: str, schema_info: dict, examples: list) -> str:
    """
    Construye un prompt combinando:
      - La descripción del esquema de la base de datos.
      - Los ejemplos de consultas leídos desde queries.sql (para que el modelo aprenda a tratar la información).
      - La consulta en lenguaje natural.
    
    Se invoca al LLM local a través de Ollama (modelo mistral:latest) para transformar el prompt en una consulta SQL.
    """
    prompt = "A partir de los siguientes ejemplos, aprende a tratar la información de la base de datos y genera la consulta SQL adecuada.\n\n"
    prompt += "Esquema de la base de datos:\n"
    for table, columns in schema_info.items():
        prompt += f"- Tabla {table}: columnas {', '.join(columns)}\n"
    prompt += "\nEjemplos de consultas (desde queries.sql):\n"
    for ex in examples:
        prompt += f"- {ex}\n"
    prompt += f"\nConsulta en lenguaje natural: {natural_language_query}\n"
    prompt += "\nGenera la consulta SQL correspondiente utilizando SQL estándar."
    
    sql_query = local_llm_generate(prompt)
    return sql_query

def local_llm_generate(prompt: str) -> str:
    """
    Invoca el modelo mistral:latest de Ollama para generar la consulta SQL a partir del prompt.
    Se usa `subprocess.run` para ejecutar el comando y capturar la salida.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral:latest"],
            input=prompt,  # Pasa el prompt como entrada estándar
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()  # Devuelve la consulta SQL generada por el modelo
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al invocar el LLM: {e.stderr}")


@app.on_event("startup")
async def startup_event():
    """
    Al iniciar la aplicación:
      - Se actualiza el esquema de la base de datos.
      - Se lanza una tarea en segundo plano para refrescar el esquema periódicamente.
    """
    global current_schema
    current_schema = get_db_schema()
    asyncio.create_task(update_schema_periodically(60))

@app.post("/query", response_model=QueryResponse)
async def process_query(query_req: QueryRequest):
    """
    Endpoint que procesa la consulta en lenguaje natural.
    1. Utiliza el esquema actualizado automáticamente.
    2. Carga los ejemplos desde /sql/queries.sql para que el modelo aprenda a tratar la información.
    3. Genera la consulta SQL a partir del prompt y del LLM local.
    4. Ejecuta la consulta en PostgreSQL y retorna el resultado.
    """
    examples = load_example_queries("sql/queries.sql")
    if not examples:
        examples = ["Ejemplo: SELECT * FROM ventas WHERE fecha >= '2023-01-01';"]
    sql_query = generate_sql_query(query_req.natural_language_query, current_schema, examples)
    try:
        with engine.connect() as conn:
            result_proxy = conn.execute(sql_query)
            result = [dict(row) for row in result_proxy]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al ejecutar la consulta SQL: {str(e)}")
    
    return QueryResponse(sql_query=sql_query, result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
