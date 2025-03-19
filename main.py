from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, text
import uvicorn
from dotenv import load_dotenv
import os
import subprocess
import asyncio
import psycopg2

# Cargar variables de entorno desde el fichero .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
SCHEMA = os.getenv("SCHEMA", "public")

# Crear la conexión a la base de datos y la instancia de MetaData
engine = create_engine(DATABASE_URL)
metadata = MetaData()

better_prompt = "Es extremadamente importante que la respuesta a esta consulta debe ser unica y exclusivamente una " \
"consulta sql standard que pueda ser ejecutada en una base de datos postgresql sin incorporar headers ni footers ni " \
"conclusiones, solo texto sql que pudiera ser ejecutado en una consola de base de datos. El comienzo de la respuesta " \
"debe empezar por la palabra SELECT y terminar con un punto y coma. Si la respuesta no cumple con estos requisitos, " \
"el modelo no podra evaluarla correctamente. Por favor, asegurate de que la respuesta sea una consulta sql valida."


app = FastAPI(title="NLDataQueries: Consulta en Lenguaje Natural a SQL")

# Variable global para almacenar el esquema actualizado
current_schema = {}

def get_db_schema() -> dict:
    """
    Realiza la introspección del esquema actual de la base de datos utilizando el esquema definido.
    """
    metadata.reflect(engine, schema=SCHEMA)
    schema_info = {}
    for table_name, table in metadata.tables.items():
        schema_info[table_name] = [col.name for col in table.columns]
    return schema_info

async def listen_for_schema_changes():
    """
    Escucha eventos de cambios en la estructura de la base de datos usando PostgreSQL NOTIFY.
    """
    global current_schema
    conn = psycopg2.connect(DATABASE_URL)
    conn.set_isolation_level(0)  # Permitir escuchar eventos
    cursor = conn.cursor()
    cursor.execute("LISTEN schema_update;")
    print("Escuchando cambios en el esquema de la base de datos...")
    
    while True:
        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            print(f"Cambio detectado en la base de datos: {notify.payload}")
            current_schema = get_db_schema()
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """
    Al iniciar la aplicación:
      - Se actualiza el esquema de la base de datos.
      - Se lanza una tarea en segundo plano para escuchar cambios en la estructura.
    """
    global current_schema
    current_schema = get_db_schema()
    asyncio.create_task(listen_for_schema_changes())

class QueryRequest(BaseModel):
    natural_language_query: str

class QueryResponse(BaseModel):
    sql_query: str
    result: list

def local_llm_generate(natural_language_query: str, schema_info: dict, examples: list) -> str:
    """
    Usa Ollama (Mistral) para generar una consulta SQL válida basada en el esquema real de la base de datos.
    """
    prompt = "Genera una consulta SQL basada en el siguiente esquema de base de datos.\n"
    prompt += "Asegúrate de que la consulta use exclusivamente las tablas y columnas listadas a continuación.\n\n"
    
    # Agregar información del esquema al prompt
    prompt += "### Esquema de la base de datos:\n"
    for table, columns in schema_info.items():
        prompt += f"- Tabla: {table} (Columnas: {', '.join(columns)})\n"

    # Incluir ejemplos previos para mejorar la precisión del modelo
    prompt += "\n### Ejemplos de consultas SQL válidas:\n"
    for ex in examples:
        prompt += f"- {ex}\n"
    
    prompt += f"\n### Consulta en lenguaje natural:\n{natural_language_query}\n"
    prompt += "\n### Responde únicamente con la consulta SQL correspondiente (sin explicaciones ni comentarios)." + better_prompt

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral:latest"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        sql_output = result.stdout.strip()

        # Extraer solo la consulta SQL y validar su contenido
        if any(table in sql_output for table in schema_info.keys()) and "SELECT" in sql_output.upper():
            return sql_output.split(";")[0] + ";"  # Devolver solo la primera consulta válida
        else:
            return "ERROR: La consulta generada no coincide con el esquema de la base de datos."

    except subprocess.CalledProcessError as e:
        return f"ERROR: Fallo al invocar el LLM: {e.stderr}"


@app.post("/query", response_model=QueryResponse)
async def process_query(query_req: QueryRequest):
    """
    Procesa la consulta en lenguaje natural, generando y ejecutando una consulta SQL válida.
    """
    examples = ["SELECT * FROM users WHERE created_at >= NOW() - INTERVAL '30 days';"]
    sql_query = local_llm_generate(query_req.natural_language_query, current_schema, examples)

    # Validar si la consulta hace referencia a tablas del esquema
    if not any(table in sql_query for table in current_schema.keys()):
        raise HTTPException(status_code=400, detail="La consulta generada no hace referencia a tablas válidas en el esquema.")

    try:
        with engine.connect() as conn:
            result_proxy = conn.execute(text(sql_query))
            result = [dict(row._mapping) for row in result_proxy]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al ejecutar la consulta SQL: {str(e)}")
    
    return QueryResponse(sql_query=sql_query, result=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
