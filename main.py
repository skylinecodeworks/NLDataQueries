from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import uvicorn
from dotenv import load_dotenv
import os
import subprocess
import asyncio
import psycopg2
import re
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List



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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite cualquier origen, restringe esto en producción
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los headers
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class HistoryResponse(BaseModel):
    total_rows: int
    page: int
    per_page: int
    results: List[dict]

class ExecuteSQLRequest(BaseModel):
    natural_language_query: str
    sql_query: str


# Variable global para almacenar el esquema actualizado
current_schema = {}

def get_db_schema() -> dict:
    """
    Realiza la introspección del esquema actual de la base de datos con detalles completos de las tablas.
    """
    inspector = inspect(engine)
    schema_info = {}
    for table_name in inspector.get_table_names(schema=SCHEMA):
        columns = inspector.get_columns(table_name, schema=SCHEMA)
        foreign_keys = inspector.get_foreign_keys(table_name, schema=SCHEMA)
        primary_keys = inspector.get_pk_constraint(table_name, schema=SCHEMA)
        
        schema_info[table_name] = {
            "columns": {col["name"]: str(col["type"]) for col in columns},
            "primary_keys": primary_keys.get("constrained_columns", []),
            "foreign_keys": {
                fk["constrained_columns"][0]: fk["referred_table"] + "(" + fk["referred_columns"][0] + ")"
                for fk in foreign_keys if fk["constrained_columns"]
            }
        }
    print("Esquema de la base de datos actualizado:", schema_info)
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
    total_rows: int
    page: int
    per_page: int

def local_llm_generate(natural_language_query: str, schema_info: dict, examples: list, limit: int, offset: int) -> str:
    """
    Usa Ollama (Mistral) para generar una consulta SQL válida basada en el esquema real de la base de datos con paginación.
    """
    prompt = (
        "Genera una consulta SQL basada en el siguiente esquema de base de datos. "
        "Asegúrate de que la consulta use exclusivamente las tablas y columnas listadas a continuación. "
        "Incluye siempre la cláusula ORDER BY para una paginación efectiva. "
        f"Usa LIMIT {limit} OFFSET {offset} para manejar la paginación.\n"
    )
    
    # Agregar información del esquema al prompt
    prompt += "### Esquema de la base de datos:\n"
    for table, details in schema_info.items():
        columns_info = ", ".join([f"{col} ({dtype})" for col, dtype in details["columns"].items()])
        prompt += f"- Tabla: {table} (Columnas: {columns_info})\n"

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

        # Limpieza del formato ```sql ... ```
        sql_output = re.sub(r"```(sql)?", "", sql_output, flags=re.IGNORECASE).strip()

        if any(table in sql_output for table in schema_info.keys()) and "SELECT" in sql_output.upper():
            return sql_output.split(";")[0] + f" LIMIT {limit} OFFSET {offset}" if "LIMIT" not in sql_output else sql_output.split(";")[0] + ";"
        else:
            return "ERROR: La consulta generada no coincide con el esquema de la base de datos."

    except subprocess.CalledProcessError as e:
        return f"ERROR: Fallo al invocar el LLM: {e.stderr}"

@app.post("/query", response_model=QueryResponse)
async def process_query(query_req: QueryRequest, page: int = Query(1, ge=1), per_page: int = Query(10, ge=1, le=100)):
    """
    Procesa la consulta en lenguaje natural con soporte de paginación y almacena el historial de consultas exitosas.
    """
    examples = ["SELECT * FROM users WHERE created_at >= NOW() - INTERVAL '30 days';"]
    offset = (page - 1) * per_page
    sql_query = local_llm_generate(query_req.natural_language_query, current_schema, examples, per_page, offset)

    if not any(table in sql_query for table in current_schema.keys()):
        raise HTTPException(status_code=400, detail="La consulta generada no hace referencia a tablas válidas en el esquema.")

    try:
        with engine.connect() as conn:
            result_proxy = conn.execute(text(sql_query))
            result = [dict(row._mapping) for row in result_proxy]
            total_rows = len(result)

            # Insertar en llm_sql_history con "natural" protegido con comillas dobles
            insert_query = text("""
                INSERT INTO llm_sql_history (sql, status, "natural") 
                VALUES (:sql_query, :status, :natural_query)
            """)
            conn.execute(insert_query, {
                "sql_query": sql_query,
                "status": True,
                "natural_query": query_req.natural_language_query
            })
            conn.commit()

    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"Error al ejecutar la consulta SQL: {str(e)}")

    return QueryResponse(sql_query=sql_query, result=result, total_rows=total_rows, page=page, per_page=per_page)


@app.get("/history", response_model=HistoryResponse)
async def get_query_history(page: int = Query(1, ge=1), per_page: int = Query(10, ge=1, le=100)):
    """
    Retorna el historial de consultas desde la tabla llm_sql_history de manera paginada, ordenado por id DESC.
    """
    offset = (page - 1) * per_page

    try:
        with engine.connect() as conn:
            count_query = text("SELECT COUNT(*) FROM llm_sql_history;")
            total_rows = conn.execute(count_query).scalar()

            history_query = text("""
                SELECT id, sql, status, "natural"
                FROM llm_sql_history
                ORDER BY id DESC
                LIMIT :limit OFFSET :offset;
            """)

            result_proxy = conn.execute(history_query, {"limit": per_page, "offset": offset})
            results = [dict(row._mapping) for row in result_proxy]

    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"Error al recuperar el historial: {str(e)}")

    return HistoryResponse(total_rows=total_rows, page=page, per_page=per_page, results=results)

@app.post("/execute_sql", response_model=QueryResponse)
async def execute_sql_query(
    request: ExecuteSQLRequest,
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100)
):
    """
    Ejecuta una consulta SQL específica recibida desde el frontend y la almacena en el historial si es válida.
    """
    offset = (page - 1) * per_page
    sql_query = request.sql_query.strip()

    # Asegurar que la consulta SQL tiene paginación
    if "LIMIT" not in sql_query.upper():
        sql_query = f"{sql_query} LIMIT {per_page} OFFSET {offset};"

    try:
        with engine.connect() as conn:
            result_proxy = conn.execute(text(sql_query))
            result = [dict(row._mapping) for row in result_proxy]
            total_rows = len(result)

            # Guardar la consulta en el historial
            insert_query = text("""
                INSERT INTO llm_sql_history (sql, status, "natural") 
                VALUES (:sql_query, :status, :natural_query)
            """)
            conn.execute(insert_query, {
                "sql_query": sql_query,
                "status": True,
                "natural_query": request.natural_language_query
            })
            conn.commit()

    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"Error al ejecutar la consulta SQL: {str(e)}")

    return QueryResponse(sql_query=sql_query, result=result, total_rows=total_rows, page=page, per_page=per_page)



@app.get("/")
async def serve_ui():
    """
    Sirve la interfaz de usuario desde el archivo ui.html.
    """
    return FileResponse(os.path.join(BASE_DIR, "ui.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
