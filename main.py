import uvicorn
import os
import subprocess
import asyncio
import psycopg2
import requests
import re
import json
import subprocess
import weaviate
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from weaviate.connect import ConnectionParams
from dotenv import load_dotenv



load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080/v1")
DATABASE_URL = os.getenv("DATABASE_URL")
SCHEMA = os.getenv("SCHEMA")

engine = create_engine(f"{DATABASE_URL}?options=-csearch_path={SCHEMA}")
metadata = MetaData()

better_prompt = (
    "Es extremadamente importante que la respuesta a esta consulta debe ser única y exclusivamente una "
    "consulta sql standard que pueda ser ejecutada en una base de datos postgresql sin incorporar headers ni footers ni "
    "conclusiones, solo texto sql que pudiera ser ejecutado en una consola de base de datos. El comienzo de la respuesta "
    "debe empezar por la palabra SELECT y terminar con un punto y coma. Si la respuesta no cumple con estos requisitos, "
    "el modelo no podrá evaluarla correctamente. Por favor, asegúrate de que la respuesta sea una consulta sql válida."
)

app = FastAPI(title="NLDataQueries: Consulta en Lenguaje Natural a SQL")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
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


class ChartSuggestionRequest(BaseModel):
    columns: Dict[str, str]
    data_sample: List[Dict[str, Any]]

class ChartSuggestionResponse(BaseModel):
    chart_type: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    group_by: Optional[str] = None
    explanation: Optional[str] = None

current_schema = {}

def get_weaviate_client():
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host="localhost",
            http_port="8080",
            http_secure=False,
            grpc_host="localhost",
            grpc_port="50051",
            grpc_secure=False,
        )
    )
    client.connect()
    return client

import weaviate

def get_weaviate_client():

    client = weaviate.WeaviateClient(
        connection_params=weaviate.ConnectionParams.from_params(
            http_host="localhost",
            http_port="8080",
            http_secure=False,
            grpc_host="localhost",
            grpc_port="50051",
            grpc_secure=False,
        )
    )
    client.connect()
    return client

def retrieve_context_from_weaviate(query: str, limit: int = 3) -> str:
    client = get_weaviate_client()
    collection = client.collections.use("PostgresManualChunk")

    response = collection.query.near_text(
        query=query,
        limit=limit,
        distance=0.7
    )

    objects = response.objects
    if not objects:
        return ""
    
    chunks = []
    for obj in objects:
        chunk_id = obj.properties.get("chunk_id", "")
        text = obj.properties.get("text", "")
        if text:
            chunks.append(f"(Chunk ID: {chunk_id}) {text}")

    return "\n\n".join(chunks)




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
    conn.set_isolation_level(0)  
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

def modify_pagination(sql_query: str, per_page: int, offset: int) -> str:
    """
    Remueve cualquier cláusula LIMIT/OFFSET existente y agrega la paginación deseada.
    """
    base_sql = re.sub(r"\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?", "", sql_query, flags=re.IGNORECASE).strip().rstrip(";")
    return f"{base_sql} LIMIT {per_page} OFFSET {offset};"

def local_llm_generate(natural_language_query: str, schema_info: dict, examples: list) -> str:
    """
    Genera una consulta SQL válida basada en el esquema real de la base de datos
    utilizando el modelo LM-Studio en lugar de Ollama.

    :param natural_language_query: Consulta en lenguaje natural proporcionada por el usuario.
    :param schema_info: Información del esquema de la base de datos.
    :param examples: Ejemplos para proporcionar contexto y mejorar la precisión del modelo.
    :return: Una consulta SQL generada como una cadena.
    """
    weaviate_context = retrieve_context_from_weaviate(natural_language_query, limit=3)

    prompt = (
        "Genera una consulta SQL basada en el siguiente esquema de base de datos. "
        "Asegúrate de que la consulta use exclusivamente las tablas y columnas listadas a continuación. "
        "Incluye siempre la cláusula ORDER BY para una paginación efectiva.\n"
        "A continuación tienes extractos relevantes del manual de Postgres y el esquema de la base de datos.\n\n"
        "### Extractos relevantes del manual:\n"
        f"{weaviate_context}\n\n"
    )

    prompt += "### Esquema de la base de datos:\n"
    for table, details in schema_info.items():
        columns_info = ", ".join([f"{col} ({dtype})" for col, dtype in details["columns"].items()])
        prompt += f"- Tabla: {table} (Columnas: {columns_info})\n"

    prompt += "\n### Ejemplos de consultas SQL válidas:\n"
    for ex in examples:
        prompt += f"- {ex}\n"

    prompt += f"\n### Consulta en lenguaje natural:\n{natural_language_query}\n"
    prompt += "### Responde únicamente con la consulta SQL correspondiente (sin explicaciones ni comentarios)." + better_prompt

    try:
        url = "http://localhost:1234/api/v0/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen2.5-coder-3b-instruct",  
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()

        sql_output = response_data['choices'][0]['message']['content']

        sql_output = re.sub(r"```(sql)?", "", sql_output, flags=re.IGNORECASE).strip()

        if any(table in sql_output for table in schema_info.keys()) and "SELECT" in sql_output.upper():
            return sql_output.split(";")[0] + ";"  
        else:
            return "ERROR: La consulta generada no coincide con el esquema de la base de datos."

    except requests.exceptions.RequestException as e:
        return f"ERROR: Fallo al invocar el LLM: {str(e)}"


@app.post("/query", response_model=QueryResponse)
async def process_query(query_req: QueryRequest, page: int = Query(1, ge=1), per_page: int = Query(10, ge=1, le=100)):
    """
    Procesa la consulta en lenguaje natural generando la consulta SQL una única vez y aplicando la paginación.
    """
    examples = ["SELECT * FROM users WHERE created_at >= NOW() - INTERVAL '30 days';"]
    offset = (page - 1) * per_page
    base_sql_query = local_llm_generate(query_req.natural_language_query, current_schema, examples)

    if not any(table in base_sql_query for table in current_schema.keys()):
        raise HTTPException(status_code=400, detail="La consulta generada no hace referencia a tablas válidas en el esquema.")

    sql_query = modify_pagination(base_sql_query, per_page, offset)

    try:
        with engine.connect() as conn:
            result_proxy = conn.execute(text(sql_query))
            result = [dict(row._mapping) for row in result_proxy]
            total_rows = len(result)

            insert_query = text("""
                INSERT INTO llm_sql_history (sql, status, "natural") 
                VALUES (:sql_query, :status, :natural_query)
            """)
            conn.execute(insert_query, {
                "sql_query": base_sql_query,  
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
    Ejecuta una consulta SQL específica (ya generada previamente) y actualiza la paginación sin volver a invocar al LLM.
    """
    offset = (page - 1) * per_page
    base_sql_query = request.sql_query.strip()

    sql_query = modify_pagination(base_sql_query, per_page, offset)

    try:
        with engine.connect() as conn:
            result_proxy = conn.execute(text(sql_query))
            result = [dict(row._mapping) for row in result_proxy]
            total_rows = len(result)

            insert_query = text("""
                INSERT INTO llm_sql_history (sql, status, "natural") 
                VALUES (:sql_query, :status, :natural_query)
            """)
            conn.execute(insert_query, {
                "sql_query": base_sql_query,
                "status": True,
                "natural_query": request.natural_language_query
            })
            conn.commit()

    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"Error al ejecutar la consulta SQL: {str(e)}")

    return QueryResponse(sql_query=sql_query, result=result, total_rows=total_rows, page=page, per_page=per_page)

@app.get("/schema")
async def get_schema():
    """
    Devuelve el modelo de datos (esquema de la base de datos) obtenido a través de la función get_db_schema.
    """
    schema = get_db_schema()
    return schema

@app.get("/")
async def serve_ui():
    """
    Sirve la interfaz de usuario desde el archivo ui.html.
    """
    return FileResponse(os.path.join(BASE_DIR, "ui.html"))

@app.get("/image/{image_name}")
async def serve_image(image_name: str):
    """
    Sirve imágenes estáticas desde la carpeta images.
    """
    return FileResponse(os.path.join(BASE_DIR, "images", image_name))

async def serve_ui():
    """
    Sirve la interfaz de usuario desde el archivo ui.html.
    """
    return FileResponse(os.path.join(BASE_DIR, "ui.html"))

@app.post("/download_csv")
async def download_csv(request: ExecuteSQLRequest):
    """
    Ejecuta la consulta SQL base (sin paginación) y devuelve los resultados en un archivo CSV.
    """
    base_sql_query = request.sql_query.strip()
    try:
        with engine.connect() as conn:
            result_proxy = conn.execute(text(base_sql_query))
            result = [dict(row._mapping) for row in result_proxy]
            columns = result_proxy.keys() if result_proxy.keys() else (result[0].keys() if result else [])
            
            import csv
            import io
            output = io.StringIO()
            csv_writer = csv.DictWriter(output, fieldnames=columns)
            csv_writer.writeheader()
            csv_writer.writerows(result)
            csv_data = output.getvalue()
            output.close()
    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"Error al ejecutar la consulta SQL: {str(e)}")
    
    from fastapi.responses import Response
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )

def local_llm_suggest_chart(columns: Dict[str, str], data_sample: List[Dict[str, Any]]) -> dict:
    """
    Llama al mismo LLM, proporcionándole la lista de columnas y una muestra de datos,
    para que sugiera el tipo de gráfico ideal y qué ejes usar.
    Espera que la respuesta sea un JSON parseable.
    """
    prompt = (
        "Te proporcionaré la descripción de un conjunto de datos con sus columnas (nombres y tipos) "
        "y algunas filas de ejemplo. Tu tarea es sugerir el mejor tipo de gráfico para visualizar "
        "estos datos y qué columnas deberían usarse como ejes.\n\n"
        "### Columnas:\n"
    )
    for col_name, col_type in columns.items():
        prompt += f"- {col_name}: {col_type}\n"
    prompt += "\n### Muestra de datos:\n"
    if data_sample:
        sample_rows = data_sample[:3]
        for i, row in enumerate(sample_rows, start=1):
            prompt += f"Fila {i}: {row}\n"
    else:
        prompt += "(No hay datos de ejemplo)\n"

    prompt += (
        "\nCon esta información, responde con un objeto JSON estricto, sin rodearlo de texto adicional, "
        "con la siguiente estructura:\n"
        "{\n"
        '  "chart_type": "line|bar|pie|scatter|area|histogram|otro",\n'
        '  "x_axis": "nombre_de_la_columna_o_null",\n'
        '  "y_axis": "nombre_de_la_columna_o_null",\n'
        '  "group_by": "nombre_de_la_columna_o_null",\n'
        '  "explanation": "Una breve justificación de tu elección"\n'
        "}\n\n"
        "Asegúrate de que la respuesta sea un JSON válido, sin agregados fuera del JSON."
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral:latest"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        llm_output = result.stdout.strip()

        llm_output = re.sub(r"```(json)?", "", llm_output, flags=re.IGNORECASE).strip()
        
        chart_suggestion = json.loads(llm_output)
        return chart_suggestion

    except subprocess.CalledProcessError as e:
        return {
            "chart_type": "error",
            "explanation": f"Error llamando al LLM: {str(e)}"
        }
    except json.JSONDecodeError:
        return {
            "chart_type": "unknown",
            "explanation": f"No se pudo parsear la salida como JSON. Respuesta bruta: {llm_output}"
        }

@app.post("/suggest_chart", response_model=ChartSuggestionResponse)
async def suggest_chart(request: ChartSuggestionRequest):
    """
    Dado un conjunto de columnas (con tipos) y un pequeño sample de datos,
    se pregunta al LLM qué tipo de gráfico sugiere y qué ejes usar.
    Retorna la sugerencia en un objeto ChartSuggestionResponse.
    """
    suggestion_dict = local_llm_suggest_chart(request.columns, request.data_sample)

    chart_type = suggestion_dict.get("chart_type", "unknown")
    x_axis = suggestion_dict.get("x_axis")
    y_axis = suggestion_dict.get("y_axis")
    group_by = suggestion_dict.get("group_by")
    explanation = suggestion_dict.get("explanation", "Sin explicación")

    return ChartSuggestionResponse(
        chart_type=chart_type,
        x_axis=x_axis,
        y_axis=y_axis,
        group_by=group_by,
        explanation=explanation
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
