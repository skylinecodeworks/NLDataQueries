# NLDataQueries

**NLDataQueries** is an AI-powered system that transforms natural language into executable PostgreSQL queries, enabling users to interact with structured data intuitively. It includes semantic search over the official PostgreSQL documentation and a simple web-based UI to visualize results.

---

## 1. Objective

The main goals of this project are:

- Enable querying structured data through natural language inputs.
- Convert user queries into valid PostgreSQL-compatible SQL.
- Execute and visualize query results via an intuitive interface.
- Use semantic search powered by embeddings to explore PostgreSQL documentation.

---

## 2. Installation

> Requirements:
> - Python 3.12+
> - PostgreSQL (with schema and data from `sql/`)
> - Weaviate server (for embeddings)
> - [Astral UV](https://docs.astral.sh/uv/getting-started/installation/) (Python environment and dependency manager)

### Install Astral UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```

### Clone the repository and set up environment
git clone https://github.com/skylinecodeworks/NLDataQueries.git
cd NLDataQueries

# Create and activate virtual environment with Astral UV
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## 3. Configuration

Create a `.env` file in the root directory with the following content:

```dotenv
DATABASE_URL=postgresql://myuser:mypassword@localhost:5432/employees
SCHEMA=employees
WEAVIATE_URL=http://localhost:8080/v1
POSTGRES_MANUAL_PDF=docs/postgres_manual.pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.5
EMBEDDING_MODEL=microsoft/codebert-base
EMBEDDING_CLASS=PostgresManualChunk
```

Ensure the following services are available:
- **PostgreSQL** running and accessible via `DATABASE_URL`.
- **Weaviate** server running on the specified host/port.
- **PostgreSQL manual PDF** available at `POSTGRES_MANUAL_PDF`.

---

## 4. Usage

### Step 1 – Load Embeddings into Weaviate

This step will extract chunks from the PostgreSQL PDF, compute embeddings, and store the most relevant in Weaviate:

```bash
python embeddings.py
```

### Step 2 – Launch the API Server

Run the API using Astral UV:

```bash
uv run main.py
```

This exposes the following endpoints:

- Swagger UI: `http://localhost:8000`
- FastAPI endpoints for:
  - SQL generation
  - Query execution
  - Semantic search
  - Chart suggestions

It provides:

- A text area for natural language input
- A panel to preview and copy the generated SQL
- Data table rendering and basic charting tools

---

## 5. Improvements

Potential enhancements include:

- Support for multiple SQL dialects
- Better natural language understanding with advanced LLMs
- Voice input support
- User authentication and persistent query history
- Deployment support via Docker and Kubernetes
- Multilingual support for queries

---

## 6. Miscellaneous

- `sql/` includes `create.sql`, `data.sql`, and `queries.sql` for bootstrapping the PostgreSQL database.
- `docs/` includes the PostgreSQL manual used for semantic indexing.
- `embeddings.py` manages the vectorization and chunk upload process.
- `bundle.sh` generates a snapshot of the project structure and code.
- `ui.html` is a static web-based interface—no frontend server needed.
- This project runs on **Astral UV**, a high-performance runtime and dependency manager for Python.

---

**License**: MIT  
**Maintainer**: [Skyline Codeworks / Tomás Pascual]