# CV Bot

CV Bot is a Semantic Kernel powered CV assistant built with FastAPI. It ingests a CV PDF, indexes the processed content in Azure AI Search, stores user-facing conversation history in Azure Cosmos DB, and answers CV questions through a routed multi-agent flow.

## High-Level Architecture

Client
-> FastAPI app in `main.py`
-> startup bootstrap in `app/config/app_startup.py`
-> ingestion resources prepared in `app/RAG/ingestion/run_ingestion.py`
-> chat service created from `app/ai/chat_main_orchestration.py`

### Runtime Components

- `main.py`
  Creates the FastAPI app and registers the system, chat, and ingestion routers.

- `app/config/app_startup.py`
  Runs startup initialization. It validates settings, prepares ingestion resources, creates the multi-agent chat service, and ensures Cosmos DB history storage exists.

- `app/api/routers/system.py`
  Exposes `/` and `/health`.

- `app/api/routers/chat.py`
  Exposes `POST /chat` and delegates requests to the multi-agent orchestration service.

- `app/api/routers/ingestion.py`
  Exposes `POST /ingestion/cv` for PDF upload and ingestion.

- `app/ai/chat_main_orchestration.py`
  Owns the main chat flow. It routes each request into one of three paths: - `cv_facts` - `skill_suggestion` - `mixed`

- `app/ai/cv_agent.py`
  The factual CV agent. It answers only from indexed CV evidence.

- `app/ai/skills_agent.py`
  The recommendation agent. It can use the factual CV agent as a tool and can also use external skills research.

- `app/tools/tools.py`
  Defines Semantic Kernel tools: - `CVSearchPlugin` for Azure AI Search retrieval - `CVAgentTool` so the skills agent can ask the real `CVAgent` factual follow-up questions - `SkillsPlugin` for external skills/context lookup

- `app/cosmos/azure_cosmos_service.py`
  Stores user-facing chat history in Cosmos DB.

- `app/RAG/ingestion/run_ingestion.py`
  Handles PDF ingestion, Document Intelligence analysis, chunking, JSONL output, Blob upload, and Search indexer execution.

## Chat Flow

User message
-> router agent classifies request
-> one of: - direct response for greetings and simple messages - `cv-agent` for factual CV questions - `skill-suggester` for recommendation questions - collaboration between both agents for mixed questions
-> final response returned to API caller
-> user-facing history written to Cosmos DB

### Mixed Collaboration Flow

Mixed user request
-> `cv-agent` gathers factual CV evidence
-> `skill-suggester` uses that evidence to produce recommendations or gap analysis
-> `cv-agent` verifies the final answer stays grounded in the CV
-> final answer returned

### Internal CV Tool Flow

`skill-suggester`
-> calls `ask_cv_agent`
-> `CVAgentTool` invokes the real `CVAgent` class
-> `CVAgent.chat(..., session_id=None)` runs factual lookup
-> answer and sources are returned to the skills agent
-> internal tool questions are not stored in Cosmos history

## Ingestion Flow

PDF upload
-> `POST /ingestion/cv`
-> file saved locally under `app/data`
-> Document Intelligence extracts layout/markdown
-> cleaned content is chunked
-> JSONL is written to `app/data/processed`
-> JSONL is uploaded to Azure Blob Storage
-> Azure AI Search indexer is triggered

## Requirements

- Python `>=3.12,<3.13`
- Azure OpenAI / Azure AI Foundry chat and embedding deployments
- Azure AI Search
- Azure Cosmos DB
- Azure Blob Storage
- Azure AI Document Intelligence
- Tavily API key for the skills suggestion tool

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

Important settings groups:

- Azure OpenAI
  - `AZURE_COGNITIVE_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
  - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`

- Azure AI Search
  - `AZURE_SEARCH_ENDPOINT`
  - `AZURE_SEARCH_ADMIN_KEY`
  - `AZURE_SEARCH_QUERY_KEY`
  - `AZURE_SEARCH_INDEX_NAME`

- Azure Cosmos DB
  - `AZURE_COSMOS_ENDPOINT`
  - `AZURE_COSMOS_KEY`
  - `AZURE_COSMOS_DATABASE_NAME`
  - `AZURE_COSMOS_CONTAINER_NAME`

- Azure AI Services / Document Intelligence
  - `AZURE_AI_SERVICES_ENDPOINT`
  - `AZURE_AI_SERVICES_KEY`

- Azure Blob Storage
  - `AZURE_BLOB_CONNECTION_STRING`
  - `AZURE_BLOB_CONTAINER_NAME`

- Skills lookup
  - `TAVILY_API_KEY`

## Install

From the repo root:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv sync
```

This installs the dependencies declared in `pyproject.toml`. If you prefer another environment manager, install the same package set into the active virtual environment before running the app.

## Run the API

From the repo root:

```powershell
python -m uvicorn main:app --reload
```

Default URLs:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## API Endpoints

### Health

```http
GET /health
```

### Chat

```http
POST /chat
Content-Type: application/json
```

Example body:

```json
{
  "message": "What Azure services are mentioned in the CV?",
  "session_id": null
}
```

### Upload and Ingest a CV

```http
POST /ingestion/cv
Content-Type: multipart/form-data
```

PowerShell example:

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/ingestion/cv -F "file=@C:\path\to\your-cv.pdf;type=application/pdf"
```

The endpoint accepts PDF files only.

## Run the Sequential Agent Flow Script

To test the routed chat flow directly without going through the HTTP API, use:

```powershell
.\.venv\Scripts\python main_agent_test.py
```

This script sends 10 sequential questions that intentionally cover:

- direct greeting flow
- factual CV-only routing
- skills-only routing
- mixed collaboration routing

It prints the session id, resolved intent, routing reasoning, final response, and collaboration turns when present.

## Automated Tests

Run the automated suite with:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The current suite covers:

- chat API behavior
- ingestion API behavior
- orchestration routing
- startup bootstrap behavior
- CV agent tool behavior
- 10 mocked CV question scenarios

These tests are mocked unit and service tests. They do not call live Azure services.

## Notes

- Startup currently provisions ingestion resources and ensures Cosmos DB storage is ready before serving requests.
- User-facing chat history is persisted in Cosmos DB.
- Internal follow-up questions from the skills agent to the factual CV agent are not written to Cosmos history.
- The factual CV agent is grounded in Azure AI Search retrieval over the ingested CV content.
