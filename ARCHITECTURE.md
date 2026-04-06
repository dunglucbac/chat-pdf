# PDF Q&A Chat Application — Architecture & Flow

## Overview

A Retrieval-Augmented Generation (RAG) application that lets users upload PDFs and ask questions about them. It uses LangChain for the AI pipeline, Pinecone as a vector store, OpenAI for LLMs and embeddings, and includes a feedback-driven A/B testing system for components.

---

## Project Structure

```
pdf/
├── app/
│   ├── web/                        # Flask web server
│   │   ├── __init__.py             # Flask app factory
│   │   ├── api.py                  # API helper functions
│   │   ├── config/                 # App configuration
│   │   ├── db/                     # Database layer
│   │   │   └── models/             # SQLAlchemy ORM models
│   │   │       ├── base.py         # BaseModel with generic CRUD
│   │   │       ├── user.py         # User model
│   │   │       ├── pdf.py          # PDF model
│   │   │       ├── conversation.py # Conversation model
│   │   │       └── message.py      # Message model
│   │   ├── views/                  # Flask blueprints (routes)
│   │   │   ├── auth_views.py       # Auth endpoints
│   │   │   ├── pdf_views.py        # PDF upload/list endpoints
│   │   │   ├── conversation_views.py # Chat endpoints
│   │   │   ├── score_views.py      # Scoring endpoints
│   │   │   └── client_views.py     # Svelte SPA catch-all
│   │   ├── hooks.py                # Middleware & decorators
│   │   ├── files.py                # File upload/download
│   │   └── tasks/
│   │       └── embeddings.py       # Celery embedding task
│   ├── chat/                       # LangChain AI pipeline
│   │   ├── chat.py                 # Main chat builder
│   │   ├── create_embeddings.py    # PDF ingestion logic
│   │   ├── score.py                # Scoring & weighted selection
│   │   ├── redis.py                # Redis client
│   │   ├── chains/
│   │   │   ├── retrieval.py        # ConversationalRetrievalChain
│   │   │   ├── traceable.py        # Langfuse tracing mixin
│   │   │   └── streamable.py       # Token streaming mixin
│   │   ├── llms/
│   │   │   ├── chatopenai.py       # LLM builder function
│   │   │   └── __init__.py         # llm_map (variants)
│   │   ├── memories/
│   │   │   ├── sql_memory.py       # Full buffer memory
│   │   │   ├── window_memory.py    # Window buffer memory
│   │   │   ├── histories/
│   │   │   │   └── sql_history.py  # SQL-backed message history
│   │   │   └── __init__.py         # memory_map (variants)
│   │   ├── vector_stores/
│   │   │   ├── pinecone.py         # Pinecone retriever builder
│   │   │   └── __init__.py         # retriever_map (variants)
│   │   ├── embeddings/
│   │   │   └── openai.py           # OpenAI embeddings instance
│   │   ├── callbacks/
│   │   │   └── stream.py           # Streaming callback handler
│   │   ├── tracing/
│   │   │   └── langfuse_client.py  # Langfuse client instance
│   │   └── models/
│   │       └── __init__.py         # ChatArgs, Metadata (Pydantic)
│   └── celery/                     # Celery worker setup
│       ├── __init__.py             # Celery app factory
│       └── worker.py               # Worker entry point
├── client/                         # Svelte SPA frontend
├── requirements.txt
└── tasks.py                        # Invoke dev tasks
```

---

## Data Models

### User
| Field    | Type   | Notes         |
|----------|--------|---------------|
| id       | UUID   | Primary key   |
| email    | string | Unique        |
| password | string | Hashed        |

### PDF
| Field   | Type   | Notes       |
|---------|--------|-------------|
| id      | UUID   | Primary key |
| name    | string | Filename    |
| user_id | UUID   | FK → User   |

### Conversation
| Field      | Type   | Notes                                  |
|------------|--------|----------------------------------------|
| id         | UUID   | Primary key                            |
| pdf_id     | UUID   | FK → PDF                               |
| user_id    | UUID   | FK → User                              |
| llm        | string | Selected LLM name (e.g. `gpt-4`)       |
| retriever  | string | Selected retriever (e.g. `pinecone_3`) |
| memory     | string | Selected memory (e.g. `sql_buffer_memory`) |
| created_on | datetime | Timestamp                            |

### Message
| Field           | Type   | Notes                        |
|-----------------|--------|------------------------------|
| id              | UUID   | Primary key                  |
| conversation_id | UUID   | FK → Conversation            |
| role            | string | `human`, `ai`, or `system`   |
| content         | text   | Message body                 |
| created_on      | datetime | Timestamp                  |

---

## End-to-End Flow

### 1. PDF Upload & Ingestion

```
User uploads PDF
    │
    ▼
POST /api/pdfs  (pdf_views.py)
    │
    ├── handle_file_upload decorator → saves file to temp dir
    ├── files.upload() → uploads to external file service
    ├── PDF record created in database
    └── process_document.delay(pdf.id)  ← Celery task queued
              │
              ▼
        Celery worker (tasks/embeddings.py)
              │
              ├── Download PDF from file service
              └── create_embeddings_for_pdf(pdf_id, temp_path)
                        │  (create_embeddings.py)
                        │
                        ├── PyPDFLoader → load pages
                        ├── RecursiveCharacterTextSplitter
                        │     chunk_size=500, overlap=100
                        ├── OpenAIEmbeddings → embed each chunk
                        └── Pinecone.add_texts()
                              metadata: { page, pdf_id, text }
```

### 2. Starting a Conversation

```
POST /api/conversations?pdf_id=<id>
    │
    ├── Creates Conversation record (no components yet)
    └── Returns conversation_id
```

### 3. Sending a Message (Chat Flow)

```
POST /api/conversations/<id>/messages  (conversation_views.py)
    │
    ├── build_chat(chat_args)  ← chat.py
    │       │
    │       ├── select_component("retriever", retriever_map, chat_args)
    │       ├── select_component("llm", llm_map, chat_args)
    │       ├── select_component("memory", memory_map, chat_args)
    │       │       │
    │       │       └── select_component() logic:
    │       │             ├── get_conversation_components() → check DB
    │       │             ├── If saved → reuse same component
    │       │             └── If new  → random_component_by_score()
    │       │                           (weighted by Redis scores)
    │       │
    │       ├── set_conversation_components() → persist names to DB
    │       ├── condense_question_llm = ChatOpenAI(streaming=False)
    │       └── StreamingConversationalRetrievalChain.from_llm(...)
    │
    ├── If streaming=true:
    │       └── chain.stream(question) → yields tokens via SSE
    └── If streaming=false:
            └── chain({"question": question}) → returns full answer
```

### 4. Inside the Retrieval Chain

```
User question
    │
    ▼
TraceableChain.__call__()
    ├── Create Langfuse trace (conversation_id, user_id, pdf_id)
    └── Add LangfuseCallbackHandler to callbacks
              │
              ▼
ConversationalRetrievalChain
    │
    ├── condense_question_llm
    │     Rephrases follow-up questions using chat history
    │     e.g. "What else?" → "What else does the PDF say about X?"
    │
    ├── retriever (Pinecone)
    │     similarity_search(standalone_question, filter={pdf_id})
    │     Returns k=2 or k=3 most relevant text chunks
    │
    └── main llm (gpt-4 or gpt-3.5-turbo)
          Given: [context chunks] + [chat history] + [question]
          Generates: answer
                │
                ▼
          StreamableChain (if streaming)
                ├── Spawns thread to run chain
                ├── StreamingHandler puts tokens → queue
                └── Main thread yields tokens from queue
```

---

## Component System (A/B Testing)

The app runs an A/B testing loop across three component types:

### Component Maps

| Type      | Variant Name          | Description                   |
|-----------|-----------------------|-------------------------------|
| llm       | `gpt-4`               | GPT-4 (higher quality)        |
| llm       | `gpt-3.5-turbo`       | GPT-3.5 (faster, cheaper)     |
| retriever | `pinecone_2`          | Retrieve top-2 chunks         |
| retriever | `pinecone_3`          | Retrieve top-3 chunks         |
| memory    | `sql_buffer_memory`   | Full conversation history     |
| memory    | `sql_window_memory`   | Last 2 messages only          |

### Selection Logic (`score.py`)

```
New conversation → random_component_by_score(component_type)
    │
    ├── Fetch from Redis:
    │     {type}_score_values  → sum of all scores
    │     {type}_score_counts  → number of votes
    │
    ├── avg_score = score_values / score_counts
    │     (floor: 0.1 to ensure all variants can be chosen)
    │
    └── random.choices(variants, weights=avg_scores)
          → returns selected variant name
```

Existing conversation → always reuses saved components (consistency).

### Feedback Loop

```
User rates conversation  →  POST /api/scores?conversation_id=<id>
    │                         body: { score: float in [-1, 1] }
    │
    └── score_conversation(conversation_id, score)
              │
              ├── Normalize score to [0, 1]
              ├── Lookup conversation's llm, retriever, memory
              └── For each component:
                    HINCRBY {component}_score_values  += normalized_score
                    HINCRBY {component}_score_counts  += 1
```

Better-performing components accumulate higher scores, increasing their selection probability in future conversations.

---

## API Reference

### Auth
| Method | Endpoint           | Description    |
|--------|--------------------|----------------|
| GET    | /api/auth/user     | Current user   |
| POST   | /api/auth/signup   | Register       |
| POST   | /api/auth/signin   | Login          |
| POST   | /api/auth/signout  | Logout         |

### PDFs
| Method | Endpoint           | Description             |
|--------|--------------------|-------------------------|
| GET    | /api/pdfs          | List user's PDFs        |
| POST   | /api/pdfs          | Upload PDF              |
| GET    | /api/pdfs/:id      | PDF details + file URL  |

### Conversations
| Method | Endpoint                              | Description                |
|--------|---------------------------------------|----------------------------|
| GET    | /api/conversations?pdf_id=            | List conversations for PDF |
| POST   | /api/conversations?pdf_id=            | Create conversation        |
| POST   | /api/conversations/:id/messages       | Send message               |

### Scores
| Method | Endpoint                        | Description              |
|--------|---------------------------------|--------------------------|
| POST   | /api/scores?conversation_id=    | Submit score (-1 to 1)   |
| GET    | /api/scores                     | Get aggregated scores    |

---

## Streaming

```
StreamingConversationalRetrievalChain
    │
    ├── StreamableChain.stream(input)
    │       ├── queue = Queue()
    │       ├── handler = StreamingHandler(queue)
    │       ├── Thread(target=chain.__call__, callbacks=[handler]).start()
    │       └── while True:
    │               token = queue.get()
    │               if token is None: break   ← on_llm_end sentinel
    │               yield token               ← on_llm_new_token
    │
    └── Flask response:
            Response(stream_with_context(...), mimetype="text/event-stream")
```

---

## Observability (Langfuse)

Every chat invocation is traced in Langfuse:

```
TraceableChain.__call__()
    ├── langfuse_instance.trace(name=conversation_id, metadata={...})
    ├── LangfuseCallbackHandler added to LangChain callbacks
    └── Trace captures: LLM calls, token counts, latency, scores
```

User feedback scores can be associated with traces for quality analysis.

---

## Infrastructure

### Services Required

| Service  | Purpose                              | Config Key            |
|----------|--------------------------------------|-----------------------|
| PostgreSQL / SQLite | Application data          | `SQLALCHEMY_DATABASE_URI` |
| Redis    | Celery broker + component scores     | `REDIS_URI`           |
| Pinecone | Vector store for embeddings          | `PINECONE_*`          |
| OpenAI   | LLM inference + embeddings           | `OPENAI_API_KEY`      |
| Langfuse | Tracing & observability              | `LANGFUSE_*`          |
| File service | PDF storage                      | `UPLOAD_URL`          |

### Running the App

```bash
# Web server
flask run

# Celery worker
celery -A app.celery.worker worker

# Dev worker (with auto-reload)
inv devworker
```

---

## Key Dependencies

| Package      | Version  | Role                        |
|--------------|----------|-----------------------------|
| Flask        | -        | Web framework               |
| SQLAlchemy   | -        | ORM                         |
| LangChain    | 0.0.352  | RAG orchestration           |
| openai       | -        | LLM + embeddings            |
| pinecone     | -        | Vector database             |
| celery       | -        | Async task queue            |
| redis        | -        | Broker + score storage      |
| langfuse     | -        | Tracing & observability     |
| pypdf        | -        | PDF parsing                 |
