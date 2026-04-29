# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

WebWhiz is a SaaS platform that lets users train AI chatbots on website data and embed them via a `<script>` tag. It uses OpenAI for embeddings and completions, MongoDB for persistent storage, and Redis for caching and the task queue.

## Commands

### Backend (NestJS — root directory)

```bash
yarn start:dev          # Run API server in watch mode
yarn build              # Compile TypeScript
yarn crawler:worker     # Start JS Celery crawler worker
yarn lint               # ESLint
yarn test               # Unit tests (Jest)
yarn test:watch         # Unit tests in watch mode
yarn test:cov           # Coverage report
yarn test:e2e           # End-to-end tests
```

Run a single test file:
```bash
yarn test -- src/path/to/file.spec.ts
```

### Frontend (React — `frontend/` directory)

```bash
cd frontend && yarn start    # Dev server
cd frontend && yarn build    # Production build
cd frontend && yarn lint     # ESLint
cd frontend && yarn test     # Jest tests
```

### Widget (`widget/` directory)

```bash
cd widget && yarn start      # Dev server (Parcel)
cd widget && yarn build      # Builds webwhiz-sdk.js
```

### Python Worker (`workers/` directory)

```bash
cd workers && celery -A worker worker --loglevel=info
```

### Database Migrations (`migrations/` directory)

```bash
cd migrations && yarn status   # Check migration status
cd migrations && yarn up       # Apply pending migrations
cd migrations && yarn down     # Revert last migration
```

### Docker (recommended for local development)

```bash
docker compose up              # Start all services
docker compose build           # Rebuild containers
```

## Architecture

The system has five runtime components:

1. **NestJS API server** (`src/`) — REST API + WebSocket gateway
2. **JS Celery worker** (`src/crawler.main.ts`) — web crawling and embedding generation
3. **Python Celery worker** (`workers/worker.py`) — cosine similarity search, PDF/HTML extraction
4. **MongoDB** — persistent data (users, knowledgebases, content chunks, chat sessions)
5. **Redis** — embeddings cache, Celery task queue, WebSocket adapter, session state

### Backend module layout (`src/`)

| Module | Role |
|---|---|
| `auth` | JWT, Passport strategies (jwt, local, headerApiKey), Google OAuth |
| `user` | User management, API keys, subscription info |
| `knowledgebase` | Core domain: KB CRUD, WebSocket chat gateway, data store |
| `chatbot` | AI completion requests, session management, manual admin chat |
| `importers` | Crawlee/Playwright web crawler, PDF, HTML text extraction |
| `openai` | OpenAI client wrapper — rate limiting, embeddings, streaming |
| `subscription` | Lemon Squeezy billing, plan limits, token tracking |
| `task` | Long-running job status tracking |
| `slack` | Slack Bolt bot integration |
| `common` | Shared config, Mongo, Redis, Celery, email, Sentry setup |

All routes are JWT-protected by default via a global `JwtAuthGuard`. Mark public endpoints with `@Public()`. Role restriction uses `@Roles(Role.Admin)`.

### Data flow for chatbot creation

1. Frontend POSTs to `/knowledgebase` → MongoDB doc created
2. Backend enqueues `tasks.crawl` in Redis (Celery)
3. JS worker crawls pages with Crawlee/Playwright, stores text chunks in `KbDataStore` collection
4. Python worker computes OpenAI embeddings, stores vectors in Redis
5. Frontend polls `/task/:id` until status is `completed`

### Chat flow

1. Widget connects via Socket.io, creates or resumes a `ChatSession`
2. User message triggers `tasks.gen_embeddings` equivalent lookup — Python worker returns nearest chunks via cosine similarity
3. Backend constructs prompt with retrieved chunks, calls OpenAI with streaming
4. Response streams back through WebSocket to widget
5. Admins can join any session via the dashboard (also Socket.io)

WebSocket uses a Redis adapter (`RedisIoAdapter`) so multiple API instances share session state.

### Frontend structure (`frontend/src/`)

- `Base.tsx` — root class component; manages auth state in localStorage, global Axios 401 interceptor, React Router v5 `Switch`
- `containers/` — smart components wired to API
- `services/` — Axios-based API clients (`knowledgebaseService`, `authServices`, `userServices`) and `SocketService`
- Auth token stored as `accesstoken` in localStorage, injected as `Authorization: Bearer` header by the global Axios instance

### Widget (`widget/`)

Standalone JS bundle built with Parcel, distributed as `webwhiz-sdk.js`. Communicates with the backend via Socket.io. Configured through query parameters (`kbId`, `baseUrl`, etc.) when embedded on customer sites.

## Environment

Copy `.env.sample` to `.env`. Required variables:

- `MONGO_URI`, `MONGO_DBNAME` — database
- `REDIS_HOST`, `REDIS_PORT` — queue and cache
- `OPENAI_KEY` — AI completions and embeddings
- `SECRET_KEY`, `ENC_KEY` — JWT and encryption
- `CLIENT_URL` — frontend origin (used in CORS and emails)

Optional: `GOOGLE_CLIENT_ID`, `LEMON_SQUEEZY_API_KEY`, `SENDGRID_API_KEY`, `SENTRY_DSN`, `SLACK_*`.
