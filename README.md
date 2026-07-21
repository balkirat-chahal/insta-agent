# Post Studio (insta-agent)

Chat agent + REST API that writes social post text with an LLM, renders dark square text-images (auto-carousel when long), stores them locally and on Cloudinary, and can publish to Instagram — only when you explicitly ask.

## Quick start

```bash
pnpm install
cp .env.example .env   # fill in keys
pnpm dev
```

- Chat UI: http://localhost:3000/chat  
- API docs: http://localhost:3000/docs  

```bash
pnpm build && pnpm start   # production (tsc → dist/)
```

### Scripts

| Script | Purpose |
|--------|---------|
| `pnpm dev` | Run with `tsx` from `src/main.ts` |
| `pnpm build` | Emit `dist/` |
| `pnpm start` | Run `node dist/main.js` |
| `pnpm typecheck` | `tsc --noEmit` |
| `pnpm lint` / `lint:fix` | Oxlint |
| `pnpm fmt` / `fmt:check` | Oxfmt |

## Environment

See `.env.example`. Important groups:

| Area | Vars |
|------|------|
| LLM | `AI_PROVIDER`, `AI_MODEL`, plus `GOOGLE_API_KEY` (Gemini) or `OPENAI_API_KEY` + optional `OPENAI_BASE_URL` (Ollama / OpenAI-compatible) |
| Cloudinary | `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` — required for create/edit uploads and IG publish |
| Instagram | `IG_ACCESS_TOKEN`, `IG_USER_ID`, `IG_API_VERSION` — only for publish / fetch feed |
| Local URLs | `PUBLIC_BASE_URL` — used for local `/images/...` links in API responses (IG uses Cloudinary HTTPS URLs) |

Startup logs print which credentials are present/missing.

**Gemini note:** Gemini 3 tool calling needs `thought_signature` on function-call parts. This repo patches `@langchain/google-genai@0.2.18` (`patches/`) so the agent loop does not 400 mid-turn.

## Layout

```
src/main.ts      HTTP (Hono + zod-openapi), chat UI, docs
src/utils.ts     env, SQLite, render, Cloudinary, Instagram, agent loop
public/chat.html Simple chat frontend (POST /api/chat only)
data/            SQLite DB + local PNGs (gitignored)
dist/            Build output
```

| Layer | Responsibility |
|-------|----------------|
| `main.ts` | Routes, validation, DTOs — little business logic |
| `utils.ts` | Tools, storage, image pipeline, IG, `runAgentTurn` |

## What you can do

Via chat (`/api/chat`) or matching REST endpoints:

- **Create** post text → render 1–10 slides → save local + Cloudinary  
- **Edit** by instruction or set exact text (re-render + re-upload)  
- **List** local posts  
- **Fetch** your Instagram feed  
- **Publish** a saved post (single or carousel) — only with explicit publish intent  

Chat UI: http://localhost:3000/chat — new chat = new `sessionId`.

---

## How the agent works

Verified against `src/utils.ts` (`runAgentTurn`, `buildTools`).

### Setup before the loop

When `POST /api/chat` calls `runAgentTurn(sessionId, userMessage)`:

1. **Per-turn context (`ctx`)** — `{ latestUserMessage, touchedPostIds, events }`.  
   `buildTools(ctx)` runs every turn. Tools are **closures over `ctx`**, so the publish hard-guard can read the *current* user text without trusting the model.

2. **Six LangChain tools** (zod schemas → function-calling JSON for the model):  
   `create_post`, `edit_post`, `set_post_text`, `list_posts`, `get_instagram_posts`, `publish_post`.  
   A `toolMap` (name → tool) dispatches calls.

3. **Model** — `getModel()` lazily builds LangChain `initChatModel` (cached). If `bindTools` is missing, the turn returns early with diagnostics. Otherwise `bindTools(tools)` attaches schemas to every `invoke`.

4. **Messages** — load history from SQLite (`chatRepo.history`), then:
   ```
   [SystemMessage(AGENT_SYSTEM_PROMPT), ...history, HumanMessage(userMessage)]
   ```
   A parallel `newMessages` array holds **only this turn** (human / AI / tool). That is what gets appended to the DB so history is not duplicated. The system prompt is re-injected each turn and is not stored as a chat row.

### The tool loop (ReAct, max 8 model steps)

```
for step 0..7:
  ai = await modelWithTools.invoke(messages)   // full context every time
  push ai onto messages + newMessages

  if ai.tool_calls is empty:
    → final answer; reply = ai.content; break

  else, for each tool call (sequential):
    output = await tool.invoke(tc.args)        // JSON string
    ToolMessage({ content, tool_call_id: tc.id, name })
    push onto messages + newMessages
  // next invoke sees tool results
```

Mechanics:

- **Stop condition = no tool calls.** The model decides when it is done. If it keeps calling tools until the cap, the default reply (“I couldn’t complete that…”) remains unless diagnostics enrich it.
- **Tools return JSON strings; they catch internally** (`{"ok":true,...}` / `{"ok":false,"error":"..."}`). The loop also wraps `invoke` so escapes become structured errors. The model can then quote failures (system prompt requires that).
- **`tool_call_id` is required** so parallel-in-one-step calls can be matched to results. Multiple calls in one step are supported but run **sequentially**.
- If the loop throws after posts were already created (e.g. provider error on the next model step), the reply notes that images may still be available via `touchedPosts`.

### Publish double-guard

1. **Soft:** system prompt + tool description — only call `publish_post` when explicitly asked.  
2. **Hard:** first line of `publish_post` runs `PUBLISH_INTENT` against `ctx.latestUserMessage`. No match → `REFUSED` JSON, no Instagram call.

`PUBLISH_INTENT` requires explicit phrasing (e.g. “publish it”, “post this to Instagram”). Mere mention of “instagram” in a creative brief does **not** match.

### Side channels: events, touched posts, diagnostics

- Tools call `pushEvent` and add IDs to `touchedPostIds`.  
- **`touchedPosts`** — resolved `postSummary` objects (local + Cloudinary URLs) for the UI without parsing prose.  
- **`diagnostics`** — `events`, `failures` (error/refused), `warnings` (intent matched but tool skipped; missing IG/Cloudinary), `toolsCalled`.  
- **`enrichReply`** appends failures/warnings to the reply unless the model already included them.  
- **`chatRepo.append(sessionId, newMessages)`** persists the turn so the next message sees prior tool outputs (e.g. postIds for “edit that”).

### One-line data flow

```
user message
  → tools closed over ctx
  → [invoke → tool_calls? → run tools → ToolMessages → repeat ≤8]
  → text reply
  → persist new messages
  → return reply + touchedPosts + diagnostics
```

Design choices that keep it robust: **tools as closures over per-turn context** (deterministic publish guard), and **structured JSON tool results instead of throwing** (loop stays alive; model stays informed).

### Create / publish pipeline (for debugging)

```
create_post / edit:
  LLM text → renderTextImages (Sharp SVG→PNG, emoji stripped)
           → upload to Cloudinary
           → SQLite posts row (image_files + image_cloud_urls)

publish_post:
  igImageUrlsFor(post)  // Cloudinary only
  → Instagram container(s) → media_publish
  → store ig_media_id
```

Instagram cannot fetch `localhost`; Cloudinary URLs are required on the post.

---

## API sketch

| Method | Path | Notes |
|--------|------|--------|
| `POST` | `/api/chat` | Agent turn; body `{ sessionId?, message }` |
| `GET` | `/api/chat/:sessionId` | Visible user/assistant history |
| `POST` | `/api/posts` | Create without chat |
| `GET` | `/api/posts` | List |
| `PATCH` | `/api/posts/:id` | Edit by instruction |
| `POST` | `/api/posts/:id/publish` | Publish |
| `GET` | `/api/instagram/posts` | Recent IG media |
| `GET` | `/images/:filename` | Local PNG |
| `GET` | `/chat` | UI |
| `GET` | `/docs` | Scalar OpenAPI |

Full schemas: `/openapi.json` or `/docs`.

## Storage

SQLite (`data/app.db` by default):

- **`posts`** — prompt, content, `image_files`, `image_cloud_urls`, `ig_media_id`, timestamps  
- **`chat_messages`** — `session_id` + serialized LangChain stored messages  

Local PNGs under `data/images/`.

## Debugging cheatsheet

| Symptom | Check |
|---------|--------|
| Reply but no images | `diagnostics.toolsCalled` — did `create_post` run? |
| Tool error in UI | `diagnostics.failures` + `[create_post]` / `[cloudinary]` / `[ig]` logs |
| Publish refused | Message vs `PUBLISH_INTENT`; not just “make an IG post” |
| Publish API error | Cloudinary URLs on post? `IG_*` set? |
| Session amnesia | Client dropped `sessionId` / New chat |
| Gemini 400 after a tool | Thought-signature patch applied? (`pnpm install` with patches) |

## License

Private project — no license file included.
