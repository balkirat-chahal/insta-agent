// =============================================================================
// utils.ts — env, database, AI, agent (tool calling), image rendering, Instagram
// =============================================================================
//
// Install:
//   npm i hono @hono/node-server @hono/zod-openapi @scalar/hono-api-reference
//   npm i langchain @langchain/core @langchain/google-genai
//   npm i drizzle-orm better-sqlite3 sharp zod dotenv
//   npm i -D typescript tsx @types/node @types/better-sqlite3
//
// Run:  npx tsx main.ts   →   open http://localhost:3000/docs
//
// NOTE: schema changed (posts now store multiple images). Delete data/app.db
// from the previous version before starting.
// =============================================================================

import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import { sqliteTable, text, integer } from "drizzle-orm/sqlite-core";
import { eq, asc, desc } from "drizzle-orm";
import sharp from "sharp";
import { z } from "zod";
import { initChatModel } from "langchain/chat_models/universal";
import { tool } from "@langchain/core/tools";
import {
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
} from "@langchain/core/messages";

// =============================================================================
// 1. ENV
// =============================================================================

export const env = {
  PORT: Number(process.env.PORT ?? 3000),

  // --- AI (LangChain universal — swap provider/model freely) ---
  // Local Ollama (OpenAI-compatible): AI_PROVIDER=openai + OPENAI_BASE_URL=http://localhost:11434/v1
  AI_PROVIDER: process.env.AI_PROVIDER ?? "google-genai",
  AI_MODEL: process.env.AI_MODEL ?? "gemini-2.0-flash",
  // Provider key (OPENAI_API_KEY / GOOGLE_API_KEY / ...) is read by LangChain
  // from process.env. OPENAI_BASE_URL points ChatOpenAI at Ollama when set.
  OPENAI_BASE_URL: process.env.OPENAI_BASE_URL ?? "",
  OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? "",

  // --- Instagram (Instagram API with Instagram Login, graph.instagram.com) ---
  IG_ACCESS_TOKEN: process.env.IG_ACCESS_TOKEN ?? "",
  IG_USER_ID: process.env.IG_USER_ID ?? "",
  IG_API_VERSION: process.env.IG_API_VERSION ?? "v25.0",

  // Public HTTPS base URL of THIS server — Instagram fetches images from here.
  PUBLIC_BASE_URL: process.env.PUBLIC_BASE_URL ?? `http://localhost:${process.env.PORT ?? 3000}`,

  // --- Storage ---
  IMAGES_DIR: process.env.IMAGES_DIR ?? path.resolve("data/images"),
  DB_FILE: process.env.DB_FILE ?? path.resolve("data/app.db"),
};

fs.mkdirSync(env.IMAGES_DIR, { recursive: true });
fs.mkdirSync(path.dirname(env.DB_FILE), { recursive: true });

// =============================================================================
// 2. DATABASE (drizzle + better-sqlite3)
// =============================================================================

export const posts = sqliteTable("posts", {
  id: text("id").primaryKey(),
  prompt: text("prompt").notNull(),
  content: text("content").notNull(),
  imageFiles: text("image_files", { mode: "json" }).$type<string[]>().notNull(), // 1 = single, >1 = carousel
  igMediaId: text("ig_media_id"),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
  updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});

export const chatMessages = sqliteTable("chat_messages", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  sessionId: text("session_id").notNull(),
  message: text("message").notNull(), // serialized LangChain StoredMessage JSON
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
});

export type Post = typeof posts.$inferSelect;

const sqlite = new Database(env.DB_FILE);
sqlite.exec(`
  CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    content TEXT NOT NULL,
    image_files TEXT NOT NULL,
    ig_media_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
`);

export const db = drizzle(sqlite);

export const postRepo = {
  create(data: { prompt: string; content: string; imageFiles: string[] }): Post {
    const now = new Date();
    const row: Post = {
      id: crypto.randomUUID(),
      prompt: data.prompt,
      content: data.content,
      imageFiles: data.imageFiles,
      igMediaId: null,
      createdAt: now,
      updatedAt: now,
    };
    db.insert(posts).values(row).run();
    return row;
  },
  get(id: string): Post | undefined {
    return db.select().from(posts).where(eq(posts.id, id)).get();
  },
  list(): Post[] {
    return db.select().from(posts).orderBy(desc(posts.createdAt)).all();
  },
  update(id: string, patch: Partial<Pick<Post, "content" | "imageFiles" | "igMediaId">>): Post | undefined {
    db.update(posts).set({ ...patch, updatedAt: new Date() }).where(eq(posts.id, id)).run();
    return this.get(id);
  },
};

export const chatRepo = {
  append(sessionId: string, messages: BaseMessage[]) {
    const stored = mapChatMessagesToStoredMessages(messages);
    for (const m of stored) {
      db.insert(chatMessages)
        .values({ sessionId, message: JSON.stringify(m), createdAt: new Date() })
        .run();
    }
  },
  history(sessionId: string): BaseMessage[] {
    const rows = db
      .select()
      .from(chatMessages)
      .where(eq(chatMessages.sessionId, sessionId))
      .orderBy(asc(chatMessages.id))
      .all();
    return mapStoredMessagesToChatMessages(rows.map((r) => JSON.parse(r.message)));
  },
};

// =============================================================================
// 3. IMAGE RENDERING — dark bg, light text, auto-scaled, carousel-aware
// =============================================================================
//
// Font size is scaled so the text fills the 1080×1080 canvas as much as
// possible. If the text can't fit even at the minimum readable size, it is
// split by sentences across multiple slides (max 10 — Instagram's carousel
// limit) and each slide is scaled independently.
// =============================================================================

const IMG_SIZE = 1080;
const PADDING = 110;
const USABLE = IMG_SIZE - PADDING * 2;
const MAX_FONT = 120;
const MIN_FONT = 42;
const LINE_HEIGHT_RATIO = 1.4;
const CHAR_WIDTH_RATIO = 0.56; // avg glyph width ≈ 0.56 × font size (bold sans)
const MAX_SLIDES = 10;

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function wrap(text: string, maxCharsPerLine: number): string[] {
  const lines: string[] = [];
  for (const paragraph of text.split(/\n+/)) {
    let line = "";
    for (const word of paragraph.split(/\s+/).filter(Boolean)) {
      const candidate = line ? `${line} ${word}` : word;
      if (candidate.length > maxCharsPerLine && line) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    }
    if (line) lines.push(line);
  }
  return lines.length ? lines : [" "];
}

/** Largest font size at which `text` fits the usable square, or null. */
function fitFont(text: string): { fontSize: number; lines: string[] } | null {
  for (let fontSize = MAX_FONT; fontSize >= MIN_FONT; fontSize -= 2) {
    const charsPerLine = Math.max(1, Math.floor(USABLE / (fontSize * CHAR_WIDTH_RATIO)));
    const lines = wrap(text, charsPerLine);
    const blockHeight = lines.length * fontSize * LINE_HEIGHT_RATIO;
    const widthOk = lines.every((l) => l.length <= charsPerLine);
    if (widthOk && blockHeight <= USABLE) return { fontSize, lines };
  }
  return null;
}

/** Split long text into slide-sized chunks by sentence (fallback: by words). */
function splitIntoSlides(text: string): string[] {
  if (fitFont(text)) return [text];

  const sentences = text.match(/[^.!?\n]+[.!?]*\s*/g)?.map((s) => s.trim()).filter(Boolean) ?? [text];
  const slides: string[] = [];
  let current = "";

  const flush = () => {
    if (current.trim()) slides.push(current.trim());
    current = "";
  };

  for (const sentence of sentences) {
    const candidate = current ? `${current} ${sentence}` : sentence;
    if (fitFont(candidate)) {
      current = candidate;
    } else if (current) {
      flush();
      current = sentence;
    } else {
      // Single sentence too long for one slide — split by words.
      let chunk = "";
      for (const word of sentence.split(/\s+/)) {
        const c = chunk ? `${chunk} ${word}` : word;
        if (fitFont(c)) chunk = c;
        else {
          if (chunk) slides.push(chunk);
          chunk = word;
        }
      }
      current = chunk;
    }
  }
  flush();
  return slides.slice(0, MAX_SLIDES);
}

async function renderSlide(text: string, slideNo: number, totalSlides: number): Promise<string> {
  const fit = fitFont(text) ?? { fontSize: MIN_FONT, lines: wrap(text, Math.floor(USABLE / (MIN_FONT * CHAR_WIDTH_RATIO))) };
  const lineHeight = fit.fontSize * LINE_HEIGHT_RATIO;
  const startY = IMG_SIZE / 2 - ((fit.lines.length - 1) * lineHeight) / 2 + fit.fontSize * 0.35;

  const textEls = fit.lines
    .map(
      (line, i) =>
        `<text x="50%" y="${(startY + i * lineHeight).toFixed(1)}" text-anchor="middle" ` +
        `font-family="DejaVu Sans, Arial, sans-serif" font-size="${fit.fontSize}" ` +
        `font-weight="700" fill="#f5f5f4">${escapeXml(line)}</text>`
    )
    .join("\n");

  const counter =
    totalSlides > 1
      ? `<text x="${IMG_SIZE - 60}" y="${IMG_SIZE - 52}" text-anchor="end" ` +
        `font-family="DejaVu Sans, Arial, sans-serif" font-size="30" fill="#a8a29e">${slideNo}/${totalSlides}</text>`
      : "";

  const svg = `
  <svg width="${IMG_SIZE}" height="${IMG_SIZE}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#0c0a09"/>
    ${textEls}
    ${counter}
  </svg>`;

  const filename = `${Date.now()}-${crypto.randomUUID().slice(0, 8)}.png`;
  await sharp(Buffer.from(svg)).png().toFile(path.join(env.IMAGES_DIR, filename));
  return filename;
}

/** Render text to 1..10 images. Returns the filenames in slide order. */
export async function renderTextImages(text: string): Promise<string[]> {
  const slides = splitIntoSlides(text);
  const files: string[] = [];
  for (let i = 0; i < slides.length; i++) {
    files.push(await renderSlide(slides[i], i + 1, slides.length));
  }
  return files;
}

export function imagePathFor(filename: string): string {
  return path.join(env.IMAGES_DIR, filename);
}

export function publicImageUrl(filename: string): string {
  return `${env.PUBLIC_BASE_URL.replace(/\/$/, "")}/images/${filename}`;
}

// =============================================================================
// 4. AI TEXT HELPERS (LangChain — provider-agnostic)
// =============================================================================

let modelPromise: ReturnType<typeof initChatModel> | null = null;

export function getModel() {
  modelPromise ??= initChatModel(env.AI_MODEL, {
    modelProvider: env.AI_PROVIDER,
    temperature: 0.8,
    ...(env.AI_PROVIDER === "openai"
      ? {
          apiKey: env.OPENAI_API_KEY || "ollama",
          configuration: env.OPENAI_BASE_URL
            ? { baseURL: env.OPENAI_BASE_URL }
            : undefined,
        }
      : {}),
  });
  return modelPromise;
}

export async function aiGeneratePostText(prompt: string): Promise<string> {
  const model = await getModel();
  const res = await model.invoke([
    {
      role: "system",
      content:
        "You write text for social media image posts. Return ONLY the post text — " +
        "no quotes, no markdown, no explanations.",
    },
    { role: "user", content: prompt },
  ]);
  return String(res.content).trim();
}

export async function aiRevisePostText(current: string, instruction: string): Promise<string> {
  const model = await getModel();
  const res = await model.invoke([
    {
      role: "system",
      content:
        "You edit social media post text. Apply the instruction to the current text. " +
        "Return ONLY the revised text — no quotes, no markdown, no explanations.",
    },
    { role: "user", content: `Current text:\n${current}\n\nInstruction:\n${instruction}` },
  ]);
  return String(res.content).trim();
}

// =============================================================================
// 5. INSTAGRAM (graph.instagram.com — single image OR carousel)
// =============================================================================

const IG_BASE = () => `https://graph.instagram.com/${env.IG_API_VERSION}`;

async function igFetch<T>(url: string, body?: Record<string, unknown>): Promise<T> {
  const res = await fetch(url, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.IG_ACCESS_TOKEN}`,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const json = (await res.json()) as any;
  if (!res.ok || json.error) throw new Error(`Instagram API error: ${JSON.stringify(json.error ?? json)}`);
  return json as T;
}

function requireIgConfig() {
  if (!env.IG_ACCESS_TOKEN || !env.IG_USER_ID) {
    throw new Error("IG_ACCESS_TOKEN and IG_USER_ID must be set in .env");
  }
}

async function waitForContainer(containerId: string) {
  for (let i = 0; i < 20; i++) {
    const s = await igFetch<{ status_code: string }>(`${IG_BASE()}/${containerId}?fields=status_code`);
    if (s.status_code === "FINISHED") return;
    if (s.status_code === "ERROR") {
      throw new Error("Instagram failed to process media (is PUBLIC_BASE_URL publicly reachable over HTTPS?)");
    }
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error("Timed out waiting for Instagram media container");
}

/** Publish 1 image as a feed post, or 2–10 images as a carousel. */
export async function igPublishImages(imageUrls: string[], caption: string): Promise<string> {
  requireIgConfig();
  if (imageUrls.length === 0) throw new Error("No images to publish");
  if (imageUrls.length > MAX_SLIDES) throw new Error(`Max ${MAX_SLIDES} images per carousel`);

  let creationId: string;

  if (imageUrls.length === 1) {
    const c = await igFetch<{ id: string }>(`${IG_BASE()}/${env.IG_USER_ID}/media`, {
      image_url: imageUrls[0],
      caption,
    });
    await waitForContainer(c.id);
    creationId = c.id;
  } else {
    // Carousel: item containers → parent CAROUSEL container
    const children: string[] = [];
    for (const url of imageUrls) {
      const item = await igFetch<{ id: string }>(`${IG_BASE()}/${env.IG_USER_ID}/media`, {
        image_url: url,
        is_carousel_item: true,
      });
      await waitForContainer(item.id);
      children.push(item.id);
    }
    const parent = await igFetch<{ id: string }>(`${IG_BASE()}/${env.IG_USER_ID}/media`, {
      media_type: "CAROUSEL",
      children: children.join(","), // Instagram expects a comma-separated string, not an array
      caption,
    });
    await waitForContainer(parent.id);
    creationId = parent.id;
  }

  const published = await igFetch<{ id: string }>(`${IG_BASE()}/${env.IG_USER_ID}/media_publish`, {
    creation_id: creationId,
  });
  return published.id;
}

export interface IgPost {
  id: string;
  caption?: string;
  media_type: string;
  media_url?: string;
  permalink?: string;
  timestamp?: string;
}

export async function igFetchMyPosts(limit = 10): Promise<IgPost[]> {
  requireIgConfig();
  const fields = "id,caption,media_type,media_url,permalink,timestamp";
  const json = await igFetch<{ data: IgPost[] }>(`${IG_BASE()}/me/media?fields=${fields}&limit=${limit}`);
  return json.data ?? [];
}

// =============================================================================
// 6. AGENT — chat with tool calling
// =============================================================================
//
// The agent can do everything the REST API can: create posts (text + images),
// edit them, list them, read your Instagram feed, and publish. Publishing is
// double-guarded:
//   1. System prompt: only publish when explicitly told to in the latest turn.
//   2. Hard guard: the publish tool refuses unless the user's latest message
//      literally contains publish intent (post/publish/upload/share/instagram).
//      Adjust PUBLISH_INTENT if it's too strict for your phrasing.
// =============================================================================

const PUBLISH_INTENT = /\b(post it|post this|post that|publish|upload|share|put it on|instagram|ig)\b/i;

export function postSummary(p: Post) {
  return {
    postId: p.id,
    text: p.content,
    slides: p.imageFiles.length,
    imageUrls: p.imageFiles.map(publicImageUrl),
    publishedToInstagram: !!p.igMediaId,
    igMediaId: p.igMediaId,
  };
}

interface AgentContext {
  latestUserMessage: string;
  touchedPostIds: Set<string>;
}

function buildTools(ctx: AgentContext) {
  return [
    tool(
      async ({ topicOrText, verbatim }) => {
        const content = verbatim ? topicOrText : await aiGeneratePostText(topicOrText);
        const imageFiles = await renderTextImages(content);
        const post = postRepo.create({ prompt: topicOrText, content, imageFiles });
        ctx.touchedPostIds.add(post.id);
        return JSON.stringify(postSummary(post));
      },
      {
        name: "create_post",
        description:
          "Create a new post: writes post text about the given topic (or uses the text verbatim), " +
          "renders it as one or more images (carousel if long), and saves it. Returns the postId — " +
          "remember it for later edits or publishing. Does NOT publish to Instagram.",
        schema: z.object({
          topicOrText: z.string().describe("Topic to write about, or the exact text if verbatim=true"),
          verbatim: z.boolean().optional().describe("If true, use topicOrText as the post text as-is"),
        }),
      }
    ),

    tool(
      async ({ postId, instruction }) => {
        const post = postRepo.get(postId);
        if (!post) return JSON.stringify({ error: `No post with id ${postId}. Use list_posts.` });
        const content = await aiRevisePostText(post.content, instruction);
        const imageFiles = await renderTextImages(content);
        const updated = postRepo.update(postId, { content, imageFiles })!;
        ctx.touchedPostIds.add(postId);
        return JSON.stringify(postSummary(updated));
      },
      {
        name: "edit_post",
        description:
          "Edit an existing post's text per an instruction and re-render its image(s). " +
          "Does NOT publish to Instagram.",
        schema: z.object({
          postId: z.string().describe("The post id"),
          instruction: z.string().describe("How to change the text, e.g. 'make it shorter and funnier'"),
        }),
      }
    ),

    tool(
      async ({ postId, text }) => {
        const post = postRepo.get(postId);
        if (!post) return JSON.stringify({ error: `No post with id ${postId}.` });
        const imageFiles = await renderTextImages(text);
        const updated = postRepo.update(postId, { content: text, imageFiles })!;
        ctx.touchedPostIds.add(postId);
        return JSON.stringify(postSummary(updated));
      },
      {
        name: "set_post_text",
        description: "Replace a post's text with exact text provided by the user and re-render the image(s).",
        schema: z.object({ postId: z.string(), text: z.string() }),
      }
    ),

    tool(
      async () => JSON.stringify(postRepo.list().slice(0, 20).map(postSummary)),
      {
        name: "list_posts",
        description: "List locally saved posts (most recent first) with their postIds.",
        schema: z.object({}),
      }
    ),

    tool(
      async ({ limit }) => {
        try {
          return JSON.stringify(await igFetchMyPosts(limit ?? 10));
        } catch (e) {
          return JSON.stringify({ error: e instanceof Error ? e.message : String(e) });
        }
      },
      {
        name: "get_instagram_posts",
        description: "Fetch the user's recent posts from their Instagram account.",
        schema: z.object({ limit: z.number().int().min(1).max(50).optional() }),
      }
    ),

    tool(
      async ({ postId, caption }) => {
        // HARD GUARD: refuse unless the latest user message shows publish intent.
        if (!PUBLISH_INTENT.test(ctx.latestUserMessage)) {
          return JSON.stringify({
            error:
              "REFUSED: The user has not explicitly asked to publish in their latest message. " +
              "Ask the user to confirm they want this posted to Instagram.",
          });
        }
        const post = postRepo.get(postId);
        if (!post) return JSON.stringify({ error: `No post with id ${postId}.` });
        try {
          const igMediaId = await igPublishImages(post.imageFiles.map(publicImageUrl), caption ?? post.content);
          const updated = postRepo.update(postId, { igMediaId })!;
          ctx.touchedPostIds.add(postId);
          return JSON.stringify({ published: true, ...postSummary(updated) });
        } catch (e) {
          return JSON.stringify({ error: e instanceof Error ? e.message : String(e) });
        }
      },
      {
        name: "publish_post",
        description:
          "Publish a saved post to Instagram (single image or carousel). " +
          "ONLY call this when the user's LATEST message explicitly asks to post/publish/upload. " +
          "Never call it proactively. If unsure, ask the user first.",
        schema: z.object({
          postId: z.string(),
          caption: z.string().optional().describe("Custom caption; defaults to the post text"),
        }),
      }
    ),
  ];
}

const AGENT_SYSTEM_PROMPT = `You are Post Studio, an assistant that helps the user create and manage social media image posts.

You can: create posts (you write the text, it gets rendered as dark-background image(s) — long text becomes a carousel), edit posts by id, list saved posts, read the user's Instagram feed, and publish posts to Instagram.

Rules:
- Every created/edited post has a postId. Always tell the user the postId and show the text you wrote.
- When the user refers to "it" / "that post" / "the last one", resolve it from the conversation or use list_posts.
- NEVER publish to Instagram unless the user's latest message explicitly asks you to post/publish/upload. Creating or editing a post is NOT permission to publish it. If ambiguous, ask for confirmation and do nothing.
- After publishing, confirm with the Instagram media id.
- Keep replies short and conversational.`;

export interface AgentResult {
  sessionId: string;
  reply: string;
  touchedPosts: ReturnType<typeof postSummary>[];
}

/** Run one chat turn: load history, tool-call loop, persist, respond. */
export async function runAgentTurn(sessionId: string, userMessage: string): Promise<AgentResult> {
  const ctx: AgentContext = { latestUserMessage: userMessage, touchedPostIds: new Set() };
  const tools = buildTools(ctx);
  const toolMap = new Map(tools.map((t) => [t.name, t]));

  const model = await getModel();
  const modelWithTools = model.bindTools!(tools);

  const history = chatRepo.history(sessionId);
  const humanMsg = new HumanMessage(userMessage);
  const messages: BaseMessage[] = [new SystemMessage(AGENT_SYSTEM_PROMPT), ...history, humanMsg];
  const newMessages: BaseMessage[] = [humanMsg];

  let reply = "I couldn't complete that — please try again.";

  for (let step = 0; step < 8; step++) {
    const ai = await modelWithTools.invoke(messages);
    messages.push(ai);
    newMessages.push(ai);

    const toolCalls = ai.tool_calls ?? [];
    if (toolCalls.length === 0) {
      reply = typeof ai.content === "string" ? ai.content : JSON.stringify(ai.content);
      break;
    }

    for (const tc of toolCalls) {
      const t = toolMap.get(tc.name);
      let output: string;
      try {
        // Tools are a heterogeneous union; invoke via DynamicStructuredTool's common shape.
        output = t
          ? String(await (t as { invoke: (args: unknown) => Promise<unknown> }).invoke(tc.args))
          : JSON.stringify({ error: `Unknown tool ${tc.name}` });
      } catch (e) {
        output = JSON.stringify({ error: e instanceof Error ? e.message : String(e) });
      }
      const toolMsg = new ToolMessage({ content: output, tool_call_id: tc.id!, name: tc.name });
      messages.push(toolMsg);
      newMessages.push(toolMsg);
    }
  }

  chatRepo.append(sessionId, newMessages);

  const touchedPosts = [...ctx.touchedPostIds]
    .map((id) => postRepo.get(id))
    .filter((p): p is Post => !!p)
    .map(postSummary);

  return { sessionId, reply, touchedPosts };
}
