// =============================================================================
// utils.ts — env, database, AI helpers, image rendering, Cloudinary, Instagram, observability
// =============================================================================
//
// Dev:  pnpm dev
// Prod: pnpm build && pnpm start
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
import { v2 as cloudinary } from "cloudinary";
import { initChatModel } from "langchain/chat_models/universal";
import {
  type BaseMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
} from "@langchain/core/messages";

export function formatError(e: unknown): string {
  if (!(e instanceof Error)) return String(e);
  const parts: string[] = [e.message];
  let cur: unknown = e.cause;
  for (let i = 0; i < 4 && cur; i++) {
    if (cur instanceof Error) {
      const code = (cur as NodeJS.ErrnoException).code;
      parts.push(code ? `${cur.message} (${code})` : cur.message);
      cur = cur.cause;
    } else {
      parts.push(String(cur));
      break;
    }
  }
  return parts.join(" ← ");
}

export function log(scope: string, message: string, extra?: Record<string, unknown>) {
  const payload = extra ? ` ${JSON.stringify(extra)}` : "";
  console.log(`[${new Date().toISOString()}] [${scope}] ${message}${payload}`);
}

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

  // Public HTTPS base URL of THIS server — used for local image URLs in responses.
  // Instagram publish uses Cloudinary URLs instead.
  PUBLIC_BASE_URL: process.env.PUBLIC_BASE_URL ?? `http://localhost:${process.env.PORT ?? 3000}`,

  // --- Cloudinary (public HTTPS image hosting for Instagram) ---
  CLOUDINARY_CLOUD_NAME: process.env.CLOUDINARY_CLOUD_NAME ?? "",
  CLOUDINARY_API_KEY: process.env.CLOUDINARY_API_KEY ?? "",
  CLOUDINARY_API_SECRET: process.env.CLOUDINARY_API_SECRET ?? "",

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
  imageFiles: text("image_files", { mode: "json" }).$type<string[]>().notNull(), // local filenames
  imageCloudUrls: text("image_cloud_urls", { mode: "json" }).$type<string[]>().notNull(), // Cloudinary secure URLs
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
    image_cloud_urls TEXT NOT NULL DEFAULT '[]',
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

// Migrate older DBs that predate Cloudinary columns.
{
  const cols = sqlite.prepare(`PRAGMA table_info(posts)`).all() as { name: string }[];
  if (!cols.some((c) => c.name === "image_cloud_urls")) {
    sqlite.exec(`ALTER TABLE posts ADD COLUMN image_cloud_urls TEXT NOT NULL DEFAULT '[]'`);
    log("db", "migrated posts: added image_cloud_urls");
  }
}

export const db = drizzle(sqlite);

export const postRepo = {
  create(data: {
    prompt: string;
    content: string;
    imageFiles: string[];
    imageCloudUrls: string[];
  }): Post {
    const now = new Date();
    const row: Post = {
      id: crypto.randomUUID(),
      prompt: data.prompt,
      content: data.content,
      imageFiles: data.imageFiles,
      imageCloudUrls: data.imageCloudUrls,
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
  update(
    id: string,
    patch: Partial<Pick<Post, "content" | "imageFiles" | "imageCloudUrls" | "igMediaId">>,
  ): Post | undefined {
    db.update(posts)
      .set({ ...patch, updatedAt: new Date() })
      .where(eq(posts.id, id))
      .run();
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

// Concrete fonts that exist on macOS. Avoid "emoji" fallbacks — Pango aborts the
// whole process when color-emoji fonts fail to load (common with Sharp/librsvg).
const SVG_FONT_FAMILY = "Arial, Helvetica, 'Helvetica Neue', 'Gurmukhi MN', sans-serif";

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/**
 * Strip emoji / pictographs before SVG render. Sharp→librsvg→Pango crashes hard
 * ("Could not load fallback font, bailing out") when it tries to load color-emoji fonts.
 */
function sanitizeForImage(text: string): string {
  return text
    .replace(/\p{Extended_Pictographic}/gu, "")
    .replace(/\p{Emoji_Presentation}/gu, "")
    .replace(/\uFE0E|\uFE0F/g, "") // text/emoji variation selectors
    .replace(/\u200D/g, "") // ZWJ
    .replace(/\u20E3/g, "") // combining enclosing keycap
    .replace(/[\u{1F3FB}-\u{1F3FF}]/gu, "") // skin tones
    .replace(/[^\S\n]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
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

  const sentences = text
    .match(/[^.!?\n]+[.!?]*\s*/g)
    ?.map((s) => s.trim())
    .filter(Boolean) ?? [text];
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
  const fit = fitFont(text) ?? {
    fontSize: MIN_FONT,
    lines: wrap(text, Math.floor(USABLE / (MIN_FONT * CHAR_WIDTH_RATIO))),
  };
  const lineHeight = fit.fontSize * LINE_HEIGHT_RATIO;
  const startY = IMG_SIZE / 2 - ((fit.lines.length - 1) * lineHeight) / 2 + fit.fontSize * 0.35;

  const textEls = fit.lines
    .map(
      (line, i) =>
        `<text x="50%" y="${(startY + i * lineHeight).toFixed(1)}" text-anchor="middle" ` +
        `font-family="${SVG_FONT_FAMILY}" font-size="${fit.fontSize}" ` +
        `font-weight="700" fill="#f5f5f4">${escapeXml(line)}</text>`,
    )
    .join("\n");

  const counter =
    totalSlides > 1
      ? `<text x="${IMG_SIZE - 60}" y="${IMG_SIZE - 52}" text-anchor="end" ` +
        `font-family="${SVG_FONT_FAMILY}" font-size="30" fill="#a8a29e">${slideNo}/${totalSlides}</text>`
      : "";

  const svg = `
  <svg width="${IMG_SIZE}" height="${IMG_SIZE}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#0c0a09"/>
    ${textEls}
    ${counter}
  </svg>`;

  const filename = `${Date.now()}-${crypto.randomUUID().slice(0, 8)}.png`;
  try {
    await sharp(Buffer.from(svg)).png().toFile(path.join(env.IMAGES_DIR, filename));
  } catch (e) {
    throw new Error(
      `Image render failed (font/SVG): ${e instanceof Error ? e.message : String(e)}. ` +
        `Text preview: ${JSON.stringify(text.slice(0, 80))}`,
    );
  }
  return filename;
}

/** Render text to 1..10 images. Returns the filenames in slide order. */
export async function renderTextImages(text: string): Promise<string[]> {
  const cleaned = sanitizeForImage(text);
  if (!cleaned) {
    throw new Error("Nothing left to render after removing emoji/symbols from the post text.");
  }
  if (cleaned !== text.trim()) {
    log("render", "stripped emoji/pictographs before image render", {
      beforeLen: text.length,
      afterLen: cleaned.length,
    });
  }
  const slides = splitIntoSlides(cleaned);
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
// 3b. CLOUDINARY — upload local renders; Instagram fetches these HTTPS URLs
// =============================================================================

export function cloudinaryConfigured(): boolean {
  return !!(env.CLOUDINARY_CLOUD_NAME && env.CLOUDINARY_API_KEY && env.CLOUDINARY_API_SECRET);
}

function requireCloudinaryConfig() {
  if (!cloudinaryConfigured()) {
    throw new Error(
      "Cloudinary is not configured. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and " +
        "CLOUDINARY_API_SECRET in .env (needed so Instagram can fetch public image URLs).",
    );
  }
}

let cloudinaryReady = false;
function ensureCloudinary() {
  requireCloudinaryConfig();
  if (!cloudinaryReady) {
    cloudinary.config({
      cloud_name: env.CLOUDINARY_CLOUD_NAME,
      api_key: env.CLOUDINARY_API_KEY,
      api_secret: env.CLOUDINARY_API_SECRET,
      secure: true,
    });
    cloudinaryReady = true;
  }
}

/** Upload one local PNG to Cloudinary; returns the secure HTTPS URL. */
export async function uploadLocalImageToCloudinary(filename: string): Promise<string> {
  ensureCloudinary();
  const localPath = imagePathFor(filename);
  if (!fs.existsSync(localPath)) {
    throw new Error(`Cannot upload missing local image: ${localPath}`);
  }
  const publicId = `insta-agent/${path.parse(filename).name}`;
  log("cloudinary", "uploading", { filename, publicId });
  try {
    const result = await cloudinary.uploader.upload(localPath, {
      public_id: publicId,
      overwrite: true,
      resource_type: "image",
      format: "png",
    });
    if (!result.secure_url) throw new Error("Cloudinary upload returned no secure_url");
    log("cloudinary", "upload ok", { filename, url: result.secure_url });
    return result.secure_url;
  } catch (e) {
    throw new Error(`Cloudinary upload failed for ${filename}: ${formatError(e)}`);
  }
}

export async function uploadImageFilesToCloudinary(filenames: string[]): Promise<string[]> {
  const urls: string[] = [];
  for (const filename of filenames) {
    urls.push(await uploadLocalImageToCloudinary(filename));
  }
  return urls;
}

/** Render locally, then upload each slide to Cloudinary. */
export async function renderAndUploadImages(text: string): Promise<{
  imageFiles: string[];
  imageCloudUrls: string[];
}> {
  const imageFiles = await renderTextImages(text);
  const imageCloudUrls = await uploadImageFilesToCloudinary(imageFiles);
  return { imageFiles, imageCloudUrls };
}

/** URLs Instagram should fetch — Cloudinary only. */
export function igImageUrlsFor(post: Post): string[] {
  const urls = (post.imageCloudUrls ?? []).filter(Boolean);
  if (urls.length === 0) {
    throw new Error(
      `Post ${post.id} has no Cloudinary URLs. Re-create or edit the post after Cloudinary is configured.`,
    );
  }
  if (urls.length !== post.imageFiles.length) {
    log("cloudinary", "warning: cloud URL count != local file count", {
      files: post.imageFiles.length,
      cloud: urls.length,
    });
  }
  return urls;
}

// =============================================================================
// 4. AI TEXT HELPERS (LangChain — provider-agnostic)
// =============================================================================

let modelPromise: ReturnType<typeof initChatModel> | null = null;

export function getModel() {
  const googleKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || "";
  modelPromise ??= initChatModel(env.AI_MODEL, {
    modelProvider: env.AI_PROVIDER,
    temperature: 0.8,
    ...(env.AI_PROVIDER === "openai"
      ? {
          apiKey: env.OPENAI_API_KEY || "ollama",
          configuration: env.OPENAI_BASE_URL ? { baseURL: env.OPENAI_BASE_URL } : undefined,
        }
      : {}),
    ...(env.AI_PROVIDER === "google-genai"
      ? {
          apiKey: googleKey || undefined,
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
        "no quotes, no markdown, no explanations. Do NOT use emoji or emoticons " +
        "(they break image rendering); use plain words only.",
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
        "Return ONLY the revised text — no quotes, no markdown, no explanations, no emoji.",
    },
    { role: "user", content: `Current text:\n${current}\n\nInstruction:\n${instruction}` },
  ]);
  return String(res.content).trim();
}

// =============================================================================
// 5. INSTAGRAM (graph.instagram.com — single image OR carousel)
// =============================================================================

const IG_BASE = () => `https://graph.instagram.com/${env.IG_API_VERSION}`;

function igErrorMessage(status: number, json: any, step: string): string {
  const err = json?.error ?? json;
  const parts = [
    `Instagram ${step} failed (HTTP ${status})`,
    err?.message ? String(err.message) : undefined,
    err?.error_user_msg ? String(err.error_user_msg) : undefined,
    err?.code != null ? `code=${err.code}` : undefined,
    err?.error_subcode != null ? `subcode=${err.error_subcode}` : undefined,
    err?.type ? `type=${err.type}` : undefined,
  ].filter(Boolean);
  const hint = !env.PUBLIC_BASE_URL.startsWith("https://")
    ? " Hint: PUBLIC_BASE_URL must be public HTTPS for Instagram to fetch images."
    : env.PUBLIC_BASE_URL.includes("localhost")
      ? " Hint: PUBLIC_BASE_URL points at localhost — Instagram cannot reach it; use a tunnel."
      : "";
  return parts.join(" — ") + hint;
}

async function igFetch<T>(
  url: string,
  body?: Record<string, unknown>,
  step = "request",
): Promise<T> {
  log("ig", `→ ${step}`, {
    method: body ? "POST" : "GET",
    url: url.replace(env.IG_ACCESS_TOKEN, "***"),
  });
  const res = await fetch(url, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.IG_ACCESS_TOKEN}`,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const json = (await res.json()) as any;
  if (!res.ok || json.error) {
    const msg = igErrorMessage(res.status, json, step);
    log("ig", `✗ ${msg}`, {
      body: body ? { ...body, caption: body.caption ? "[redacted]" : undefined } : undefined,
    });
    throw new Error(msg);
  }
  log("ig", `✓ ${step}`, { id: json.id, status_code: json.status_code });
  return json as T;
}

function requireIgConfig() {
  const missing: string[] = [];
  if (!env.IG_ACCESS_TOKEN) missing.push("IG_ACCESS_TOKEN");
  if (!env.IG_USER_ID) missing.push("IG_USER_ID");
  if (missing.length) {
    throw new Error(
      `Cannot talk to Instagram — missing ${missing.join(" and ")} in .env. ` +
        `Chat/create still work; only publish and reading the IG feed need these.`,
    );
  }
}

async function waitForContainer(containerId: string) {
  for (let i = 0; i < 20; i++) {
    const s = await igFetch<{ status_code: string }>(
      `${IG_BASE()}/${containerId}?fields=status_code`,
      undefined,
      `container status (${containerId})`,
    );
    if (s.status_code === "FINISHED") return;
    if (s.status_code === "ERROR") {
      throw new Error(
        `Instagram failed to process media container ${containerId}. ` +
          `Usually PUBLIC_BASE_URL is not publicly reachable over HTTPS, or the image URL 404s. ` +
          `Current PUBLIC_BASE_URL=${env.PUBLIC_BASE_URL}`,
      );
    }
    log("ig", `container ${containerId} status=${s.status_code}, waiting…`);
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error(`Timed out waiting for Instagram media container ${containerId}`);
}

/** Publish 1 image as a feed post, or 2–10 images as a carousel. */
export async function igPublishImages(imageUrls: string[], caption: string): Promise<string> {
  requireIgConfig();
  if (imageUrls.length === 0) throw new Error("No images to publish");
  if (imageUrls.length > MAX_SLIDES) throw new Error(`Max ${MAX_SLIDES} images per carousel`);
  log("ig", "publish start", { count: imageUrls.length, urls: imageUrls });

  let creationId: string;

  if (imageUrls.length === 1) {
    const c = await igFetch<{ id: string }>(
      `${IG_BASE()}/${env.IG_USER_ID}/media`,
      { image_url: imageUrls[0], caption },
      "create single-image container",
    );
    await waitForContainer(c.id);
    creationId = c.id;
  } else {
    // Carousel: item containers → parent CAROUSEL container
    const children: string[] = [];
    for (let i = 0; i < imageUrls.length; i++) {
      const url = imageUrls[i];
      const item = await igFetch<{ id: string }>(
        `${IG_BASE()}/${env.IG_USER_ID}/media`,
        { image_url: url, is_carousel_item: true },
        `create carousel item ${i + 1}/${imageUrls.length}`,
      );
      await waitForContainer(item.id);
      children.push(item.id);
    }
    const parent = await igFetch<{ id: string }>(
      `${IG_BASE()}/${env.IG_USER_ID}/media`,
      {
        media_type: "CAROUSEL",
        children: children.join(","), // Instagram expects a comma-separated string, not an array
        caption,
      },
      "create carousel parent",
    );
    await waitForContainer(parent.id);
    creationId = parent.id;
  }

  const published = await igFetch<{ id: string }>(
    `${IG_BASE()}/${env.IG_USER_ID}/media_publish`,
    { creation_id: creationId },
    "media_publish",
  );
  log("ig", "publish done", { igMediaId: published.id });
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
  const json = await igFetch<{ data: IgPost[] }>(
    `${IG_BASE()}/me/media?fields=${fields}&limit=${limit}`,
    undefined,
    "fetch my media",
  );
  return json.data ?? [];
}
