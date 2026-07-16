// =============================================================================
// main.ts — Hono + zod-openapi + Scalar (/docs)
// Run: npx tsx main.ts
// =============================================================================

import { serve } from "@hono/node-server";
import { OpenAPIHono, createRoute, z } from "@hono/zod-openapi";
import { Scalar } from "@scalar/hono-api-reference";
import fs from "node:fs";
import crypto from "node:crypto";
import {
  env,
  postRepo,
  chatRepo,
  postSummary,
  runAgentTurn,
  aiGeneratePostText,
  aiRevisePostText,
  renderAndUploadImages,
  imagePathFor,
  publicImageUrl,
  igPublishImages,
  igImageUrlsFor,
  igFetchMyPosts,
} from "./utils";

const app = new OpenAPIHono();

// -----------------------------------------------------------------------------
// Schemas
// -----------------------------------------------------------------------------

const PostSchema = z
  .object({
    id: z.string().uuid(),
    prompt: z.string(),
    content: z.string(),
    imageFiles: z.array(z.string()),
    imageUrls: z.array(z.string()).openapi({ description: "Local server URLs" }),
    imageCloudUrls: z.array(z.string()).openapi({ description: "Cloudinary HTTPS URLs (used for Instagram)" }),
    isCarousel: z.boolean(),
    igMediaId: z.string().nullable(),
    createdAt: z.string(),
    updatedAt: z.string(),
  })
  .openapi("Post");

const PostSummarySchema = z
  .object({
    postId: z.string(),
    text: z.string(),
    slides: z.number(),
    imageFiles: z.array(z.string()).optional(),
    imageUrls: z.array(z.string()),
    imageCloudUrls: z.array(z.string()),
    publishedToInstagram: z.boolean(),
    igMediaId: z.string().nullable(),
  })
  .openapi("PostSummary");

const ErrorSchema = z.object({ error: z.string() }).openapi("Error");

const AgentEventSchema = z
  .object({
    ts: z.string(),
    scope: z.string(),
    status: z.enum(["ok", "error", "refused", "info"]),
    message: z.string(),
    detail: z.record(z.unknown()).optional(),
  })
  .openapi("AgentEvent");

const DiagnosticsSchema = z
  .object({
    events: z.array(AgentEventSchema),
    failures: z.array(z.string()),
    warnings: z.array(z.string()),
    toolsCalled: z.array(z.string()),
  })
  .openapi("Diagnostics");

function toDto(p: NonNullable<ReturnType<typeof postRepo.get>>) {
  return {
    id: p.id,
    prompt: p.prompt,
    content: p.content,
    imageFiles: p.imageFiles,
    imageUrls: p.imageFiles.map(publicImageUrl),
    imageCloudUrls: p.imageCloudUrls ?? [],
    isCarousel: p.imageFiles.length > 1,
    igMediaId: p.igMediaId,
    createdAt: p.createdAt.toISOString(),
    updatedAt: p.updatedAt.toISOString(),
  };
}

// -----------------------------------------------------------------------------
// POST /api/chat — the agent. One endpoint = full conversation with tools.
// -----------------------------------------------------------------------------

app.openapi(
  createRoute({
    method: "post",
    path: "/api/chat",
    tags: ["Agent"],
    summary: "Chat with the agent (creates/edits posts, reads IG, publishes only when told)",
    description:
      "Send a message; keep passing the returned sessionId to continue the conversation. " +
      "The agent can create posts, edit them, list them, fetch your Instagram feed, and " +
      "publish — but it will only publish when your message explicitly asks it to. " +
      "touchedPosts contains any posts created/edited/published this turn so a frontend " +
      "can render the images immediately.",
    request: {
      body: {
        content: {
          "application/json": {
            schema: z.object({
              sessionId: z.string().uuid().optional().openapi({
                description: "Omit on the first message; reuse afterwards",
              }),
              message: z.string().min(1).openapi({
                example: "Make me a post about why side projects die, make it a bit long and dramatic",
              }),
            }),
          },
        },
      },
    },
    responses: {
      200: {
        description: "Agent reply",
        content: {
          "application/json": {
            schema: z.object({
              sessionId: z.string(),
              reply: z.string(),
              touchedPosts: z.array(PostSummarySchema),
              diagnostics: DiagnosticsSchema.openapi({
                description:
                  "What happened this turn: tool timeline, failures, and warnings. " +
                  "Check failures/warnings when create or publish does not work.",
              }),
            }),
          },
        },
      },
      500: { description: "Agent error", content: { "application/json": { schema: ErrorSchema } } },
    },
  }),
  async (c) => {
    const { sessionId, message } = c.req.valid("json");
    const sid = sessionId ?? crypto.randomUUID();
    console.log(`[${new Date().toISOString()}] [http] POST /api/chat`, { sessionId: sid, message });
    try {
      const result = await runAgentTurn(sid, message);
      if (result.diagnostics.failures.length || result.diagnostics.warnings.length) {
        console.warn(`[${new Date().toISOString()}] [http] /api/chat issues`, {
          sessionId: sid,
          failures: result.diagnostics.failures,
          warnings: result.diagnostics.warnings,
          toolsCalled: result.diagnostics.toolsCalled,
        });
      }
      return c.json(result, 200);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      console.error(`[${new Date().toISOString()}] [http] /api/chat crashed`, { sessionId: sid, error });
      return c.json({ error }, 500);
    }
  }
);

// -----------------------------------------------------------------------------
// GET /api/chat/{sessionId} — history (for rebuilding the UI on reload)
// -----------------------------------------------------------------------------

app.openapi(
  createRoute({
    method: "get",
    path: "/api/chat/{sessionId}",
    tags: ["Agent"],
    summary: "Get a chat session's visible history (user + assistant turns)",
    request: { params: z.object({ sessionId: z.string().uuid() }) },
    responses: {
      200: {
        description: "History",
        content: {
          "application/json": {
            schema: z.array(z.object({ role: z.string(), content: z.string() })),
          },
        },
      },
    },
  }),
  (c) => {
    const { sessionId } = c.req.valid("param");
    const visible = chatRepo
      .history(sessionId)
      .filter((m) => m.getType() === "human" || (m.getType() === "ai" && typeof m.content === "string" && m.content))
      .map((m) => ({ role: m.getType() === "human" ? "user" : "assistant", content: String(m.content) }));
    return c.json(visible);
  }
);

// -----------------------------------------------------------------------------
// Direct REST endpoints (same capabilities, no agent)
// -----------------------------------------------------------------------------

app.openapi(
  createRoute({
    method: "post",
    path: "/api/posts",
    tags: ["Posts"],
    summary: "AI-generate post text, render image(s), save locally + in sqlite",
    request: {
      body: {
        content: {
          "application/json": {
            schema: z.object({
              prompt: z.string().min(1).openapi({ example: "a motivational post about Monday mornings" }),
            }),
          },
        },
      },
    },
    responses: {
      200: { description: "Created post", content: { "application/json": { schema: PostSchema } } },
      500: { description: "Create failed", content: { "application/json": { schema: ErrorSchema } } },
    },
  }),
  async (c) => {
    const { prompt } = c.req.valid("json");
    try {
      const content = await aiGeneratePostText(prompt);
      const { imageFiles, imageCloudUrls } = await renderAndUploadImages(content);
      const post = postRepo.create({ prompt, content, imageFiles, imageCloudUrls });
      return c.json(toDto(post), 200);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      console.error(`[${new Date().toISOString()}] [http] POST /api/posts failed`, { error });
      return c.json({ error }, 500);
    }
  }
);

app.openapi(
  createRoute({
    method: "get",
    path: "/api/posts",
    tags: ["Posts"],
    summary: "List locally saved posts",
    responses: {
      200: { description: "Posts", content: { "application/json": { schema: z.array(PostSchema) } } },
    },
  }),
  (c) => c.json(postRepo.list().map(toDto))
);

app.openapi(
  createRoute({
    method: "patch",
    path: "/api/posts/{id}",
    tags: ["Posts"],
    summary: "Edit a post's text via AI instruction; image(s) re-rendered",
    request: {
      params: z.object({ id: z.string().uuid() }),
      body: {
        content: {
          "application/json": {
            schema: z.object({
              instruction: z.string().min(1).openapi({ example: "make it funnier and add an emoji" }),
            }),
          },
        },
      },
    },
    responses: {
      200: { description: "Updated post", content: { "application/json": { schema: PostSchema } } },
      404: { description: "Not found", content: { "application/json": { schema: ErrorSchema } } },
      500: { description: "Update failed", content: { "application/json": { schema: ErrorSchema } } },
    },
  }),
  async (c) => {
    const { id } = c.req.valid("param");
    const { instruction } = c.req.valid("json");
    const post = postRepo.get(id);
    if (!post) return c.json({ error: "Post not found" }, 404);
    const content = await aiRevisePostText(post.content, instruction);
    try {
      const { imageFiles, imageCloudUrls } = await renderAndUploadImages(content);
      return c.json(toDto(postRepo.update(id, { content, imageFiles, imageCloudUrls })!), 200);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      console.error(`[${new Date().toISOString()}] [http] PATCH /api/posts failed`, { id, error });
      return c.json({ error }, 500);
    }
  }
);

app.openapi(
  createRoute({
    method: "post",
    path: "/api/posts/{id}/publish",
    tags: ["Instagram"],
    summary: "Publish a saved post to Instagram (carousel if multiple images)",
    request: {
      params: z.object({ id: z.string().uuid() }),
      body: {
        content: {
          "application/json": {
            schema: z.object({ caption: z.string().optional() }),
          },
        },
        required: false,
      },
    },
    responses: {
      200: { description: "Published", content: { "application/json": { schema: PostSchema } } },
      404: { description: "Not found", content: { "application/json": { schema: ErrorSchema } } },
      502: { description: "Instagram error", content: { "application/json": { schema: ErrorSchema } } },
    },
  }),
  async (c) => {
    const { id } = c.req.valid("param");
    const body = await c.req.json().catch(() => ({} as { caption?: string }));
    const post = postRepo.get(id);
    if (!post) return c.json({ error: "Post not found" }, 404);
    try {
      const igMediaId = await igPublishImages(igImageUrlsFor(post), body.caption ?? post.content);
      return c.json(toDto(postRepo.update(id, { igMediaId })!), 200);
    } catch (e) {
      return c.json({ error: e instanceof Error ? e.message : String(e) }, 502);
    }
  }
);

app.openapi(
  createRoute({
    method: "get",
    path: "/api/instagram/posts",
    tags: ["Instagram"],
    summary: "Fetch my recent Instagram posts",
    request: {
      query: z.object({
        limit: z.coerce.number().int().min(1).max(50).optional().openapi({ example: 10 }),
      }),
    },
    responses: {
      200: {
        description: "Instagram posts",
        content: {
          "application/json": {
            schema: z.array(
              z.object({
                id: z.string(),
                caption: z.string().optional(),
                media_type: z.string(),
                media_url: z.string().optional(),
                permalink: z.string().optional(),
                timestamp: z.string().optional(),
              })
            ),
          },
        },
      },
      502: { description: "Instagram error", content: { "application/json": { schema: ErrorSchema } } },
    },
  }),
  async (c) => {
    try {
      const { limit } = c.req.valid("query");
      return c.json(await igFetchMyPosts(limit ?? 10), 200);
    } catch (e) {
      return c.json({ error: e instanceof Error ? e.message : String(e) }, 502);
    }
  }
);

// -----------------------------------------------------------------------------
// GET /images/{filename} — serve generated images (Instagram fetches these)
// -----------------------------------------------------------------------------

app.get("/images/:filename", (c) => {
  const filename = c.req.param("filename");
  if (filename.includes("..") || filename.includes("/")) return c.text("Bad filename", 400);
  const filePath = imagePathFor(filename);
  if (!fs.existsSync(filePath)) return c.text("Not found", 404);
  return c.body(fs.readFileSync(filePath), 200, { "Content-Type": "image/png" });
});

// -----------------------------------------------------------------------------
// OpenAPI spec + Scalar docs
// -----------------------------------------------------------------------------

app.doc("/openapi.json", {
  openapi: "3.1.0",
  info: {
    title: "AI Post Studio",
    version: "2.0.0",
    description:
      "Chat agent + REST API: generate post text with AI, render dark text-images " +
      "(auto carousel), edit by reference, and publish to Instagram — only when asked.",
  },
});

app.get("/docs", Scalar({ url: "/openapi.json", theme: "purple" }));
app.get("/", (c) => c.redirect("/docs"));

serve({ fetch: app.fetch, port: env.PORT }, (info) => {
  console.log(`🚀 http://localhost:${info.port}/docs`);
});
