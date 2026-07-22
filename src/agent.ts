// =============================================================================
// agent.ts — chat agent tool-calling loop (ReAct)
// =============================================================================

import { z } from "zod";
import { tool } from "@langchain/core/tools";
import {
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import {
  env,
  formatError,
  log,
  postRepo,
  chatRepo,
  type Post,
  aiGeneratePostText,
  aiRevisePostText,
  renderAndUploadImages,
  renderTextImages,
  uploadImageFilesToCloudinary,
  publicImageUrl,
  cloudinaryConfigured,
  igPublishImages,
  igImageUrlsFor,
  igFetchMyPosts,
  getModel,
} from "./utils.js";

// =============================================================================
// Agent — tool calling
// =============================================================================
//
// The agent can do everything the REST API can: create posts (text + images),
// edit them, list them, read your Instagram feed, and publish. Publishing is
// double-guarded:
//   1. System prompt: only publish when explicitly told to in the latest turn.
//   2. Hard guard: the publish tool refuses unless the user's latest message
//      shows explicit publish intent (see PUBLISH_INTENT).
// =============================================================================

// Require an explicit publish action — do NOT match mere mentions of "instagram".
const PUBLISH_INTENT =
  /\b(post it|post this|post that|publish(?:\s+it)?|upload(?:\s+it)?|share it|put it on|post(?:\s+(?:it|this|that))?\s+to\s+(?:instagram|ig)|(?:publish|upload|share).{0,24}(?:instagram|ig))\b/i;
const CREATE_INTENT = /\b(create|make|write|generate|draft)\b.*\b(post|carousel|image|slide)/i;

export type AgentEventStatus = "ok" | "error" | "refused" | "info";

export interface AgentEvent {
  ts: string;
  scope: string;
  status: AgentEventStatus;
  message: string;
  detail?: Record<string, unknown>;
}

export function postSummary(p: Post) {
  return {
    postId: p.id,
    text: p.content,
    slides: p.imageFiles.length,
    imageFiles: p.imageFiles,
    imageUrls: p.imageFiles.map(publicImageUrl),
    imageCloudUrls: p.imageCloudUrls ?? [],
    publishedToInstagram: !!p.igMediaId,
    igMediaId: p.igMediaId,
  };
}

interface AgentContext {
  latestUserMessage: string;
  touchedPostIds: Set<string>;
  events: AgentEvent[];
}

function pushEvent(
  ctx: AgentContext,
  scope: string,
  status: AgentEventStatus,
  message: string,
  detail?: Record<string, unknown>,
) {
  const event: AgentEvent = { ts: new Date().toISOString(), scope, status, message, detail };
  ctx.events.push(event);
  log(scope, `${status === "ok" ? "✓" : status === "error" ? "✗" : "•"} ${message}`, detail);
}

function toolError(message: string, detail?: Record<string, unknown>) {
  return JSON.stringify({ ok: false, error: message, ...detail });
}

function toolOk(data: unknown) {
  return JSON.stringify({
    ok: true,
    ...(typeof data === "object" && data ? data : { result: data }),
  });
}

function buildTools(ctx: AgentContext) {
  return [
    tool(
      async ({ topicOrText, verbatim }) => {
        try {
          pushEvent(ctx, "create_post", "info", "generating text + images", {
            topicOrText,
            verbatim: !!verbatim,
          });
          const content = verbatim ? topicOrText : await aiGeneratePostText(topicOrText);
          const { imageFiles, imageCloudUrls } = await renderAndUploadImages(content);
          const post = postRepo.create({
            prompt: topicOrText,
            content,
            imageFiles,
            imageCloudUrls,
          });
          ctx.touchedPostIds.add(post.id);
          pushEvent(ctx, "create_post", "ok", `created post ${post.id}`, {
            slides: imageFiles.length,
            imageFiles,
            imageCloudUrls,
          });
          return toolOk(postSummary(post));
        } catch (e) {
          const message = `create_post failed: ${formatError(e)}`;
          pushEvent(ctx, "create_post", "error", message);
          return toolError(message);
        }
      },
      {
        name: "create_post",
        description:
          "Create a new post: writes post text about the given topic (or uses the text verbatim), " +
          "renders it as one or more images (carousel if long), and saves it. Returns the postId — " +
          "remember it for later edits or publishing. Does NOT publish to Instagram.",
        schema: z.object({
          topicOrText: z
            .string()
            .describe("Topic to write about, or the exact text if verbatim=true"),
          verbatim: z
            .boolean()
            .optional()
            .describe("If true, use topicOrText as the post text as-is"),
        }),
      },
    ),

    tool(
      async ({ postId, instruction }) => {
        try {
          const post = postRepo.get(postId);
          if (!post) {
            const message = `No post with id ${postId}. Use list_posts.`;
            pushEvent(ctx, "edit_post", "error", message);
            return toolError(message);
          }
          pushEvent(ctx, "edit_post", "info", `editing ${postId}`, { instruction });
          const content = await aiRevisePostText(post.content, instruction);
          const { imageFiles, imageCloudUrls } = await renderAndUploadImages(content);
          const updated = postRepo.update(postId, { content, imageFiles, imageCloudUrls })!;
          ctx.touchedPostIds.add(postId);
          pushEvent(ctx, "edit_post", "ok", `updated post ${postId}`, {
            slides: imageFiles.length,
            imageCloudUrls,
          });
          return toolOk(postSummary(updated));
        } catch (e) {
          const message = `edit_post failed: ${formatError(e)}`;
          pushEvent(ctx, "edit_post", "error", message);
          return toolError(message);
        }
      },
      {
        name: "edit_post",
        description:
          "Edit an existing post's text per an instruction and re-render its image(s). " +
          "Does NOT publish to Instagram.",
        schema: z.object({
          postId: z.string().describe("The post id"),
          instruction: z
            .string()
            .describe("How to change the text, e.g. 'make it shorter and funnier'"),
        }),
      },
    ),

    tool(
      async ({ postId, text }) => {
        try {
          const post = postRepo.get(postId);
          if (!post) {
            const message = `No post with id ${postId}.`;
            pushEvent(ctx, "set_post_text", "error", message);
            return toolError(message);
          }
          const imageFiles = await renderTextImages(text);
          const imageCloudUrls = await uploadImageFilesToCloudinary(imageFiles);
          const updated = postRepo.update(postId, { content: text, imageFiles, imageCloudUrls })!;
          ctx.touchedPostIds.add(postId);
          pushEvent(ctx, "set_post_text", "ok", `set text on ${postId}`, {
            slides: imageFiles.length,
            imageCloudUrls,
          });
          return toolOk(postSummary(updated));
        } catch (e) {
          const message = `set_post_text failed: ${formatError(e)}`;
          pushEvent(ctx, "set_post_text", "error", message);
          return toolError(message);
        }
      },
      {
        name: "set_post_text",
        description:
          "Replace a post's text with exact text provided by the user and re-render the image(s).",
        schema: z.object({ postId: z.string(), text: z.string() }),
      },
    ),

    tool(
      async () => {
        const list = postRepo.list().slice(0, 20).map(postSummary);
        pushEvent(ctx, "list_posts", "ok", `listed ${list.length} posts`);
        return toolOk({ posts: list });
      },
      {
        name: "list_posts",
        description: "List locally saved posts (most recent first) with their postIds.",
        schema: z.object({}),
      },
    ),

    tool(
      async ({ limit }) => {
        try {
          const posts = await igFetchMyPosts(limit ?? 10);
          pushEvent(ctx, "get_instagram_posts", "ok", `fetched ${posts.length} IG posts`);
          return toolOk({ posts });
        } catch (e) {
          const message = `get_instagram_posts failed: ${formatError(e)}`;
          pushEvent(ctx, "get_instagram_posts", "error", message);
          return toolError(message);
        }
      },
      {
        name: "get_instagram_posts",
        description: "Fetch the user's recent posts from their Instagram account.",
        schema: z.object({ limit: z.number().int().min(1).max(50).optional() }),
      },
    ),

    tool(
      async ({ postId, caption }) => {
        // HARD GUARD: refuse unless the latest user message shows publish intent.
        if (!PUBLISH_INTENT.test(ctx.latestUserMessage)) {
          const message =
            "REFUSED: Latest user message does not look like an explicit publish request. " +
            'Say something like "publish it" / "post this to Instagram". ' +
            `Message was: ${JSON.stringify(ctx.latestUserMessage)}`;
          pushEvent(ctx, "publish_post", "refused", message, { postId });
          return toolError(message, { reason: "no_publish_intent" });
        }
        const post = postRepo.get(postId);
        if (!post) {
          const message = `No post with id ${postId}.`;
          pushEvent(ctx, "publish_post", "error", message);
          return toolError(message);
        }
        try {
          pushEvent(ctx, "publish_post", "info", `publishing ${postId}`, {
            slides: post.imageFiles.length,
            imageCloudUrls: post.imageCloudUrls,
          });
          const igMediaId = await igPublishImages(igImageUrlsFor(post), caption ?? post.content);
          const updated = postRepo.update(postId, { igMediaId })!;
          ctx.touchedPostIds.add(postId);
          pushEvent(ctx, "publish_post", "ok", `published ${postId} → IG ${igMediaId}`);
          return toolOk({ published: true, ...postSummary(updated) });
        } catch (e) {
          const message = `publish_post failed: ${formatError(e)}`;
          pushEvent(ctx, "publish_post", "error", message, {
            publicBaseUrl: env.PUBLIC_BASE_URL,
            hasToken: !!env.IG_ACCESS_TOKEN,
            hasUserId: !!env.IG_USER_ID,
            hasCloudinary: cloudinaryConfigured(),
            imageCloudUrls: postRepo.get(postId)?.imageCloudUrls ?? [],
          });
          return toolError(message);
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
      },
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
- If a tool returns ok:false or an error, you MUST tell the user the exact error message and what to fix. Do not hide or soften failures.
- Keep replies short and conversational, but never omit failure reasons.`;

export interface AgentResult {
  sessionId: string;
  reply: string;
  touchedPosts: ReturnType<typeof postSummary>[];
  /** Structured timeline of what happened this turn — use this when the reply is vague. */
  diagnostics: {
    events: AgentEvent[];
    failures: string[];
    warnings: string[];
    toolsCalled: string[];
  };
}

function buildDiagnostics(ctx: AgentContext, toolsCalled: string[]): AgentResult["diagnostics"] {
  const failures = ctx.events
    .filter((e) => e.status === "error" || e.status === "refused")
    .map((e) => e.message);
  const warnings: string[] = [];

  if (
    CREATE_INTENT.test(ctx.latestUserMessage) &&
    !toolsCalled.includes("create_post") &&
    ctx.touchedPostIds.size === 0
  ) {
    warnings.push(
      "User message looks like a create request, but create_post was not called. The model may have skipped the tool.",
    );
  }
  if (PUBLISH_INTENT.test(ctx.latestUserMessage) && !toolsCalled.includes("publish_post")) {
    warnings.push(
      "User message looks like a publish request, but publish_post was not called. " +
        (!env.IG_ACCESS_TOKEN || !env.IG_USER_ID
          ? "IG credentials may be missing, or the model skipped the tool."
          : "The model may have skipped the tool or needs a clearer postId."),
    );
  }
  if (PUBLISH_INTENT.test(ctx.latestUserMessage) && (!env.IG_ACCESS_TOKEN || !env.IG_USER_ID)) {
    warnings.push(
      "IG_ACCESS_TOKEN / IG_USER_ID not set — publish will fail until they are configured.",
    );
  }
  if (!cloudinaryConfigured()) {
    warnings.push(
      "Cloudinary is not configured (CLOUDINARY_CLOUD_NAME / API_KEY / API_SECRET) — create/edit/publish need it for public image URLs.",
    );
  }

  return { events: ctx.events, failures, warnings, toolsCalled };
}

function enrichReply(reply: string, diagnostics: AgentResult["diagnostics"]): string {
  if (!diagnostics.failures.length && !diagnostics.warnings.length) return reply;

  const bits: string[] = [];
  if (diagnostics.failures.length) {
    bits.push("What failed:\n" + diagnostics.failures.map((f) => `• ${f}`).join("\n"));
  }
  if (diagnostics.warnings.length) {
    bits.push("Warnings:\n" + diagnostics.warnings.map((w) => `• ${w}`).join("\n"));
  }

  const appendix = bits.join("\n\n");
  // Avoid duplicating if the model already pasted the same errors.
  if (diagnostics.failures.every((f) => reply.includes(f.slice(0, 40)))) return reply;
  return `${reply}\n\n---\n${appendix}`;
}

/** Run one chat turn: load history, tool-call loop, persist, respond. */
export async function runAgentTurn(sessionId: string, userMessage: string): Promise<AgentResult> {
  const ctx: AgentContext = {
    latestUserMessage: userMessage,
    touchedPostIds: new Set(),
    events: [],
  };
  const tools = buildTools(ctx);
  const toolMap = new Map(tools.map((t) => [t.name, t]));
  const toolsCalled: string[] = [];

  pushEvent(ctx, "agent", "info", "turn start", {
    sessionId,
    message: userMessage,
    provider: env.AI_PROVIDER,
    model: env.AI_MODEL,
  });

  let model;
  try {
    model = await getModel();
  } catch (e) {
    const message = `Failed to init model (${env.AI_PROVIDER}/${env.AI_MODEL}): ${formatError(e)}`;
    pushEvent(ctx, "agent", "error", message);
    const diagnostics = buildDiagnostics(ctx, toolsCalled);
    return {
      sessionId,
      reply: enrichReply("I couldn't reach the language model.", diagnostics),
      touchedPosts: [],
      diagnostics,
    };
  }

  if (typeof model.bindTools !== "function") {
    const message = `Model ${env.AI_PROVIDER}/${env.AI_MODEL} does not support tool calling (bindTools missing).`;
    pushEvent(ctx, "agent", "error", message);
    const diagnostics = buildDiagnostics(ctx, toolsCalled);
    return {
      sessionId,
      reply: enrichReply(message, diagnostics),
      touchedPosts: [],
      diagnostics,
    };
  }

  const modelWithTools = model.bindTools(tools);

  const history = chatRepo.history(sessionId);
  const humanMsg = new HumanMessage(userMessage);
  const messages: BaseMessage[] = [new SystemMessage(AGENT_SYSTEM_PROMPT), ...history, humanMsg];
  const newMessages: BaseMessage[] = [humanMsg];

  let reply = "I couldn't complete that — please try again.";

  try {
    for (let step = 0; step < 8; step++) {
      pushEvent(ctx, "agent", "info", `model step ${step + 1}`);
      const ai = await modelWithTools.invoke(messages);
      messages.push(ai);
      newMessages.push(ai);

      const toolCalls = ai.tool_calls ?? [];
      if (toolCalls.length === 0) {
        reply = typeof ai.content === "string" ? ai.content : JSON.stringify(ai.content);
        pushEvent(ctx, "agent", "ok", "final reply (no more tools)", {
          replyPreview: reply.slice(0, 160),
        });
        break;
      }

      for (const tc of toolCalls) {
        toolsCalled.push(tc.name);
        pushEvent(ctx, "agent", "info", `tool call ${tc.name}`, { args: tc.args });
        const t = toolMap.get(tc.name);
        let output: string;
        try {
          output = t
            ? String(await (t as { invoke: (args: unknown) => Promise<unknown> }).invoke(tc.args))
            : toolError(`Unknown tool ${tc.name}`);
          if (!t) pushEvent(ctx, "agent", "error", `Unknown tool ${tc.name}`);
        } catch (e) {
          const message = `Tool ${tc.name} threw: ${formatError(e)}`;
          pushEvent(ctx, tc.name, "error", message);
          output = toolError(message);
        }
        const toolMsg = new ToolMessage({ content: output, tool_call_id: tc.id!, name: tc.name });
        messages.push(toolMsg);
        newMessages.push(toolMsg);
      }
    }
  } catch (e) {
    const message = `Agent loop failed: ${formatError(e)}`;
    pushEvent(ctx, "agent", "error", message);
    if (ctx.touchedPostIds.size > 0) {
      reply =
        "I ran into a model error after updating your post(s). The images should still be available below — try sending another message if you need edits or publishing.";
    } else {
      reply = "Something went wrong while talking to the model.";
    }
  }

  chatRepo.append(sessionId, newMessages);

  const touchedPosts = [...ctx.touchedPostIds]
    .map((id) => postRepo.get(id))
    .filter((p): p is Post => !!p)
    .map(postSummary);

  const diagnostics = buildDiagnostics(ctx, toolsCalled);
  reply = enrichReply(reply, diagnostics);

  pushEvent(ctx, "agent", "info", "turn end", {
    failures: diagnostics.failures.length,
    warnings: diagnostics.warnings.length,
    toolsCalled,
  });

  return { sessionId, reply, touchedPosts, diagnostics };
}
