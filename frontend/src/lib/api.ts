const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// Carries the HTTP status (when known) so callers can tell a rate limit
// (429) apart from the model still warming up (503) apart from a plain
// network failure (no status) instead of showing one generic error for all
// of them.
export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface GeneratePayload {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  use_cache: boolean;
  web_search: boolean;
  // LoRA adapter for this request only: omit for the server default,
  // "none" for the plain base model.
  adapter?: string;
}

export interface StreamMetrics {
  time_taken_seconds: number;
  tokens_per_second: number;
  tokens_generated: number;
  sources?: { title: string; snippet: string; link: string }[];
  safety_net_prefix?: string;
}

export interface HealthStatus {
  status: string;
  checkpoint: string;
  parameters: number;
  device: string;
  error_details?: string;
  uptime_seconds?: number;
  total_requests?: number;
  avg_tokens_per_second?: number;
  model_size?: string;
  layers?: number;
  heads?: number;
  emb_dim?: number;
  context?: number;
  default_adapter?: string | null;
}

export const api = {
  async health() {
    try {
      const res = await fetch(`${BACKEND_URL}/health`);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      // Ignore
    }
    return null;
  },

  async generate(payload: GeneratePayload, signal?: AbortSignal) {
    const res = await fetch(`${BACKEND_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!res.ok) throw new ApiError(`Backend error: ${res.status}`, res.status);
    return await res.json();
  },

  async generateStream(
    payload: GeneratePayload,
    onToken: (token: string) => void,
    onComplete: (metrics: StreamMetrics) => void,
    onError: (err: any) => void,
    signal?: AbortSignal,
    onSources?: (sources: { title: string; snippet: string; link: string }[]) => void
  ) {
    try {
      const res = await fetch(`${BACKEND_URL}/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
      });

      if (!res.ok) {
        throw new ApiError(`HTTP error! status: ${res.status}`, res.status);
      }
      if (!res.body) {
        throw new Error("No response body");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8", { fatal: false });
      let buffer = "";
      let finished = false;

      // Returns true once the stream has delivered its terminal event.
      const handleLine = (line: string): boolean => {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) return false;
        const dataStr = trimmed.slice(6);
        if (dataStr === "[DONE]") return false; // Standard SSE close

        let parsed: any;
        try {
          parsed = JSON.parse(dataStr);
        } catch (e) {
          console.warn("Failed to parse SSE line:", line);
          return false;
        }
        if (parsed.error !== undefined) {
          // Server-side generation failure: surface it instead of silently
          // waiting for a `done` event that will never come.
          throw new ApiError(String(parsed.error), 500);
        }
        if (parsed.sources !== undefined && !parsed.done) {
          onSources?.(parsed.sources);
        } else if (parsed.token !== undefined) {
          onToken(parsed.token);
        } else if (parsed.done) {
          onComplete({
            time_taken_seconds: parsed.time_taken_seconds,
            tokens_per_second: parsed.tokens_per_second,
            tokens_generated: parsed.tokens_generated,
            sources: parsed.sources,
            safety_net_prefix: parsed.safety_net_prefix,
          });
          return true;
        }
        return false;
      };

      while (!finished) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (handleLine(line)) {
            finished = true;
            break;
          }
        }
      }

      // Flush remaining stream if any
      buffer += decoder.decode();
      if (!finished) {
        for (const line of buffer.split("\n")) {
          if (handleLine(line)) {
            finished = true;
            break;
          }
        }
      }

      if (!finished) {
        // Connection dropped mid-generation (proxy timeout, server restart).
        throw new ApiError("The response stream ended unexpectedly.");
      }

    } catch (e: any) {
      if (e.name === "AbortError") {
        // Just resolve gracefully so partial text is kept
        onComplete({
            time_taken_seconds: 0,
            tokens_per_second: 0,
            tokens_generated: 0
        });
      } else {
        onError(e);
      }
    }
  },

  async startFinetune(payload: { examples: any[]; adapter_name: string; steps: number; lr: number }) {
    const res = await fetch(`${BACKEND_URL}/finetune`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        if (typeof body.detail === "string") detail = body.detail;
      } catch (e) {}
      throw new ApiError(detail, res.status);
    }
    return await res.json();
  },

  async getFinetuneStatus(jobId: string) {
    const res = await fetch(`${BACKEND_URL}/finetune/${jobId}`);
    if (!res.ok) throw new Error("Failed to get status");
    return await res.json();
  },

  // Adapters are chosen per request (GeneratePayload.adapter). Changing the
  // server-wide default is an admin-only operation and not exposed here.
  async getAdapters(): Promise<{ adapters: string[]; default: string | null }> {
    try {
      const res = await fetch(`${BACKEND_URL}/adapters`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return { adapters: [], default: null };
  },

  async submitFeedback(payload: { prompt: string; response: string; rating: string; correction?: string }) {
    const res = await fetch(`${BACKEND_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to submit feedback");
    return await res.json();
  },

  async getStarterDataset() {
    try {
      const res = await fetch(`${BACKEND_URL}/starter-dataset`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return { dataset: [] };
  },

  async getTrainingPlotBlob(): Promise<Blob | null> {
    try {
      const res = await fetch(`${BACKEND_URL}/training/plot`);
      if (res.ok) return await res.blob();
    } catch (e) {}
    return null;
  },
};
