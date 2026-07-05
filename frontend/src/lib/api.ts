const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export interface GeneratePayload {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  use_cache: boolean;
  web_search: boolean;
}

export interface StreamMetrics {
  time_taken_seconds: number;
  tokens_per_second: number;
  tokens_generated: number;
  sources?: { title: string; snippet: string; link: string }[];
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
    if (!res.ok) throw new Error("Backend error");
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
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      if (!res.body) {
        throw new Error("No response body");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8", { fatal: false });
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed.startsWith("data: ")) {
            const dataStr = trimmed.slice(6);
            if (dataStr === "[DONE]") continue; // Standard SSE close
            
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.sources !== undefined && onSources) {
                onSources(parsed.sources);
              } else if (parsed.token !== undefined) {
                onToken(parsed.token);
              } else if (parsed.done) {
                onComplete({
                  time_taken_seconds: parsed.time_taken_seconds,
                  tokens_per_second: parsed.tokens_per_second,
                  tokens_generated: parsed.tokens_generated,
                  sources: parsed.sources,
                });
              }
            } catch (e) {
              console.warn("Failed to parse SSE line:", line);
            }
          }
        }
      }
      
      // Flush remaining stream if any
      const finalStr = decoder.decode();
      if (finalStr) {
         buffer += finalStr;
         const lines = buffer.split("\n");
         for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data: ")) {
                try {
                  const parsed = JSON.parse(trimmed.slice(6));
                  if (parsed.token !== undefined) onToken(parsed.token);
                  if (parsed.done) onComplete(parsed);
                } catch (e) {}
            }
         }
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
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async getFinetuneStatus(jobId: string) {
    const res = await fetch(`${BACKEND_URL}/finetune/${jobId}`);
    if (!res.ok) throw new Error("Failed to get status");
    return await res.json();
  },

  async getAdapters() {
    try {
      const res = await fetch(`${BACKEND_URL}/adapters`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return { adapters: [] };
  },

  async activateAdapter(name: string) {
    const res = await fetch(`${BACKEND_URL}/adapters/${name}/activate`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to activate");
    return await res.json();
  },

  async deactivateAdapter() {
    const res = await fetch(`${BACKEND_URL}/adapters/deactivate`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to deactivate");
    return await res.json();
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

  async getFeedback() {
    try {
      const res = await fetch(`${BACKEND_URL}/feedback`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return { feedback: [] };
  }
};
