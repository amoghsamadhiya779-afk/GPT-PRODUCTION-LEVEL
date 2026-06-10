"use client";

import React from "react";
import { Cpu, HelpCircle, Layers, Zap, Star } from "lucide-react";

export default function ArchitectureView() {
  const codeAttention = `class MultiHeadAttention(nn.Module):
    def forward(self, x, layer_past=None, use_cache=False):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)      # Projection
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Split into multiple heads
        # ...
        
        # Concatenate keys/values with cache
        if layer_past is not None:
            past_keys, past_values = layer_past
            keys = torch.cat((past_keys, keys), dim=-2)
            values = torch.cat((past_values, values), dim=-2)
            
        present = (keys, values)
        
        # Attention scores calculation
        attn_scores = queries @ keys.transpose(2, 3)
        
        # Causal mask applied to prevent future attention
        if num_tokens > 1:
            mask_bool = self.mask.bool()[:total_tokens, :total_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)`;

  const codeGelu = `class GELU(nn.Module):
    def forward(self, x):
        # Tanh approximation of GELU
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))`;

  return (
    <div className="flex-1 overflow-y-auto px-4 md:px-8 py-8 flex flex-col items-center select-none">
      <div className="w-full max-w-4xl space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-primary flex items-center gap-2">
            <Cpu className="w-7 h-7 text-accent" />
            Model Architecture
          </h1>
          <p className="text-sm text-secondary mt-1">
            Modular transformer block implementation details mapped to custom GPT-2 specifications.
          </p>
        </div>

        {/* Hyperparameters Configuration */}
        <div className="p-5 rounded-2xl border border-border bg-surface/30 backdrop-blur-sm">
          <h3 className="text-sm font-semibold text-primary uppercase tracking-wider mb-4 flex items-center gap-1.5">
            <Star className="w-4 h-4 text-accent" />
            GPT-2 Standard Configuration (124M Parameters)
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            {[
              { label: "Vocabulary Size", value: "50,257 tokens" },
              { label: "Context Window", value: "1,024 tokens" },
              { label: "Embedding Dim (d_in)", value: "768 dimensions" },
              { label: "Attention Heads", value: "12 heads" },
              { label: "Transformer Layers", value: "12 blocks" },
              { label: "Dropout Rate", value: "10% (0.1)" },
              { label: "Bias Terms", value: "Enabled in QKV projections" },
              { label: "Weight Shares", value: "WTE tied with Head output" },
            ].map((config, i) => (
              <div key={i} className="p-3.5 rounded-xl border border-border/50 bg-elevated/15">
                <div className="text-[10px] text-muted uppercase font-medium tracking-wider">{config.label}</div>
                <div className="text-sm font-semibold text-secondary mt-1">{config.value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Two-Column Code and Logic Display */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Attention and Cache Card */}
          <div className="p-5 rounded-2xl border border-border bg-surface/30 backdrop-blur-sm flex flex-col justify-between">
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-primary uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-accent" />
                Causal Attention & Caching
              </h3>
              <p className="text-xs text-secondary leading-relaxed">
                The attention mechanism is built from raw PyTorch tensors. By maintaining the Key-Value states across generation steps, inference time is reduced from O(N²) to O(1) step.
              </p>
            </div>
            <div className="mt-4 rounded-xl border border-border bg-elevated/20 p-4 font-mono text-[11px] text-secondary overflow-x-auto max-h-[350px] leading-relaxed">
              <pre>{codeAttention}</pre>
            </div>
          </div>

          {/* LayerNorm and Activations Card */}
          <div className="p-5 rounded-2xl border border-border bg-surface/30 backdrop-blur-sm flex flex-col justify-between">
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-primary uppercase tracking-wider flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-accent" />
                Pre-LayerNorm & GELU Activation
              </h3>
              <p className="text-xs text-secondary leading-relaxed">
                To guarantee training stability, Layer Normalization (LayerNorm) is applied before entering Multi-Head Self-Attention and Feed-Forward layers. GELU activation functions use the tanh-approximation formula matching the GPT-2 paper.
              </p>
            </div>
            <div className="mt-4 rounded-xl border border-border bg-elevated/20 p-4 font-mono text-[11px] text-secondary overflow-x-auto max-h-[350px] leading-relaxed">
              <pre>{codeGelu}</pre>
            </div>
          </div>
        </div>

        {/* Visual Workflow Notes */}
        <div className="p-4 rounded-xl border border-border bg-elevated/15 flex gap-3 text-xs text-secondary leading-relaxed">
          <HelpCircle className="w-5 h-5 text-accent shrink-0 mt-0.5 animate-pulse" />
          <div>
            <span className="font-semibold text-primary block mb-0.5">How the Token Generation Stream works:</span>
            The user inputs prompt text. The tiktoken BPE tokenizer encodes it to a tensor of token IDs. The model performs a forward pass, loading embeddings, passing through 12 stacked transformer blocks, and generating probability logits. The output head samples next-token indices (greedy/temperature-scaled/top-k), which are appended, decoded, and stream-revealed to the viewport.
          </div>
        </div>
      </div>
    </div>
  );
}
