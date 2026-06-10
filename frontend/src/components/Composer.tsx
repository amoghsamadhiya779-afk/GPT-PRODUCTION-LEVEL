"use client";

import React, { useRef, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, Paperclip, Square, RotateCcw, Copy, Check } from "lucide-react";

interface ComposerProps {
  onSend: (text: string) => void;
  isGenerating: boolean;
  onStop: () => void;
  onRegenerate?: () => void;
  lastAssistantMessage?: string;
}

export default function Composer({
  onSend,
  isGenerating,
  onStop,
  onRegenerate,
  lastAssistantMessage,
}: ComposerProps) {
  const [input, setInput] = useState("");
  const [isCopied, setIsCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-expand textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  }, [input]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isGenerating) return;

    onSend(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopy = () => {
    if (!lastAssistantMessage) return;
    navigator.clipboard.writeText(lastAssistantMessage);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-6 pt-2">
      {/* Floating Action Bar above Composer */}
      {(!isGenerating && (onRegenerate || lastAssistantMessage)) && (
        <div className="flex justify-center gap-2 mb-3">
          {onRegenerate && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onRegenerate}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full border border-border bg-surface/50 text-secondary hover:text-primary hover:bg-surface/80 backdrop-blur-sm transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Regenerate response
            </motion.button>
          )}

          {lastAssistantMessage && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full border border-border bg-surface/50 text-secondary hover:text-primary hover:bg-surface/80 backdrop-blur-sm transition-colors"
            >
              {isCopied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-success" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy response
                </>
              )}
            </motion.button>
          )}
        </div>
      )}

      {/* Composer Input Area */}
      <form
        onSubmit={handleSubmit}
        className="relative flex items-end w-full bg-surface/60 border border-border rounded-[24px] shadow-lg focus-within:border-accent/35 focus-within:shadow-accent-glow backdrop-blur-md transition-all p-2 pl-4"
      >
        {/* Attach Button */}
        <button
          type="button"
          className="p-2 mb-1.5 rounded-full text-muted hover:text-primary hover:bg-elevated/45 transition-colors hover:scale-105 active:scale-95 flex-shrink-0"
          title="Attach files (Mock)"
        >
          <Paperclip className="w-4.5 h-4.5" />
        </button>

        {/* Text Area */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send a prompt context to GPT-2..."
          rows={1}
          className="flex-1 max-h-[200px] border-0 bg-transparent py-3 px-2 focus:ring-0 focus:outline-none resize-none overflow-y-auto text-sm md:text-base text-primary placeholder-muted leading-relaxed"
        />

        {/* Send / Stop Action Button */}
        <div className="flex items-center justify-center p-1.5 flex-shrink-0">
          {isGenerating ? (
            <motion.button
              type="button"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onStop}
              className="p-3 bg-destructive hover:bg-destructive/90 text-white rounded-full flex items-center justify-center shadow-md shadow-destructive/20 transition-all hover:scale-105 active:scale-95"
              title="Stop generating"
            >
              <Square className="w-4 h-4 fill-white" />
            </motion.button>
          ) : (
            <motion.button
              type="submit"
              disabled={!input.trim()}
              whileHover={input.trim() ? { scale: 1.02 } : {}}
              whileTap={input.trim() ? { scale: 0.98 } : {}}
              className={`p-3 rounded-full flex items-center justify-center transition-all ${
                input.trim()
                  ? "bg-accent hover:bg-accent-hover text-accent-foreground shadow-md shadow-accent/20 hover:scale-105 active:scale-95 cursor-pointer"
                  : "bg-elevated/40 text-muted cursor-not-allowed"
              }`}
              title="Send prompt"
            >
              <ArrowUp className="w-4.5 h-4.5 font-bold" />
            </motion.button>
          )}
        </div>
      </form>
    </div>
  );
}
