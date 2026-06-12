"use client";

import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useTheme } from "./ThemeProvider";
import { Cpu, User, Sparkles, AlertTriangle } from "lucide-react";
import Logo from "@/components/Logo";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  // Stats
  latency?: number;
  tokensPerSecond?: number;
  tokensGenerated?: number;
  useCache?: boolean;
  sources?: { title: string; snippet: string; link: string }[] | null;
}

interface ChatWindowProps {
  messages: Message[];
  isGenerating: boolean;
}

export default function ChatWindow({ messages, isGenerating }: ChatWindowProps) {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const { theme } = useTheme();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 flex flex-col items-center">
      {/* Container max-width 900px */}
      <div className="w-full max-w-[900px] flex-1 flex flex-col justify-between">
        {messages.length === 0 ? (
          /* Landing State */
          <div className="flex-1 flex flex-col items-center justify-center text-center py-20 px-4 select-none">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.4 }}
              className="w-14 h-14 rounded-2xl bg-accent/5 border border-accent/15 flex items-center justify-center mb-6 shadow-lg shadow-accent-glow"
            >
              <Logo size={28} className="text-accent" />
            </motion.div>
            <motion.h1
              initial={{ y: 8, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.1, duration: 0.35 }}
              className="text-2xl md:text-3xl font-semibold tracking-tight text-primary"
            >
              GPT-2 Production Playground
            </motion.h1>
            <motion.p
              initial={{ y: 8, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.35 }}
              className="mt-3 text-sm md:text-base text-secondary max-w-lg leading-relaxed"
            >
              Inference playground executing directly on custom GPT-2 checkpoints built from scratch in PyTorch.
            </motion.p>
            
            {/* Quick Tips */}
            <motion.div
              initial={{ y: 8, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.35 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-12 w-full max-w-2xl"
            >
              <div className="p-4 rounded-xl border border-border bg-elevated/15 text-left">
                <h3 className="text-xs font-semibold text-primary flex items-center gap-1.5 uppercase tracking-wider">
                  <Sparkles className="w-3.5 h-3.5 text-accent" />
                  Pretrained Weights
                </h3>
                <p className="text-xs text-secondary mt-1.5 leading-relaxed">
                  Loaded with official OpenAI GPT-2 124M parameter weights, mapped layer by layer to generate clean English.
                </p>
              </div>
              <div className="p-4 rounded-xl border border-border bg-elevated/15 text-left">
                <h3 className="text-xs font-semibold text-primary flex items-center gap-1.5 uppercase tracking-wider">
                  <Cpu className="w-3.5 h-3.5 text-accent" />
                  O(N) KV-Caching
                </h3>
                <p className="text-xs text-secondary mt-1.5 leading-relaxed">
                  Causal self-attention cache optimization avoids recalculating keys and values across autoregressive steps.
                </p>
              </div>
            </motion.div>
          </div>
        ) : (
          /* Message List */
          <div className="space-y-6 md:space-y-8">
            {messages.map((message) => {
              const isUser = message.role === "user";
              return (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className={`flex gap-4 max-w-full ${isUser ? "justify-end" : "justify-start"}`}
                >
                  {/* Avatar left */}
                  {!isUser && (
                    <div className="w-8 h-8 rounded-lg bg-accent/5 border border-accent/15 flex items-center justify-center flex-shrink-0">
                      <Logo size={16} className="text-accent" />
                    </div>
                  )}

                  {/* Bubble content */}
                  <div className={`flex flex-col gap-1.5 max-w-[85%] md:max-w-[75%]`}>
                    <div
                      className={`px-4.5 py-3 rounded-2xl text-sm md:text-base leading-relaxed break-words shadow-sm ${
                        isUser
                          ? "bg-elevated border border-border/60 text-primary rounded-tr-none"
                          : "bg-surface/40 border border-border/30 text-secondary rounded-tl-none"
                      }`}
                    >
                      <span className="whitespace-pre-wrap">{message.content}</span>
                      
                      {/* Streaming Indicator */}
                      {message.isStreaming && (
                        <span className="inline-block w-2 h-4 ml-1 bg-accent/80 animate-pulse align-middle" />
                      )}
                    </div>

                    {/* Telemetry metadata footer */}
                    {!isUser && !message.isStreaming && (message.latency !== undefined) && (
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-1 text-[10.5px] font-mono text-muted select-none">
                        <span>Latency: <strong>{message.latency.toFixed(3)}s</strong></span>
                        <span className="text-border">•</span>
                        <span>Speed: <strong>{message.tokensPerSecond?.toFixed(1)} t/s</strong></span>
                        <span className="text-border">•</span>
                        <span>Tokens: <strong>{message.tokensGenerated}</strong></span>
                        <span className="text-border">•</span>
                        <span>KV-Cache: <strong className={message.useCache ? "text-accent" : "text-muted"}>{message.useCache ? "ON" : "OFF"}</strong></span>
                      </div>
                    )}

                    {/* Web Search Sources */}
                    {!isUser && !message.isStreaming && message.sources && message.sources.length > 0 && (
                      <div className="mt-2.5 p-3 rounded-xl border border-border/40 bg-elevated/15 text-left space-y-2 w-full">
                        <h4 className="text-[10.5px] font-semibold text-accent flex items-center gap-1.5 uppercase tracking-wider select-none">
                          <Sparkles className="w-3.5 h-3.5 text-accent animate-pulse" />
                          Retrieved Search Sources
                        </h4>
                        <div className="grid grid-cols-1 gap-2 mt-1.5">
                          {message.sources.map((src, idx) => (
                            <a
                              key={idx}
                              href={src.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block p-2.5 rounded-lg border border-border/30 bg-surface/20 hover:bg-surface/55 hover:border-accent/40 transition-all duration-200 group"
                            >
                              <div className="text-[11.5px] font-semibold text-primary group-hover:text-accent transition-colors truncate">
                                {src.title}
                              </div>
                              <div className="text-[10.5px] text-secondary/80 leading-relaxed line-clamp-2 mt-0.5">
                                {src.snippet}
                              </div>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Avatar right */}
                  {isUser && (
                    <div className="w-8 h-8 rounded-lg bg-elevated border border-border/80 flex items-center justify-center flex-shrink-0">
                      <User className="w-4 h-4 text-secondary" />
                    </div>
                  )}
                </motion.div>
              );
            })}
            
            {/* Generating Skeleton */}
            {isGenerating && messages[messages.length - 1]?.role === "user" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex gap-4 justify-start"
              >
                <div className="w-8 h-8 rounded-lg bg-accent/5 border border-accent/15 flex items-center justify-center flex-shrink-0">
                  <Logo size={16} className="text-accent animate-spin" />
                </div>
                <div className="flex flex-col gap-1.5 w-[140px]">
                  <div className="px-4 py-3 rounded-2xl bg-surface/35 border border-border/20 text-secondary rounded-tl-none flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}
