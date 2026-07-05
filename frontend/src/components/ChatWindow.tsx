"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useTheme } from "./ThemeProvider";
import { Cpu, User, Sparkles, AlertTriangle, GraduationCap, Atom, Compass, ThumbsUp, ThumbsDown, Edit3, Check } from "lucide-react";
import Logo from "@/components/Logo";
import { api } from "@/lib/api";

function MessageFeedback({ prompt, response }: { prompt: string; response: string }) {
  const [rating, setRating] = useState<"up" | "down" | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [correction, setCorrection] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleRate = async (r: "up" | "down") => {
    setRating(r);
    if (r === "up") {
      try {
        await api.submitFeedback({ prompt, response, rating: "up" });
        setSubmitted(true);
      } catch (e) {}
    } else {
      setIsEditing(true);
    }
  };

  const handleCorrect = async () => {
    try {
      await api.submitFeedback({ prompt, response, rating: "down", correction });
      setSubmitted(true);
      setIsEditing(false);
    } catch (e) {}
  };

  if (submitted) {
    return <div className="text-[10px] text-success flex items-center gap-1"><Check className="w-3 h-3"/> Feedback submitted</div>;
  }

  return (
    <div className="flex flex-col gap-2 mt-2">
      {!isEditing ? (
        <div className="flex items-center gap-2">
          <button onClick={() => handleRate("up")} className={`p-1 rounded hover:bg-surface/50 text-secondary hover:text-success transition-colors ${rating === 'up' ? 'text-success' : ''}`}>
            <ThumbsUp className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => handleRate("down")} className={`p-1 rounded hover:bg-surface/50 text-secondary hover:text-destructive transition-colors ${rating === 'down' ? 'text-destructive' : ''}`}>
            <ThumbsDown className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setIsEditing(true)} className="p-1 rounded hover:bg-surface/50 text-secondary hover:text-accent transition-colors flex items-center gap-1 text-[10px] ml-2">
            <Edit3 className="w-3.5 h-3.5" />
            Correct this answer
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-2 bg-elevated/30 p-2 rounded-lg border border-border">
          <textarea
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            placeholder="Provide the correct response..."
            className="w-full text-xs bg-surface border border-border rounded p-2 focus:outline-none focus:border-accent"
            rows={2}
          />
          <div className="flex justify-end gap-2">
            <button onClick={() => setIsEditing(false)} className="text-[10px] text-muted hover:text-secondary px-2 py-1">Cancel</button>
            <button onClick={handleCorrect} className="text-[10px] bg-accent text-accent-foreground px-2 py-1 rounded hover:bg-accent/90">Submit</button>
          </div>
        </div>
      )}
    </div>
  );
}

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
  persona?: "Math Tutor" | "Physics Helper" | "General Assistant" | null;
  onSelectPersona?: (persona: "Math Tutor" | "Physics Helper" | "General Assistant") => void;
  onSelectPrompt?: (promptText: string) => void;
}

export default function ChatWindow({ 
  messages, 
  isGenerating, 
  persona, 
  onSelectPersona, 
  onSelectPrompt 
}: ChatWindowProps) {
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
          /* Landing State: Persona Selector */
          <div className="flex-1 flex flex-col items-center justify-center py-10 px-4 select-none">
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
              className="text-2xl md:text-3xl font-semibold tracking-tight text-primary text-center"
            >
              Select a Persona to Start Learning
            </motion.h1>
            <motion.p
              initial={{ y: 8, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.35 }}
              className="mt-3 text-sm md:text-base text-secondary max-w-lg leading-relaxed text-center"
            >
              Choose a custom tutor persona. The model is fine-tuned to answer prompts inside these domains and recommend next steps.
            </motion.p>
            
            {/* Persona Grid */}
            <motion.div
              initial={{ y: 12, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.4 }}
              className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-10 w-full max-w-3xl"
            >
              {/* Math Tutor Card */}
              <button
                onClick={() => onSelectPersona?.("Math Tutor")}
                className="p-5 rounded-2xl border border-border bg-surface/30 hover:border-accent/40 hover:bg-elevated/10 text-left transition-all duration-300 hover:scale-102 flex flex-col justify-between cursor-pointer h-[200px]"
              >
                <div>
                  <div className="w-10 h-10 rounded-xl bg-sky-500/10 flex items-center justify-center mb-4 text-sky-400">
                    <GraduationCap className="w-5 h-5" />
                  </div>
                  <h3 className="text-sm font-semibold text-primary">Math Tutor</h3>
                  <p className="text-xs text-secondary mt-1.5 leading-relaxed pr-2">
                    Algebra, Calculus, and Geometry equations. Formula derivation and problem-solving steps.
                  </p>
                </div>
                <div className="text-[10px] font-mono text-muted uppercase tracking-wider mt-4">
                  Geometry • Calculus • Algebra
                </div>
              </button>

              {/* Physics Helper Card */}
              <button
                onClick={() => onSelectPersona?.("Physics Helper")}
                className="p-5 rounded-2xl border border-border bg-surface/30 hover:border-accent/40 hover:bg-elevated/10 text-left transition-all duration-300 hover:scale-102 flex flex-col justify-between cursor-pointer h-[200px]"
              >
                <div>
                  <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center mb-4 text-purple-400">
                    <Atom className="w-5 h-5" />
                  </div>
                  <h3 className="text-sm font-semibold text-primary">Physics Helper</h3>
                  <p className="text-xs text-secondary mt-1.5 leading-relaxed pr-2">
                    Classical mechanics, kinematics, force calculations, and thermodynamics.
                  </p>
                </div>
                <div className="text-[10px] font-mono text-muted uppercase tracking-wider mt-4">
                  Mechanics • Energy • Forces
                </div>
              </button>

              {/* General Assistant Card */}
              <button
                onClick={() => onSelectPersona?.("General Assistant")}
                className="p-5 rounded-2xl border border-border bg-surface/30 hover:border-accent/40 hover:bg-elevated/10 text-left transition-all duration-300 hover:scale-102 flex flex-col justify-between cursor-pointer h-[200px]"
              >
                <div>
                  <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center mb-4 text-amber-400">
                    <Compass className="w-5 h-5" />
                  </div>
                  <h3 className="text-sm font-semibold text-primary">General Assistant</h3>
                  <p className="text-xs text-secondary mt-1.5 leading-relaxed pr-2">
                    Importance of mathematics, history, prime numbers, infinity, and theory.
                  </p>
                </div>
                <div className="text-[10px] font-mono text-muted uppercase tracking-wider mt-4">
                  History • Infinity • Theory
                </div>
              </button>
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
                      <div className="flex flex-col gap-2 w-full mt-1">
                        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-1 text-[10.5px] font-mono text-muted select-none">
                          <span>Latency: <strong>{message.latency.toFixed(3)}s</strong></span>
                          <span className="text-border">•</span>
                          <span>Speed: <strong>{message.tokensPerSecond?.toFixed(1)} t/s</strong></span>
                          <span className="text-border">•</span>
                          <span>Tokens: <strong>{message.tokensGenerated}</strong></span>
                          <span className="text-border">•</span>
                          <span>KV-Cache: <strong className={message.useCache ? "text-accent" : "text-muted"}>{message.useCache ? "ON" : "OFF"}</strong></span>
                        </div>
                        
                        {/* Feedback Area */}
                        <div className="px-1">
                          <MessageFeedback prompt={messages[messages.findIndex(m => m.id === message.id) - 1]?.content || ""} response={message.content} />
                        </div>
                      </div>
                    )}

                    {/* Web Search Sources */}
                    {!isUser && message.sources && message.sources.length > 0 && (
                      <div className="mt-2.5 p-3 rounded-xl border border-border/40 bg-elevated/15 text-left space-y-2 w-full">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <h4 className="text-[10.5px] font-semibold text-accent flex items-center gap-1.5 uppercase tracking-wider select-none">
                            <Sparkles className="w-3.5 h-3.5 text-accent animate-pulse" />
                            Retrieved Search Sources
                          </h4>
                          <span className="text-[9px] bg-accent/10 text-accent px-1.5 py-0.5 rounded-full uppercase tracking-wider font-semibold border border-accent/20">
                            Grounded on live web results
                          </span>
                        </div>
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

            {/* Clickable prompt suggestions below assistant responses */}
            {!isGenerating && messages[messages.length - 1]?.role === "assistant" && persona && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 border border-border/40 bg-elevated/10 p-4 rounded-2xl flex flex-col gap-3"
              >
                <div className="text-[10px] font-semibold text-accent uppercase tracking-wider flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  Recommended Next Steps
                </div>
                <div className="flex flex-wrap gap-2 mt-1">
                  {persona === "Math Tutor" && (
                    <>
                      <button
                        onClick={() => onSelectPrompt?.("Explain the Pythagorean theorem")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        Explain the Pythagorean theorem
                      </button>
                      <button
                        onClick={() => onSelectPrompt?.("What is a derivative?")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        What is a derivative?
                      </button>
                      <button
                        onClick={() => onSelectPrompt?.("How do you solve linear equations?")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        How do you solve linear equations?
                      </button>
                    </>
                  )}
                  {persona === "Physics Helper" && (
                    <>
                      <button
                        onClick={() => onSelectPrompt?.("What is Newton's second law?")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        What is Newton's second law?
                      </button>
                      <button
                        onClick={() => onSelectPrompt?.("Explain gravitational potential energy")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        Explain gravitational potential energy
                      </button>
                      <button
                        onClick={() => onSelectPrompt?.("How does speed relate to velocity?")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        How does speed relate to velocity?
                      </button>
                    </>
                  )}
                  {persona === "General Assistant" && (
                    <>
                      <button
                        onClick={() => onSelectPrompt?.("Why is math important?")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        Why is math important?
                      </button>
                      <button
                        onClick={() => onSelectPrompt?.("Explain the concept of infinity")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        Explain the concept of infinity
                      </button>
                      <button
                        onClick={() => onSelectPrompt?.("What are prime numbers?")}
                        className="px-3 py-1.5 text-xs rounded-xl border border-border bg-surface hover:border-accent hover:bg-surface/80 transition-all text-secondary hover:text-primary hover:-translate-y-0.5 shadow-sm cursor-pointer"
                      >
                        What are prime numbers?
                      </button>
                    </>
                  )}
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
