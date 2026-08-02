"use client";

import React, { useState, useEffect, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import ChatWindow, { Message } from "@/components/ChatWindow";
import Composer from "@/components/Composer";
import SettingsPanel, { ModelSettings } from "@/components/SettingsPanel";
import ParticleBackground from "@/components/ParticleBackground";
import AnalyticsView from "@/components/AnalyticsView";
import ArchitectureView from "@/components/ArchitectureView";
import TeachView from "@/components/TeachView";
import { useTheme } from "@/components/ThemeProvider";
import { MessageSquare, LineChart, Cpu, GraduationCap, Settings, Plus, Play, Pause, Activity, Menu, X, AlertTriangle } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import Logo from "@/components/Logo";
import BootSequence from "@/components/BootSequence";
import { Toaster } from "@/components/ui/toast";
import { api, ApiError, HealthStatus } from "@/lib/api";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  persona?: "Socrates" | "Einstein" | "Shakespeare" | null;
}

// Persona instructions prepended to the prompt. These are prompt-engineering
// framing on top of the general instruction-tuned model -- there are no
// persona-specific fine-tuned adapters shipped, so this is what actually
// drives the persona's behavior (plus the sampling presets below).
const PERSONA_FRAMING: Record<"Socrates" | "Einstein" | "Shakespeare", string> = {
  Socrates:
    "Respond in the voice and style of Socrates: use the Socratic method, ask probing questions, and reason through philosophical inquiry.",
  Einstein:
    "Respond in the voice and style of Albert Einstein: explain physics and scientific concepts with clarity, curiosity, and vivid analogies.",
  Shakespeare:
    "Respond in the voice and style of William Shakespeare: use rich, poetic, dramatic Early Modern English.",
};

export default function Home() {
  const { theme } = useTheme();
  const [currentNav, setCurrentNav] = useState<"chat" | "analytics" | "architecture" | "teach">("chat");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [settings, setSettings] = useState<ModelSettings>({
    model: "gpt-2-small",
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
    repetitionPenalty: 1.15,
    maxTokens: 100,
    useCache: true,
    webSearch: false,
  });

  useEffect(() => {
    try {
      const saved = localStorage.getItem("gptStudioSettings");
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (saved) {
        const parsed = JSON.parse(saved);
        // Migrate old defaults to new improved defaults
        let migrated = false;
        if (parsed.temperature === 0.8) { parsed.temperature = 0.7; migrated = true; }
        if (parsed.repetitionPenalty === 1.0) { parsed.repetitionPenalty = 1.3; migrated = true; }
        if (parsed.repetitionPenalty === 1.3) { parsed.repetitionPenalty = 1.15; migrated = true; }
        if (parsed.topP === undefined || parsed.topP === null) { parsed.topP = 0.9; migrated = true; }
        
        setSettings(parsed);
        if (migrated) {
          localStorage.setItem("gptStudioSettings", JSON.stringify(parsed));
        }
      }
    } catch (e) {}
  }, []);

  const handleSettingsChange = (newSettings: ModelSettings) => {
    setSettings(newSettings);
    try {
      localStorage.setItem("gptStudioSettings", JSON.stringify(newSettings));
    } catch (e) {}
  };

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Chat sessions state
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: "1", title: "New Playground Session", messages: [], persona: null },
  ]);
  const [backendInfo, setBackendInfo] = useState<HealthStatus | null>(null);
  const [isBannerDismissed, setIsBannerDismissed] = useState(false);
  
  const [currentSessionId, setCurrentSessionId] = useState<string>("1");
  const [sessionsLoaded, setSessionsLoaded] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Restore chat history from localStorage on mount. Any message still
  // marked isStreaming is from a generation that was interrupted by the
  // reload -- no generation is actually in flight anymore, so clear it
  // rather than leaving a permanently "typing..." bubble.
  useEffect(() => {
    try {
      const saved = localStorage.getItem("gptStudioSessions");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          const sanitized = parsed.map((s: ChatSession) => ({
            ...s,
            messages: s.messages.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m)),
          }));
          // eslint-disable-next-line react-hooks/set-state-in-effect
          setSessions(sanitized);
          // eslint-disable-next-line react-hooks/set-state-in-effect
          setCurrentSessionId(sanitized[0].id);
        }
      }
    } catch (e) {}
    setSessionsLoaded(true);
  }, []);

  // Persist chat history after the initial restore above has run, so we
  // don't immediately overwrite saved history with the default empty session.
  useEffect(() => {
    if (!sessionsLoaded) return;
    try {
      localStorage.setItem("gptStudioSessions", JSON.stringify(sessions));
    } catch (e) {}
  }, [sessions, sessionsLoaded]);

  useEffect(() => {
    if (backendInfo?.status === "active") {
      setIsBannerDismissed(false);
    }
  }, [backendInfo?.status]);

  // Auto check backend health periodically
  useEffect(() => {
    const checkHealth = async () => {
      const data = await api.health();
      if (data) {
        setBackendInfo(data as HealthStatus);
      } else {
          setBackendInfo({
            status: "offline",
            checkpoint: "None",
            parameters: 0,
            device: "cpu",
          });
        }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const currentSession = sessions.find((s) => s.id === currentSessionId) || sessions[0];
  const messages = currentSession?.messages || [];

  // Personas are prompt-based framing over the general instruction-tuned model,
  // not dedicated fine-tuned adapters -- no persona-specific adapters are shipped,
  // so we never touch adapter state here (see PERSONA_FRAMING below for what
  // actually gets sent to the model).
  const getWelcomeMessage = (persona: "Socrates" | "Einstein" | "Shakespeare") => {
    switch (persona) {
      case "Socrates":
        return (
          "Greetings. I am styled after Socrates. Let us engage in dialectic inquiry and ponder the truths of our world together.\n\n" +
          "Here are some prompts to try:\n" +
          "- **'What is the meaning of a good life?'**\n" +
          "- **'Why do people believe in things they cannot see?'**\n" +
          "- **'What makes a society just?'**"
        );
      case "Einstein":
        return (
          "Hello! I am styled after Albert Einstein. I focus on physics, natural sciences, and fundamental analysis. Let's explore the universe's mechanics!\n\n" +
          "Here are some prompts to try:\n" +
          "- **'Why is the sky blue?'**\n" +
          "- **'What are black holes?'**\n" +
          "- **'How do airplanes stay in the air?'**"
        );
      case "Shakespeare":
        return (
          "Hark! I am styled after William Shakespeare, at thy service for poetry, drama, and creative musings. Let us weave tales together!\n\n" +
          "Here are some prompts to try:\n" +
          "- **'Write a poem about the sea.'**\n" +
          "- **'Describe a rainy day in London.'**\n" +
          "- **'Tell me a dramatic story about betrayal.'**"
        );
      default:
        return "";
    }
  };

  const handleSelectPersona = (persona: "Socrates" | "Einstein" | "Shakespeare") => {
    const welcomeMsg: Message = {
      id: "welcome-" + Date.now(),
      role: "assistant",
      content: getWelcomeMessage(persona),
      personaBadge: persona,
    };

    setSessions((prev) =>
      prev.map((s) =>
        s.id === currentSessionId
          ? {
              ...s,
              persona: persona,
              messages: [welcomeMsg],
              title: `${persona} Session`,
            }
          : s
      )
    );

    let newSettings = { ...settings };

    // Apply generation presets
    if (persona === "Einstein") {
      newSettings = { ...newSettings, temperature: 0.5, webSearch: true };
    } else if (persona === "Socrates") {
      newSettings = { ...newSettings, temperature: 0.7, maxTokens: 150, webSearch: false };
    } else if (persona === "Shakespeare") {
      newSettings = { ...newSettings, temperature: 0.9, repetitionPenalty: 1.1, webSearch: false };
    }

    handleSettingsChange(newSettings);
  };

  const handleNewChat = async () => {
    const newId = String(Date.now());
    const newSession: ChatSession = {
      id: newId,
      title: "New Playground Session",
      messages: [],
      persona: null,
    };
    setSessions([newSession, ...sessions]);
    setCurrentSessionId(newId);
    setCurrentNav("chat");

    try {
      await api.deactivateAdapter();
      handleSettingsChange({ ...settings, activeAdapter: null });
    } catch (e) {
      console.warn("Failed to deactivate adapter on new chat");
    }
  };

  const handleSelectChat = (id: string) => {
    setCurrentSessionId(id);
    setCurrentNav("chat");
  };

  const handleDeleteChat = (id: string) => {
    setSessions((prev) => {
      const remaining = prev.filter((s) => s.id !== id);
      if (remaining.length === 0) {
        const fresh: ChatSession = {
          id: String(Date.now()),
          title: "New Playground Session",
          messages: [],
          persona: null,
        };
        if (currentSessionId === id) setCurrentSessionId(fresh.id);
        return [fresh];
      }
      if (currentSessionId === id) setCurrentSessionId(remaining[0].id);
      return remaining;
    });
  };

  const handleClearPersona = () => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === currentSessionId
          ? { ...s, persona: null, messages: [] }
          : s
      )
    );
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsGenerating(false);
  };

  const updateSessionMessages = (sessionId: string, newMessages: Message[]) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId
          ? {
              ...s,
              messages: newMessages,
              title:
                s.title === "New Playground Session" && newMessages.length > 0
                  ? newMessages[0].content.substring(0, 30) + "..."
                  : s.title,
            }
          : s
      )
    );
  };

  const handleSend = async (promptText: string) => {
    if (isGenerating) return;

    abortControllerRef.current = new AbortController();
    setIsGenerating(true);

    const userMessage: Message = {
      id: String(Date.now()),
      role: "user",
      content: promptText,
    };

    const newMessages = [...messages, userMessage];
    updateSessionMessages(currentSessionId, newMessages);

    const assistantMsgId = String(Date.now() + 1);
    const assistantMessage: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
      personaBadge: currentSession?.persona && !settings.webSearch ? currentSession.persona : undefined,
    };

    updateSessionMessages(currentSessionId, [...newMessages, assistantMessage]);

    try {
      if (backendInfo && backendInfo.status !== "offline") {
        // Frame the persona as an instruction the base instruction-tuned model
        // can actually act on -- there is no persona-specific fine-tuned
        // adapter, so this is prompt engineering, not a model swap.
        const finalPrompt = currentSession?.persona && !settings.webSearch
          ? `${PERSONA_FRAMING[currentSession.persona]}\n\n${promptText}`
          : promptText;

        const payload = {
          prompt: finalPrompt,
          max_new_tokens: settings.maxTokens,
          temperature: settings.temperature,
          top_k: settings.topK,
          top_p: settings.topP,
          repetition_penalty: settings.repetitionPenalty,
          use_cache: settings.useCache,
          web_search: settings.webSearch || false,
        };

        let currentContent = "";
        
        await api.generateStream(
          payload,
          (token) => {
            currentContent += token;
            setSessions((prev) =>
              prev.map((s) =>
                s.id === currentSessionId
                  ? {
                      ...s,
                      messages: s.messages.map((m) =>
                        m.id === assistantMsgId ? { ...m, content: currentContent } : m
                      ),
                    }
                  : s
              )
            );
          },
          (metrics) => {
            const finalContent = metrics.safety_net_prefix 
              ? metrics.safety_net_prefix + currentContent
              : currentContent;
              
            setSessions((prev) =>
              prev.map((s) =>
                s.id === currentSessionId
                  ? {
                      ...s,
                      messages: s.messages.map((m) =>
                        m.id === assistantMsgId
                          ? {
                              ...m,
                              content: finalContent,
                              isStreaming: false,
                              latency: metrics.time_taken_seconds,
                              tokensPerSecond: metrics.tokens_per_second,
                              tokensGenerated: metrics.tokens_generated,
                              useCache: settings.useCache,
                              sources: metrics.sources || null,
                            }
                          : m
                      ),
                    }
                  : s
              )
            );
            setIsGenerating(false);
          },
          (err) => {
            throw err;
          },
          abortControllerRef.current.signal,
          (sources) => {
            setSessions((prev) =>
              prev.map((s) =>
                s.id === currentSessionId
                  ? {
                      ...s,
                      messages: s.messages.map((m) =>
                        m.id === assistantMsgId ? { ...m, sources } : m
                      ),
                    }
                  : s
              )
            );
          }
        );

      } else {
        // 2. Standalone CPU Fallback Simulation
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const result = {
          prompt: promptText,
          generated_text: `${promptText}\n\n[MOCK MODE] Since the serving backend is offline, this mock text validates the client layout. Please start the FastAPI backend to run local GPT-2 inference.`,
          tokens_generated: 30,
          time_taken_seconds: 0.25,
          tokens_per_second: 120.0,
        };

        const fullText = result.generated_text;
        // Strip prompt from mock response
        const newText = fullText.includes(`${promptText}\n\n`) ? fullText.split(`${promptText}\n\n`)[1] : fullText;
        const tokens = newText.split(/(\s+)/); // Split by words/whitespaces
        let currentContent = "";

        for (let i = 0; i < tokens.length; i++) {
          if (abortControllerRef.current?.signal.aborted) break;

          currentContent += tokens[i];
          setSessions((prev) =>
            prev.map((s) =>
              s.id === currentSessionId
                ? {
                    ...s,
                    messages: s.messages.map((m) =>
                      m.id === assistantMsgId ? { ...m, content: currentContent } : m
                    ),
                  }
                : s
            )
          );
          // Wait 20ms per token
          await new Promise((resolve) => setTimeout(resolve, 20));
        }

        // Finalize completed message stats
        setSessions((prev) =>
          prev.map((s) =>
            s.id === currentSessionId
              ? {
                  ...s,
                  messages: s.messages.map((m) =>
                    m.id === assistantMsgId
                      ? {
                          ...m,
                          content: currentContent,
                          isStreaming: false,
                          latency: result.time_taken_seconds,
                          tokensPerSecond: result.tokens_per_second,
                          tokensGenerated: result.tokens_generated,
                          useCache: settings.useCache,
                          sources: null,
                        }
                      : m
                  ),
                }
              : s
          )
        );
        setIsGenerating(false);
      }
    } catch (e) {
      console.error("Failed to generate response:", e);

      let errorText = "[Error: Failed to connect or generation aborted]";
      if (e instanceof ApiError) {
        if (e.status === 429) {
          errorText = "[Error: Too many requests -- please wait a moment before trying again]";
        } else if (e.status === 503) {
          errorText = "[Error: The model is warming up or busy -- please try again shortly]";
        } else if (e.status && e.status >= 500) {
          errorText = "[Error: The server hit a problem generating a response -- please try again]";
        } else if (e.status) {
          errorText = `[Error: Request failed (${e.status})]`;
        }
      }

      setSessions((prev) =>
        prev.map((s) =>
          s.id === currentSessionId
            ? {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantMsgId
                    ? {
                        ...m,
                        content: m.content + "\n\n" + errorText,
                        isStreaming: false,
                      }
                    : m
                ),
              }
            : s
        )
      );
      setIsGenerating(false);
    }
  };

  const handleRegenerate = () => {
    // Find last user message
    const userMsgs = messages.filter((m) => m.role === "user");
    if (userMsgs.length === 0) return;

    const lastUserMsg = userMsgs[userMsgs.length - 1];
    // Delete last assistant msg
    const cleanMessages = messages.filter((_, idx) => idx < messages.length - 1);
    updateSessionMessages(currentSessionId, cleanMessages);
    handleSend(lastUserMsg.content);
  };

  const lastAssistantMessage = messages
    .filter((m) => m.role === "assistant" && !m.isStreaming)
    .pop()?.content;

  return (
    <div className="flex w-full h-dvh overflow-hidden bg-background text-foreground transition-colors duration-500 font-sans">
      <BootSequence />
      <ParticleBackground />
      <Toaster />

      {/* Collapsible Sidebar (Desktop) */}
      <Sidebar
        onOpenSettings={() => setIsSettingsOpen(true)}
        onSelectNav={setCurrentNav}
        currentNav={currentNav}
        onNewChat={handleNewChat}
        history={sessions.map((s) => ({ id: s.id, title: s.title }))}
        currentChatId={currentSessionId}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Mobile Header Bar */}
        <header className="flex md:hidden items-center justify-between px-4 h-16 border-b border-border bg-surface/50 backdrop-blur-md z-30 select-none w-full">
          <div className="flex items-center gap-3">
            <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
              <SheetTrigger className="p-2 rounded-lg border border-border bg-elevated/40 hover:bg-elevated/80 text-secondary transition-all cursor-pointer">
                <Menu className="w-5 h-5" />
              </SheetTrigger>
              <SheetContent side="left" className="p-0 border-r border-border bg-surface w-[280px]">
                <div className="flex flex-col h-full">
                  <div className="p-4 h-16 border-b border-border flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <Logo size={20} className="text-accent shrink-0" />
                      <span className="font-semibold text-sm tracking-tight text-primary">
                        GPT Studio<span className="text-accent">.</span>
                      </span>
                    </div>
                  </div>

                  {/* Mobile View Navigation */}
                  <div className="px-2 py-3 space-y-1 border-b border-border/60">
                    <div className="px-3 mb-2 text-[10px] font-semibold text-muted uppercase tracking-wider">
                      Navigation
                    </div>
                    {([
                      { id: "chat", name: "Playground Chat", icon: MessageSquare },
                      { id: "analytics", name: "Training Telemetry", icon: LineChart },
                      { id: "architecture", name: "Model Architecture", icon: Cpu },
                      { id: "teach", name: "Teach Mode", icon: GraduationCap },
                    ] as const).map((item) => {
                      const isActive = currentNav === item.id;
                      const Icon = item.icon;
                      return (
                        <button
                          key={item.id}
                          onClick={() => {
                            setCurrentNav(item.id);
                            setIsMobileMenuOpen(false);
                          }}
                          className={`w-full flex items-center gap-3 p-2.5 rounded-lg text-sm font-medium transition-all ${
                            isActive
                              ? "bg-primary/5 text-primary border border-primary/10 shadow-sm"
                              : "text-secondary hover:bg-elevated/30 hover:text-primary border border-transparent"
                          } px-3`}
                        >
                          <Icon className={`w-4.5 h-4.5 flex-shrink-0 ${isActive ? "text-accent" : "text-muted"}`} />
                          <span>{item.name}</span>
                        </button>
                      );
                    })}
                  </div>

                  {/* Actions & Session List (Only shown if on Chat view) */}
                  {currentNav === "chat" && (
                    <>
                      <div className="p-3">
                        <button
                          onClick={() => {
                            handleNewChat();
                            setIsMobileMenuOpen(false);
                          }}
                          className="w-full flex items-center justify-center gap-2 px-4 py-2 border border-border bg-elevated/25 text-secondary hover:text-primary transition-all rounded-lg font-medium text-sm h-10"
                        >
                          <Plus className="w-4 h-4 text-accent" />
                          New Chat
                        </button>
                      </div>
                      <div className="flex-1 overflow-y-auto px-2 space-y-1">
                        <div className="px-3 mb-2 text-[10px] font-semibold text-muted uppercase tracking-wider">
                          Recent Chats
                        </div>
                        {sessions.map((chat) => (
                          <div
                            key={chat.id}
                            className={`flex items-center rounded-lg text-sm transition-all ${
                              currentSessionId === chat.id
                                ? "bg-elevated/45 text-primary border border-border"
                                : "text-secondary hover:bg-elevated/20"
                            }`}
                          >
                            <button
                              onClick={() => {
                                handleSelectChat(chat.id);
                                setIsMobileMenuOpen(false);
                              }}
                              className="flex-1 min-w-0 text-left flex items-center gap-2.5 px-3 py-2"
                            >
                              <MessageSquare className={`w-4 h-4 flex-shrink-0 ${currentSessionId === chat.id ? "text-accent" : "text-muted"}`} />
                              <span className="truncate flex-1">{chat.title}</span>
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteChat(chat.id);
                              }}
                              aria-label={`Delete chat: ${chat.title}`}
                              title="Delete chat"
                              className="p-1.5 mr-1.5 rounded-md text-muted hover:text-red-500 hover:bg-red-500/10 transition-all flex-shrink-0"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                  
                  {currentNav !== "chat" && (
                    <div className="flex-1 flex items-center justify-center p-6 text-center text-xs text-muted">
                      Select Playground Chat to view active sessions.
                    </div>
                  )}

                  <div className="p-3 border-t border-border mt-auto">
                    <button
                      onClick={() => {
                        setIsSettingsOpen(true);
                        setIsMobileMenuOpen(false);
                      }}
                      className="w-full flex items-center gap-3 p-2.5 rounded-lg text-sm font-medium text-secondary hover:bg-elevated/40 hover:text-primary transition-all"
                    >
                      <Settings className="w-4.5 h-4.5 text-muted" />
                      Settings
                    </button>
                  </div>
                </div>
              </SheetContent>
            </Sheet>

            <Logo size={18} className="text-accent shrink-0" />
            <span className="font-semibold text-sm tracking-tight text-primary">
              GPT Studio<span className="text-accent">.</span>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setIsSettingsOpen(true);
                setIsMobileMenuOpen(false);
              }}
              className="p-2 rounded-lg border border-border bg-elevated/40 hover:bg-elevated/80 text-secondary transition-all"
            >
              <Settings className="w-4.5 h-4.5" />
            </button>
          </div>
        </header>

        {/* View Switcher Routing */}
        {currentNav === "chat" ? (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            {/* Top Workspace Header (Desktop Only) */}
            <div className="hidden md:flex items-center justify-between px-6 h-16 border-b border-border bg-surface/10 backdrop-blur-[2px] z-20">
              <div>
                <h2 className="text-sm font-semibold tracking-tight text-primary">Model Inference Playground</h2>
                <p className="text-[10px] text-muted font-mono uppercase tracking-wider mt-0.5">
                  Custom GPT-2 Engine
                </p>
              </div>
              <div className="flex items-center gap-3">
                {/* Connection Pill */}
                {backendInfo ? (
                  <div className="flex items-center gap-2 border border-border bg-elevated/15 px-3 py-1.5 rounded-full text-xs font-medium">
                    <Activity className={`w-3.5 h-3.5 ${backendInfo.status === "active" ? "text-success animate-pulse" : "text-warning"}`} />
                    <span className="text-secondary">
                      {backendInfo.status === "active" ? "Connected" : "Fallback Standby"}
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 border border-border bg-elevated/15 px-3 py-1.5 rounded-full text-xs font-medium">
                    <Activity className="w-3.5 h-3.5 text-muted animate-pulse" />
                    <span className="text-secondary">Pinging...</span>
                  </div>
                )}
              </div>
            </div>

            {/* Warm-Up Banner */}
            {backendInfo && backendInfo.status !== "active" && backendInfo.status !== "offline" && !isBannerDismissed && (
              <div className="bg-warning/10 border-b border-warning/20 px-4 py-2 flex items-center justify-between text-xs text-warning z-10 shadow-sm shrink-0">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Model warming up — responses come from a fallback engine until weights finish loading.</span>
                </div>
                <button onClick={() => setIsBannerDismissed(true)} aria-label="Dismiss banner" className="p-1 hover:bg-warning/20 rounded-md transition-colors">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Chat Workspace Message Window */}
            <ChatWindow 
              messages={messages} 
              isGenerating={isGenerating} 
              persona={currentSession?.persona || null}
              onSelectPersona={handleSelectPersona}
              onClearPersona={handleClearPersona}
              onSelectPrompt={handleSend}
            />

            {/* Chat Workspace Fixed Bottom Composer */}
            <Composer
              onSend={handleSend}
              isGenerating={isGenerating}
              onStop={handleStop}
              onRegenerate={messages.length >= 2 ? handleRegenerate : undefined}
              lastAssistantMessage={lastAssistantMessage}
              isOffline={backendInfo?.status === "offline"}
              webSearch={settings.webSearch || false}
              onToggleWebSearch={(val) => handleSettingsChange({ ...settings, webSearch: val })}
              disabled={!currentSession?.persona}
            />
          </div>
        ) : currentNav === "analytics" ? (
          <AnalyticsView backendUrl={BACKEND_URL} currentSession={currentSession} />
        ) : currentNav === "architecture" ? (
          <ArchitectureView backendInfo={backendInfo} />
        ) : (
          <TeachView />
        )}
      </div>

      {/* Floating Settings Drawer */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onChange={handleSettingsChange}
        backendInfo={backendInfo}
        activePersona={currentSession?.persona || null}
      />
    </div>
  );
}
