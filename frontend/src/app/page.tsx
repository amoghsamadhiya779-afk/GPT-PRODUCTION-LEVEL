"use client";

import React, { useState, useEffect, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import ChatWindow, { Message } from "@/components/ChatWindow";
import Composer from "@/components/Composer";
import SettingsPanel, { ModelSettings } from "@/components/SettingsPanel";
import ParticleBackground from "@/components/ParticleBackground";
import AnalyticsView from "@/components/AnalyticsView";
import ArchitectureView from "@/components/ArchitectureView";
import { useTheme } from "@/components/ThemeProvider";
import { Settings, Cpu, LineChart, Menu, Plus, Activity, MessageSquare } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import Logo from "@/components/Logo";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
}

export default function Home() {
  const { theme } = useTheme();
  const [currentNav, setCurrentNav] = useState<"chat" | "analytics" | "architecture">("chat");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [settings, setSettings] = useState<ModelSettings>({
    model: "gpt-2-small",
    temperature: 0.8,
    topK: 50,
    topP: 0.9,
    maxTokens: 100,
    useCache: true,
  });

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [backendInfo, setBackendInfo] = useState<{
    status: string;
    checkpoint: string;
    parameters: number;
    device: string;
  } | undefined>(undefined);

  // Chat sessions state
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: "1", title: "Once upon a time context", messages: [] },
  ]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("1");

  const stopRef = useRef<boolean>(false);

  // Auto check backend health periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`);
        if (res.ok) {
          const data = await res.json();
          setBackendInfo({
            status: data.status,
            checkpoint: data.checkpoint,
            parameters: data.parameters,
            device: data.device,
          });
        } else {
          setBackendInfo({
            status: "offline",
            checkpoint: "None",
            parameters: 0,
            device: "cpu",
          });
        }
      } catch (e) {
        setBackendInfo({
          status: "offline",
          checkpoint: "None (Standalone Standby)",
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

  const handleNewChat = () => {
    const newId = String(Date.now());
    const newSession: ChatSession = {
      id: newId,
      title: "New Playground Session",
      messages: [],
    };
    setSessions([newSession, ...sessions]);
    setCurrentSessionId(newId);
    setCurrentNav("chat");
  };

  const handleSelectChat = (id: string) => {
    setCurrentSessionId(id);
    setCurrentNav("chat");
  };

  const handleStop = () => {
    stopRef.current = true;
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

    // Reset stop flag
    stopRef.current = false;
    setIsGenerating(true);

    const userMessage: Message = {
      id: String(Date.now()),
      role: "user",
      content: promptText,
    };

    const newMessages = [...messages, userMessage];
    updateSessionMessages(currentSessionId, newMessages);

    // Placeholder for streaming assistant response
    const assistantMsgId = String(Date.now() + 1);
    const assistantMessage: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    updateSessionMessages(currentSessionId, [...newMessages, assistantMessage]);

    try {
      let result;
      // 1. If backend is online, query the FastAPI server
      if (backendInfo && backendInfo.status !== "offline") {
        const payload = {
          prompt: promptText,
          max_new_tokens: settings.maxTokens,
          temperature: settings.temperature,
          top_k: settings.topK,
          use_cache: settings.useCache,
        };

        const res = await fetch(`${BACKEND_URL}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          result = await res.json();
        } else {
          throw new Error("Backend generated an error.");
        }
      } else {
        // 2. Standalone CPU Fallback Simulation
        await new Promise((resolve) => setTimeout(resolve, 1000));
        result = {
          prompt: promptText,
          generated_text: `${promptText} is a wonderful seed. Since the serving backend is offline, this mock text validates the client layout. Please start the FastAPI backend to run local GPT-2 inference.`,
          tokens_generated: 30,
          time_taken_seconds: 0.25,
          tokens_per_second: 120.0,
        };
      }

      // Typewriter streaming reveal animation
      if (result) {
        const fullText = result.generated_text;
        const newText = fullText.substring(promptText.length);
        const tokens = newText.split(/(\s+)/); // Split by words/whitespaces
        let currentContent = promptText;

        for (let i = 0; i < tokens.length; i++) {
          if (stopRef.current) break;

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
                        }
                      : m
                  ),
                }
              : s
          )
        );
      }
    } catch (e) {
      console.error("Failed to generate response:", e);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === currentSessionId
            ? {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantMsgId
                    ? {
                        ...m,
                        content: "Error: Failed to connect to the model inference service.",
                        isStreaming: false,
                      }
                    : m
                ),
              }
            : s
        )
      );
    } finally {
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
      <ParticleBackground />

      {/* Collapsible Sidebar (Desktop) */}
      <Sidebar
        onOpenSettings={() => setIsSettingsOpen(true)}
        onSelectNav={setCurrentNav}
        currentNav={currentNav}
        onNewChat={handleNewChat}
        history={sessions.map((s) => ({ id: s.id, title: s.title }))}
        currentChatId={currentSessionId}
        onSelectChat={handleSelectChat}
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
                          <button
                            key={chat.id}
                            onClick={() => {
                              handleSelectChat(chat.id);
                              setIsMobileMenuOpen(false);
                            }}
                            className={`w-full text-left flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all ${
                              currentSessionId === chat.id
                                ? "bg-elevated/45 text-primary border border-border"
                                : "text-secondary hover:bg-elevated/20"
                            }`}
                          >
                            <MessageSquare className={`w-4 h-4 flex-shrink-0 ${currentSessionId === chat.id ? "text-accent" : "text-muted"}`} />
                            <span className="truncate flex-1">{chat.title}</span>
                          </button>
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

            {/* Chat Workspace Message Window */}
            <ChatWindow messages={messages} isGenerating={isGenerating} />

            {/* Chat Workspace Fixed Bottom Composer */}
            <Composer
              onSend={handleSend}
              isGenerating={isGenerating}
              onStop={handleStop}
              onRegenerate={messages.length >= 2 ? handleRegenerate : undefined}
              lastAssistantMessage={lastAssistantMessage}
            />
          </div>
        ) : currentNav === "analytics" ? (
          <AnalyticsView backendUrl={BACKEND_URL} />
        ) : (
          <ArchitectureView />
        )}
      </div>

      {/* Floating Settings Drawer */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onChange={setSettings}
        backendInfo={backendInfo}
      />
    </div>
  );
}
