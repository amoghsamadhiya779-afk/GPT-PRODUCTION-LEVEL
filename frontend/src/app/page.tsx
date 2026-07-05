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
import { Settings, Cpu, LineChart, Menu, Plus, Activity, MessageSquare, GraduationCap } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import Logo from "@/components/Logo";
import BootSequence from "@/components/BootSequence";
import { api } from "@/lib/api";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  persona?: "Math Tutor" | "Physics Helper" | "General Assistant" | null;
}

export default function Home() {
  const { theme } = useTheme();
  const [currentNav, setCurrentNav] = useState<"chat" | "analytics" | "architecture" | "teach">("chat");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [settings, setSettings] = useState<ModelSettings>({
    model: "gpt-2-small",
    temperature: 0.8,
    topK: 50,
    topP: 0.9,
    repetitionPenalty: 1.0,
    maxTokens: 100,
    useCache: true,
    webSearch: false,
  });

  useEffect(() => {
    try {
      const saved = localStorage.getItem("gptStudioSettings");
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (saved) setSettings(JSON.parse(saved));
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
  const [backendInfo, setBackendInfo] = useState<{
    status: string;
    checkpoint: string;
    parameters: number;
    device: string;
  } | undefined>(undefined);

  // Chat sessions state
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: "1", title: "New Playground Session", messages: [], persona: null },
  ]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("1");

  const abortControllerRef = useRef<AbortController | null>(null);

  // Auto check backend health periodically
  useEffect(() => {
    const checkHealth = async () => {
      const data = await api.health();
      if (data) {
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
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const currentSession = sessions.find((s) => s.id === currentSessionId) || sessions[0];
  const messages = currentSession?.messages || [];

  const getWelcomeMessage = (persona: "Math Tutor" | "Physics Helper" | "General Assistant") => {
    switch (persona) {
      case "Math Tutor":
        return (
          "Hello! I am your Math Tutor. I specialize in algebra, geometry, calculus, and mathematical equations. Let's explore mathematical concepts together!\n\n" +
          "Here are some prompts you can try to focus on what I have been trained on:\n" +
          "- **'Explain the Pythagorean theorem'**\n" +
          "- **'What is a derivative?'**\n" +
          "- **'How do you solve linear equations?'**"
        );
      case "Physics Helper":
        return (
          "Hello! I am your Physics Helper. I focus on classical mechanics, kinematics, forces, and thermodynamics. Let's solve physical mathematics together!\n\n" +
          "Here are some prompts you can try to focus on what I have been trained on:\n" +
          "- **'What is Newton's second law?'**\n" +
          "- **'Explain gravitational potential energy'**\n" +
          "- **'How does speed relate to velocity?'**"
        );
      case "General Assistant":
        return (
          "Hello! I am your General Assistant. I can help you understand general math theories, numbers, and logic. Let's start learning!\n\n" +
          "Here are some prompts you can try to focus on what I have been trained on:\n" +
          "- **'Why is math important?'**\n" +
          "- **'Explain the concept of infinity'**\n" +
          "- **'What are prime numbers?'**"
        );
      default:
        return "";
    }
  };

  const handleSelectPersona = (persona: "Math Tutor" | "Physics Helper" | "General Assistant") => {
    const welcomeMsg: Message = {
      id: "welcome-" + Date.now(),
      role: "assistant",
      content: getWelcomeMessage(persona),
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
  };

  const handleNewChat = () => {
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
  };

  const handleSelectChat = (id: string) => {
    setCurrentSessionId(id);
    setCurrentNav("chat");
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
    };

    updateSessionMessages(currentSessionId, [...newMessages, assistantMessage]);

    try {
      if (backendInfo && backendInfo.status !== "offline") {
        const payload = {
          prompt: promptText,
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
      setSessions((prev) =>
        prev.map((s) =>
          s.id === currentSessionId
            ? {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantMsgId
                    ? {
                        ...m,
                        content: m.content + "\n\n[Error: Failed to connect or generation aborted]",
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
            <ChatWindow 
              messages={messages} 
              isGenerating={isGenerating} 
              persona={currentSession?.persona || null}
              onSelectPersona={handleSelectPersona}
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
            />
          </div>
        ) : currentNav === "analytics" ? (
          <AnalyticsView backendUrl={BACKEND_URL} currentSession={currentSession} />
        ) : currentNav === "architecture" ? (
          <ArchitectureView />
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
      />
    </div>
  );
}
