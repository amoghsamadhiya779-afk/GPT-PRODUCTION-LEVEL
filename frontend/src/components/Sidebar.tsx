"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  MessageSquare,
  Plus,
  Settings,
  ChevronLeft,
  ChevronRight,
  Cpu,
  GraduationCap,
  LineChart,
  History,
  Info,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import Logo from "@/components/Logo";

interface SidebarProps {
  onOpenSettings: () => void;
  onSelectNav: (view: "chat" | "analytics" | "architecture" | "teach") => void;
  currentNav: "chat" | "analytics" | "architecture" | "teach";
  onNewChat: () => void;
  history: Array<{ id: string; title: string }>;
  currentChatId?: string;
  onSelectChat: (id: string) => void;
  onDeleteChat?: (id: string) => void;
}

export default function Sidebar({
  onOpenSettings,
  onSelectNav,
  currentNav,
  onNewChat,
  history,
  currentChatId,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Transition settings matching the prompt specifications
  const sidebarTransition = {
    type: "spring" as const,
    stiffness: 260,
    damping: 28,
  };

  const navItems = [
    { id: "chat", name: "Playground Chat", icon: MessageSquare },
    { id: "analytics", name: "Training Telemetry", icon: LineChart },
    { id: "architecture", name: "Model Architecture", icon: Cpu },
    { id: "teach", name: "Teach Mode", icon: GraduationCap },
  ] as const;

  return (
    <motion.aside
      animate={{ width: isCollapsed ? 72 : 280 }}
      transition={sidebarTransition}
      className="hidden md:flex flex-col h-full bg-surface border-r border-border backdrop-blur-md z-30 select-none relative overflow-hidden flex-shrink-0"
    >
      {/* Sidebar Header */}
      <div className="flex items-center justify-between p-4 h-16 border-b border-border">
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2.5"
          >
            <Logo size={20} className="text-accent shrink-0" />
            <span className="font-semibold text-sm tracking-tight text-primary">
              GPT Studio<span className="text-accent">.</span>
            </span>
          </motion.div>
        )}
        {isCollapsed && (
          <div className="flex items-center justify-center mx-auto">
            <Logo size={20} className="text-accent" />
          </div>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-pressed={isCollapsed}
          className={`absolute top-5 right-[-12px] w-6 h-6 bg-surface border border-border rounded-full flex items-center justify-center text-muted hover:text-primary hover:border-accent shadow-sm transition-colors z-40 ${
            isCollapsed ? "right-2" : ""
          }`}
        >
          {isCollapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Action: New Chat */}
      <div className="p-3">
        <Button
          onClick={onNewChat}
          variant="outline"
          className={`w-full justify-start gap-2 border-border bg-elevated/20 text-secondary hover:text-primary transition-all hover:scale-102 active:scale-98 h-10 ${
            isCollapsed ? "p-0 items-center justify-center" : "px-4"
          }`}
        >
          <Plus className="w-4 h-4 text-accent flex-shrink-0" />
          {!isCollapsed && <span className="text-sm font-medium">New Chat</span>}
        </Button>
      </div>

      {/* Main Navigation */}
      <div className="px-2 py-3 space-y-1 border-b border-border/60">
        {!isCollapsed && (
          <div className="px-3 mb-2 text-[10px] font-semibold text-muted uppercase tracking-wider">
            Navigation
          </div>
        )}
        {navItems.map((item) => {
          const isActive = currentNav === item.id;
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onSelectNav(item.id)}
              className={`w-full flex items-center gap-3 p-2.5 rounded-lg text-sm font-medium transition-all hover:scale-[1.01] active:scale-[0.99] ${
                isActive
                  ? "bg-primary/5 text-primary border border-primary/10 shadow-sm"
                  : "text-secondary hover:bg-elevated/30 hover:text-primary border border-transparent"
              } ${isCollapsed ? "justify-center" : "px-3"}`}
            >
              <Icon className={`w-4.5 h-4.5 flex-shrink-0 ${isActive ? "text-accent" : "text-muted"}`} />
              {!isCollapsed && <span>{item.name}</span>}
            </button>
          );
        })}
      </div>

      {/* History / Recent Conversations */}
      <div className="flex-1 overflow-y-auto px-2 py-4 space-y-1">
        {history.length > 0 && !isCollapsed && (
          <div className="px-3 mb-2 flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted uppercase tracking-wider">
              Recent Chats
            </span>
            <History className="w-3.5 h-3.5 text-muted" />
          </div>
        )}

        {isCollapsed ? (
          <div className="flex flex-col items-center gap-4 text-muted pt-2">
            {history.map((chat) => (
              <button
                key={chat.id}
                onClick={() => {
                  onSelectNav("chat");
                  onSelectChat(chat.id);
                }}
                className={`w-8 h-8 rounded-lg flex items-center justify-center text-secondary hover:text-primary transition-all hover:scale-105 active:scale-95 border ${
                  currentChatId === chat.id
                    ? "bg-primary/5 border-primary/20 text-accent font-semibold"
                    : "border-transparent bg-elevated/10"
                }`}
                title={chat.title}
              >
                <MessageSquare className="w-4.5 h-4.5" />
              </button>
            ))}
          </div>
        ) : (
          <div className="space-y-0.5">
            {history.map((chat) => {
              const isSelected = currentChatId === chat.id && currentNav === "chat";
              return (
                <div
                  key={chat.id}
                  className={`group flex items-center rounded-lg text-sm transition-all ${
                    isSelected
                      ? "bg-elevated/40 text-primary border border-border/80 font-medium"
                      : "text-secondary hover:bg-elevated/20 hover:text-primary border border-transparent"
                  }`}
                >
                  <button
                    onClick={() => {
                      onSelectNav("chat");
                      onSelectChat(chat.id);
                    }}
                    className="flex-1 min-w-0 text-left flex items-center gap-2.5 px-3 py-2"
                  >
                    <MessageSquare className={`w-4 h-4 flex-shrink-0 ${isSelected ? "text-accent" : "text-muted"}`} />
                    <span className="truncate flex-1 pr-1">{chat.title}</span>
                  </button>
                  {onDeleteChat && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChat(chat.id);
                      }}
                      aria-label={`Delete chat: ${chat.title}`}
                      title="Delete chat"
                      className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 p-1.5 mr-1.5 rounded-md text-muted hover:text-red-500 hover:bg-red-500/10 transition-all flex-shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              );
            })}
            {history.length === 0 && (
              <div className="px-3 py-4 text-xs text-muted text-center border border-dashed border-border rounded-xl">
                No recent chats
              </div>
            )}
          </div>
        )}
      </div>

      {/* Sidebar Footer */}
      <div className="p-3 border-t border-border mt-auto bg-surface/40">
        <button
          onClick={onOpenSettings}
          className={`w-full flex items-center gap-3 p-2.5 rounded-lg text-sm font-medium text-secondary hover:bg-elevated/40 hover:text-primary transition-all hover:scale-102 active:scale-98 ${
            isCollapsed ? "justify-center" : "px-3"
          }`}
        >
          <Settings className="w-4.5 h-4.5 text-muted animate-spin-hover" />
          {!isCollapsed && <span>Settings</span>}
        </button>
        {!isCollapsed && (
          <div className="mt-3 flex items-center gap-2 px-3 text-[10px] text-muted">
            <Info className="w-3 h-3" />
            <span>GPT-2 Scratch Implementation</span>
          </div>
        )}
      </div>
    </motion.aside>
  );
}
