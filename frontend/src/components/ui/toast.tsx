"use client";

import React, { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Info, X } from "lucide-react";

interface ToastItem {
  id: number;
  message: string;
  variant: "success" | "error" | "info";
}

// Minimal module-level pub/sub so `toast()` can be called from anywhere
// (event handlers, async callbacks) without needing a Context provider
// wrapping the tree. Mount a single <Toaster /> near the root.
let toastId = 0;
let listeners: Array<(toasts: ToastItem[]) => void> = [];
let toasts: ToastItem[] = [];

function emit() {
  listeners.forEach((l) => l(toasts));
}

function dismissToast(id: number) {
  toasts = toasts.filter((t) => t.id !== id);
  emit();
}

export function toast(message: string, variant: ToastItem["variant"] = "info") {
  const id = ++toastId;
  toasts = [...toasts, { id, message, variant }];
  emit();
  setTimeout(() => dismissToast(id), 4000);
}

export function Toaster() {
  const [items, setItems] = useState<ToastItem[]>(toasts);

  useEffect(() => {
    listeners.push(setItems);
    return () => {
      listeners = listeners.filter((l) => l !== setItems);
    };
  }, []);

  const iconFor = (variant: ToastItem["variant"]) => {
    if (variant === "success") return <CheckCircle2 className="w-4 h-4 text-success flex-shrink-0" />;
    if (variant === "error") return <XCircle className="w-4 h-4 text-destructive flex-shrink-0" />;
    return <Info className="w-4 h-4 text-accent flex-shrink-0" />;
  };

  if (items.length === 0) return null;

  return (
    <div
      className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm w-[calc(100%-2rem)] sm:w-auto"
      aria-live="polite"
      role="status"
    >
      {items.map((t) => (
        <div
          key={t.id}
          className="flex items-start gap-2 px-4 py-3 rounded-xl border border-border bg-surface/95 backdrop-blur-md shadow-lg text-sm text-primary"
        >
          {iconFor(t.variant)}
          <span className="flex-1">{t.message}</span>
          <button
            onClick={() => dismissToast(t.id)}
            aria-label="Dismiss notification"
            className="text-muted hover:text-primary flex-shrink-0"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}
