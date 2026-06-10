"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { LineChart, BarChart2, Info, RefreshCw, Cpu, Award } from "lucide-react";

interface AnalyticsViewProps {
  backendUrl: string;
}

export default function AnalyticsView({ backendUrl }: AnalyticsViewProps) {
  const [plotUrl, setPlotUrl] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPlot = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${backendUrl}/training/plot`);
      if (res.ok) {
        // Create a blob URL to render the image
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setPlotUrl(url);
      } else {
        throw new Error("Telemetry plot not found on server.");
      }
    } catch (e) {
      console.warn("Failed to fetch training plot:", e);
      setError("Active telemetry plot unavailable. Load checkpoints to log data.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPlot();
  }, [backendUrl]);

  return (
    <div className="flex-1 overflow-y-auto px-4 md:px-8 py-8 flex flex-col items-center select-none">
      <div className="w-full max-w-4xl space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-primary flex items-center gap-2">
            <LineChart className="w-7 h-7 text-accent" />
            Training Telemetry
          </h1>
          <p className="text-sm text-secondary mt-1">
            Visual logs, parameter curves, and optimization performance tracked via MLflow.
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { label: "Learning Rate", value: "5e-4", desc: "Cosine scheduler" },
            { label: "Weight Decay", value: "0.1", desc: "AdamW optimizer" },
            { label: "Context Size", value: "256 tokens", desc: "Window length" },
            { label: "Dataset", value: "The Verdict", desc: "Edith Wharton" },
          ].map((stat, i) => (
            <div key={i} className="p-4 rounded-xl border border-border bg-surface/30 backdrop-blur-sm">
              <div className="text-[10px] font-semibold text-muted uppercase tracking-wider">{stat.label}</div>
              <div className="text-xl font-semibold text-primary mt-1">{stat.value}</div>
              <div className="text-xs text-muted mt-0.5">{stat.desc}</div>
            </div>
          ))}
        </div>

        {/* Plot and Details */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Plot Box */}
          <div className="lg:col-span-2 p-5 rounded-2xl border border-border bg-surface/30 backdrop-blur-sm flex flex-col min-h-[350px] justify-between">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-semibold text-primary uppercase tracking-wider flex items-center gap-1.5">
                <BarChart2 className="w-4 h-4 text-accent" />
                Cross-Entropy Loss Progression
              </h3>
              <button
                onClick={fetchPlot}
                className="p-1 rounded-lg border border-border hover:bg-elevated/40 text-secondary transition-all hover:scale-105 active:scale-95"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
              </button>
            </div>

            <div className="flex-1 flex items-center justify-center border border-border/40 rounded-xl bg-elevated/10 overflow-hidden relative p-2">
              {isLoading ? (
                <div className="flex flex-col items-center gap-2 text-muted">
                  <RefreshCw className="w-6 h-6 animate-spin text-accent" />
                  <span className="text-xs">Loading training logs...</span>
                </div>
              ) : error ? (
                <div className="flex flex-col items-center gap-2 text-muted text-center p-6">
                  <Info className="w-6 h-6 text-muted" />
                  <span className="text-xs max-w-xs">{error}</span>
                </div>
              ) : (
                <img
                  src={plotUrl}
                  alt="Training Loss Plot"
                  className="max-h-[300px] object-contain rounded-lg shadow-sm"
                />
              )}
            </div>
          </div>

          {/* Training Details */}
          <div className="p-5 rounded-2xl border border-border bg-surface/30 backdrop-blur-sm space-y-4">
            <h3 className="text-sm font-semibold text-primary uppercase tracking-wider flex items-center gap-1.5 border-b border-border/60 pb-3">
              <Award className="w-4 h-4 text-accent" />
              Optimization Properties
            </h3>
            <div className="space-y-3.5 text-xs">
              <p className="text-secondary leading-relaxed">
                The model is pre-trained using a sliding-window next-token prediction objective. The training configurations feature:
              </p>
              <ul className="space-y-2 text-secondary list-disc pl-4">
                <li>Linear warmup scheduler over initial steps.</li>
                <li>Gradient clipping at 1.0 threshold for stable backpropagation.</li>
                <li>Train/Val splits evaluated periodically (eval frequency of 5 steps).</li>
                <li>MLflow logging tracking loss metrics, weights, and run parameter configs.</li>
              </ul>
              <div className="p-3 rounded-lg border border-border bg-elevated/25 text-[11px] text-muted leading-normal flex items-start gap-2">
                <Info className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                <span>
                  Checkpoints are saved automatically when validation loss achieves new local minimums.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
