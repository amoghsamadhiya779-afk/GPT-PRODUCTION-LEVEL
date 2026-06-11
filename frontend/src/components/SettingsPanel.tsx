"use client";

import React from "react";
import { useTheme } from "./ThemeProvider";
import { motion, AnimatePresence } from "framer-motion";
import { X, Sun, Moon, Cpu, Zap, Activity } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";

export interface ModelSettings {
  model: string;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  maxTokens: number;
  useCache: boolean;
}

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  settings: ModelSettings;
  onChange: (settings: ModelSettings) => void;
  backendInfo?: {
    status: string;
    checkpoint: string;
    parameters: number;
    device: string;
  };
}

export default function SettingsPanel({
  isOpen,
  onClose,
  settings,
  onChange,
  backendInfo,
}: SettingsPanelProps) {
  const { theme, setTheme } = useTheme();

  const updateSetting = <K extends keyof ModelSettings>(key: K, value: ModelSettings[K]) => {
    onChange({
      ...settings,
      [key]: value,
    });
  };

  const models = [
    { id: "gpt-2-small", name: "GPT-2 Small (124M)", icon: Cpu, desc: "Pre-trained weights" },
    { id: "gpt-2-tiny", name: "GPT-2 Tiny (13M)", icon: Zap, desc: "Fast demo checkpoint" },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.4 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 z-40"
          />

          {/* Panel */}
          <motion.div
            initial={{ x: "100%", opacity: 0.5 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: "100%", opacity: 0.5 }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 bottom-0 w-full max-w-md bg-surface/80 border-l border-border backdrop-blur-xl p-6 shadow-2xl z-50 overflow-y-auto flex flex-col justify-between"
          >
            <div>
              {/* Header */}
              <div className="flex items-center justify-between pb-6 border-b border-border">
                <div>
                  <h2 className="text-lg font-semibold tracking-tight text-primary">Inference Settings</h2>
                  <p className="text-xs text-muted">Configure the runtime parameters of the custom GPT model.</p>
                </div>
                <button
                  onClick={onClose}
                  className="p-1.5 rounded-lg border border-border bg-elevated/40 hover:bg-elevated/80 text-secondary transition-all hover:scale-105 active:scale-95"
                >
                  <X className="w-4.5 h-4.5" />
                </button>
              </div>

              {/* Model Selection */}
              <div className="space-y-3 py-6 border-b border-border">
                <label className="text-xs font-semibold text-muted uppercase tracking-wider">Model</label>
                <div className="grid grid-cols-1 gap-2">
                  {models.map((model) => {
                    const isSelected = settings.model === model.id;
                    const Icon = model.icon;
                    return (
                      <button
                        key={model.id}
                        onClick={() => updateSetting("model", model.id)}
                        className={`flex items-center justify-between p-3.5 rounded-xl border text-left transition-all hover:scale-[1.01] active:scale-[0.99] ${
                          isSelected
                            ? "bg-primary/5 border-primary/20 text-primary"
                            : "bg-elevated/20 border-border text-secondary hover:bg-elevated/40"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <Icon className={`w-5 h-5 ${isSelected ? "text-accent" : "text-muted"}`} />
                          <div>
                            <div className="text-sm font-medium">{model.name}</div>
                            <div className="text-xs text-muted">{model.desc}</div>
                          </div>
                        </div>
                        {isSelected && (
                          <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Hyperparameters */}
              <div className="space-y-6 py-6 border-b border-border">
                <label className="text-xs font-semibold text-muted uppercase tracking-wider block">
                  Parameters
                </label>

                {/* Temperature */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="font-medium text-secondary">Temperature</span>
                    <span className="font-mono text-xs text-muted border border-border px-2 py-0.5 rounded bg-elevated/30">
                      {settings.temperature.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    value={[settings.temperature]}
                    onValueChange={(val) => updateSetting("temperature", Array.isArray(val) ? val[0] : val)}
                    min={0.0}
                    max={1.5}
                    step={0.05}
                    className="py-2"
                  />
                  <span className="text-[11px] text-muted block leading-normal">
                    Controls randomness: lower temperatures are deterministic, higher are creative.
                  </span>
                </div>

                {/* Max Tokens */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="font-medium text-secondary">Max Tokens</span>
                    <span className="font-mono text-xs text-muted border border-border px-2 py-0.5 rounded bg-elevated/30">
                      {settings.maxTokens}
                    </span>
                  </div>
                  <Slider
                    value={[settings.maxTokens]}
                    onValueChange={(val) => updateSetting("maxTokens", Array.isArray(val) ? val[0] : val)}
                    min={10}
                    max={300}
                    step={10}
                    className="py-2"
                  />
                  <span className="text-[11px] text-muted block leading-normal">
                    The maximum number of sequence tokens to generate in a single run.
                  </span>
                </div>

                {/* Top-K Limit */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="font-medium text-secondary">Top-K Limit</span>
                    <span className="font-mono text-xs text-muted border border-border px-2 py-0.5 rounded bg-elevated/30">
                      {settings.topK}
                    </span>
                  </div>
                  <Slider
                    value={[settings.topK]}
                    onValueChange={(val) => updateSetting("topK", Array.isArray(val) ? val[0] : val)}
                    min={1}
                    max={100}
                    step={1}
                    className="py-2"
                  />
                  <span className="text-[11px] text-muted block leading-normal">
                    Caps sampling to the K most probable tokens at each step.
                  </span>
                </div>

                {/* Top-P (Mock/Extended Parameter) */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="font-medium text-secondary">Top-P (Nucleus)</span>
                    <span className="font-mono text-xs text-muted border border-border px-2 py-0.5 rounded bg-elevated/30">
                      {settings.topP.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    value={[settings.topP]}
                    onValueChange={(val) => updateSetting("topP", Array.isArray(val) ? val[0] : val)}
                    min={0.0}
                    max={1.0}
                    step={0.05}
                    className="py-2"
                  />
                  <span className="text-[11px] text-muted block leading-normal">
                    Limits sample vocabulary pool dynamically to cumulative probability threshold.
                  </span>
                </div>

                {/* Repetition Penalty */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="font-medium text-secondary">Repetition Penalty</span>
                    <span className="font-mono text-xs text-muted border border-border px-2 py-0.5 rounded bg-elevated/30">
                      {settings.repetitionPenalty.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    value={[settings.repetitionPenalty]}
                    onValueChange={(val) => updateSetting("repetitionPenalty", Array.isArray(val) ? val[0] : val)}
                    min={1.0}
                    max={2.0}
                    step={0.05}
                    className="py-2"
                  />
                  <span className="text-[11px] text-muted block leading-normal">
                    Penalizes repeated tokens to discourage redundancy and loops.
                  </span>
                </div>

                {/* KV Cache Toggle */}
                <div className="flex items-center justify-between p-3 rounded-xl border border-border bg-elevated/10">
                  <div>
                    <div className="text-sm font-medium text-secondary">Key-Value Caching</div>
                    <div className="text-[11px] text-muted">Accelerates autoregressive inference speed to O(1) step.</div>
                  </div>
                  <Switch
                    checked={settings.useCache}
                    onCheckedChange={(val) => updateSetting("useCache", val)}
                  />
                </div>
              </div>

              {/* Theme Settings */}
              <div className="space-y-3 py-6">
                <label className="text-xs font-semibold text-muted uppercase tracking-wider block">
                  Appearance Theme
                </label>
                <div className="flex p-1 rounded-xl bg-elevated/30 border border-border">
                  <button
                    onClick={() => setTheme("dark")}
                    className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium rounded-lg transition-all hover:scale-[1.01] active:scale-[0.99] ${
                      theme === "dark"
                        ? "bg-background text-primary shadow-sm border border-border"
                        : "text-secondary hover:text-primary"
                    }`}
                  >
                    <Moon className="w-4 h-4" />
                    Dark Mode
                  </button>
                  <button
                    onClick={() => setTheme("light")}
                    className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium rounded-lg transition-all hover:scale-[1.01] active:scale-[0.99] ${
                      theme === "light"
                        ? "bg-background text-primary shadow-sm border border-border"
                        : "text-secondary hover:text-primary"
                    }`}
                  >
                    <Sun className="w-4 h-4" />
                    Light Mode
                  </button>
                </div>
              </div>
            </div>

            {/* Backend Info Footer */}
            {backendInfo && (
              <div className="mt-6 p-4 rounded-xl border border-border bg-elevated/20 text-xs space-y-2.5">
                <div className="flex items-center gap-2 font-semibold text-primary pb-2 border-b border-border/60">
                  <Activity className="w-4 h-4 text-accent animate-pulse" />
                  <span>Serving Telemetry</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Backend:</span>
                  <span className={`font-medium ${backendInfo.status === "active" ? "text-success" : "text-warning"}`}>
                    {backendInfo.status.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Hardware Device:</span>
                  <span className="font-medium text-secondary uppercase">{backendInfo.device}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Total Parameters:</span>
                  <span className="font-medium text-secondary font-mono">
                    {backendInfo.parameters.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center overflow-hidden">
                  <span className="text-muted shrink-0 mr-2">Checkpoint:</span>
                  <span className="font-medium text-secondary font-mono text-[10px] truncate max-w-[180px]" title={backendInfo.checkpoint}>
                    {backendInfo.checkpoint.split(/[/\\]/).pop()}
                  </span>
                </div>
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
