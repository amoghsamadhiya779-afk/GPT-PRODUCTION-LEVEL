import React, { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { Activity, Play, Plus, Trash2, FileJson, CheckCircle } from "lucide-react";

interface Example {
  instruction: string;
  response: string;
}

export default function TeachView() {
  const [examples, setExamples] = useState<Example[]>([{ instruction: "", response: "" }]);
  const [adapterName, setAdapterName] = useState("");
  const [steps, setSteps] = useState(20);
  const [lr, setLr] = useState(0.001);
  const [isTraining, setIsTraining] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const [losses, setLosses] = useState<number[]>([]);
  const intervalRef = useRef<any>(null);

  // Poll status if jobId is present
  useEffect(() => {
    if (!jobId || !isTraining) return;

    intervalRef.current = setInterval(async () => {
      try {
        const state = await api.getFinetuneStatus(jobId);
        setStatus(state);
        if (state.current_loss !== null) {
          setLosses((prev) => [...prev, state.current_loss]);
        }

        if (state.status === "done" || state.status === "failed") {
          setIsTraining(false);
          clearInterval(intervalRef.current);
          if (state.status === "done") {
            alert("Fine-tuning completed successfully! You can activate it in Settings.");
          } else {
            setError("Fine-tuning failed.");
          }
        }
      } catch (err) {
        console.error(err);
      }
    }, 2000);

    return () => clearInterval(intervalRef.current);
  }, [jobId, isTraining]);

  const loadFeedback = async () => {
    try {
      const data = await api.getFeedback();
      const corrections = data.feedback.filter((f: any) => f.correction);
      if (corrections.length > 0) {
        const newEx = corrections.map((f: any) => ({
          instruction: f.prompt,
          response: f.correction,
        }));
        setExamples(newEx);
      } else {
        alert("No feedback with corrections found.");
      }
    } catch (err: any) {
      alert("Failed to load feedback.");
    }
  };

  const loadStarterDataset = async () => {
    try {
      const data = await api.getStarterDataset();
      if (data.dataset && data.dataset.length > 0) {
        setExamples(data.dataset);
      } else {
        alert("Starter dataset is empty or not found.");
      }
    } catch (err: any) {
      alert("Failed to load starter dataset.");
    }
  };

  const startTraining = async () => {
    setError(null);
    if (!adapterName.match(/^[a-zA-Z0-9_-]+$/)) {
      setError("Adapter name must contain only letters, numbers, hyphens, and underscores.");
      return;
    }
    const validExamples = examples.filter((e) => e.instruction.trim() && e.response.trim());
    if (validExamples.length === 0) {
      setError("Please provide at least one valid example.");
      return;
    }

    try {
      const res = await api.startFinetune({
        examples: validExamples,
        adapter_name: adapterName,
        steps: steps,
        lr: lr,
      });
      setJobId(res.job_id);
      setIsTraining(true);
      setLosses([]);
      setStatus({ status: "queued", step: 0, total_steps: steps, current_loss: null, eta_seconds: null });
    } catch (err: any) {
      setError(err.message || "Failed to start training.");
    }
  };

  const addExample = () => setExamples([...examples, { instruction: "", response: "" }]);
  const removeExample = (idx: number) => setExamples(examples.filter((_, i) => i !== idx));

  // Render a simple SVG sparkline for loss
  const renderSparkline = () => {
    if (losses.length < 2) return null;
    const max = Math.max(...losses);
    const min = Math.min(...losses);
    const range = max - min || 1;
    const pts = losses.map((l, i) => {
      const x = (i / (losses.length - 1)) * 100;
      const y = 100 - ((l - min) / range) * 100;
      return `${x},${y}`;
    }).join(" ");

    return (
      <svg viewBox="0 -10 100 120" className="w-full h-16 stroke-accent fill-transparent overflow-visible mt-2">
        <polyline points={pts} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background overflow-y-auto p-6 lg:p-12 relative z-0">
      <div className="max-w-4xl mx-auto w-full space-y-8">
        <div>
          <h1 className="text-3xl font-light text-foreground tracking-tight">Teach Mode</h1>
          <p className="text-secondary mt-2">
            Fine-tune the model directly using Low-Rank Adaptation (LoRA). Provide examples or pull from user feedback.
          </p>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-6">
            <div className="bg-surface border border-border rounded-xl p-6 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-foreground">Training Examples</h3>
                <div className="flex gap-4">
                  <button
                    onClick={loadStarterDataset}
                    className="flex items-center gap-2 text-xs font-medium text-accent hover:text-accent/80 transition-colors"
                  >
                    <FileJson className="w-4 h-4" />
                    Load Starter Dataset
                  </button>
                  <button
                    onClick={loadFeedback}
                    className="flex items-center gap-2 text-xs font-medium text-accent hover:text-accent/80 transition-colors"
                  >
                    <FileJson className="w-4 h-4" />
                    Load from Feedback
                  </button>
                </div>
              </div>

              <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                {examples.map((ex, idx) => (
                  <div key={idx} className="relative group bg-elevated/30 border border-border rounded-lg p-4">
                    <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => removeExample(idx)} className="text-muted hover:text-destructive">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <label className="text-[10px] uppercase tracking-wider font-semibold text-secondary mb-1 block">
                          Instruction (Prompt)
                        </label>
                        <textarea
                          value={ex.instruction}
                          onChange={(e) => {
                            const newEx = [...examples];
                            newEx[idx].instruction = e.target.value;
                            setExamples(newEx);
                          }}
                          placeholder="E.g. What is the capital of France?"
                          className="w-full bg-transparent border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent resize-none"
                          rows={2}
                          disabled={isTraining}
                        />
                      </div>
                      <div>
                        <label className="text-[10px] uppercase tracking-wider font-semibold text-secondary mb-1 block">
                          Response (Target)
                        </label>
                        <textarea
                          value={ex.response}
                          onChange={(e) => {
                            const newEx = [...examples];
                            newEx[idx].response = e.target.value;
                            setExamples(newEx);
                          }}
                          placeholder="E.g. The capital of France is Paris."
                          className="w-full bg-transparent border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent resize-none"
                          rows={2}
                          disabled={isTraining}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <button
                onClick={addExample}
                disabled={isTraining}
                className="mt-4 flex items-center gap-2 text-sm text-secondary hover:text-foreground transition-colors py-2 px-3 border border-dashed border-border rounded-lg w-full justify-center"
              >
                <Plus className="w-4 h-4" />
                Add Example
              </button>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-surface border border-border rounded-xl p-6 shadow-sm">
              <h3 className="text-sm font-medium text-foreground mb-4">Configuration</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="text-xs font-medium text-secondary mb-1.5 block">Adapter Name</label>
                  <input
                    type="text"
                    value={adapterName}
                    onChange={(e) => setAdapterName(e.target.value)}
                    placeholder="my-lora-adapter"
                    className="w-full bg-elevated/50 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent"
                    disabled={isTraining}
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-secondary mb-1.5 block">Steps (1-200)</label>
                  <input
                    type="number"
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    min={1}
                    max={200}
                    className="w-full bg-elevated/50 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent"
                    disabled={isTraining}
                  />
                </div>
                
                <div>
                  <label className="text-xs font-medium text-secondary mb-1.5 block">Learning Rate</label>
                  <input
                    type="number"
                    value={lr}
                    onChange={(e) => setLr(Number(e.target.value))}
                    step={0.001}
                    className="w-full bg-elevated/50 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent"
                    disabled={isTraining}
                  />
                </div>
              </div>

              <div className="mt-6">
                <button
                  onClick={startTraining}
                  disabled={isTraining || !adapterName}
                  aria-label="Start fine-tuning"
                  className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isTraining 
                      ? "bg-accent/20 text-accent/50 cursor-not-allowed" 
                      : !adapterName
                      ? "bg-muted text-muted-foreground cursor-not-allowed"
                      : "bg-accent text-accent-foreground hover:bg-accent/90"
                  }`}
                >
                  {isTraining ? (
                    <Activity className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  {isTraining ? "Training..." : "Start fine-tuning"}
                </button>
              </div>
            </div>

            {isTraining && status && (
              <div className="bg-surface border border-border rounded-xl p-6 shadow-sm overflow-hidden relative">
                <h3 className="text-sm font-medium text-foreground mb-4">Training Progress</h3>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-secondary">Progress</span>
                      <span className="text-accent font-medium">
                        {status.step} / {status.total_steps}
                      </span>
                    </div>
                    <div className="w-full bg-elevated h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-accent h-full transition-all duration-500 ease-out"
                        style={{ width: `${(status.step / status.total_steps) * 100}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div>
                      <span className="text-secondary block mb-1">Current Loss</span>
                      <span className="text-foreground font-mono">
                        {status.current_loss ? status.current_loss.toFixed(4) : "--"}
                      </span>
                    </div>
                    <div>
                      <span className="text-secondary block mb-1">ETA</span>
                      <span className="text-foreground font-mono">
                        {status.eta_seconds ? `${Math.ceil(status.eta_seconds)}s` : "--"}
                      </span>
                    </div>
                  </div>

                  <div>
                     {renderSparkline()}
                  </div>
                </div>
              </div>
            )}
            
            {status?.status === "done" && (
              <div className="bg-success/10 border border-success text-success rounded-xl p-4 flex items-center gap-3">
                <CheckCircle className="w-5 h-5 flex-shrink-0" />
                <div className="text-sm leading-tight">
                  <p className="font-medium">Training Complete</p>
                  <p className="opacity-90 text-xs mt-0.5">Adapter '{adapterName}' is ready to use in Settings.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
