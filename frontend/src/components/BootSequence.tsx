"use client";

import React, { useEffect, useState, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";

const TARGET_NODES = [
  { cx: 16, cy: 3 },
  { cx: 28, cy: 10 },
  { cx: 28, cy: 22 },
  { cx: 16, cy: 29 },
  { cx: 4, cy: 22 },
  { cx: 4, cy: 10 },
  { cx: 16, cy: 16 }, // Center
];

export default function BootSequence() {
  const [shouldRender, setShouldRender] = useState(false);
  const [phase, setPhase] = useState(0);
  const [telemetry, setTelemetry] = useState<string[]>([]);
  const skipped = useRef(false);

  // Generate random scatter positions once
  const randomScatter = useMemo(() => {
    return TARGET_NODES.map(() => ({
      // eslint-disable-next-line react-hooks/purity
      x: Math.random() * 200 - 100 + 16,
      // eslint-disable-next-line react-hooks/purity
      y: Math.random() * 200 - 100 + 16,
    }));
  }, []);

  // Hydration & initial setup
  useEffect(() => {
    const booted = sessionStorage.getItem("hasBooted");
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    
    if (booted || prefersReducedMotion) {
      sessionStorage.setItem("hasBooted", "true");
      return;
    }
    
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setShouldRender(true);
  }, []);

  const completeBoot = () => {
    if (skipped.current) return;
    skipped.current = true;
    sessionStorage.setItem("hasBooted", "true");
    setShouldRender(false);
  };

  useEffect(() => {
    if (!shouldRender) return;

    // Attach skip listeners
    const handleSkip = () => completeBoot();
    window.addEventListener("keydown", handleSkip);
    window.addEventListener("click", handleSkip);
    window.addEventListener("touchstart", handleSkip);

    let isMounted = true;

    // Fetch telemetry with 1.2s timeout
    const fetchHealth = async () => {
      const msgs = ["> initializing model..."];
      setTelemetry([...msgs]);

      const healthPromise = api.health();
      const timeoutPromise = new Promise((resolve) => setTimeout(() => resolve("TIMEOUT"), 1200));

      const result = await Promise.race([healthPromise, timeoutPromise]) as any;
      
      if (!isMounted) return;

      if (result && result !== "TIMEOUT" && result.status !== "offline") {
        msgs.push(`> checkpoint: ${result.checkpoint}`);
        msgs.push(`> parameters: ${result.parameters}`);
        msgs.push(`> device: ${result.device}`);
        msgs.push("> system online.");
      } else {
        msgs.push("> connection failed.");
        msgs.push("> standalone mode.");
      }
      setTelemetry([...msgs]);
    };

    fetchHealth();

    // Sequence timing
    const t1 = setTimeout(() => { if (isMounted) setPhase(1); }, 1000); // Connect lines
    const t2 = setTimeout(() => { if (isMounted) setPhase(2); }, 1500); // Pulse
    const t3 = setTimeout(() => { completeBoot(); }, 2500); // End

    return () => {
      isMounted = false;
      window.removeEventListener("keydown", handleSkip);
      window.removeEventListener("click", handleSkip);
      window.removeEventListener("touchstart", handleSkip);
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [shouldRender]);

  return (
    <AnimatePresence>
      {shouldRender && (
        <motion.div
          key="boot-sequence"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4, ease: "easeInOut" }}
          className="fixed inset-0 z-[100] bg-background flex flex-col items-center justify-center overflow-hidden"
        >
          {/* Main Logo Container */}
          <div className="relative w-48 h-48 sm:w-64 sm:h-64 flex items-center justify-center">
            {/* Pulse Effect */}
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={phase >= 2 ? { scale: [1, 2.5], opacity: [0, 0.15, 0] } : {}}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="absolute inset-0 bg-accent rounded-full blur-2xl"
            />

            <svg
              viewBox="0 0 32 32"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="w-24 h-24 sm:w-32 sm:h-32 text-accent overflow-visible"
            >
              {/* Connecting Lines */}
              <motion.g
                initial={{ opacity: 0 }}
                animate={phase >= 1 ? { opacity: 1 } : { opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                <path
                  d="M16 3L28 10V22L16 29L4 22V10L16 3Z"
                  stroke="currentColor"
                  strokeWidth="1.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <line x1="16" y1="3" x2="16" y2="12.5" stroke="currentColor" strokeWidth="0.8" strokeDasharray="1 1" />
                <line x1="16" y1="19.5" x2="16" y2="29" stroke="currentColor" strokeWidth="0.8" strokeDasharray="1 1" />
                <line x1="4" y1="10" x2="13" y2="14.2" stroke="currentColor" strokeWidth="0.8" />
                <line x1="28" y1="22" x2="19" y2="17.8" stroke="currentColor" strokeWidth="0.8" />
                <line x1="28" y1="10" x2="19" y2="14.2" stroke="currentColor" strokeWidth="0.8" />
                <line x1="4" y1="22" x2="13" y2="17.8" stroke="currentColor" strokeWidth="0.8" />
              </motion.g>

              {/* Converging Nodes */}
              {TARGET_NODES.map((target, i) => (
                <motion.circle
                  key={i}
                  r={i === 6 ? 3.5 : 2}
                  fill="currentColor"
                  initial={{ cx: randomScatter[i].x, cy: randomScatter[i].y, opacity: 0 }}
                  animate={{ cx: target.cx, cy: target.cy, opacity: 1 }}
                  transition={{
                    duration: 1.0,
                    ease: [0.25, 0.1, 0.25, 1],
                  }}
                />
              ))}
            </svg>
          </div>

          {/* Telemetry Console */}
          <div className="absolute bottom-16 sm:bottom-24 w-full max-w-sm px-6 h-32 flex flex-col justify-end text-[11px] sm:text-xs font-mono text-accent/80 select-none pointer-events-none">
            {telemetry.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: i * 0.12 }}
                className="leading-loose whitespace-nowrap"
              >
                {msg}
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
