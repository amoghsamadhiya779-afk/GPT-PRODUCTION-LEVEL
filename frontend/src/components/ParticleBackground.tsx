"use client";

import React, { useEffect, useRef } from "react";
import { useTheme } from "./ThemeProvider";

interface Particle {
  x: number;
  y: number;
  size: number;
  vx: number;
  vy: number;
  alpha: number;
  glow: boolean;
}

export default function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    // Mouse coordinates for subtle parallax
    let mouseX = width / 2;
    let mouseY = height / 2;
    let targetMouseX = width / 2;
    let targetMouseY = height / 2;

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    const handleMouseMove = (e: MouseEvent) => {
      targetMouseX = e.clientX;
      targetMouseY = e.clientY;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("mousemove", handleMouseMove);

    // Generate stars and particles
    const particles: Particle[] = [];
    const particleCount = Math.floor((width * height) / 14000); // Scale based on screen size

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * 1.2 + 0.4,
        vx: (Math.random() - 0.5) * 0.08,
        vy: (Math.random() - 0.5) * 0.08 - 0.04, // Drifts slowly upwards
        alpha: Math.random() * 0.4 + 0.1,
        glow: Math.random() > 0.85,
      });
    }

    // Static background stars (very small, no drift)
    const staticStars: { x: number; y: number; size: number; alpha: number }[] = [];
    const staticStarCount = Math.floor((width * height) / 9000);
    for (let i = 0; i < staticStarCount; i++) {
      staticStars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * 0.6 + 0.1,
        alpha: Math.random() * 0.25 + 0.05,
      });
    }

    // Loop
    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Interpolate mouse coordinates for smooth parallax
      mouseX += (targetMouseX - mouseX) * 0.05;
      mouseY += (targetMouseY - mouseY) * 0.05;

      const parallaxX = (mouseX - width / 2) * 0.012;
      const parallaxY = (mouseY - height / 2) * 0.012;

      const isDark = theme === "dark";

      // 1. Draw radial gradient glow (neutral monochromatic glow)
      const radialGlow = ctx.createRadialGradient(
        width / 2 + parallaxX * 2,
        height * 0.3 + parallaxY * 2,
        0,
        width / 2 + parallaxX * 2,
        height * 0.3 + parallaxY * 2,
        width * 0.7
      );

      if (isDark) {
        radialGlow.addColorStop(0, "rgba(39, 39, 42, 0.12)"); // Zinc-800 very soft glow
        radialGlow.addColorStop(0.5, "rgba(24, 24, 27, 0.03)"); // Zinc-900 very soft glow
        radialGlow.addColorStop(1, "rgba(9, 9, 11, 0)");
      } else {
        radialGlow.addColorStop(0, "rgba(244, 244, 245, 0.3)"); // Zinc-100 very soft glow
        radialGlow.addColorStop(1, "rgba(250, 250, 250, 0)");
      }
      ctx.fillStyle = radialGlow;
      ctx.fillRect(0, 0, width, height);

      // 2. Draw static stars
      ctx.fillStyle = isDark ? "#FAFAFA" : "#09090B";
      staticStars.forEach((star) => {
        // Shift with parallax
        let sx = star.x + parallaxX * 0.25;
        let sy = star.y + parallaxY * 0.25;

        // Wrap around screen boundaries
        if (sx < 0) sx = width + (sx % width);
        if (sx > width) sx = sx % width;
        if (sy < 0) sy = height + (sy % height);
        if (sy > height) sy = sy % height;

        ctx.globalAlpha = star.alpha;
        ctx.beginPath();
        ctx.arc(sx, sy, star.size, 0, Math.PI * 2);
        ctx.fill();
      });

      // 3. Draw and update drifting particles
      particles.forEach((p) => {
        // Update positions
        p.x += p.vx + parallaxX * 0.08;
        p.y += p.vy + parallaxY * 0.08;

        // Warp boundaries
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        // Draw particle
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = isDark ? "#FAFAFA" : "#09090B"; // Monochromatic light or dark colors

        if (p.glow) {
          ctx.shadowBlur = 8;
          ctx.shadowColor = isDark ? "rgba(250, 250, 250, 0.5)" : "rgba(9, 9, 11, 0.3)";
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        // Reset shadow
        ctx.shadowBlur = 0;
      });

      ctx.globalAlpha = 1.0;
      animationId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animationId);
    };
  }, [theme]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-none -z-10 transition-colors duration-500"
    />
  );
}
