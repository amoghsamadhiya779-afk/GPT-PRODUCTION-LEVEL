"use client";

import React from "react";

interface LogoProps {
  className?: string;
  size?: number;
}

export default function Logo({ className = "", size = 24 }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`text-primary hover:scale-105 transition-transform duration-300 ${className}`}
    >
      {/* Hexagonal Attention Node outline */}
      <path
        d="M16 3L28 10V22L16 29L4 22V10L16 3Z"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Central Node representing query token */}
      <circle cx="16" cy="16" r="3.5" fill="currentColor" />
      {/* Key-Value Attention vectors converging to center */}
      <line x1="16" y1="3" x2="16" y2="12.5" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2 2" />
      <line x1="16" y1="19.5" x2="16" y2="29" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2 2" />
      <line x1="4" y1="10" x2="13" y2="14.2" stroke="currentColor" strokeWidth="1.5" />
      <line x1="28" y1="22" x2="19" y2="17.8" stroke="currentColor" strokeWidth="1.5" />
      <line x1="28" y1="10" x2="19" y2="14.2" stroke="currentColor" strokeWidth="1.5" />
      <line x1="4" y1="22" x2="13" y2="17.8" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}
