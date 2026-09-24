import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Only absolute http(s) URLs may be used as link targets. Source links come
// from third-party search results; anything else (javascript:, data:, ...)
// would run in our origin when clicked.
export function safeHttpUrl(url: string | undefined | null): string | undefined {
  if (!url) return undefined
  try {
    const parsed = new URL(url)
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : undefined
  } catch {
    return undefined
  }
}
