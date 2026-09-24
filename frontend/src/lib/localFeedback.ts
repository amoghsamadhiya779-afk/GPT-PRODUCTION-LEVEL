// Corrections the user submitted from this browser, used to seed Teach Mode.
//
// Feedback sent to the server is pooled across all users, so reading it back
// is admin-only. Teach Mode instead trains on the user's *own* corrections,
// which we keep locally when they are submitted.

const STORAGE_KEY = "gptStudioCorrections";
const MAX_STORED = 200;

export interface Correction {
  instruction: string;
  response: string;
}

function isCorrection(value: unknown): value is Correction {
  const v = value as Correction;
  return typeof v?.instruction === "string" && typeof v?.response === "string";
}

export function loadCorrections(): Correction[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.filter(isCorrection) : [];
  } catch {
    return [];
  }
}

export function saveCorrection(correction: Correction): void {
  if (!correction.instruction.trim() || !correction.response.trim()) return;
  try {
    const next = [...loadCorrections(), correction].slice(-MAX_STORED);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Storage full or unavailable (private mode): the server copy still exists.
  }
}
