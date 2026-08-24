import type { StoredAnalysis } from "@/types/prediction";

const key = "bovasante:analyses";
export function getAnalyses(): StoredAnalysis[] {
  try { return JSON.parse(localStorage.getItem(key) ?? "[]") as StoredAnalysis[]; } catch { return []; }
}
export function saveAnalysis(analysis: StoredAnalysis) {
  localStorage.setItem(key, JSON.stringify([analysis, ...getAnalyses()]));
}
