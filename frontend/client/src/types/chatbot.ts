export interface ChatPart {
  text?: string;
  function_call?: { name: string; args: Record<string, unknown> };
  function_response?: { name: string; response: Record<string, unknown> };
}
export interface ChatHistoryItem { role: string; parts: ChatPart[] }
export interface ChatPrediction {
  predicted_class: string; predicted_class_fr: string; confidence: number;
  probabilities: Record<string, number>; reliability_note: string;
}
export interface ChatResponse { reply: string; history: ChatHistoryItem[]; prediction: ChatPrediction | null }
