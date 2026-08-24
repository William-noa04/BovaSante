import { readJson } from "./http";
import type { ChatHistoryItem, ChatResponse } from "@/types/chatbot";

const baseUrl = (import.meta.env.VITE_CHATBOT_API_URL ?? "http://127.0.0.1:8001").replace(/\/$/, "");

export const chatbotApi = {
  async send(message: string, history: ChatHistoryItem[]): Promise<ChatResponse> {
    return readJson<ChatResponse>(await fetch(`${baseUrl}/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, history }),
    }));
  },
  health: () => fetch(`${baseUrl}/health`).then(readJson<{ status: string; model_loaded: boolean }>),
};
