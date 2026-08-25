import type { ChatHistoryItem } from "@/types/chatbot";

export interface StoredConversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  history: ChatHistoryItem[];
}

const key = "bovasante:conversations";

export function getConversations(): StoredConversation[] {
  try {
    const list = JSON.parse(localStorage.getItem(key) ?? "[]") as StoredConversation[];
    return [...list].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  } catch {
    return [];
  }
}

export function saveConversation(conversation: StoredConversation) {
  const rest = getConversations().filter((c) => c.id !== conversation.id);
  localStorage.setItem(key, JSON.stringify([conversation, ...rest]));
}

export function deleteConversation(id: string) {
  localStorage.setItem(key, JSON.stringify(getConversations().filter((c) => c.id !== id)));
}
