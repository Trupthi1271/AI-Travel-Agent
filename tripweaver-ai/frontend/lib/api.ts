const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface TraceInfo {
  path: string;
  latency_ms: number;
  query_type?: string;
  destination?: string;
  tools_called?: string[];
  iterations?: number;
  error?: string | null;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  trace: TraceInfo;
}

export interface Destination {
  destination: string;
  count: number;
}

export async function sendMessage(
  message: string,
  sessionId: string,
  history: ChatMessage[]
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, history }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

export async function getPopularDestinations(): Promise<Destination[]> {
  const res = await fetch(`${API_URL}/popular-destinations`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.destinations || [];
}

export async function getSessionHistory(sessionId: string) {
  const res = await fetch(`${API_URL}/session/${sessionId}/history`);
  if (!res.ok) return { searches: [], itineraries: [] };
  return res.json();
}
