"use client";
import { useState, useRef } from "react";
import { Menu, X, PlusCircle } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import MessageList, { type Message } from "@/components/MessageList";
import ChatInput from "@/components/ChatInput";
import { sendMessage } from "@/lib/api";

function getSessionId(): string {
  if (typeof window === "undefined") return "ssr";
  let id = sessionStorage.getItem("tripweaver_session");
  if (!id) {
    id = Math.random().toString(36).slice(2, 10);
    sessionStorage.setItem("tripweaver_session", id);
  }
  return id;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Use ref to always have latest messages without stale closure
  const messagesRef = useRef<Message[]>([]);
  const loadingRef = useRef(false);

  const handleSend = async (text: string) => {
    if (!text.trim() || loadingRef.current) return;
    setError(null);
    setIsSidebarOpen(false); // Close sidebar on mobile when sending

    const userMsg: Message = { role: "user", content: text };
    const updated = [...messagesRef.current, userMsg];
    messagesRef.current = updated;
    setMessages(updated);

    loadingRef.current = true;
    setLoading(true);

    try {
      const sessionId = getSessionId();
      const history = messagesRef.current.slice(-8).map(m => ({
        role: m.role,
        content: m.role === "assistant" ? m.content.slice(0, 200) : m.content,
      }));

      const data = await sendMessage(text, sessionId, history);

      const withResponse = [
        ...messagesRef.current,
        { role: "assistant" as const, content: data.response, trace: data.trace },
      ];
      messagesRef.current = withResponse;
      setMessages(withResponse);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Something went wrong";
      setError(msg);
      const withError = [
        ...messagesRef.current,
        { role: "assistant" as const, content: `❌ ${msg}` },
      ];
      messagesRef.current = withError;
      setMessages(withError);
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  };

  const handleClear = () => {
    messagesRef.current = [];
    setMessages([]);
    setError(null);
  };

  const userMessages = messages.filter(m => m.role === "user").length;

  return (
    <div className="flex h-screen overflow-hidden bg-[#0E1117] text-[#FAFAFA]">
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 lg:hidden" 
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-72 transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0
        ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"}
      `}>
        <Sidebar
          onQuickAction={handleSend}
          onClear={handleClear}
          messageCount={userMessages}
          onClose={() => setIsSidebarOpen(false)}
        />
      </div>

      {/* Main chat area */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 bg-[#1E2130] border-b border-[#2D3250] flex-shrink-0">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 -ml-2 rounded-lg hover:bg-[#2D3250] lg:hidden"
            >
              <Menu size={20} />
            </button>
            <div>
              <h1 className="text-sm font-bold text-white leading-tight">✈️ AI Travel Concierge</h1>
              <p className="text-[10px] text-[#888] uppercase tracking-wider font-semibold">Powered by Groq · Amadeus</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <button 
              onClick={handleClear}
              className="hidden sm:flex items-center gap-2 text-xs text-[#888] hover:text-white transition-colors"
            >
              <PlusCircle size={14} />
              New Chat
            </button>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#262840] border border-[#2D3250]">
              <span className="w-2 h-2 rounded-full bg-[#27AE60] animate-pulse" />
              <span className="text-[11px] font-medium text-[#888]">Live</span>
            </div>
          </div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="mx-4 mt-4 p-3 rounded-lg bg-[#3d1a1a] text-[#ff6b6b] border border-[#7d2d2d] flex justify-between items-center text-sm">
            <div className="flex items-center gap-2">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
            <button onClick={() => setError(null)} className="opacity-60 hover:opacity-100 transition-opacity">
              <X size={16} />
            </button>
          </div>
        )}

        {/* Messages */}
        <MessageList messages={messages} loading={loading} />

        {/* Input */}
        <div className="p-4 bg-gradient-to-t from-[#0E1117] via-[#0E1117] to-transparent">
          <ChatInput onSend={handleSend} disabled={loading} />
        </div>
      </main>
    </div>
  );
}
