"use client";
import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Cloud, Hotel, Plane, Wallet, Map as MapIcon, Utensils, Globe } from "lucide-react";
import ResponseCard from "./ResponseCard";
import TraceExpander from "./TraceExpander";
import type { TraceInfo } from "@/lib/api";

export interface Message {
  role: "user" | "assistant";
  content: string;
  trace?: TraceInfo;
}

interface Props {
  messages: Message[];
  loading: boolean;
}

const FEATURES = [
  { icon: Cloud,    title: "Live Weather",  desc: "Real-time forecasts", color: "text-blue-400" },
  { icon: Hotel,    title: "Hotels",        desc: "Amadeus data",       color: "text-green-400" },
  { icon: Plane,    title: "Flights",       desc: "Live fares",         color: "text-purple-400" },
  { icon: Wallet,   title: "Budget",        desc: "Smart planning",     color: "text-orange-400" },
  { icon: MapIcon,  title: "Itinerary",    desc: "Multi-day plans",    color: "text-[#FF6B35]" },
  { icon: Utensils, title: "Restaurants",  desc: "Local food spots",   color: "text-red-400" },
];

export default function MessageList({ messages, loading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8 space-y-6">
      {/* Welcome screen */}
      <AnimatePresence>
        {messages.length === 0 && !loading && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto text-center"
          >
            <div className="w-16 h-16 rounded-2xl bg-[#FF6B35]/10 flex items-center justify-center text-[#FF6B35] mb-6 shadow-xl shadow-orange-500/10">
              <Globe size={32} />
            </div>
            <h2 className="text-2xl font-bold text-white mb-3 tracking-tight">How can I help you travel?</h2>
            <p className="text-sm text-[#888] mb-10 max-w-md leading-relaxed">
              Your intelligent travel concierge for India. Ask about destinations, flights, hotels, or plan a complete itinerary.
            </p>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
              {FEATURES.map((f) => (
                <div
                  key={f.title}
                  className="group p-4 rounded-xl bg-[#1E2130] border border-[#2D3250] hover:border-[#FF6B35]/30 hover:bg-[#262840] transition-all text-left"
                >
                  <f.icon size={18} className={`${f.color} mb-2.5 group-hover:scale-110 transition-transform`} />
                  <div className="text-[13px] font-bold text-white mb-0.5">{f.title}</div>
                  <div className="text-[11px] text-[#888]">{f.desc}</div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Message thread */}
      <div className="max-w-3xl mx-auto space-y-8 pb-4">
        {messages.map((msg, i) => (
          <motion.div 
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.role === "user" ? (
              <div className="max-w-[85%] sm:max-w-[75%] px-4 py-3 rounded-2xl rounded-tr-sm bg-[#FF6B35] text-white text-sm shadow-lg shadow-orange-600/10 leading-relaxed font-medium">
                {msg.content}
              </div>
            ) : (
              <div className="flex gap-4 w-full">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[#2D3250] border border-[#3D4466] flex items-center justify-center text-xs shadow-sm self-start">
                  🌍
                </div>
                <div className="flex-1 bg-[#1E2130] border border-[#2D3250] rounded-2xl rounded-tl-sm p-4 shadow-sm min-w-0">
                  <ResponseCard content={msg.content} />
                  {msg.trace && (
                    <div className="mt-4 pt-4 border-t border-[#2D3250]">
                      <TraceExpander trace={msg.trace} />
                    </div>
                  )}
                </div>
              </div>
            )}
          </motion.div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-4"
          >
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[#2D3250] border border-[#3D4466] flex items-center justify-center text-xs self-start">
              🌍
            </div>
            <div className="bg-[#1E2130] border border-[#2D3250] rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-3">
              <div className="flex gap-1">
                {[0, 1, 2].map(j => (
                  <motion.span
                    key={j}
                    animate={{ y: [0, -4, 0] }}
                    transition={{ 
                      repeat: Infinity, 
                      duration: 0.6, 
                      delay: j * 0.15,
                      ease: "easeInOut"
                    }}
                    className="w-1.5 h-1.5 rounded-full bg-[#FF6B35]"
                  />
                ))}
              </div>
              <span className="text-xs font-bold text-[#888] uppercase tracking-widest">Thinking</span>
            </div>
          </motion.div>
        )}
      </div>

      <div ref={bottomRef} className="h-4" />
    </div>
  );
}
