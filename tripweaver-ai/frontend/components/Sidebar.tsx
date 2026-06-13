"use client";
import { useEffect, useState } from "react";
import { X, Zap, Star, Trash2, Globe } from "lucide-react";
import { getPopularDestinations, type Destination } from "@/lib/api";

const QUICK_ACTIONS = [
  { label: "🌤 Goa Weather",       query: "What's the weather in Goa?" },
  { label: "🏨 Jaipur Hotels",     query: "Suggest hotels in Jaipur" },
  { label: "✈️ Delhi → Goa",        query: "Flights from Delhi to Goa" },
  { label: "💰 Budget ₹15k",       query: "My budget is 15000 INR for 3 days" },
  { label: "📅 Manali Plan",       query: "Plan a 3-day budget trip to Manali" },
  { label: "🍽️ Mumbai Eats",       query: "Best restaurants in Mumbai" },
];

interface Props {
  onQuickAction: (query: string) => void;
  onClear: () => void;
  messageCount: number;
  onClose?: () => void;
}

export default function Sidebar({ onQuickAction, onClear, messageCount, onClose }: Props) {
  const [popular, setPopular] = useState<Destination[]>([]);

  useEffect(() => {
    getPopularDestinations().then(setPopular).catch(() => {});
  }, []);

  return (
    <aside className="h-full flex flex-col bg-[#1E2130] border-r border-[#2D3250] overflow-y-auto">
      {/* Logo Area */}
      <div className="flex items-center justify-between p-5 border-b border-[#2D3250]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#FF6B35] flex items-center justify-center text-white">
            <Globe size={18} />
          </div>
          <div>
            <div className="font-bold text-white text-sm tracking-tight">TripWeaver AI</div>
            <div className="text-[10px] text-[#888] font-medium uppercase tracking-wider">Travel Concierge</div>
          </div>
        </div>
        {onClose && (
          <button onClick={onClose} className="p-2 -mr-2 rounded-lg hover:bg-[#2D3250] lg:hidden text-[#888]">
            <X size={18} />
          </button>
        )}
      </div>

      <div className="flex-1 p-4 space-y-6">
        {/* Quick Actions */}
        <section>
          <div className="flex items-center gap-2 mb-3 px-1 text-[11px] font-bold text-[#555] uppercase tracking-widest">
            <Zap size={12} className="text-[#FF6B35]" />
            Quick Actions
          </div>
          <div className="grid grid-cols-1 gap-1.5">
            {QUICK_ACTIONS.map((a, i) => (
              <button
                key={i}
                onClick={() => onQuickAction(a.query)}
                className="text-left text-xs px-3 py-2.5 rounded-lg bg-[#262840] text-[#ccc] hover:bg-[#2D3250] hover:text-white transition-all border border-transparent hover:border-[#3D4466]"
              >
                {a.label}
              </button>
            ))}
          </div>
        </section>

        {/* Popular destinations */}
        {popular.length > 0 && (
          <section>
            <div className="flex items-center gap-2 mb-3 px-1 text-[11px] font-bold text-[#555] uppercase tracking-widest">
              <Star size={12} className="text-yellow-500" />
              Popular Now
            </div>
            <div className="flex flex-wrap gap-2">
              {popular.slice(0, 8).map((d, i) => (
                <button
                  key={i}
                  onClick={() => onQuickAction(`Plan a 3-day trip to ${d.destination}`)}
                  className="text-[11px] px-2.5 py-1.5 rounded-md bg-[#262840] text-[#888] border border-[#2D3250] hover:border-[#FF6B35] hover:text-[#FF6B35] transition-all"
                >
                  {d.destination}
                </button>
              ))}
            </div>
          </section>
        )}
      </div>

      {/* Footer Stats + Clear */}
      <div className="p-4 mt-auto border-t border-[#2D3250] bg-[#1a1d2b]">
        <div className="mb-4 px-2">
          <div className="flex justify-between items-center text-[11px] text-[#555] font-medium mb-1.5">
            <span>Session Queries</span>
            <span>{messageCount}</span>
          </div>
          <div className="w-full h-1 bg-[#2D3250] rounded-full overflow-hidden">
            <div 
              className="h-full bg-[#FF6B35] transition-all duration-500" 
              style={{ width: `${Math.min(messageCount * 10, 100)}%` }}
            />
          </div>
        </div>
        
        <button
          onClick={onClear}
          className="w-full flex items-center justify-center gap-2 text-xs py-2.5 rounded-lg bg-[#2D3250]/50 text-[#888] hover:bg-red-500/10 hover:text-red-400 transition-all border border-transparent hover:border-red-500/20"
        >
          <Trash2 size={14} />
          Clear Conversation
        </button>
        
        <p className="text-[10px] mt-4 text-center text-[#444] font-medium tracking-tight">
          V1.2.0 · Built with Groq
        </p>
      </div>
    </aside>
  );
}
