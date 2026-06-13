"use client";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, Check, Info, Hotel, Plane, Wallet, Map as MapIcon, MapPin, Utensils, MessageSquare } from "lucide-react";

interface Props {
  content: string;
}

type CardType = "weather" | "hotel" | "flight" | "budget" | "itinerary" | "places" | "restaurant" | "general";

function detectType(content: string): CardType {
  const c = content.toLowerCase();
  
  // Weather keywords & emojis
  if (c.includes("weather") || c.includes("forecast") || c.includes("🌡") || c.includes("🌤") || c.includes("☁️") || c.includes("🌧")) 
    return "weather";
    
  // Hotel keywords & emojis
  if (c.includes("hotel") || c.includes("accommodation") || c.includes("stay") || c.includes("resort") || c.includes("🏨")) 
    return "hotel";
    
  // Flight keywords & emojis
  if (c.includes("flight") || c.includes("airline") || c.includes("airport") || c.includes("✈️")) 
    return "flight";
    
  // Budget keywords & emojis
  if (c.includes("budget") || c.includes("cost") || c.includes("inr") || c.includes("price") || c.includes("₹") || c.includes("💰")) 
    return "budget";
    
  // Restaurant keywords & emojis
  if (c.includes("restaurant") || c.includes("eat") || c.includes("food") || c.includes("dining") || c.includes("🍽️") || c.includes("🌮") || c.includes("🍔")) 
    return "restaurant";
    
  // Itinerary keywords & emojis
  if (c.includes("day 1") || c.includes("day 2") || c.includes("itinerary") || (c.includes("plan") && c.includes("trip")) || c.includes("🗺️")) 
    return "itinerary";
    
  // Places keywords & emojis
  if (c.includes("top places") || c.includes("sightseeing") || c.includes("visit") || c.includes("📍") || c.includes("🏛️")) 
    return "places";
    
  return "general";
}

const CARD_META: Record<CardType, { icon: any; label: string; color: string; bg: string }> = {
  weather:    { icon: Info,          label: "Weather Report",   color: "#4A90D9", bg: "bg-[#4A90D9]/10" },
  hotel:      { icon: Hotel,         label: "Accommodation",    color: "#27AE60", bg: "bg-[#27AE60]/10" },
  flight:     { icon: Plane,         label: "Flight Results",   color: "#8E44AD", bg: "bg-[#8E44AD]/10" },
  budget:     { icon: Wallet,        label: "Budget Breakdown", color: "#E67E22", bg: "bg-[#E67E22]/10" },
  itinerary:  { icon: MapIcon,       label: "Trip Itinerary",  color: "#FF6B35", bg: "bg-[#FF6B35]/10" },
  places:     { icon: MapPin,        label: "Top Places",       color: "#16A085", bg: "bg-[#16A085]/10" },
  restaurant: { icon: Utensils,      label: "Restaurants",     color: "#C0392B", bg: "bg-[#C0392B]/10" },
  general:    { icon: MessageSquare, label: "",                 color: "#555",    bg: "bg-transparent"  },
};

export default function ResponseCard({ content }: Props) {
  const [copied, setCopied] = useState(false);
  const type = detectType(content);
  const meta = CARD_META[type];
  const Icon = meta.icon;

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full relative group">
      <div className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={handleCopy}
          className="p-1.5 rounded-lg bg-[#262840] border border-[#2D3250] text-[#888] hover:text-white transition-colors"
          title="Copy to clipboard"
        >
          {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
        </button>
      </div>

      {meta.label && (
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-t-lg mb-3 text-xs font-bold uppercase tracking-wider ${meta.bg}`}
          style={{ borderLeft: `3px solid ${meta.color}`, color: meta.color }}
        >
          <Icon size={14} />
          <span>{meta.label}</span>
        </div>
      )}
      
      <div className="prose-chat text-[13px] leading-relaxed text-[#D1D5DB]">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
            h2: ({node, ...props}) => <h2 className="text-base font-bold text-white mt-4 mb-2 first:mt-0" {...props} />,
            h3: ({node, ...props}) => <h3 className="text-sm font-bold text-white mt-3 mb-1.5" {...props} />,
            p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
            ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1" {...props} />,
            li: ({node, ...props}) => <li className="marker:text-[#FF6B35]" {...props} />,
            table: ({node, ...props}) => (
              <div className="overflow-x-auto my-4 rounded-xl border border-[#2D3250] bg-[#1a1d2b]">
                <table className="min-w-full divide-y divide-[#2D3250]" {...props} />
              </div>
            ),
            thead: ({node, ...props}) => <thead className="bg-[#262840]" {...props} />,
            th: ({node, ...props}) => <th className="px-4 py-3 text-left text-xs font-bold text-[#888] uppercase tracking-wider" {...props} />,
            td: ({node, ...props}) => <td className="px-4 py-3 text-sm border-t border-[#2D3250]" {...props} />,
            tr: ({node, ...props}) => <tr className="hover:bg-white/[0.02] transition-colors" {...props} />,
            strong: ({node, ...props}) => <strong className="font-bold text-white" {...props} />,
            hr: ({node, ...props}) => <hr className="my-6 border-[#2D3250]" {...props} />,
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  );
}
