"use client";
import { useState, useRef, type KeyboardEvent } from "react";
import { SendHorizonal, Loader2 } from "lucide-react";

interface Props {
  onSend: (message: string) => void;
  disabled: boolean;
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const msg = value.trim();
    if (!msg || disabled) return;
    onSend(msg);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className={`
        relative flex items-end gap-2 p-2 rounded-2xl border transition-all duration-200
        ${disabled ? "bg-[#1E2130]/50 border-[#2D3250]" : "bg-[#1E2130] border-[#2D3250] shadow-lg shadow-black/20 focus-within:border-[#FF6B35]/50 focus-within:ring-4 focus-within:ring-[#FF6B35]/5"}
      `}>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Ask about weather, hotels, flights, or plan a trip..."
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none bg-transparent text-[14px] px-3 py-2.5 outline-none placeholder-[#555] text-white leading-relaxed min-h-[44px]"
          style={{ maxHeight: "160px" }}
        />
        
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className={`
            flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200
            ${disabled || !value.trim() 
              ? "bg-[#2D3250] text-[#555] cursor-not-allowed" 
              : "bg-[#FF6B35] text-white hover:bg-[#e05a25] hover:scale-105 active:scale-95 shadow-lg shadow-orange-600/20"
            }
          `}
        >
          {disabled ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <SendHorizonal size={18} />
          )}
        </button>
      </div>
      <p className="mt-3 text-[10px] text-center text-[#555] font-medium uppercase tracking-widest">
        Press <span className="text-[#888]">Shift + Enter</span> for new line
      </p>
    </div>
  );
}
