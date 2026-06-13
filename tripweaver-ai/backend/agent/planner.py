import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from agent.memory import TravelMemory
from agent.tools import ALL_TOOLS
from database.db import save_search, get_preferences, save_preferences, format_preferences_for_prompt

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH))

SYSTEM_PROMPT = """You are TripWeaver, an AI Travel Concierge for Indian travelers.

TOOL ROUTING — call the correct tool FIRST, then format the response:

| Query type | Tool |
|---|---|
| weather / rain / temperature / forecast | weather_tool |
| hotels / stay / accommodation | hotel_tool |
| budget — user gives amount AND days | budget_tool |
| flights / airfare | flight_tool |
| attractions / places / things to do | places_tool |
| restaurants / food / where to eat / cafes | restaurant_tool |
| festivals / visa / news / safety | web_search_tool |
| "save this trip" | save_itinerary_tool |
| "my history" / "saved trips" | search_history_tool |

━━━ MULTI-TOOL PLANNING RULE ━━━
When user says "plan a trip", "plan a X-day trip", or "plan my trip to X":
AUTOMATICALLY call ALL of these tools in sequence:
  1. weather_tool(city) — get current conditions
  2. places_tool(city) — get attractions
  3. hotel_tool(city) — get accommodation options
  4. budget_tool("AMOUNT,DAYS") — if budget mentioned, else estimate
Then merge ALL results into one complete travel package response.
DO NOT call just one tool for trip planning.

RULES:
1. ALWAYS call the tool first. NEVER answer from memory for weather, hotels, flights, places, or restaurants.
2. After the tool returns, paste its output then add your insights.
3. NEVER invent hotel names, flight numbers, prices, restaurant names, or attraction names.
4. Call ONE tool for single queries. Call MULTIPLE tools for trip planning.

━━━ RESPONSE FORMAT ━━━

**Weather:**
## 🌤️ Weather in [City]
[Full weather_tool output]
---
### 🗺️ Best Places Given This Weather
| # | Place | Why |
|---|---|---|
| 1 | **[Place]** | [reason] |
---
### 💡 Quick Tips
- **Pack:** [items] · **Tip:** [advice]

**Hotels:**
## 🏨 Hotels in [City]
[Full hotel_tool output]
---
### 💡 Booking Tips
- [peak season advice] · [city tip]

**Flights:**
## ✈️ Flights: [Origin] → [Destination]
[Full flight_tool output]
---
### 💡 Tips · Compare on MakeMyTrip · Early morning = cheapest

**Budget:**
## 💰 Budget Breakdown
[Full budget_tool output]

**Restaurants:**
## 🍽️ Where to Eat in [City]
[Full restaurant_tool output]
---
### 💡 Food Tips
- [local specialty to try] · [best area for street food]

**Full Trip Plan** (multi-tool response):
## 🗺️ [X]-Day Trip to [City]

### 🌤️ Weather & Best Time
[paste weather_tool output summary]

### 🗺️ Top Attractions
[paste places_tool output]

### 🏨 Where to Stay
[paste hotel_tool output — top 3-5 options]

### 🍽️ Where to Eat
[paste restaurant_tool output — top recommendations]

### Day 1 — [Theme]
| Time | Activity | Cost |
|---|---|---|
| 🌅 Morning | [activity] | ₹X |
| ☀️ Afternoon | [activity] | ₹X |
| 🌙 Evening | [activity] | ₹X |

### Day 2 — [Theme] (repeat)
### Day 3 — [Theme] (repeat)

---
### 💰 Budget Summary
| Category | Cost |
|---|---|
| 🏨 Accommodation | ₹X |
| 🍽️ Food | ₹X |
| 🚌 Transport | ₹X |
| 🎯 Activities | ₹X |
| **Total** | **₹X** |

---
### ✈️ Travel Tips
- **Best time:** · **Getting there:** · **Don't miss:**

**Places:**
## 🗺️ Top Places in [City]
[Full places_tool output]

General: Use ₹ · --- dividers · Keep concise · Tailor to travel style"""

class TripPlanner:
    def __init__(self):
        # LLM with Gemini fallback
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        self.llm = None
        if groq_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=groq_key,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024,
                    timeout=30,
                )
            except Exception:
                self.llm = None

        if self.llm is None and gemini_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=gemini_key,
                model="gemini-2.0-flash",
                temperature=0.3,
                max_tokens=1024,
            )

        if self.llm is None:
            raise RuntimeError("No LLM available. Set GROQ_API_KEY or GEMINI_API_KEY in .env")

        # Initialize memory
        self.memory = TravelMemory()

        # LangChain AgentExecutor — primary execution path (reliable, fast)
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm_with_tools, ALL_TOOLS, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=30,
            return_intermediate_steps=True,
        )

    def _resolve_input(self, user_input: str) -> str:
        """
        Only inject context destination for vague follow-ups that contain
        NO explicit city name. If the user names a city, pass through unchanged.
        """
        u = user_input.lower().strip()
        dest = self.memory.context.destination

        if not dest:
            return user_input

        # If the input already contains any known city name, don't override
        known_cities = list(self.memory.context.__class__.__init__.__defaults__ or [])
        from agent.memory import TravelContext
        # Check against all known destinations in the static list
        known_destinations = [
            "goa", "jaipur", "manali", "delhi", "mumbai", "kerala",
            "udaipur", "shimla", "bangalore", "chennai", "kolkata",
            "agra", "varanasi", "rishikesh", "darjeeling", "ooty",
            "ladakh", "kashmir", "rajasthan", "coorg", "munnar",
            "leh", "kochi", "cochin", "pune", "hyderabad", "amritsar",
            "jodhpur", "mysore", "mysuru", "hampi", "kodaikanal",
        ]
        # If user explicitly named a city, pass through unchanged
        if any(city in u for city in known_destinations):
            return user_input

        # Only inject for truly vague follow-ups with no city mentioned
        weather_followups = [
            "what is the weather there", "what's the weather there",
            "how's the weather", "how is the weather", "weather there",
            "will it rain", "is it hot", "is it cold",
            "what's the weather", "what is the weather",
            "weather forecast", "check weather",
        ]
        if any(phrase in u for phrase in weather_followups):
            return f"What is the weather in {dest}?"

        hotel_followups = [
            "suggest hotels", "find hotels", "what about hotels",
            "where to stay", "hotels there", "accommodation there",
            "suggest accommodation", "find hostels",
        ]
        if any(phrase in u for phrase in hotel_followups):
            return f"Suggest hotels in {dest}"

        places_followups = [
            "what to see there", "things to do there", "places there",
            "what should i see", "sightseeing there", "attractions there",
        ]
        if any(phrase in u for phrase in places_followups):
            return f"What are the top places to visit in {dest}?"

        return user_input

    def chat(self, user_input: str, session_id: str = "default") -> str:
        resolved_input = self._resolve_input(user_input)

        # Log search to DB (non-blocking)
        try:
            save_search(
                session_id=session_id,
                query=user_input,
                query_type=self._classify_query(user_input),
                destination=self.memory.context.destination,
            )
        except Exception:
            pass

        # Inject saved preferences
        prefs = {}
        try:
            prefs = get_preferences(session_id)
        except Exception:
            pass

        try:
            # Only pass user messages as chat history — AI responses cause context contamination
            from langchain_core.messages import HumanMessage as HM
            raw_history = self.memory.get_chat_history()
            chat_history = [m for m in raw_history if isinstance(m, HM)][-4:]

            if prefs:
                pref_str = format_preferences_for_prompt(prefs)
                if pref_str:
                    from langchain_core.messages import SystemMessage as SM
                    chat_history = [SM(content=pref_str)] + chat_history

            result = self.agent_executor.invoke({
                "input": resolved_input,
                "chat_history": chat_history,
            })

            response = result.get("output", "")
            if not response or "agent stopped" in response.lower():
                steps = result.get("intermediate_steps", [])
                if steps:
                    response = steps[-1][1] if steps else ""
                if not response:
                    raise ValueError("Empty response")

        except Exception:
            # Fallback: plain LLM chain without tools
            try:
                fallback_prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                chain = fallback_prompt | self.llm | StrOutputParser()
                response = chain.invoke({
                    "input": resolved_input,
                    "chat_history": self.memory.get_chat_history(),
                })
            except Exception as e:
                return f"I apologize, I encountered an error: {str(e)}. Please try again."

        self.memory.add_user_message(user_input)
        self.memory.messages.append(AIMessage(content=response))

        # Persist preferences
        try:
            ctx = self.memory.context
            updates = {}
            if ctx.travel_style:  updates["travel_style"]  = ctx.travel_style
            if ctx.accommodation: updates["accommodation"]  = ctx.accommodation
            if ctx.interests:     updates["interests"]      = ctx.interests
            if updates:
                save_preferences(session_id, **updates)
        except Exception:
            pass

        return response

    @staticmethod
    def _classify_query(text: str) -> str:
        """Classify a query into a type for DB logging."""
        t = text.lower()
        # itinerary first — "budget trip" should be itinerary not budget
        if any(w in t for w in ["plan", "itinerary", "trip to", "visit for"]):
            return "itinerary"
        if any(w in t for w in ["weather", "rain", "temperature", "forecast"]):
            return "weather"
        if any(w in t for w in ["hotel", "stay", "accommodation", "hostel"]):
            return "hotel"
        if any(w in t for w in ["flight", "fly", "airfare", "airline"]):
            return "flight"
        if any(w in t for w in ["budget", "cost", "expense"]):
            return "budget"
        if any(w in t for w in ["place", "attraction", "see", "do"]):
            return "places"
        return "general"

    def clear_memory(self):
        self.memory.clear_memory()

    def get_conversation_summary(self) -> str:
        chat_history = self.memory.get_chat_history()
        if not chat_history:
            return "No conversation history available."
        summary = "Recent conversation:\n"
        for msg in chat_history[-4:]:
            if isinstance(msg, HumanMessage):
                summary += f"User: {msg.content[:100]}...\n"
            elif isinstance(msg, AIMessage):
                summary += f"AI: {msg.content[:100]}...\n"
        return summary

        