# 🤖 TripWeaver AI Agents

TripWeaver AI utilizes a multi-agent orchestration pattern to provide comprehensive travel assistance. Our system decomposes complex travel queries into specialized tasks handled by dedicated agents.

## 🏗️ Orchestration Layer
- **Core Orchestrator (LangGraph):** Manages the state machine and conversation flow. It determines which agent or tool to invoke based on user intent and previous context.
- **LLM Engine (Groq / Llama 3):** Powers the reasoning and natural language generation, providing near-instantaneous responses.

## 🛠️ Specialized Agents & Tools

### 🌤️ Weather Agent
- **Capability:** Provides real-time weather forecasts and historical climate data.
- **Integration:** Fetches data for specific travel dates to help users pack and plan activities.

### 🏨 Accommodation Agent (Amadeus)
- **Capability:** Searches for hotels, resorts, and stays.
- **Data Source:** Connects to the **Amadeus API** to provide live availability, pricing, and hotel details.

### ✈️ Flight Agent (Amadeus)
- **Capability:** Finds optimal flight routes and current fares.
- **Data Source:** Queries **Amadeus** for real-time flight data across major carriers in India.

### 🗺️ Itinerary Architect
- **Capability:** Generates structured multi-day travel plans.
- **Logic:** Combines data from other agents (weather, local attractions) to create a balanced, time-efficient schedule.

### 💰 Budget Analyst
- **Capability:** Estimates total trip costs based on user preferences.
- **Focus:** Provides breakdowns for transport, stay, food, and sightseeing in INR.

## 🔄 Interaction Flow
1. **Input:** User asks a travel-related question.
2. **Analysis:** The Orchestrator identifies required data points (e.g., "Delhi to Goa flight" + "Weather in Goa").
3. **Execution:** Parallel calls are made to specialized agents/tools.
4. **Synthesis:** The Itinerary Architect compiles the findings into a cohesive, markdown-formatted response.
5. **Output:** The frontend renders the data using specialized UI cards.
