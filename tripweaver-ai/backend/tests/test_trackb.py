"""
Track B Test Suite — Weeks 1-4
================================
Tests for:
  - LangGraph StateGraph (graph.py)
  - Structured logger and metrics (logger.py)
  - Agent state management (TravelAgentState)
  - CI/CD pipeline config exists

Run with:  pytest tests/test_trackb.py -v
"""

import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Logger & Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogger:

    def test_get_logger_returns_logger(self):
        from agent.logger import get_logger
        log = get_logger("test")
        assert log is not None
        assert log.name == "test"

    def test_logger_does_not_crash_on_info(self):
        from agent.logger import logger
        logger.info("Test log message — should not raise")

    def test_metrics_store_records_tool_call(self):
        from agent.logger import MetricsStore
        m = MetricsStore()
        m.record_tool_call("weather_tool", 120.5)
        assert m.tool_calls["weather_tool"] == 1
        assert m.tool_latency["weather_tool"] == [120.5]
        assert m.tool_errors["weather_tool"] == 0

    def test_metrics_store_records_tool_error(self):
        from agent.logger import MetricsStore
        m = MetricsStore()
        m.record_tool_call("hotel_tool", 200.0, error=True)
        assert m.tool_errors["hotel_tool"] == 1

    def test_metrics_store_records_agent_run(self):
        from agent.logger import MetricsStore
        m = MetricsStore()
        m.record_agent_run(350.0, query_type="weather")
        assert m.agent_runs == 1
        assert m.query_types["weather"] == 1
        assert m.agent_latency == [350.0]

    def test_metrics_avg_latency(self):
        from agent.logger import MetricsStore
        m = MetricsStore()
        m.record_agent_run(100.0)
        m.record_agent_run(200.0)
        m.record_agent_run(300.0)
        assert m.avg_latency(m.agent_latency) == 200.0

    def test_metrics_summary_structure(self):
        from agent.logger import MetricsStore
        m = MetricsStore()
        m.record_tool_call("weather_tool", 100.0)
        m.record_agent_run(200.0, query_type="weather")
        summary = m.summary()
        assert "agent_runs" in summary
        assert "tool_calls" in summary
        assert "avg_agent_ms" in summary
        assert "query_types" in summary

    def test_metrics_reset(self):
        from agent.logger import MetricsStore
        m = MetricsStore()
        m.record_agent_run(100.0)
        m.reset()
        assert m.agent_runs == 0
        assert len(m.agent_latency) == 0

    def test_log_tool_call_decorator(self):
        from agent.logger import log_tool_call, MetricsStore
        m = MetricsStore()

        @log_tool_call("TestTool")
        def dummy_tool(x):
            return f"result: {x}"

        result = dummy_tool("Goa")
        assert result == "result: Goa"

    def test_log_tool_call_decorator_on_error(self):
        from agent.logger import log_tool_call

        @log_tool_call("FailingTool")
        def bad_tool():
            raise ValueError("tool failed")

        with pytest.raises(ValueError, match="tool failed"):
            bad_tool()

    def test_log_agent_run_decorator(self):
        from agent.logger import log_agent_run

        class FakePlanner:
            @log_agent_run
            def chat(self, user_input: str) -> str:
                return f"response to: {user_input}"

        p = FakePlanner()
        result = p.chat("What's the weather in Goa?")
        assert "response to" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LangGraph — State & Routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphState:

    def test_travel_agent_state_structure(self):
        from agent.graph import TravelAgentState
        from langchain_core.messages import HumanMessage
        state: TravelAgentState = {
            "messages":        [HumanMessage(content="test")],
            "query_type":      "weather",
            "destination":     "Goa",
            "session_id":      "test_session",
            "iteration":       0,
            "error":           None,
            "tool_calls_made": [],
        }
        assert state["query_type"] == "weather"
        assert state["destination"] == "Goa"
        assert state["iteration"] == 0

    def test_classify_query_weather(self):
        from agent.graph import _classify_query
        assert _classify_query("What's the weather in Goa?") == "weather"
        assert _classify_query("Will it rain in Manali?") == "weather"
        assert _classify_query("temperature in Delhi") == "weather"

    def test_classify_query_hotel(self):
        from agent.graph import _classify_query
        assert _classify_query("Suggest hotels in Jaipur") == "hotel"
        assert _classify_query("where to stay in Mumbai") == "hotel"

    def test_classify_query_flight(self):
        from agent.graph import _classify_query
        assert _classify_query("Flights from Delhi to Goa") == "flight"
        assert _classify_query("cheapest airfare to Manali") == "flight"

    def test_classify_query_budget(self):
        from agent.graph import _classify_query
        assert _classify_query("My budget is ₹15000 for 3 days") == "budget"
        assert _classify_query("how much will it cost?") == "budget"

    def test_classify_query_itinerary(self):
        from agent.graph import _classify_query
        assert _classify_query("Plan a 3-day trip to Manali") == "itinerary"

    def test_classify_query_general(self):
        from agent.graph import _classify_query
        assert _classify_query("Hello, how are you?") == "general"

    def test_extract_destination_known_city(self):
        from agent.graph import _extract_destination
        assert _extract_destination("What's the weather in Goa?") == "Goa"
        assert _extract_destination("Plan a trip to Manali") == "Manali"
        assert _extract_destination("Hotels in Jaipur") == "Jaipur"

    def test_extract_destination_unknown_returns_none(self):
        from agent.graph import _extract_destination
        result = _extract_destination("Hello, how are you?")
        assert result is None

    def test_should_continue_routes_to_tools(self):
        from agent.graph import should_continue
        from langchain_core.messages import AIMessage

        # Message with tool calls → should route to tools
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "weather_tool", "args": {"city": "Goa"}, "id": "1"}]
        )
        state = {
            "messages": [ai_msg],
            "iteration": 1,
            "error": None,
            "tool_calls_made": [],
        }
        assert should_continue(state) == "tools"

    def test_should_continue_routes_to_format(self):
        from agent.graph import should_continue
        from langchain_core.messages import AIMessage

        # Message without tool calls → should route to format
        ai_msg = AIMessage(content="Here is the weather in Goa...")
        state = {
            "messages": [ai_msg],
            "iteration": 1,
            "error": None,
            "tool_calls_made": [],
        }
        assert should_continue(state) == "format"

    def test_should_continue_max_iterations(self):
        from agent.graph import should_continue
        from langchain_core.messages import AIMessage

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "weather_tool", "args": {}, "id": "1"}]
        )
        state = {
            "messages": [ai_msg],
            "iteration": 3,   # at max
            "error": None,
            "tool_calls_made": [],
        }
        assert should_continue(state) == "end"

    def test_should_continue_on_error(self):
        from agent.graph import should_continue
        from langchain_core.messages import AIMessage

        state = {
            "messages": [AIMessage(content="error occurred")],
            "iteration": 1,
            "error": "Something went wrong",
            "tool_calls_made": [],
        }
        assert should_continue(state) == "end"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LangGraph — Graph compilation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphCompilation:

    def test_graph_builds_without_error(self):
        from agent.graph import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_module_level_instance_exists(self):
        # Graph is lazy — built on first call, not at import time
        from agent.graph import _get_graph
        graph = _get_graph()
        assert graph is not None

    def test_classify_node_updates_state(self):
        from agent.graph import classify_node
        from langchain_core.messages import HumanMessage

        state = {
            "messages":        [HumanMessage(content="What's the weather in Goa?")],
            "query_type":      "general",
            "destination":     None,
            "session_id":      "test",
            "iteration":       0,
            "error":           None,
            "tool_calls_made": [],
        }
        result = classify_node(state)
        assert result["query_type"] == "weather"
        assert result["destination"] == "Goa"
        assert result["iteration"] == 0

    def test_format_node_passes_through(self):
        from agent.graph import format_node
        from langchain_core.messages import AIMessage

        state = {
            "messages":        [AIMessage(content="Here is the weather...")],
            "query_type":      "weather",
            "destination":     "Goa",
            "session_id":      "test",
            "iteration":       1,
            "error":           None,
            "tool_calls_made": [],
        }
        result = format_node(state)
        assert result["messages"][-1].content == "Here is the weather..."


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CI/CD Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestCICD:

    def _ci_path(self):
        # .github/workflows/ci.yml is at the repo root (tripweaver-ai/)
        return Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "ci.yml"

    def test_github_actions_workflow_exists(self):
        assert self._ci_path().exists(), f"CI/CD workflow not found at {self._ci_path()}"

    def test_workflow_has_test_job(self):
        p = self._ci_path()
        if p.exists():
            content = p.read_text()
            assert "pytest" in content
            assert "python" in content.lower()

    def test_workflow_has_lint_job(self):
        p = self._ci_path()
        if p.exists():
            content = p.read_text()
            assert "ruff" in content or "lint" in content.lower()

    def test_workflow_triggers_on_main(self):
        p = self._ci_path()
        if p.exists():
            content = p.read_text()
            assert "main" in content

    def test_logs_directory_in_gitignore(self):
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore.read_text()
        assert "logs/" in content or "*.log" in content


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Integration — run_graph with mocked LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphIntegration:

    @patch("agent.graph._get_llm_with_tools")
    def test_run_graph_returns_string(self, mock_get_llm):
        """Graph should return a (string, dict) tuple."""
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="The weather in Goa is sunny and 30°C.")
        mock_get_llm.return_value = mock_llm

        from agent.graph import run_graph
        result = run_graph(user_input="What's the weather in Goa?", chat_history=[], session_id="test")
        assert isinstance(result, tuple)
        response, trace = result
        assert isinstance(response, str)
        assert len(response) > 0
        assert trace["path"] == "LangGraph StateGraph"

    @patch("agent.graph._get_llm_with_tools")
    def test_run_graph_handles_llm_error(self, mock_get_llm):
        """Graph should return error string, not raise."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        mock_get_llm.return_value = mock_llm

        from agent.graph import run_graph
        result = run_graph(user_input="What's the weather in Goa?", chat_history=[])
        assert isinstance(result, tuple)
        response, trace = result
        assert isinstance(response, str)
