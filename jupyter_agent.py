from jupyter_handler import JupyterNotebook
import json
import logging
import os
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tavily import TavilyClient

# Phoenix tracing imports
try:
    from openinference.instrumentation import using_session
    PHOENIX_AVAILABLE = True
    print("Phoenix session tracking imports successful")
except ImportError:
    PHOENIX_AVAILABLE = False
    print("Phoenix session tracking not available - missing openinference packages")

# Configure logging for utils module
logger = logging.getLogger(__name__)

# Initialize Tavily client
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_and_execute_jupyter_code_cell",
            "description": "A Python code execution environment that runs code in a Jupyter notebook interface. This is stateful - variables and imports persist between executions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_and_execute_current_cell",
            "description": "Edit the current/last code cell and execute the new code. Use this to fix errors or modify the previous code instead of creating a new cell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The updated Python code to replace the current cell with and execute."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": "Execute shell/system commands like ls, cat, mkdir, etc. This runs independently of Python and provides terminal-style output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (e.g., 'ls -la', 'cat file.txt', 'mkdir new_folder')."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Search the web for current information, documentation, tutorials, and solutions to coding problems. Use this to get context before starting tasks or when encountering errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (max 400 characters). Be specific and include relevant keywords."
                    }
                },
                "required": ["query"]
            }
        }
    },
]

# TOOLS = TOOLS[:1]

MAX_TURNS = 20


def create_phoenix_session_context(session_id: str, user_id: str = None, metadata: Dict = None):
    """
    Create a Phoenix session context for tracing LLM interactions.
    
    Args:
        session_id: Unique identifier for the session
        user_id: Optional user identifier
        metadata: Additional metadata to include in traces
        
    Returns:
        Context manager for Phoenix session tracking
    """
    if not PHOENIX_AVAILABLE:
        # Return a no-op context manager if Phoenix is not available
        from contextlib import nullcontext
        return nullcontext()
    
    try:
        # Use using_session for proper session grouping in Phoenix
        # This ensures all LLM calls within this context are grouped under the same session
        logger.debug(f"Creating Phoenix session context for session_id: {session_id}")
        return using_session(session_id)
    except Exception as e:
        logger.warning(f"Failed to create Phoenix session context for {session_id}: {e}")
        # Fallback to no-op context if Phoenix session creation fails
        from contextlib import nullcontext
        return nullcontext()


class SessionStateManager:
    """Manages comprehensive session state in a single JSON file"""
    
    def __init__(self, session_id: str, base_dir: str = './temp/'):
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / session_id
        self.state_file = self.session_dir / 'session_state.json'
        self.session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SessionStateManager initialized for {session_id}")
    
    def create_initial_state(self, hardware_config: Dict, api_config: Dict, 
                           environment: Dict, system_prompt: str) -> Dict:
        """Create initial session state structure"""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        initial_state = {
            "session_id": self.session_id,
            "created_at": timestamp,
            "last_updated": timestamp,
            "version": "1.0",
            
            "hardware_config": hardware_config,
            "api_config": api_config,
            "environment": environment,
            
            "conversation_history": [
                {
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": timestamp,
                    "metadata": {"type": "system_initialization"}
                }
            ],
            
            "llm_interactions": [],  # Complete API call logs
            "tool_executions": [],   # All tool calls and results
            
            "notebook_data": {
                "cells": [],
                "metadata": {
                    "kernel_info": {"name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"},
                },
                "nbformat": 4,
                "nbformat_minor": 0
            },
            
            "execution_state": {
                "current_turn": 0,
                "max_turns": MAX_TURNS,
                "is_running": False,
                "is_paused": False,
                "last_execution_successful": None,
                "sandbox_active": False,
                "sandbox_info": None
            },
            
            "session_stats": {
                "total_messages": 1,
                "total_code_executions": 0,
                "total_searches": 0,
                "total_errors": 0,
                "session_duration_seconds": 0
            }
        }
        
        logger.info("Created initial session state for %s", self.session_id)
        return initial_state
    
    def load_state(self) -> Optional[Dict]:
        """Load session state from file with improved error handling"""
        if not self.state_file.exists():
            logger.info(f"No existing session state found for {self.session_id}")
            return None
            
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info(f"Loaded session state for {self.session_id} with {len(state.get('conversation_history', []))} messages")
            return state
        except json.JSONDecodeError as e:
            logger.error(f"JSON corruption in session state for {self.session_id}: {str(e)}")
            logger.info(f"Creating backup of corrupted file: {self.state_file}.corrupted")
            try:
                import shutil
                shutil.copy2(self.state_file, str(self.state_file) + ".corrupted")
                logger.info(f"Backup created successfully")
            except Exception as backup_error:
                logger.warning(f"Failed to create backup: {backup_error}")
            return None
        except Exception as e:
            logger.error(f"Failed to load session state for {self.session_id}: {str(e)}")
            return None
    
    def save_state(self, state: Dict) -> bool:
        """Save session state to file with improved error handling"""
        try:
            # Update last_updated timestamp
            state["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # Update session stats
            if "session_stats" not in state:
                state["session_stats"] = {}
            
            created_at = datetime.datetime.fromisoformat(state["created_at"])
            current_time = datetime.datetime.now(datetime.timezone.utc)
            state["session_stats"]["session_duration_seconds"] = int((current_time - created_at).total_seconds())
            state["session_stats"]["total_messages"] = len(state.get("conversation_history", []))
            
            # Validate JSON serializability before writing
            try:
                json.dumps(state, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                logger.error(f"State contains non-serializable data: {e}")
                logger.info("Attempting to clean non-serializable data...")
                state = self._clean_non_serializable_data(state)
            
            # Write to temporary file first, then rename for atomic operation
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.state_file)
            
            logger.debug(f"Saved session state for {self.session_id} ({len(json.dumps(state))} characters)")
            return True
        except Exception as e:
            logger.error(f"Failed to save session state for {self.session_id}: {str(e)}")
            # Clean up temp file if it exists
            temp_file = self.state_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            return False
    
    def _clean_non_serializable_data(self, obj):
        """Recursively clean non-serializable data from objects"""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                try:
                    json.dumps(value)
                    cleaned[key] = self._clean_non_serializable_data(value)
                except (TypeError, ValueError):
                    logger.warning(f"Removing non-serializable field: {key}")
                    cleaned[key] = f"<non-serializable: {type(value).__name__}>"
            return cleaned
        elif isinstance(obj, list):
            cleaned = []
            for item in obj:
                try:
                    json.dumps(item)
                    cleaned.append(self._clean_non_serializable_data(item))
                except (TypeError, ValueError):
                    cleaned.append(f"<non-serializable: {type(item).__name__}>")
            return cleaned
        else:
            return obj
    
    def log_llm_interaction(self, state: Dict, request_data: Dict, response_data: Dict, 
                           model: str, turn: int) -> None:
        """Log complete LLM API interaction"""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        interaction = {
            "timestamp": timestamp,
            "turn": turn,
            "model": model,
            "request": {
                "messages_count": len(request_data.get("messages", [])),
                "tools_count": len(request_data.get("tools", [])),
                "model": request_data.get("model"),
                "tool_choice": request_data.get("tool_choice")
            },
            "response": {
                "content": response_data.get("choices", [{}])[0].get("message", {}).get("content"),
                "tool_calls": response_data.get("choices", [{}])[0].get("message", {}).get("tool_calls"),
                "finish_reason": response_data.get("choices", [{}])[0].get("finish_reason"),
                "usage": response_data.get("usage")
            }
        }
        
        if "llm_interactions" not in state:
            state["llm_interactions"] = []
        state["llm_interactions"].append(interaction)
        
        # Log Phoenix session information for easy debugging
        logger.debug(f"Logged LLM interaction for turn {turn} in session {self.session_id}")
        logger.debug(f"Phoenix session tracking: session_id={self.session_id}, turn={turn}, model={model}")
        
        # Log usage information if available for monitoring
        usage = response_data.get("usage")
        if usage:
            logger.info(f"Session {self.session_id} turn {turn}: "
                       f"prompt_tokens={usage.get('prompt_tokens', 0)}, "
                       f"completion_tokens={usage.get('completion_tokens', 0)}, "
                       f"total_tokens={usage.get('total_tokens', 0)}")
    
    def log_tool_execution(self, state: Dict, tool_call_id: str, tool_name: str, 
                          tool_args: Dict, result: str, execution_data: Any = None) -> None:
        """Log tool execution with full details"""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Safely serialize execution_data to prevent JSON corruption
        safe_execution_data = None
        if execution_data is not None:
            try:
                # Convert execution_data to a safe, serializable format
                if hasattr(execution_data, '__dict__'):
                    safe_execution_data = {
                        "type": type(execution_data).__name__,
                        "error": str(execution_data.error) if hasattr(execution_data, 'error') and execution_data.error else None,
                        "has_results": hasattr(execution_data, 'results') and bool(execution_data.results),
                        "has_stdout": hasattr(execution_data, 'logs') and hasattr(execution_data.logs, 'stdout') and bool(execution_data.logs.stdout),
                        "has_stderr": hasattr(execution_data, 'logs') and hasattr(execution_data.logs, 'stderr') and bool(execution_data.logs.stderr)
                    }
                else:
                    # For simple types, convert to string safely
                    safe_execution_data = str(execution_data)[:200]  # Limit length
            except Exception as e:
                logger.warning(f"Failed to serialize execution_data for {tool_call_id}: {e}")
                safe_execution_data = {"serialization_error": str(e)}
        
        tool_execution = {
            "timestamp": timestamp,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": tool_args,
            "result_summary": result[:500] + "..." if len(result) > 500 else result,
            "result_length": len(result),
            "execution_data": safe_execution_data,
            "success": execution_data is None or (hasattr(execution_data, 'error') and execution_data.error is None) if execution_data else True
        }
        
        if "tool_executions" not in state:
            state["tool_executions"] = []
        state["tool_executions"].append(tool_execution)
        
        # Update stats
        if tool_name == "add_and_execute_jupyter_code_cell":
            state["session_stats"]["total_code_executions"] = state["session_stats"].get("total_code_executions", 0) + 1
        elif tool_name == "tavily_search":
            state["session_stats"]["total_searches"] = state["session_stats"].get("total_searches", 0) + 1
            
        if not tool_execution["success"]:
            state["session_stats"]["total_errors"] = state["session_stats"].get("total_errors", 0) + 1
        
        logger.debug(f"Logged tool execution {tool_name} ({tool_call_id}) in session {self.session_id}")
    
    def add_message(self, state: Dict, role: str, content: str, 
                   tool_calls: List = None, tool_call_id: str = None, 
                   raw_execution: Any = None, metadata: Dict = None) -> None:
        """Add message to conversation history with full context"""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if raw_execution:
            message["raw_execution"] = raw_execution
        if metadata:
            message["metadata"] = metadata
            
        state["conversation_history"].append(message)
        logger.debug(f"Added {role} message to session {self.session_id} conversation history")
    
    def update_execution_state(self, state: Dict, **kwargs) -> None:
        """Update execution state fields"""
        for key, value in kwargs.items():
            if key in state["execution_state"]:
                state["execution_state"][key] = value
                logger.debug(f"Updated execution state {key}={value} for session {self.session_id}")
                
        # Try to sync with global EXECUTION_STATES for UI consistency (if available)
        try:
            import sys
            if 'app' in sys.modules:
                execution_states = getattr(sys.modules['app'], 'EXECUTION_STATES', None)
                if execution_states and self.session_id in execution_states:
                    for key, value in kwargs.items():
                        execution_states[self.session_id][key] = value
        except (ImportError, AttributeError):
            pass  # Ignore if we can't sync with global state
    
    def update_notebook_data(self, state: Dict, notebook_data: Dict) -> None:
        """Update notebook data in session state"""
        state["notebook_data"] = notebook_data
        logger.debug(f"Updated notebook data for session {self.session_id} ({len(notebook_data.get('cells', []))} cells)")
    
    def get_conversation_history(self, state: Dict) -> List[Dict]:
        """Get conversation history suitable for LLM API calls"""
        return state.get("conversation_history", [])
    
    def validate_and_repair_conversation(self, state: Dict) -> None:
        """Validate and repair conversation history to ensure tool calls have responses"""
        conversation = state.get("conversation_history", [])
        if not conversation:
            return
            
        pending_tool_calls = set()
        valid_messages = []
        
        for message in conversation:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                # Track tool calls
                for tool_call in message["tool_calls"]:
                    pending_tool_calls.add(tool_call["id"])
                valid_messages.append(message)
                
            elif message.get("role") == "tool" and message.get("tool_call_id"):
                # Remove from pending when we find a response
                pending_tool_calls.discard(message["tool_call_id"])
                valid_messages.append(message)
                
            else:
                # Regular message (system, user, assistant without tool calls)
                valid_messages.append(message)
        
        # If there are incomplete tool calls, remove the assistant messages that created them
        if pending_tool_calls:
            logger.warning(f"Found incomplete tool calls in conversation: {pending_tool_calls}")
            logger.warning("Removing incomplete assistant messages to repair conversation")
            
            repaired_messages = []
            for message in valid_messages:
                if (message.get("role") == "assistant" and 
                    message.get("tool_calls") and 
                    any(tc["id"] in pending_tool_calls for tc in message["tool_calls"])):
                    logger.debug("Removing assistant message with incomplete tool calls")
                    continue
                repaired_messages.append(message)
            
            # Update conversation history
            state["conversation_history"] = repaired_messages
            logger.info(f"Repaired conversation: {len(conversation)} -> {len(repaired_messages)} messages")
            
            # Save the repaired state
            self.save_state(state)
    
    def session_exists(self) -> bool:
        """Check if session state file exists"""
        return self.state_file.exists()
    
    def get_session_summary(self, state: Dict) -> str:
        """Get human-readable session summary"""
        stats = state.get("session_stats", {})
        created = datetime.datetime.fromisoformat(state["created_at"])
        
        return f"""Session {self.session_id}:
- Created: {created.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Messages: {stats.get('total_messages', 0)}
- Code Executions: {stats.get('total_code_executions', 0)}
- Web Searches: {stats.get('total_searches', 0)}
- Errors: {stats.get('total_errors', 0)}
- Duration: {stats.get('session_duration_seconds', 0)}s
- Hardware: {state.get('hardware_config', {}).get('gpu_type', 'unknown')}
- Model: {state.get('api_config', {}).get('model_name', 'unknown')}"""


def execute_code(sbx, code):
    logger.debug(f"Executing code in sandbox ({len(code)} characters)")
    execution = sbx.run_code(code, on_stdout=lambda data: logger.debug(f'stdout: {data}'))
    output = ""
    if len(execution.logs.stdout) > 0:
        output += "\n".join(execution.logs.stdout)
        logger.debug(f"Execution produced {len(execution.logs.stdout)} stdout lines")
    if len(execution.logs.stderr) > 0:
        output += "\n".join(execution.logs.stderr)
        logger.debug(f"Execution produced {len(execution.logs.stderr)} stderr lines")
    if execution.error is not None:
        output += execution.error.traceback
        logger.warning(f"Execution error: {execution.error.name}: {execution.error.value}")
    logger.debug(f"Code execution completed, output length: {len(output)}")
    return output, execution


def parse_exec_result_llm(execution, max_code_output=1000):
    logger.debug(f"Parsing execution result for LLM (max_output: {max_code_output})")
    output = []

    def truncate_if_needed(text):
        if len(text) > max_code_output:
            return (text[:max_code_output] + f"\n[Output is truncated as it is more than {max_code_output} characters]")
        return text

    if execution.results:
        results_text_parts = []
        plot_count = 0
        
        for result in execution.results:
            if hasattr(result, 'text') and result.text:
                results_text_parts.append(result.text)
            elif hasattr(result, 'png') and result.png:
                plot_count += 1
                results_text_parts.append(f"[Plot {plot_count} generated and displayed]")
            elif hasattr(result, 'html') and result.html:
                results_text_parts.append("[HTML output generated]")
        
        if results_text_parts:
            results_text = "\n".join(results_text_parts)
            output.append(truncate_if_needed(results_text))
        
        logger.debug(f"Added {len(execution.results)} execution results (including {plot_count} plots)")
    if execution.logs.stdout:
        stdout_text = "\n".join(execution.logs.stdout)
        output.append(truncate_if_needed(stdout_text))
        logger.debug(f"Added stdout output ({len(execution.logs.stdout)} lines)")
    if execution.logs.stderr:
        stderr_text = "\n".join(execution.logs.stderr)
        output.append(truncate_if_needed(stderr_text))
        logger.debug(f"Added stderr output ({len(execution.logs.stderr)} lines)")
    if execution.error is not None:
        output.append(truncate_if_needed(execution.error.traceback))
        logger.debug(f"Added error traceback: {execution.error.name}")
    
    final_output = "\n".join(output)
    logger.debug(f"Parsed execution result for LLM: {len(final_output)} characters")
    return final_output

def clean_messages_for_api(messages):
    """
    Create a clean copy of messages without raw_execution fields and metadata for API calls.
    Also validates that tool calls have corresponding tool responses.
    This prevents 413 errors and API validation errors.
    """
    logger.debug(f"Cleaning {len(messages)} messages for API call")
    cleaned_messages = []
    raw_execution_count = 0
    metadata_count = 0
    pending_tool_calls = set()
    
    for message in messages:
        cleaned_message = message.copy()
        
        # Remove raw_execution data
        if "raw_execution" in cleaned_message:
            cleaned_message.pop("raw_execution")
            raw_execution_count += 1
            
        # Remove metadata and timestamp
        if "metadata" in cleaned_message:
            cleaned_message.pop("metadata")
            metadata_count += 1
        if "timestamp" in cleaned_message:
            cleaned_message.pop("timestamp")
        
        # Track tool calls and responses for validation
        if cleaned_message.get("role") == "assistant" and cleaned_message.get("tool_calls"):
            for tool_call in cleaned_message["tool_calls"]:
                pending_tool_calls.add(tool_call["id"])
        elif cleaned_message.get("role") == "tool" and cleaned_message.get("tool_call_id"):
            pending_tool_calls.discard(cleaned_message["tool_call_id"])
            
        cleaned_messages.append(cleaned_message)
    
    # If there are pending tool calls without responses, remove the assistant message with tool calls
    if pending_tool_calls:
        logger.warning(f"Found {len(pending_tool_calls)} tool calls without responses: {pending_tool_calls}")
        logger.warning("Removing incomplete tool call messages to prevent API errors")
        
        # Remove messages with incomplete tool calls
        filtered_messages = []
        for message in cleaned_messages:
            if (message.get("role") == "assistant" and 
                message.get("tool_calls") and 
                any(tc["id"] in pending_tool_calls for tc in message["tool_calls"])):
                logger.debug("Removing assistant message with incomplete tool calls")
                continue
            filtered_messages.append(message)
        
        cleaned_messages = filtered_messages
    
    logger.debug(f"Cleaned messages: removed raw_execution from {raw_execution_count}, metadata from {metadata_count}")
    logger.debug(f"Final cleaned message count: {len(cleaned_messages)}")
    return cleaned_messages


def tavily_search(query):
    """
    Perform web search using Tavily API with automatic year addition and formatting.
    
    Args:
        query (str): Search query (max 400 characters)
        
    Returns:
        str: Formatted search results for LLM consumption
    """
    if not tavily_client:
        logger.error("Tavily client not initialized - API key missing")
        return "❌ Search unavailable: Tavily API key not configured"
    
    # Validate query length
    if len(query) > 400:
        logger.warning(f"Query too long ({len(query)} chars), truncating to 400")
        query = query[:400]
    
    # Add current year to query for more recent results
    current_year = datetime.datetime.now().year
    if str(current_year) not in query:
        # Only add year if query has room for it
        year_addition = f" {current_year}"
        if len(query + year_addition) <= 400:
            query += year_addition
            logger.debug(f"Added current year to query: {current_year}")
    
    logger.info(f"Performing Tavily search: '{query}' ({len(query)} chars)")
    
    try:
        # Perform search with optimized parameters
        response = tavily_client.search(
            query=query,
            search_depth="basic",  # Use basic for faster results
            max_results=5,         # Limit results to avoid overwhelming context
            include_answer=True,   # Include AI-generated answer
            include_raw_content=False,  # Don't include raw content to save tokens
            include_images=False   # Don't include images
        )
        
        logger.info(f"Search completed: {len(response.get('results', []))} results found")
        
        # Format results for LLM consumption
        formatted_results = format_search_results_for_llm(response)
        
        logger.debug(f"Formatted search results: {len(formatted_results)} characters")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Tavily search failed: {str(e)}")
        return f"❌ Search failed: {str(e)}"


def format_search_results_for_llm(response):
    """Format Tavily search results for LLM consumption"""
    
    query = response.get('query', 'Unknown query')
    results = response.get('results', [])
    answer = response.get('answer', '')
    
    formatted = f"🔍 **Web Search Results for:** {query}\n\n"
    
    if answer:
        formatted += f"**Quick Answer:** {answer}\n\n"
    
    if results:
        formatted += f"**Found {len(results)} relevant sources:**\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            content = result.get('content', '')
            score = result.get('score', 0)
            
            # Truncate content to reasonable length
            # if len(content) > 300:
            #     content = content[:300] + "..."
            
            formatted += f"**{i}. {title}** (Relevance: {score:.2f})\n"
            formatted += f"   🔗 {url}\n"
            formatted += f"   📄 {content}\n\n"
    else:
        formatted += "No results found.\n"
    
    return formatted


def run_interactive_notebook_with_session_state(client, model, session_state_manager, session_state, sbx, stop_event=None, tools=None):
    logger.info(f"Starting interactive notebook with session state for {session_state_manager.session_id}")
    
    # Get conversation history from session state
    messages = session_state_manager.get_conversation_history(session_state)
    notebook = JupyterNotebook(messages)
    
    # Update execution state
    session_state_manager.update_execution_state(session_state, is_running=True, sandbox_active=True, current_phase="initializing")
    
    # Use provided tools or default to all tools
    if tools is None:
        tools = TOOLS
    
    try:
        sbx_info = sbx.get_info()
        notebook.add_sandbox_countdown(sbx_info.started_at, sbx_info.end_at)
        
        # Store sandbox info in session state
        session_state["execution_state"]["sandbox_info"] = {
            "started_at": sbx_info.started_at.isoformat(),
            "end_at": sbx_info.end_at.isoformat(),
            "timeout_seconds": int((sbx_info.end_at - sbx_info.started_at).total_seconds())
        }
        
        logger.debug(f"Added sandbox countdown: {sbx_info.started_at} to {sbx_info.end_at}")
    except Exception as e:
        logger.warning(f"Failed to get sandbox info: {str(e)}")
    
    logger.debug("Initial notebook yield in 'generating' mode")
    
    # Update notebook data in session state
    session_state_manager.update_notebook_data(session_state, notebook.data)
    
    # Save initial state
    session_state_manager.save_state(session_state)
    
    yield notebook.render(mode="generating"), notebook.data, messages
    
    max_code_output = 1000
    turns = session_state["execution_state"]["current_turn"]
    done = False
    previous_execution_had_error = False
    previous_execution_had_warnings = False
    
    logger.info(f"Starting interactive loop from turn {turns} with max_output={max_code_output}, max_turns={MAX_TURNS}")

    while not done and (turns <= MAX_TURNS) and (stop_event is None or not stop_event.is_set()):
        turns += 1
        logger.info(f"Starting turn {turns}/{MAX_TURNS}")
        
        try:
            # Update phase to generating
            session_state_manager.update_execution_state(session_state, current_phase="generating")
            
            # Refresh messages from session state before API call
            messages = session_state_manager.get_conversation_history(session_state)
            logger.debug(f"Making API call to {model} with {len(messages)} messages")

            # Prepare request data for logging
            request_data = {
                "messages": clean_messages_for_api(messages),
                "model": model,
                "tools": tools,
                "tool_choice": "auto"
            }
            
            # Prepare session metadata for Phoenix tracing
            session_metadata = {
                "turn": turns,
                "max_turns": MAX_TURNS,
                "model": model,
                "tools_count": len(tools),
                "messages_count": len(messages),
                "current_phase": "generating"
            }
            
            # Add hardware config if available
            if "hardware_config" in session_state:
                hw_config = session_state["hardware_config"]
                session_metadata.update({
                    "gpu_type": hw_config.get("gpu_type", "unknown"),
                    "cpu_cores": hw_config.get("cpu_cores", "unknown"),
                    "memory_gb": hw_config.get("memory_gb", "unknown")
                })
            
            # Wrap OpenAI API call with Phoenix session context for proper grouping
            with create_phoenix_session_context(
                session_id=session_state_manager.session_id,
                user_id=None,  # Could be extracted from request context if available
                metadata=session_metadata
            ):
                logger.debug(f"Making OpenAI API call with Phoenix session context: {session_state_manager.session_id}")
                response = client.chat.completions.create(**request_data)
                logger.debug("API call successful within Phoenix session context")
            
            # Log the complete LLM interaction
            session_state_manager.log_llm_interaction(
                session_state, request_data, response.model_dump(), model, turns
            )
        except Exception as e:
            # Handle inference client errors
            logger.error(f"Inference failed on turn {turns}: {str(e)}")
            
            # Add detailed error information to the notebook
            error_message = str(e)
            if "429" in error_message or "too_many_requests" in error_message.lower():
                detailed_error = f"""**API Rate Limit Exceeded** 🚫

The inference service has reached its rate limit. This typically means:
- Too many requests have been sent in a short period
- Daily quota has been exceeded
- Service is temporarily overloaded

**What you can try:**
- Wait a few minutes and try again
- If using Cerebras API, check your daily quota
- Try using a different model or service
- Contact support if the issue persists

**Technical details:**
```
{error_message}
```"""
            elif "401" in error_message or "unauthorized" in error_message.lower():
                detailed_error = f"""**Authentication Error** 🔐

There's an issue with API authentication:
- API key might be missing or invalid
- API key might have expired
- Insufficient permissions

**Technical details:**
```
{error_message}
```"""
            elif "500" in error_message or "internal" in error_message.lower():
                detailed_error = f"""**Server Error** 🔧

The inference service encountered an internal error:
- Service might be temporarily unavailable
- Try again in a few moments
- If the issue persists, it's likely a service-side problem

**Technical details:**
```
{error_message}
```"""
            else:
                detailed_error = f"""**Inference Service Error** ⚠️

An error occurred while communicating with the AI service:

**Technical details:**
```
{error_message}
```

**What you can try:**
- Check your internet connection
- Try again in a few moments
- If the problem persists, contact support"""
            
            notebook.add_error(detailed_error)
            
            # Add error to session state
            session_state_manager.add_message(
                session_state, "assistant", detailed_error, 
                metadata={"type": "error", "error_type": "api_error", "turn": turns}
            )
            
            # Update execution state
            session_state_manager.update_execution_state(
                session_state, is_running=False, last_execution_successful=False
            )
            
            # Update notebook data and save state
            session_state_manager.update_notebook_data(session_state, notebook.data)
            session_state_manager.save_state(session_state)
            
            yield notebook.render(mode="error"), notebook.data, messages
            return

        # Get the response content and tool calls
        full_response = response.choices[0].message.content or ""
        tool_calls = response.choices[0].message.tool_calls or []
        
        logger.debug(f"Turn {turns}: Response content length: {len(full_response)}, Tool calls: {len(tool_calls)}")

        # Add markdown cell for assistant's thinking
        if full_response.strip():
            logger.debug(f"Adding assistant response as markdown ({len(full_response)} chars)")
            notebook.add_markdown(full_response, "assistant")
        else:
            logger.debug("Skipping empty assistant response")

        # Handle tool calls and add assistant message to session state only
        if tool_calls:
            logger.info(f"Processing {len(tool_calls)} tool calls on turn {turns}")
            # Add assistant message to session state (messages will be derived from this)
            session_state_manager.add_message(
                session_state, "assistant", full_response,
                tool_calls=[{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in tool_calls],
                metadata={"turn": turns, "type": "thinking"}
            )
            logger.debug(f"Added assistant message with {len(tool_calls)} tool calls to session state")
        elif full_response.strip():
            # If no tool calls but we have content, add regular assistant message
            session_state_manager.add_message(
                session_state, "assistant", full_response,
                metadata={"turn": turns, "type": "thinking"}
            )
            logger.debug("Added regular assistant message to session state")
        
        for i, tool_call in enumerate(tool_calls):
            logger.debug(f"Processing tool call {i+1}/{len(tool_calls)}: {tool_call.function.name}")

            if tool_call.function.name == "add_and_execute_jupyter_code_cell":
                # Update phase to executing code
                session_state_manager.update_execution_state(session_state, current_phase="executing_code")
                
                logger.debug(f"Processing code execution tool call: {tool_call.id}")
                tool_args = json.loads(tool_call.function.arguments)
                code = tool_args["code"]
                logger.debug(f"Code to execute: {len(code)} characters")
                
                # Determine if we should reuse the last cell or create a new one
                # Reuse if there were errors (not just warnings) in the previous execution
                should_reuse_cell = (previous_execution_had_error and 
                                   notebook.get_last_cell_type() == "code")
                
                if should_reuse_cell:
                    logger.info("Reusing last code cell due to previous execution error")
                    # Update the existing cell's code instead of creating a new one
                    notebook.update_last_code_cell(code)
                else:
                    logger.debug("Creating new code cell")
                    # Create a new cell (normal behavior)
                    notebook.add_code(code)
                
                logger.debug("Yielding notebook in 'executing' mode")
                yield notebook.render(mode="executing"), notebook.data, messages

                try:
                    # Check for stop event before execution
                    if stop_event and stop_event.is_set():
                        logger.info("Stop event detected before code execution")
                        stopped_message = """**Execution Stopped** ⏸️

The execution was stopped by user request before the code could run."""
                        notebook.add_markdown(stopped_message, "assistant")
                        yield notebook.render(mode="stopped"), notebook.data, messages
                        return
                    
                    # Execution sandbox call - might timeout
                    logger.info("Executing code in sandbox")
                    execution = sbx.run_code(code)
                    notebook.append_execution(execution)
                    
                    # Update error and warning tracking for next iteration
                    previous_execution_had_error = notebook.has_execution_error(execution)
                    previous_execution_had_warnings = notebook.has_execution_warnings(execution)
                    # Log tool execution in session state
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_response_content = parse_exec_result_llm(execution, max_code_output=max_code_output)
                    session_state_manager.log_tool_execution(
                        session_state, tool_call.id, "add_and_execute_jupyter_code_cell",
                        tool_args, tool_response_content, execution
                    )
                    
                    if previous_execution_had_error:
                        logger.warning("Code execution resulted in error")
                    elif previous_execution_had_warnings:
                        logger.info("Code execution completed with warnings")
                    else:
                        logger.info("Code execution completed successfully")
                    
                except Exception as e:
                    # Handle sandbox timeout/execution errors
                    logger.error(f"Code execution failed: {str(e)}")
                    
                    # Add detailed error information for code execution failures
                    error_message = str(e)
                    if "timeout" in error_message.lower():
                        detailed_error = f"""**Code Execution Timeout** ⏰

The code execution took too long and was terminated:
- Code may have entered an infinite loop
- Processing large datasets can cause timeouts
- Complex computations may exceed time limits

**What you can try:**
- Optimize your code for better performance
- Break down complex operations into smaller steps
- Increase the timeout limit in settings
- Check for infinite loops or blocking operations

**Technical details:**
```
{error_message}
```"""
                    else:
                        detailed_error = f"""**Code Execution Failed** 💥

An error occurred while executing the code in the sandbox:

**Technical details:**
```
{error_message}
```

**What you can try:**
- Check the code for syntax errors
- Verify all required packages are available
- Try simplifying the code
- Check the sandbox logs for more details"""
                    
                    notebook.add_error(detailed_error)
                    yield notebook.render(mode="error"), notebook.data, messages
                    return

                # Prepare tool response (already computed above)
                raw_execution = notebook.parse_exec_result_nb(execution)
                
                logger.debug(f"Tool response: {len(tool_response_content)} chars content, {len(raw_execution)} raw outputs")

                # Add tool response to session state only
                session_state_manager.add_message(
                    session_state, "tool", tool_response_content,
                    tool_call_id=tool_call.id, raw_execution=raw_execution,
                    metadata={"turn": turns, "execution_successful": not previous_execution_had_error}
                )
            elif tool_call.function.name == "tavily_search":
                # Update phase to searching
                session_state_manager.update_execution_state(session_state, current_phase="searching")
                
                logger.debug(f"Processing search tool call: {tool_call.id}")
                tool_args = json.loads(tool_call.function.arguments)
                query = tool_args["query"]
                logger.debug(f"Search query: '{query}' ({len(query)} chars)")
                
                # Add search status to notebook
                notebook.add_markdown("🔍 **Searching the web...**", "assistant")
                yield notebook.render(mode="generating"), notebook.data, messages
                
                try:
                    # Perform search
                    search_results = tavily_search(query)
                    logger.info("Search completed successfully")
                    
                    # Log search tool execution
                    tool_args = json.loads(tool_call.function.arguments)
                    session_state_manager.log_tool_execution(
                        session_state, tool_call.id, "tavily_search",
                        tool_args, search_results
                    )
                    
                    # Add search results to notebook
                    notebook.add_markdown(search_results, "assistant")
                    
                    # Add tool response to session state only
                    session_state_manager.add_message(
                        session_state, "tool", search_results,
                        tool_call_id=tool_call.id,
                        metadata={"turn": turns, "search_successful": True}
                    )
                    
                except Exception as e:
                    error_message = f"❌ Search failed: {str(e)}"
                    logger.error(f"Search tool call failed: {str(e)}")
                    
                    # Log failed search
                    tool_args = json.loads(tool_call.function.arguments)
                    session_state_manager.log_tool_execution(
                        session_state, tool_call.id, "tavily_search",
                        tool_args, error_message
                    )
                    
                    # Add error to notebook
                    notebook.add_markdown(error_message, "assistant")
                    
                    # Add error response to session state only
                    session_state_manager.add_message(
                        session_state, "tool", error_message,
                        tool_call_id=tool_call.id,
                        metadata={"turn": turns, "search_successful": False, "error": str(e)}
                    )
            elif tool_call.function.name == "edit_and_execute_current_cell":
                # Update phase to executing code
                session_state_manager.update_execution_state(session_state, current_phase="executing_code")
                
                logger.debug(f"Processing edit current cell tool call: {tool_call.id}")
                tool_args = json.loads(tool_call.function.arguments)
                code = tool_args["code"]
                logger.debug(f"Code to execute in current cell: {len(code)} characters")
                
                # Check if we have a code cell to edit
                if notebook.get_last_cell_type() == "code":
                    logger.info("Editing last code cell with new code")
                    notebook.update_last_code_cell(code)
                else:
                    logger.info("No code cell to edit, creating new cell")
                    notebook.add_code(code)
                
                logger.debug("Yielding notebook in 'executing' mode")
                yield notebook.render(mode="executing"), notebook.data, messages

                try:
                    # Check for stop event before execution
                    if stop_event and stop_event.is_set():
                        logger.info("Stop event detected before code execution")
                        stopped_message = """**Execution Stopped** ⏸️

The execution was stopped by user request before the code could run."""
                        notebook.add_markdown(stopped_message, "assistant")
                        yield notebook.render(mode="stopped"), notebook.data, messages
                        return
                    
                    # Execution sandbox call - might timeout
                    logger.info("Executing edited code in sandbox")
                    execution = sbx.run_code(code)
                    notebook.append_execution(execution)
                    
                    # Update error and warning tracking for next iteration
                    previous_execution_had_error = notebook.has_execution_error(execution)
                    previous_execution_had_warnings = notebook.has_execution_warnings(execution)
                    # Log tool execution in session state
                    tool_response_content = parse_exec_result_llm(execution, max_code_output=max_code_output)
                    session_state_manager.log_tool_execution(
                        session_state, tool_call.id, "edit_and_execute_current_cell",
                        tool_args, tool_response_content, execution
                    )
                    
                    if previous_execution_had_error:
                        logger.warning("Edited code execution resulted in error")
                    elif previous_execution_had_warnings:
                        logger.info("Edited code execution completed with warnings")
                    else:
                        logger.info("Edited code execution completed successfully")
                    
                except Exception as e:
                    # Handle sandbox timeout/execution errors
                    logger.error(f"Edited code execution failed: {str(e)}")
                    
                    # Add detailed error information for code execution failures
                    error_message = str(e)
                    if "timeout" in error_message.lower():
                        detailed_error = f"""**Code Execution Timeout** ⏰

The edited code execution took too long and was terminated:
- Code may have entered an infinite loop
- Processing large datasets can cause timeouts
- Complex computations may exceed time limits

**What you can try:**
- Optimize your code for better performance
- Break down complex operations into smaller steps
- Increase the timeout limit in settings
- Check for infinite loops or blocking operations

**Technical details:**
```
{error_message}
```"""
                    else:
                        detailed_error = f"""**Code Execution Failed** 💥

An error occurred while executing the edited code in the sandbox:

**Technical details:**
```
{error_message}
```

**What you can try:**
- Check the code for syntax errors
- Verify all required packages are available
- Try simplifying the code
- Check the sandbox logs for more details"""
                    
                    notebook.add_error(detailed_error)
                    yield notebook.render(mode="error"), notebook.data, messages
                    return

                # Prepare tool response
                raw_execution = notebook.parse_exec_result_nb(execution)
                
                logger.debug(f"Tool response: {len(tool_response_content)} chars content, {len(raw_execution)} raw outputs")

                # Add tool response to session state only
                session_state_manager.add_message(
                    session_state, "tool", tool_response_content,
                    tool_call_id=tool_call.id, raw_execution=raw_execution,
                    metadata={"turn": turns, "execution_successful": not previous_execution_had_error, "action": "edit_cell"}
                )
            elif tool_call.function.name == "execute_shell_command":
                # Update phase to executing shell command
                session_state_manager.update_execution_state(session_state, current_phase="executing_shell")
                
                logger.debug(f"Processing shell command tool call: {tool_call.id}")
                tool_args = json.loads(tool_call.function.arguments)
                command = tool_args["command"]
                logger.debug(f"Shell command to execute: '{command}'")
                
                # Add shell command to notebook with special styling
                notebook.add_shell_command(command)
                
                logger.debug("Yielding notebook in 'executing' mode")
                yield notebook.render(mode="executing"), notebook.data, messages

                try:
                    # Check for stop event before execution
                    if stop_event and stop_event.is_set():
                        logger.info("Stop event detected before shell execution")
                        stopped_message = """**Execution Stopped** ⏸️

The execution was stopped by user request before the shell command could run."""
                        notebook.add_markdown(stopped_message, "assistant")
                        yield notebook.render(mode="stopped"), notebook.data, messages
                        return
                    
                    # Execute shell command in sandbox using raw shell execution
                    logger.info(f"Executing raw shell command in sandbox: {command}")
                    
                    try:
                        # Use the new raw shell execution method
                        if hasattr(sbx, 'run_shell'):
                            shell_execution = sbx.run_shell(command, timeout=30)
                            logger.info("Shell command executed using raw shell method")
                        else:
                            # Fallback: Execute shell command using Python subprocess within sandbox
                            shell_code = f"""
import subprocess
import sys

try:
    result = subprocess.run(
        {repr(command)}, 
        shell=True, 
        capture_output=True, 
        text=True, 
        timeout=30
    )
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Exit code: {{result.returncode}}")
    
except subprocess.TimeoutExpired:
    print("Error: Command timed out after 30 seconds")
except Exception as e:
    print(f"Error executing command: {{e}}")
"""
                            shell_execution = sbx.run_code(shell_code)
                            logger.info("Shell command executed via Python subprocess fallback")
                        
                        # Add shell execution results to notebook
                        notebook.append_shell_execution(shell_execution)
                        
                        # Prepare response content for LLM
                        shell_response_content = parse_exec_result_llm(shell_execution, max_code_output=max_code_output)
                        
                        # Log tool execution in session state
                        session_state_manager.log_tool_execution(
                            session_state, tool_call.id, "execute_shell_command",
                            tool_args, shell_response_content, shell_execution
                        )
                        
                        # Check for errors
                        shell_had_error = notebook.has_execution_error(shell_execution)
                        
                        if shell_had_error:
                            logger.warning("Shell command execution resulted in error")
                        else:
                            logger.info("Shell command execution completed successfully")
                        
                    except Exception as shell_error:
                        logger.error(f"Shell command execution failed: {str(shell_error)}")
                        
                        # Create error message
                        detailed_error = f"""**Shell Command Failed** 🔧

An error occurred while executing the shell command:

**Command:** `{command}`

**Technical details:**
```
{str(shell_error)}
```

**What you can try:**
- Check if the command exists in the sandbox environment
- Verify command syntax
- Try a simpler version of the command
- Check if required tools/packages are installed"""
                        
                        notebook.add_error(detailed_error)
                        
                        # Log failed execution
                        session_state_manager.log_tool_execution(
                            session_state, tool_call.id, "execute_shell_command",
                            tool_args, detailed_error
                        )
                        
                        yield notebook.render(mode="error"), notebook.data, messages
                        return
                    
                except Exception as e:
                    # Handle general execution errors
                    logger.error(f"Shell command execution failed: {str(e)}")
                    
                    detailed_error = f"""**Shell Execution Error** ⚠️

An unexpected error occurred while executing the shell command:

**Command:** `{command}`

**Technical details:**
```
{str(e)}
```"""
                    
                    notebook.add_error(detailed_error)
                    yield notebook.render(mode="error"), notebook.data, messages
                    return

                # Prepare tool response for LLM and session state
                raw_execution = notebook.parse_exec_result_nb(shell_execution)
                
                logger.debug(f"Shell tool response: {len(shell_response_content)} chars content")

                # Add tool response to session state
                session_state_manager.add_message(
                    session_state, "tool", shell_response_content,
                    tool_call_id=tool_call.id, raw_execution=raw_execution,
                    metadata={"turn": turns, "command": command, "execution_successful": not shell_had_error, "action": "shell_command"}
                )
            else:
                logger.warning(f"Unknown tool call function: {tool_call.function.name}")

        if not tool_calls:
            logger.info(f"No tool calls on turn {turns}, conversation ending")
            if len(full_response.strip())==0:
                logger.error("Assistant provided no content and no tool calls")
                notebook.add_error(f"No tool call and empty assistant response:\n{response.model_dump_json(indent=2)}")

            # Only add the final assistant message if we didn't already add it above
            # (in the elif full_response.strip() block)
            if full_response.strip():
                # Since we're now only using session state, we can safely add the message
                # The session state manager will handle any deduplication if needed
                session_state_manager.add_message(
                    session_state, "assistant", full_response,
                    metadata={"turn": turns, "type": "final_response"}
                )
                logger.debug("Added final assistant response to session state")
            
            done = True
            
        # Update session state after each turn
        session_state_manager.update_execution_state(
            session_state, current_turn=turns, last_execution_successful=not previous_execution_had_error
        )
        session_state_manager.update_notebook_data(session_state, notebook.data)
        session_state_manager.save_state(session_state)
        
        if done:
            logger.info(f"Interactive notebook completed after {turns} turns")
            session_state_manager.update_execution_state(
                session_state, is_running=False, sandbox_active=True
            )
            session_state_manager.save_state(session_state)
            yield notebook.render(mode="done"), notebook.data, messages
        else:
            logger.debug(f"Turn {turns} completed, yielding in 'generating' mode")
            yield notebook.render(mode="generating"), notebook.data, messages
    
    if turns > MAX_TURNS:
        logger.warning(f"Interactive notebook reached maximum turns ({MAX_TURNS})")
        error_msg = f"**Maximum Turns Reached** 🔄\n\nThe conversation has reached the maximum number of turns ({MAX_TURNS}). This is a safety limit to prevent infinite loops.\n\n**What you can try:**\n- Start a new conversation\n- Clear the notebook and begin fresh\n- Contact support if you need a higher turn limit"
        notebook.add_error(error_msg)
        
        # Add error to session state
        session_state_manager.add_message(
            session_state, "assistant", error_msg,
            metadata={"type": "error", "error_type": "max_turns_exceeded", "turn": turns}
        )
        
        # Update final state
        session_state_manager.update_execution_state(
            session_state, is_running=False, last_execution_successful=False
        )
        session_state_manager.update_notebook_data(session_state, notebook.data)
        session_state_manager.save_state(session_state)
        
        yield notebook.render(mode="error"), notebook.data, messages
    elif stop_event and stop_event.is_set():
        logger.info("Interactive notebook stopped by user")
        
        # Add a stopped message to the notebook
        stopped_message = """**Execution Stopped** ⏸️

The execution was stopped by user request. You can resume by clicking Run again."""
        notebook.add_markdown(stopped_message, "assistant")
        
        # Add stopped message to session state
        session_state_manager.add_message(
            session_state, "assistant", stopped_message,
            metadata={"type": "status", "status_type": "stopped_by_user", "turn": turns}
        )
        
        # Update state to indicate pause
        session_state_manager.update_execution_state(
            session_state, is_running=False, is_paused=True
        )
        session_state_manager.update_notebook_data(session_state, notebook.data)
        session_state_manager.save_state(session_state)
        
        yield notebook.render(mode="stopped"), notebook.data, messages


def run_interactive_notebook(client, model, messages, sbx, stop_event=None, tools=None):
    """Backward compatibility wrapper for the new session state system"""
    logger.warning("Using legacy run_interactive_notebook - this should be replaced with session state version")
    
    # Create a temporary session for backward compatibility
    import uuid
    temp_session_id = str(uuid.uuid4())[:8]
    session_manager = SessionStateManager(temp_session_id)
    
    # Create basic session state
    session_state = session_manager.create_initial_state(
        hardware_config={"gpu_type": "unknown", "cpu_cores": 2, "memory_gb": 8, "timeout_sec": 300},
        api_config={"model_name": model, "provider_type": "unknown"},
        environment={"variables": "", "files_uploaded": []},
        system_prompt=messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""
    )
    
    # Initialize conversation history with provided messages
    session_state["conversation_history"] = messages
    
    # Use the new session-based function
    yield from run_interactive_notebook_with_session_state(
        client, model, session_manager, session_state, sbx, stop_event, tools
    )