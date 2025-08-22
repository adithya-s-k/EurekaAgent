from jupyter_handler import JupyterNotebook
import json
import logging
import os
import datetime
from tavily import TavilyClient

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
    Create a clean copy of messages without raw_execution fields for API calls.
    This prevents 413 errors caused by large execution data.
    """
    logger.debug(f"Cleaning {len(messages)} messages for API call")
    cleaned_messages = []
    raw_execution_count = 0
    for message in messages:
        cleaned_message = message.copy()
        if "raw_execution" in cleaned_message:
            cleaned_message.pop("raw_execution")
            raw_execution_count += 1
        cleaned_messages.append(cleaned_message)
    
    logger.debug(f"Removed raw_execution from {raw_execution_count} messages")
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


def run_interactive_notebook(client, model, messages, sbx, stop_event=None, tools=None):
    logger.info(f"Starting interactive notebook with {len(messages)} initial messages")
    notebook = JupyterNotebook(messages)
    
    # Use provided tools or default to all tools
    if tools is None:
        tools = TOOLS
    
    try:
        sbx_info = sbx.get_info()
        notebook.add_sandbox_countdown(sbx_info.started_at, sbx_info.end_at)
        logger.debug(f"Added sandbox countdown: {sbx_info.started_at} to {sbx_info.end_at}")
    except Exception as e:
        logger.warning(f"Failed to get sandbox info: {str(e)}")
    
    logger.debug("Initial notebook yield in 'generating' mode")
    yield notebook.render(mode="generating"), notebook.data, messages
    
    max_code_output = 1000
    turns = 0
    done = False
    previous_execution_had_error = False
    previous_execution_had_warnings = False
    
    logger.info(f"Starting interactive loop with max_output={max_code_output}, max_turns={MAX_TURNS}")

    while not done and (turns <= MAX_TURNS) and (stop_event is None or not stop_event.is_set()):
        turns += 1
        logger.info(f"Starting turn {turns}/{MAX_TURNS}")
        
        try:
            # Inference client call - might fail
            logger.debug(f"Making API call to {model} with {len(messages)} messages")
            response = client.chat.completions.create(
                messages=clean_messages_for_api(messages),
                model=model,
                tools=tools,
                tool_choice="auto",
            )
            logger.debug("API call successful")
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

        # Handle tool calls
        if tool_calls:
            logger.info(f"Processing {len(tool_calls)} tool calls on turn {turns}")
        
        for i, tool_call in enumerate(tool_calls):
            logger.debug(f"Processing tool call {i+1}/{len(tool_calls)}: {tool_call.function.name}")
            # Add assistant message with tool call to history
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
            messages.append(assistant_message)
            logger.debug("Added assistant message with tool call to history")

            if tool_call.function.name == "add_and_execute_jupyter_code_cell":
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

                # Prepare tool response
                tool_response_content = parse_exec_result_llm(execution, max_code_output=max_code_output)
                raw_execution = notebook.parse_exec_result_nb(execution)
                
                logger.debug(f"Tool response: {len(tool_response_content)} chars content, {len(raw_execution)} raw outputs")
                
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response_content,
                        "raw_execution": raw_execution
                    }
                )
            elif tool_call.function.name == "tavily_search":
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
                    
                    # Add search results to notebook
                    notebook.add_markdown(search_results, "assistant")
                    
                    # Add tool response to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_results
                        }
                    )
                    
                except Exception as e:
                    error_message = f"❌ Search failed: {str(e)}"
                    logger.error(f"Search tool call failed: {str(e)}")
                    
                    # Add error to notebook
                    notebook.add_markdown(error_message, "assistant")
                    
                    # Add error response to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_message
                        }
                    )
            else:
                logger.warning(f"Unknown tool call function: {tool_call.function.name}")

        if not tool_calls:
            logger.info(f"No tool calls on turn {turns}, conversation ending")
            if len(full_response.strip())==0:
                logger.error("Assistant provided no content and no tool calls")
                notebook.add_error(f"No tool call and empty assistant response:\n{response.model_dump_json(indent=2)}")
            messages.append({"role": "assistant", "content": full_response})
            done = True
            
        if done:
            logger.info(f"Interactive notebook completed after {turns} turns")
            yield notebook.render(mode="done"), notebook.data, messages
        else:
            logger.debug(f"Turn {turns} completed, yielding in 'generating' mode")
            yield notebook.render(mode="generating"), notebook.data, messages
    
    if turns > MAX_TURNS:
        logger.warning(f"Interactive notebook reached maximum turns ({MAX_TURNS})")
        notebook.add_error(f"**Maximum Turns Reached** 🔄\n\nThe conversation has reached the maximum number of turns ({MAX_TURNS}). This is a safety limit to prevent infinite loops.\n\n**What you can try:**\n- Start a new conversation\n- Clear the notebook and begin fresh\n- Contact support if you need a higher turn limit")
        yield notebook.render(mode="error"), notebook.data, messages
    elif stop_event and stop_event.is_set():
        logger.info("Interactive notebook stopped by user")
        # Add a stopped message to the notebook
        stopped_message = """**Execution Stopped** ⏸️

The execution was stopped by user request. You can:
- Click "▶️ Continue" to resume from where you left off
- Start a new task by typing a new message
- Clear the notebook to start fresh"""
        notebook.add_markdown(stopped_message, "assistant")
        yield notebook.render(mode="stopped"), notebook.data, messages