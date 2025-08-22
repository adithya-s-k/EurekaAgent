import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config
import json
import copy
from jinja2 import DictLoader
import datetime
import logging

# Configure logging for jupyter_handler module
logger = logging.getLogger(__name__)


system_template = """\
<details>
  <summary style="display: flex; align-items: center; cursor: pointer; margin-bottom: 12px;">
    <h3 style="color: #374151; margin: 0; margin-right: 8px; font-size: 14px; font-weight: 600;">System</h3>
    <span class="arrow" style="margin-right: 12px; font-size: 12px;">▶</span>
    <div style="flex: 1; height: 2px; background-color: #374151;"></div>
  </summary>
  <div style="margin-top: 8px; padding: 8px; background-color: #f9fafb; border-radius: 4px; border-left: 3px solid #374151; margin-bottom: 16px;">
    {}
  </div>
</details>

<style>
details > summary .arrow {{
  display: inline-block;
  transition: transform 0.2s;
}}
details[open] > summary .arrow {{
  transform: rotate(90deg);
}}
details > summary {{
  list-style: none;
}}
details > summary::-webkit-details-marker {{
  display: none;
}}
</style>
"""

user_template = """\
<div style="display: flex; align-items: center; margin-bottom: 12px;">
    <h3 style="color: #166534; margin: 0; margin-right: 12px; font-size: 14px; font-weight: 600;">User</h3>
    <div style="flex: 1; height: 2px; background-color: #166534;"></div>
</div>
<div style="margin-bottom: 16px;">{}</div>"""

assistant_thinking_template = """\
<div style="display: flex; align-items: center; margin-bottom: 12px;">
    <h3 style="color: #1d5b8e; margin: 0; margin-right: 12px; font-size: 14px; font-weight: 600;">Assistant</h3>
    <div style="flex: 1; height: 2px; background-color: #1d5b8e;"></div>
</div>
<div style="margin-bottom: 16px;">{}</div>"""

assistant_final_answer_template = """<div class="alert alert-block alert-warning">
<b>Assistant:</b> Final answer: {}
</div>
"""

header_message = """<div style="text-align: center; padding: 24px 16px; margin-bottom: 24px;">
  <h1 style="color: #1e3a8a; font-size: 48px; font-weight: 700; margin: 0 0 8px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    🔬 Eureka Agent
  </h1>
  <p style="color: #6b7280; font-size: 11px; margin: 0; display: flex; align-items: center; justify-content: center; gap: 6px;">
    <img style="height: 16px; width: auto; opacity: 0.7;" 
         src="https://huggingface.co/spaces/lvwerra/jupyter-agent-2/resolve/main/jupyter-agent-2.png" 
         alt="Jupyter Agent 2" />
    <span>Forked from Jupyter Agent 2</span>
  </p>
</div>
"""

bad_html_bad = """input[type="file"] {
  display: block;
}"""


EXECUTING_WIDGET = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #e3f2fd; border-radius: 6px; border-left: 3px solid #2196f3;">
    <div style="display: flex; gap: 4px;">
        <div style="width: 6px; height: 6px; background-color: #2196f3; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite;"></div>
        <div style="width: 6px; height: 6px; background-color: #2196f3; border-radius: 50%; animation: pulse 1.5s ease-in-out 0.1s infinite;"></div>
        <div style="width: 6px; height: 6px; background-color: #2196f3; border-radius: 50%; animation: pulse 1.5s ease-in-out 0.2s infinite;"></div>
    </div>
    <span style="color: #1976d2; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Executing code...
    </span>
</div>

<style>
@keyframes pulse {
    0%, 80%, 100% {
        opacity: 0.3;
        transform: scale(0.8);
    }
    40% {
        opacity: 1;
        transform: scale(1);
    }
}
</style>
"""

GENERATING_WIDGET = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #f3e5f5; border-radius: 6px; border-left: 3px solid #9c27b0;">
    <div style="width: 80px; height: 4px; background-color: #e1bee7; border-radius: 2px; overflow: hidden;">
        <div style="width: 30%; height: 100%; background-color: #9c27b0; border-radius: 2px; animation: progress 2s ease-in-out infinite;"></div>
    </div>
    <span style="color: #7b1fa2; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Generating...
    </span>
</div>

<style>
@keyframes progress {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(250%); }
}
</style>
"""

DONE_WIDGET = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #e8f5e8; border-radius: 6px; border-left: 3px solid #4caf50;">
    <div style="width: 16px; height: 16px; background-color: #4caf50; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
        <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
            <path d="M1 4L3.5 6.5L9 1" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    <span style="color: #2e7d32; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Generation complete
    </span>
</div>
"""

DONE_WIDGET = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #e8f5e8; border-radius: 6px; border-left: 3px solid #4caf50; animation: fadeInOut 4s ease-in-out forwards;">
    <div style="width: 16px; height: 16px; background-color: #4caf50; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
        <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
            <path d="M1 4L3.5 6.5L9 1" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    <span style="color: #2e7d32; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Generation complete
    </span>
</div>

<style>
@keyframes fadeInOut {
    0% { opacity: 0; transform: translateY(10px); }
    15% { opacity: 1; transform: translateY(0); }
    85% { opacity: 1; transform: translateY(0); }
    100% { opacity: 0; transform: translateY(-10px); }
}
</style>
"""

STOPPED_WIDGET = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #fff3e0; border-radius: 6px; border-left: 3px solid #ff9800;">
    <div style="width: 16px; height: 16px; background-color: #ff9800; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 10px;">
        ⏸
    </div>
    <span style="color: #f57c00; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Execution stopped by user
    </span>
</div>
"""

ERROR_WIDGET = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #ffebee; border-radius: 6px; border-left: 3px solid #f44336;">
    <div style="width: 16px; height: 16px; background-color: #f44336; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 10px;">
        ⚠
    </div>
    <span style="color: #c62828; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Execution failed - check error details above
    </span>
</div>
"""

ERROR_HTML = """\
<div style="display: flex; align-items: center; gap: 8px; padding: 12px; background-color: #ffebee; border-radius: 6px; border-left: 3px solid #f44336; margin: 8px 0;">
    <div style="width: 20px; height: 20px; background-color: #f44336; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">
        !
    </div>
    <div style="color: #c62828; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <strong>Error:</strong> {}
    </div>
</div>"""

STOPPED_SANDBOX_HTML = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #f5f5f5; border-radius: 6px; border-left: 3px solid #9e9e9e; margin-bottom: 16px;">
    <div style="width: 16px; height: 16px; background-color: #9e9e9e; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 10px;">
        ⏹
    </div>
    <div style="flex: 1;">
        <div style="margin-bottom: 4px; font-size: 13px; color: #757575; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-weight: 500;">
            Sandbox stopped
        </div>
        <div style="width: 100%; height: 8px; background-color: #e0e0e0; border-radius: 4px; overflow: hidden;">
            <div style="height: 100%; background-color: #9e9e9e; border-radius: 4px; width: 100%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 11px; color: #757575; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <span>Started: {start_time}</span>
            <span>Expired: {end_time}</span>
        </div>
    </div>
</div>
"""

TIMEOUT_HTML = """
<div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #fff3e0; border-radius: 6px; border-left: 3px solid #ff9800; margin-bottom: 16px;">
    <div style="width: 16px; height: 16px; background-color: #ff9800; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 10px;">
        ⏱
    </div>
    <div style="flex: 1;">
        <div style="margin-bottom: 4px; font-size: 13px; color: #f57c00; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-weight: 500;">
            The E2B Sandbox for code execution has a timeout of {total_seconds} seconds.
        </div>
        <div style="width: 100%; height: 8px; background-color: #ffe0b3; border-radius: 4px; overflow: hidden;">
            <div id="progress-bar-{unique_id}" style="height: 100%; background: linear-gradient(90deg, #ff9800 0%, #f57c00 50%, #f44336 100%); border-radius: 4px; width: {current_progress}%; animation: progress-fill-{unique_id} {remaining_seconds}s linear forwards;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 11px; color: #f57c00; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <span>Started: {start_time}</span>
            <span>Expires: {end_time}</span>
        </div>
    </div>
</div>

<style>
@keyframes progress-fill-{unique_id} {{
    from {{ width: {current_progress}%; }}
    to {{ width: 100%; }}
}}
</style>
"""

TIMEOUT_HTML = """
<div style="display: flex; align-items: center; gap: 8px; padding: 6px 10px; background-color: #fafafa; border-radius: 4px; border-left: 2px solid #d1d5db; margin-bottom: 8px; font-size: 12px;">
    <div style="width: 12px; height: 12px; background-color: #d1d5db; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 8px;">
        ⏱
    </div>
    <div style="flex: 1;">
        <div style="margin-bottom: 2px; font-size: 11px; color: #6b7280; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-weight: 400;">
            Sandbox timeout: {total_seconds}s
        </div>
        <div style="width: 100%; height: 6px; background-color: #e5e7eb; border-radius: 3px; overflow: hidden;">
            <div id="progress-bar-{unique_id}" style="height: 100%; background-color: #6b7280; border-radius: 3px; width: {current_progress}%; animation: progress-fill-{unique_id} {remaining_seconds}s linear forwards;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 2px; font-size: 10px; color: #9ca3af; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <span>Started: {start_time}</span>
            <span>Expires: {end_time}</span>
        </div>
    </div>
</div>

<style>
@keyframes progress-fill-{unique_id} {{
    from {{ width: {current_progress}%; }}
    to {{ width: 100%; }}
}}
</style>
"""

# just make the code font a bit smaller
custom_css = """
<style type="text/css">
/* Code font size */
.highlight pre, .highlight code,
div.input_area pre, div.output_area pre {
    font-size: 12px !important;
    line-height: 1.4 !important;
}

/* Fix prompt truncation */
.jp-InputPrompt, .jp-OutputPrompt {
    text-overflow: clip !important;
}
</style>
"""

# Configure the exporter
config = Config()
html_exporter = HTMLExporter(config=config, template_name="classic")


class JupyterNotebook:
    def __init__(self, messages=None, session_state_data=None):
        self.exec_count = 0
        self.countdown_info = None
        
        # If session_state_data is provided, use it directly
        if session_state_data and "notebook_data" in session_state_data:
            logger.info("Initializing JupyterNotebook from session state")
            self.data = session_state_data["notebook_data"]
            # Count existing code cells to maintain execution count
            self.exec_count = len([cell for cell in self.data.get("cells", []) 
                                 if cell.get("cell_type") == "code" and cell.get("execution_count")])
            logger.info(f"JupyterNotebook initialized from session state with {len(self.data['cells'])} cells, exec_count={self.exec_count}")
            return
        
        # Legacy initialization path
        if messages is None:
            messages = []
        logger.debug(f"Initializing JupyterNotebook with {len(messages)} messages")
        self.data, self.code_cell_counter = self.create_base_notebook(messages)
        logger.info(f"JupyterNotebook initialized with {len(self.data['cells'])} cells")


    def create_base_notebook(self, messages):
        logger.debug("Creating base notebook structure")
        base_notebook = {
            "metadata": {
                "kernel_info": {"name": "python3"},
                "language_info": {
                    "name": "python",
                    "version": "3.12",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 0,
            "cells": []
        }
        
        # Add header
        base_notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": header_message
        })
        logger.debug("Added header cell to notebook")

        # Set initial data
        self.data = base_notebook
        
        # Add empty code cell if no messages
        if len(messages) == 0:
            self.data["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": "",
                "outputs": []
            })
            logger.debug("Added empty code cell for new notebook")
            return self.data, 0

        # Process messages using existing methods
        logger.info(f"Processing {len(messages)} messages for notebook creation")
        i = 0
        while i < len(messages):
            message = messages[i]
            logger.debug(f"Processing message {i+1}/{len(messages)}: {message['role']}")
            
            if message["role"] == "system":
                logger.debug("Adding system message as markdown")
                self.add_markdown(message["content"], "system")
                
            elif message["role"] == "user":
                logger.debug("Adding user message as markdown")
                self.add_markdown(message["content"], "user")
                
            elif message["role"] == "assistant":
                if "tool_calls" in message:
                    logger.debug(f"Processing assistant message with {len(message['tool_calls'])} tool calls")
                    # Add assistant thinking if there's content
                    if message.get("content"):
                        logger.debug("Adding assistant thinking content")
                        self.add_markdown(message["content"], "assistant")
                    
                    # Process tool calls - we know the next message(s) will be tool responses
                    for tool_call in message["tool_calls"]:
                        if tool_call["function"]["name"] == "add_and_execute_jupyter_code_cell":
                            logger.debug(f"Processing code execution tool call: {tool_call['id']}")
                            tool_args = json.loads(tool_call["function"]["arguments"])
                            code = tool_args["code"]
                            logger.debug(f"Code cell contains {len(code)} characters")
                            
                            # Get the next tool response (guaranteed to exist)
                            tool_message = messages[i + 1]
                            if tool_message["role"] == "tool" and tool_message.get("tool_call_id") == tool_call["id"]:
                                logger.debug(f"Found matching tool response for {tool_call['id']}")
                                # Use the raw execution directly!
                                execution = tool_message["raw_execution"]
                                self.add_code_execution(code, execution, parsed=True)
                                logger.debug(f"Added code execution cell with {len(execution)} outputs")
                                i += 1  # Skip the tool message since we just processed it
                            else:
                                logger.warning(f"No matching tool response found for tool call {tool_call['id']}")
                else:
                    # Regular assistant message
                    logger.debug("Adding regular assistant message")
                    self.add_markdown(message["content"], "assistant")
                    
            elif message["role"] == "tool":
                # Skip - should have been handled with corresponding tool_calls
                # This shouldn't happen given our assumptions, but just in case
                logger.debug("Skipping tool message (should have been processed with tool_calls)")
                pass
                
            i += 1
        
        return self.data, 0

    def _update_countdown_cell(self):
        if not self.countdown_info:
            logger.debug("No countdown info available, skipping countdown update")
            return
        
        logger.debug("Updating countdown cell")
            
        start_time = self.countdown_info['start_time']
        end_time = self.countdown_info['end_time']
        
        current_time = datetime.datetime.now(datetime.timezone.utc)
        remaining_time = end_time - current_time
        
        # Show stopped message if expired
        if remaining_time.total_seconds() <= 0:
            logger.info("Sandbox has expired, showing stopped message")
            # Format display for stopped sandbox
            start_display = start_time.strftime("%H:%M")
            end_display = end_time.strftime("%H:%M")
            
            stopped_html = STOPPED_SANDBOX_HTML.format(
                start_time=start_display,
                end_time=end_display
            )
            
            # Update countdown cell to show stopped message
            stopped_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": stopped_html
            }
            
            # Find and update existing countdown cell
            for i, cell in enumerate(self.data["cells"]):
                if cell.get("cell_type") == "markdown" and ("⏱" in str(cell.get("source", "")) or "⏹" in str(cell.get("source", ""))):
                    self.data["cells"][i] = stopped_cell
                    logger.debug(f"Updated countdown cell at position {i} with stopped message")
                    break
            
            return
        
        # Calculate current progress
        total_duration = end_time - start_time
        elapsed_time = current_time - start_time
        current_progress = (elapsed_time.total_seconds() / total_duration.total_seconds()) * 100
        current_progress = max(0, min(100, current_progress))
        logger.debug(f"Countdown progress: {current_progress:.1f}% ({remaining_time.total_seconds():.0f}s remaining)")
        
        # Format display
        start_display = start_time.strftime("%H:%M")
        end_display = end_time.strftime("%H:%M")
        remaining_seconds = int(remaining_time.total_seconds())
        remaining_minutes = remaining_seconds // 60
        remaining_secs = remaining_seconds % 60
        remaining_display = f"{remaining_minutes}:{remaining_secs:02d}"
        
        # Generate unique ID to avoid CSS conflicts when updating
        unique_id = int(current_time.timestamp() * 1000) % 100000
        
        # Calculate total timeout duration in seconds
        total_seconds = int(total_duration.total_seconds())
        
        countdown_html = TIMEOUT_HTML.format(
            start_time=start_display,
            end_time=end_display,
            current_progress=current_progress,
            remaining_seconds=remaining_seconds,
            unique_id=unique_id,
            total_seconds=total_seconds
        )
        
        # Update or insert the countdown cell
        countdown_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": countdown_html
        }
        
        # Find existing countdown cell by looking for the timer emoji
        found_countdown = False
        for i, cell in enumerate(self.data["cells"]):
            if cell.get("cell_type") == "markdown" and "⏱" in str(cell.get("source", "")):
                # Update existing countdown cell
                self.data["cells"][i] = countdown_cell
                found_countdown = True
                logger.debug(f"Updated existing countdown cell at position {i}")
                break
        
        if not found_countdown:
            # Insert new countdown cell at position 1 (after header)
            self.data["cells"].insert(1, countdown_cell)
            logger.debug("Inserted new countdown cell at position 1")

    def add_sandbox_countdown(self, start_time, end_time):
        logger.info(f"Adding sandbox countdown: {start_time} to {end_time}")
        # Store the countdown info for later updates
        self.countdown_info = {
            'start_time': start_time,
            'end_time': end_time,
            'cell_index': 1  # Remember where we put it
        }

    def add_code_execution(self, code, execution, parsed=False):
        self.exec_count += 1
        logger.debug(f"Adding code execution cell #{self.exec_count} with {len(code)} chars of code")
        outputs = execution if parsed else self.parse_exec_result_nb(execution)
        logger.debug(f"Code execution has {len(outputs)} outputs")
        self.data["cells"].append({
            "cell_type": "code",
            "execution_count": self.exec_count,
            "metadata": {},
            "source": code,
            "outputs": outputs
            })
        
    def add_code(self, code):
        """Add a code cell without execution results"""
        self.exec_count += 1
        logger.debug(f"Adding code cell #{self.exec_count} with {len(code)} chars (no execution)")
        self.data["cells"].append({
            "cell_type": "code",
            "execution_count": self.exec_count,
            "metadata": {},
            "source": code,
            "outputs": []
        })

    def append_execution(self, execution):
        """Append execution results to the immediate previous cell if it's a code cell"""
        if (len(self.data["cells"]) > 0 and 
            self.data["cells"][-1]["cell_type"] == "code"):
            outputs = self.parse_exec_result_nb(execution)
            self.data["cells"][-1]["outputs"] = outputs
            logger.debug(f"Appended {len(outputs)} outputs to last code cell")
        else:
            logger.error("Cannot append execution: previous cell is not a code cell")
            raise ValueError("Cannot append execution: previous cell is not a code cell")

    def has_execution_error(self, execution):
        """Check if an execution result contains an error"""
        has_error = execution.error is not None
        logger.debug(f"Execution error check: {has_error}")
        return has_error

    def has_execution_warnings(self, execution):
        """Check if an execution result contains warnings (stderr output but no error)"""
        has_warnings = (execution.error is None and 
                       execution.logs.stderr and 
                       len(execution.logs.stderr) > 0)
        logger.debug(f"Execution warning check: {has_warnings}")
        return has_warnings

    def update_last_code_cell(self, code):
        """Update the source code of the last code cell"""
        if (len(self.data["cells"]) > 0 and 
            self.data["cells"][-1]["cell_type"] == "code"):
            logger.debug(f"Updating last code cell with {len(code)} chars")
            self.data["cells"][-1]["source"] = code
            # Clear previous outputs when updating code
            self.data["cells"][-1]["outputs"] = []
            logger.debug("Cleared previous outputs from updated code cell")
        else:
            logger.error("Cannot update: last cell is not a code cell")
            raise ValueError("Cannot update: last cell is not a code cell")

    def get_last_cell_type(self):
        """Get the type of the last cell, or None if no cells exist"""
        if len(self.data["cells"]) > 0:
            cell_type = self.data["cells"][-1]["cell_type"]
            logger.debug(f"Last cell type: {cell_type}")
            return cell_type
        logger.debug("No cells exist, returning None")
        return None
                
    def add_markdown(self, markdown, role="markdown"):
        logger.debug(f"Adding markdown cell with role '{role}' ({len(markdown)} chars)")
        if role == "system":
            system_message = markdown if markdown else "default"
            markdown_formatted = system_template.format(system_message.replace('\n', '<br>'))
        elif role == "user":
            markdown_formatted = user_template.format(markdown.replace('\n', '<br>'))
        elif role == "assistant":
            markdown_formatted = assistant_thinking_template.format(markdown)
            markdown_formatted = markdown_formatted.replace('<think>', '&lt;think&gt;')
            markdown_formatted = markdown_formatted.replace('</think>', '&lt;/think&gt;')
        else:
            # Default case for raw markdown
            markdown_formatted = markdown

        self.data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": markdown_formatted
        })

    def add_error(self, error_message):
        """Add an error message cell to the notebook"""
        logger.warning(f"Adding error cell: {error_message}")
        error_html = ERROR_HTML.format(error_message)
    
        self.data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": error_html
        })

    def add_final_answer(self, answer):
        logger.info(f"Adding final answer cell ({len(answer)} chars)")
        self.data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": assistant_final_answer_template.format(answer)
            })

    def parse_exec_result_nb(self, execution):
        """Convert an E2B Execution object to Jupyter notebook cell output format"""
        logger.debug("Parsing execution result for notebook format")
        outputs = []
        
        if execution.logs.stdout:
            stdout_text = ''.join(execution.logs.stdout)
            logger.debug(f"Adding stdout output ({len(stdout_text)} chars)")
            outputs.append({
                'output_type': 'stream',
                'name': 'stdout',
                'text': stdout_text
            })
        
        if execution.logs.stderr:
            stderr_text = ''.join(execution.logs.stderr)
            # Filter out plot data from stderr before displaying
            plot_start = stderr_text.find("__PLOT_DATA__")
            plot_end = stderr_text.find("__END_PLOT_DATA__")
            if plot_start != -1 and plot_end != -1:
                # Remove plot data from stderr text
                clean_stderr = stderr_text[:plot_start] + stderr_text[plot_end + len("__END_PLOT_DATA__"):]
                stderr_text = clean_stderr.strip()
            
            # Only add stderr output if there's content after filtering
            if stderr_text:
                logger.debug(f"Adding stderr output ({len(stderr_text)} chars)")
                outputs.append({
                    'output_type': 'stream',
                    'name': 'stderr',
                    'text': stderr_text
                })

        if execution.error:
            logger.debug(f"Adding error output: {execution.error.name}: {execution.error.value}")
            outputs.append({
                'output_type': 'error',
                'ename': execution.error.name,
                'evalue': execution.error.value,
                'traceback': [line for line in execution.error.traceback.split('\n')]
            })

        for i, result in enumerate(execution.results):
            logger.debug(f"Processing execution result {i+1}/{len(execution.results)}")
            output = {
                'output_type': 'execute_result' if result.is_main_result else 'display_data',
                'metadata': {},
                'data': {}
            }
            
            if result.text:
                output['data']['text/plain'] = result.text
            if result.html:
                output['data']['text/html'] = result.html
            if result.png:
                output['data']['image/png'] = result.png
            if result.svg:
                output['data']['image/svg+xml'] = result.svg
            if result.jpeg:
                output['data']['image/jpeg'] = result.jpeg
            if result.pdf:
                output['data']['application/pdf'] = result.pdf
            if result.latex:
                output['data']['text/latex'] = result.latex
            if result.json:
                output['data']['application/json'] = result.json
            if result.javascript:
                output['data']['application/javascript'] = result.javascript

            if result.is_main_result and execution.execution_count is not None:
                output['execution_count'] = execution.execution_count

            if output['data']:
                logger.debug(f"Added result output with data types: {list(output['data'].keys())}")
                outputs.append(output)
            else:
                logger.debug("Skipping result with no data")

        logger.debug(f"Parsed execution result into {len(outputs)} outputs")
        return outputs

    def filter_base64_images(self, message):
        """Filter out base64 encoded images from message content"""
        if isinstance(message, dict) and 'nbformat' in message:
            for output in message['nbformat']:
                if 'data' in output:
                    for key in list(output['data'].keys()):
                        if key.startswith('image/') or key == 'application/pdf':
                            output['data'][key] = '<placeholder_image>'
        return message
    
    def render(self, mode="default"):
        logger.debug(f"Rendering notebook in '{mode}' mode with {len(self.data['cells'])} cells")
        if self.countdown_info is not None:
            self._update_countdown_cell()

        render_data = copy.deepcopy(self.data)
        
        if mode == "generating":
            render_data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": GENERATING_WIDGET
            })

        elif mode == "executing":
            logger.debug("Adding executing widget to render")
            render_data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": EXECUTING_WIDGET
            })

        elif mode == "done":
            logger.debug("Adding done widget to render")
            render_data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": DONE_WIDGET
            })
        
        elif mode == "stopped":
            logger.debug("Adding stopped widget to render")
            render_data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": STOPPED_WIDGET
            })
        
        elif mode == "error":
            logger.debug("Adding error widget to render")
            render_data["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ERROR_WIDGET
            })
        
        elif mode != "default":
            logger.error(f"Invalid render mode: {mode}")
            raise ValueError(f"Render mode should be generating, executing, done, stopped, or error. Given: {mode}.")
        
        notebook = nbformat.from_dict(render_data)
        notebook_body, _ = html_exporter.from_notebook_node(notebook)
        notebook_body = notebook_body.replace(bad_html_bad, "")
        logger.debug(f"Rendered notebook HTML ({len(notebook_body)} chars)")

        # make code font a bit smaller with custom css
        if "<head>" in notebook_body:
            notebook_body = notebook_body.replace("</head>", f"{custom_css}</head>")
            logger.debug("Applied custom CSS to notebook")
        return notebook_body
    
    @classmethod
    def from_session_state(cls, session_state_data):
        """Create JupyterNotebook instance from session state data"""
        return cls(session_state_data=session_state_data)
    
    def get_session_notebook_data(self):
        """Get notebook data in format suitable for session state"""
        return self.data.copy()
    
    def update_from_session_state(self, session_state_data):
        """Update notebook data from session state"""
        if "notebook_data" in session_state_data:
            self.data = session_state_data["notebook_data"].copy()
            # Update execution count based on existing cells
            self.exec_count = len([cell for cell in self.data.get("cells", []) 
                                 if cell.get("cell_type") == "code" and cell.get("execution_count")])
            logger.debug(f"Updated notebook from session state: {len(self.data['cells'])} cells, exec_count={self.exec_count}")
    
def main():
    """Create a mock notebook to test styling"""
    # Create mock messages
    mock_messages = [
        {"role": "system", "content": "You are a helpful AI assistant that can write and execute Python code."},
        {"role": "user", "content": "Can you help me create a simple plot of a sine wave?"},
        {"role": "assistant", "content": "I'll help you create a sine wave plot using matplotlib. Let me write the code for that."},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "add_and_execute_jupyter_code_cell", "arguments": '{"code": "import numpy as np\\nimport matplotlib.pyplot as plt\\n\\n# Create x values\\nx = np.linspace(0, 4*np.pi, 100)\\ny = np.sin(x)\\n\\n# Create the plot\\nplt.figure(figsize=(10, 6))\\nplt.plot(x, y, \'b-\', linewidth=2)\\nplt.title(\'Sine Wave\')\\nplt.xlabel(\'x\')\\nplt.ylabel(\'sin(x)\')\\nplt.grid(True)\\nplt.show()"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "raw_execution": [{"output_type": "stream", "name": "stdout", "text": "Plot created successfully!"}]}
    ]
    
    # Create notebook
    notebook = JupyterNotebook(mock_messages)
    
    # Add a timeout countdown (simulating a sandbox that started 2 minutes ago with 5 minute timeout)
    start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=2)
    end_time = start_time + datetime.timedelta(minutes=5)
    notebook.add_sandbox_countdown(start_time, end_time)
    
    # Render and save
    html_output = notebook.render()
    
    with open("mock_notebook.html", "w", encoding="utf-8") as f:
        f.write(html_output)
    
    print("Mock notebook saved as 'mock_notebook.html'")
    print("Open it in your browser to see the styling changes.")

def create_notebook_from_session_state(session_state):
    """Helper function to create JupyterNotebook from session state"""
    return JupyterNotebook.from_session_state(session_state)


if __name__ == "__main__":
    main()