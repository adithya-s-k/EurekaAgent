import os
import logging
import gradio as gr
from gradio.utils import get_space
from modal_sandbox import create_modal_sandbox
from pathlib import Path
import json
import threading
from openai import OpenAI, AzureOpenAI
from jupyter_handler import JupyterNotebook

if not get_space():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except (ImportError, ModuleNotFoundError):
        pass
from utils import (
    run_interactive_notebook,
)

def create_notification_html(message, notification_type="info"):
    """
    Create HTML for notification messages
    
    Args:
        message: The notification message
        notification_type: Type of notification ('info', 'success', 'warning', 'error')
    """
    colors = {
        'info': '#3498db',
        'success': '#27ae60', 
        'warning': '#f39c12',
        'error': '#e74c3c'
    }
    
    icons = {
        'info': '🔄',
        'success': '✅',
        'warning': '⚠️', 
        'error': '❌'
    }
    
    color = colors.get(notification_type, colors['info'])
    icon = icons.get(notification_type, icons['info'])
    
    return f"""
    <div style="
        background-color: {color}20;
        border-left: 4px solid {color};
        padding: 12px 16px;
        margin: 10px 0;
        border-radius: 4px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        color: #2c3e50;
    ">
        <strong>{icon} {message}</strong>
    </div>
    """


def initialize_phoenix_tracing():
    """Initialize Phoenix tracing with proper error handling"""
    try:
        from phoenix.otel import register
        
        phoenix_api_key = os.getenv("PHOENIX_API_KEY")
        collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        
        if not phoenix_api_key:
            logger.info("Phoenix API key not found, skipping Phoenix tracing initialization")
            return None
            
        if not collector_endpoint:
            logger.info("Phoenix collector endpoint not found, skipping Phoenix tracing initialization")
            return None
            
        logger.info("Initializing Phoenix tracing...")
        
        # Set required environment variables
        os.environ["PHOENIX_API_KEY"] = phoenix_api_key
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = collector_endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"


        # Configure the Phoenix tracer with reduced auto-instrumentation to avoid conflicts
        tracer_provider = register(
            project_name="jupyter-agent-2",
            auto_instrument=True,  # Disable auto-instrument to prevent OpenAI client conflicts
            set_global_tracer_provider=True
        )
        
        logger.info("Phoenix tracing initialized successfully")
        return tracer_provider
        
    except ImportError:
        logger.info("Phoenix not installed, skipping tracing initialization")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Phoenix tracer (non-critical): {str(e)}")
        return None



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jupyter_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Phoenix tracing
tracer_provider = initialize_phoenix_tracing()


MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET")
HF_TOKEN = os.environ.get("HF_TOKEN")
SANDBOXES = {}
SANDBOX_TIMEOUT = 300
STOP_EVENTS = {}  # Store stop events for each session
EXECUTION_STATES = {}  # Store execution states for each session

# GPU configuration options for the UI
GPU_OPTIONS = [
    ("CPU Only", "cpu"),
    ("NVIDIA T4 (16GB)", "T4"),
    ("NVIDIA L4 (24GB)", "L4"), 
    ("NVIDIA A100 40GB", "A100-40GB"),
    ("NVIDIA A100 80GB", "A100-80GB"),
    ("NVIDIA H100 (80GB)", "H100")
]
TMP_DIR = './tmp/'
# model="Qwen/Qwen3-Coder-480B-A35B-Instruct:cerebras"
# model="qwen-3-coder-480b"


def initialize_openai_client():
    """Initialize OpenAI client with proper error handling and fallbacks"""
    client = None
    model_name = None
    
    try:
        # Option 1: Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
            logger.info("Initializing Azure OpenAI client")
            client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            )
            model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default fallback
            logger.info(f"Azure OpenAI client initialized with model: {model_name}")
            
        # Option 2: Custom Provider (Cerebras, etc.)  
        elif os.environ.get("PROVIDER_API_ENDPOINT") and os.environ.get("PROVIDER_API_KEY"):
            logger.info("Initializing custom provider OpenAI client")
            client = OpenAI(
                base_url=os.environ.get("PROVIDER_API_ENDPOINT"),
                api_key=os.environ.get("PROVIDER_API_KEY")
            )
            model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default fallback
            logger.info(f"Custom provider client initialized with model: {model_name}")
            
        # Option 3: Standard OpenAI
        elif os.environ.get("OPENAI_API_KEY"):
            logger.info("Initializing standard OpenAI client")
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default fallback
            logger.info(f"OpenAI client initialized with model: {model_name}")
            
        # Option 4: Default OpenAI (will use OPENAI_API_KEY env var)
        else:
            logger.info("Initializing default OpenAI client (using OPENAI_API_KEY env var)")
            client = OpenAI()  # Will automatically use OPENAI_API_KEY
            model_name = os.environ.get("MODEL_NAME", "gpt-4")
            logger.info(f"Default OpenAI client initialized with model: {model_name}")
            
        # Test the client with a simple request
        logger.info("Testing client connection...")
        try:
            # Simple test to verify the client works
            test_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info("Client connection test successful")
        except Exception as test_error:
            logger.error(f"Client connection test failed: {str(test_error)}")
            # Don't raise here, let the main application handle it
            
        return client, model_name
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        logger.error("Available environment variables:")
        env_vars = [
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", 
            "PROVIDER_API_ENDPOINT", "PROVIDER_API_KEY",
            "OPENAI_API_KEY", "MODEL_NAME"
        ]
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                logger.error(f"  {var}: {'*' * min(len(value), 10)}... (length: {len(value)})")
            else:
                logger.error(f"  {var}: NOT SET")
        raise

try:
    client, model_name = initialize_openai_client()
except Exception as e:
    logger.error(f"Critical error: Could not initialize any OpenAI client: {str(e)}")
    raise



init_notebook = JupyterNotebook()

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
    logger.info(f"Created temporary directory: {TMP_DIR}")
else:
    logger.info(f"Using existing temporary directory: {TMP_DIR}")

with open(TMP_DIR+"jupyter-agent.ipynb", 'w', encoding='utf-8') as f:
    json.dump(JupyterNotebook().data, f, indent=2)
logger.info(f"Initialized default notebook file: {TMP_DIR}jupyter-agent.ipynb")

try:
    with open("system_prompt.txt", "r") as f:
        DEFAULT_SYSTEM_PROMPT = f.read()
    logger.info("Loaded system prompt from ds-system-prompt.txt")
except FileNotFoundError:
    logger.warning("ds-system-prompt.txt not found, using fallback system prompt")


def execute_jupyter_agent(
    user_input, files, message_history, gpu_type, cpu_cores, memory_gb, timeout_sec, request: gr.Request
):
    session_id = request.session_hash
    logger.info(f"Starting execution for session {session_id}")
    logger.info(f"Hardware config: GPU={gpu_type}, CPU={cpu_cores}, Memory={memory_gb}GB, Timeout={timeout_sec}s")
    logger.info(f"User input length: {len(user_input)} characters")
    
    # Initialize or reset stop event for this session
    STOP_EVENTS[session_id] = threading.Event()
    EXECUTION_STATES[session_id] = {"running": True, "paused": False}

    # Set up save directory early for notifications
    session_dir = os.path.join(TMP_DIR, request.session_hash)
    os.makedirs(session_dir, exist_ok=True)
    save_dir = os.path.join(session_dir, 'jupyter-agent.ipynb')
    
    # Create initial notebook file so it exists for Gradio
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(init_notebook.data, f, indent=2)
    logger.info(f"Initialized notebook for session {session_id}")

    if request.session_hash not in SANDBOXES:
        logger.info(f"Creating new Modal sandbox for session {session_id}")
        
        # Show initialization notification
        gpu_info = gpu_type.upper() if gpu_type != "cpu" else "CPU Only"
        if gpu_type in ["T4", "L4", "A100-40GB", "A100-80GB", "H100"]:
            gpu_info = f"NVIDIA {gpu_type}"
            
        init_message = f"Initializing {gpu_info} sandbox with {cpu_cores} CPU cores and {memory_gb}GB RAM..."
        notification_html = create_notification_html(init_message, "info")
        yield notification_html, message_history, save_dir
        
        # Create Modal sandbox with user-specified configuration
        environment_vars = {}
        if MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
            environment_vars.update({
                "MODAL_TOKEN_ID": MODAL_TOKEN_ID,
                "MODAL_TOKEN_SECRET": MODAL_TOKEN_SECRET
            })
            logger.debug(f"Modal credentials configured for session {session_id}")
        
        try:
            SANDBOXES[request.session_hash] = create_modal_sandbox(
                gpu_config=gpu_type,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                timeout=int(timeout_sec),
                environment_vars=environment_vars
            )
            logger.info(f"Successfully created Modal sandbox for session {session_id}")
            
            # Show success notification
            success_message = f"✨ {gpu_info} sandbox ready! Environment initialized with all packages."
            success_notification_html = create_notification_html(success_message, "success")
            yield success_notification_html, message_history, save_dir
            
        except Exception as e:
            logger.error(f"Failed to create Modal sandbox for session {session_id}: {str(e)}")
            # Show error notification
            error_message = f"Failed to initialize sandbox: {str(e)}"
            error_notification_html = create_notification_html(error_message, "error")
            yield error_notification_html, message_history, save_dir
            raise
    else:
        logger.info(f"Reusing existing Modal sandbox for session {session_id}")
        # Show reuse notification
        gpu_info = gpu_type.upper() if gpu_type != "cpu" else "CPU Only"
        if gpu_type in ["T4", "L4", "A100-40GB", "A100-80GB", "H100"]:
            gpu_info = f"NVIDIA {gpu_type}"
        reuse_message = f"Using existing {gpu_info} sandbox - ready to execute!"
        reuse_notification_html = create_notification_html(reuse_message, "success")
        yield reuse_notification_html, message_history, save_dir
    
    sbx = SANDBOXES[request.session_hash]
    logger.debug(f"Notebook will be saved to: {save_dir}")
    
    # Initial notebook render
    yield init_notebook.render(), message_history, save_dir



    filenames = []
    if files is not None:
        logger.info(f"Processing {len(files)} uploaded files for session {session_id}")
        for filepath in files:
            filpath = Path(filepath)
            try:
                # Get file size for verification
                file_size = os.path.getsize(filepath)
                
                with open(filepath, "rb") as file:
                    logger.info(f"Uploading file {filepath} ({file_size} bytes) to session {session_id}")
                    sbx.files.write(filpath.name, file)
                    
                    # Verify upload succeeded
                    if sbx.files.verify_file_upload(filpath.name, file_size):
                        filenames.append(filpath.name)
                        logger.debug(f"Successfully uploaded and verified {filpath.name}")
                    else:
                        logger.error(f"File upload verification failed for {filpath.name}")
                        raise RuntimeError(f"File upload verification failed for {filpath.name}")
                        
            except Exception as e:
                logger.error(f"Failed to upload file {filepath} for session {session_id}: {str(e)}")
                raise
    else:
        logger.info(f"No files to upload for session {session_id}")

    sytem_prompt = DEFAULT_SYSTEM_PROMPT
    # Initialize message_history if it doesn't exist
    if len(message_history) == 0:
        logger.info(f"Initializing new conversation for session {session_id}")
        
        # Format files section
        if files is None:
            files_section = "- None"
        else:
            files_section = "- " + "\n- ".join(filenames)
            logger.info(f"System prompt includes {len(filenames)} files: {filenames}")
        
        # Format GPU information
        gpu_info = gpu_type.upper() if gpu_type != "cpu" else "CPU Only"
        if gpu_type in ["T4", "L4", "A100-40GB", "A100-80GB", "H100"]:
            gpu_info = f"NVIDIA {gpu_type}"
        
        # Format available packages based on hardware configuration
        packages_list = sbx.available_packages
        packages_section = "\n".join([f"- {package}" for package in packages_list])
        
        # Format the complete system prompt with named placeholders
        sytem_prompt = sytem_prompt.replace("{AVAILABLE_FILES}", files_section)
        sytem_prompt = sytem_prompt.replace("{GPU_TYPE}", gpu_info)
        sytem_prompt = sytem_prompt.replace("{CPU_CORES}", str(cpu_cores))
        sytem_prompt = sytem_prompt.replace("{MEMORY_GB}", str(memory_gb))
        sytem_prompt = sytem_prompt.replace("{TIMEOUT_SECONDS}", str(timeout_sec))
        sytem_prompt = sytem_prompt.replace("{AVAILABLE_PACKAGES}", packages_section)

        message_history.append(
            {
                "role": "system",
                "content": sytem_prompt,
            }
        )
    else:
        logger.info(f"Continuing existing conversation for session {session_id} (history length: {len(message_history)})")
    
    message_history.append({"role": "user", "content": user_input})
    logger.debug(f"Added user message to history for session {session_id}")

    logger.debug(f"Message history for session {session_id}: {len(message_history)} messages")

    logger.info(f"Starting interactive notebook execution for session {session_id}")
    try:
        for notebook_html, notebook_data, messages in run_interactive_notebook(
            client, model_name, message_history, sbx, STOP_EVENTS[session_id]
        ):
            message_history = messages
            logger.debug(f"Interactive notebook yield for session {session_id}")
            yield notebook_html, message_history, TMP_DIR+"jupyter-agent.ipynb"
    except Exception as e:
        logger.error(f"Error during interactive notebook execution for session {session_id}: {str(e)}")
        raise
    
    try:
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=2)
        logger.info(f"Final notebook saved for session {session_id}: {save_dir}")
    except Exception as e:
        logger.error(f"Failed to save final notebook for session {session_id}: {str(e)}")
        raise
    
    yield notebook_html, message_history, save_dir
    logger.info(f"Completed execution for session {session_id}")
    
    # Update execution state
    if session_id in EXECUTION_STATES:
        EXECUTION_STATES[session_id]["running"] = False

def clear(msg_state, request: gr.Request):
    session_id = request.session_hash
    logger.info(f"Clearing session {session_id}")
    
    if request.session_hash in SANDBOXES:
        try:
            logger.info(f"Killing Modal sandbox for session {session_id}")
            SANDBOXES[request.session_hash].kill()
            SANDBOXES.pop(request.session_hash)
            logger.info(f"Successfully cleared sandbox for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing sandbox for session {session_id}: {str(e)}")
    else:
        logger.info(f"No sandbox found to clear for session {session_id}")

    msg_state = []
    logger.info(f"Reset message state for session {session_id}")
    
    # Clean up stop events and execution states
    if session_id in STOP_EVENTS:
        STOP_EVENTS.pop(session_id)
    if session_id in EXECUTION_STATES:
        EXECUTION_STATES.pop(session_id)
        
    return init_notebook.render(), msg_state

def stop_execution(request: gr.Request):
    """Stop the current execution for this session"""
    session_id = request.session_hash
    logger.info(f"Stopping execution for session {session_id}")
    
    if session_id in STOP_EVENTS:
        STOP_EVENTS[session_id].set()
        logger.info(f"Stop signal sent for session {session_id}")
        
        # Update execution state
        if session_id in EXECUTION_STATES:
            EXECUTION_STATES[session_id]["running"] = False
            EXECUTION_STATES[session_id]["paused"] = True
        
        return "⏸️ Execution stopped - use Continue to resume"
    else:
        logger.warning(f"No active execution found for session {session_id}")
        return "❌ No active execution to stop"

def continue_execution(user_input, files, message_history, gpu_type, cpu_cores, memory_gb, timeout_sec, request: gr.Request):
    """Continue execution after it was stopped"""
    session_id = request.session_hash
    logger.info(f"Continuing execution for session {session_id}")
    
    # Reset stop event and execution state
    if session_id in STOP_EVENTS:
        STOP_EVENTS[session_id].clear()
        logger.info(f"Cleared stop event for session {session_id}")
    
    if session_id in EXECUTION_STATES:
        EXECUTION_STATES[session_id]["running"] = True
        EXECUTION_STATES[session_id]["paused"] = False
        logger.info(f"Reset execution state for session {session_id}")
    
    # Continue with normal execution - yield from the generator
    yield from execute_jupyter_agent(user_input, files, message_history, gpu_type, cpu_cores, memory_gb, timeout_sec, request)

def get_execution_status(request: gr.Request):
    """Get the current execution status for UI updates"""
    session_id = request.session_hash
    
    if session_id not in EXECUTION_STATES:
        return "⚪ Ready"
    
    state = EXECUTION_STATES[session_id]
    if state["running"]:
        if session_id in STOP_EVENTS and STOP_EVENTS[session_id].is_set():
            return "⏸️ Stopped"
        else:
            return "🟢 Running"
    elif state.get("paused", False):
        return "⏸️ Paused"
    else:
        return "⚪ Ready"


css = """
#component-0 {
    height: 100vh;
    overflow-y: auto;
    padding: 20px;
}

.gradio-container {
    height: 100vh !important;
}

.contain {
    height: 100vh !important;
}
"""


# Create the interface
with gr.Blocks() as demo:
    msg_state = gr.State(value=[])

    html_output = gr.HTML(value=JupyterNotebook().render())
    
    user_input = gr.Textbox(
        # value="Write code to multiply three numbers: 10048, 32, 19", lines=3, label="User input"
        # value="Execute this complete Lotka-Volterra solution in a single cell:\n\n```python\n# Complete solution in a single cell\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy.integrate import solve_ivp\n\n# Set a compatible matplotlib style\nplt.style.use('default')\nplt.rcParams['figure.facecolor'] = 'white'\nplt.rcParams['axes.grid'] = True\n\n# Define the Lotka-Volterra model\ndef lotka_volterra(t, z, alpha, beta, delta, gamma):\n    x, y = z\n    dxdt = alpha * x - beta * x * y\n    dydt = delta * x * y - gamma * y\n    return [dxdt, dydt]\n\n# Parameters and initial conditions\nalpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5\nz0 = [40, 9]\nt_span = (0, 15)\nt_eval = np.linspace(0, 15, 1000)\n\n# Solve equations\nsolution = solve_ivp(lotka_volterra, t_span, z0, args=(alpha, beta, delta, gamma), t_eval=t_eval)\nx, y = solution.y[0], solution.y[1]\n\n# Create plots\nfig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n\n# Time evolution\nax1.plot(solution.t, x, label='Prey', linewidth=2.5, color='blue')\nax1.plot(solution.t, y, label='Predator', linewidth=2.5, color='red')\nax1.set_xlabel('Time')\nax1.set_ylabel('Population')\nax1.set_title('Population Dynamics Over Time')\nax1.legend()\nax1.grid(True)\n\n# Phase diagram\nax2.plot(x, y, linewidth=2.5, color='orange')\nax2.scatter(x[0], y[0], color='green', s=100, label='Start')\nax2.scatter(x[-1], y[-1], color='red', s=100, label='End')\nax2.set_xlabel('Prey Population')\nax2.set_ylabel('Predator Population')\nax2.set_title('Phase Diagram')\nax2.legend()\nax2.grid(True)\n\nplt.tight_layout()\nplt.show()\n\nprint(f'Prey range: {x.min():.2f} to {x.max():.2f}')\nprint(f'Predator range: {y.min():.2f} to {y.max():.2f}')\n```", label="Agent task"
        value="train a 5 neuron neural network to classify the iris dataset",
    )
    
    with gr.Row():
        generate_btn = gr.Button("Run!", variant="primary")
        stop_btn = gr.Button("⏸️ Stop", variant="secondary")
        continue_btn = gr.Button("▶️ Continue", variant="secondary")
        clear_btn = gr.Button("Clear Notebook", variant="stop")
    
    # Status display
    status_display = gr.Textbox(
        value="⚪ Ready", 
        label="Execution Status", 
        interactive=False,
        max_lines=1
    )
    
    with gr.Accordion("Hardware Configuration ⚙️", open=False):
        with gr.Row():
            gpu_type = gr.Dropdown(
                choices=GPU_OPTIONS,
                value="cpu",
                label="GPU Type",
                info="Select hardware acceleration"
            )
            cpu_cores = gr.Slider(
                minimum=0.25,
                maximum=16,
                value=2.0,
                step=0.25,
                label="CPU Cores",
                info="Number of CPU cores"
            )
        with gr.Row():
            memory_gb = gr.Slider(
                minimum=0.5,
                maximum=64,
                value=8.0,
                step=0.5,
                label="Memory (GB)",
                info="RAM allocation"
            )
            timeout_sec = gr.Slider(
                minimum=60,
                maximum=1800,
                value=300,
                step=60,
                label="Timeout (seconds)",
                info="Maximum execution time"
            )
        
        hardware_info = gr.Markdown("""
        **Hardware Options:**
        - **CPU Only**: Free, good for basic tasks
        - **T4**: Low-cost GPU, good for small models
        - **L4**: Mid-range GPU, better performance
        - **A100 40/80GB**: High-end GPU for large models
        - **H100**: Latest flagship GPU for maximum performance
        
        ⚠️ **Note**: GPU instances cost more. Choose based on your workload.
        """)
    
    with gr.Accordion("Upload files ⬆ | Download notebook⬇", open=False):
        files = gr.File(label="Upload files to use", file_count="multiple")
        file = gr.File(TMP_DIR+"jupyter-agent.ipynb", label="Download Jupyter Notebook")


    generate_btn.click(
        fn=execute_jupyter_agent,
        inputs=[user_input, files, msg_state, gpu_type, cpu_cores, memory_gb, timeout_sec],
        outputs=[html_output, msg_state, file],
        show_progress="hidden",
    )

    stop_btn.click(
        fn=stop_execution,
        outputs=[status_display],
        show_progress="hidden",
    )

    continue_btn.click(
        fn=continue_execution,
        inputs=[user_input, files, msg_state, gpu_type, cpu_cores, memory_gb, timeout_sec],
        outputs=[html_output, msg_state, file],
        show_progress="hidden",
    )

    clear_btn.click(fn=clear, inputs=[msg_state], outputs=[html_output, msg_state])

    # Periodic status update
    # demo.load(
    #     fn=get_execution_status,
    #     inputs=None,
    #     outputs=[status_display],
    #     every=2,  # Update every 2 seconds
    #     show_progress="hidden"
    # )

    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=""" () => {
    if (document.querySelectorAll('.dark').length) {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    }
}
"""
    )

logger.info("Starting Gradio application")
demo.launch(ssr_mode=False)
