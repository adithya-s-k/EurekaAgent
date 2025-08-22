import os
import logging
import gradio as gr
from gradio.utils import get_space
from modal_sandbox import create_modal_sandbox
from pathlib import Path
import json
from datetime import datetime
import threading
import re
from openai import OpenAI, AzureOpenAI
from jupyter_handler import JupyterNotebook

if not get_space():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except (ImportError, ModuleNotFoundError):
        pass
from jupyter_agent import (
    run_interactive_notebook_with_session_state,
    SessionStateManager,
)

TMP_DIR = './temp/'

# Environment and API key management utilities
def get_environment():
    """Get the current environment (dev/prod)"""
    return os.environ.get("ENVIRONMENT", "prod").lower()

def is_dev_environment():
    """Check if running in development environment"""
    return get_environment() == "dev"

def get_required_api_keys():
    """Get dictionary of required API keys and their current status"""
    required_keys = {
        "MODAL_TOKEN_ID": {
            "value": os.environ.get("MODAL_TOKEN_ID"),
            "required": True,
            "description": "Modal Token ID for sandbox access"
        },
        "MODAL_TOKEN_SECRET": {
            "value": os.environ.get("MODAL_TOKEN_SECRET"),
            "required": True,
            "description": "Modal Token Secret for sandbox access"
        },
        "HF_TOKEN": {
            "value": os.environ.get("HF_TOKEN"),
            "required": False,
            "description": "Hugging Face Token for model access"
        },
        "PROVIDER_API_KEY": {
            "value": os.environ.get("PROVIDER_API_KEY"),
            "required": True,
            "description": "AI Provider API Key (Anthropic, OpenAI, etc.)"
        },
        "PROVIDER_API_ENDPOINT": {
            "value": os.environ.get("PROVIDER_API_ENDPOINT"),
            "required": True,
            "description": "AI Provider API Endpoint"
        },
        "MODEL_NAME": {
            "value": os.environ.get("MODEL_NAME"),
            "required": True,
            "description": "Model name to use"
        },
        "TAVILY_API_KEY": {
            "value": os.environ.get("TAVILY_API_KEY"),
            "required": False,
            "description": "Tavily API Key for web search functionality"
        }
    }
    return required_keys

def get_missing_api_keys():
    """Get list of missing required API keys"""
    required_keys = get_required_api_keys()
    missing_keys = {}
    
    for key, config in required_keys.items():
        if config["required"] and not config["value"]:
            missing_keys[key] = config
    
    return missing_keys

def validate_api_key_format(key_name, key_value):
    """Basic validation for API key formats"""
    if not key_value or not key_value.strip():
        return False, "API key cannot be empty"
    
    key_value = key_value.strip()
    
    # Basic format validation
    if key_name == "MODAL_TOKEN_ID" and not key_value.startswith("ak-"):
        return False, "Modal Token ID should start with 'ak-'"
    elif key_name == "MODAL_TOKEN_SECRET" and not key_value.startswith("as-"):
        return False, "Modal Token Secret should start with 'as-'"
    elif key_name == "HF_TOKEN" and not key_value.startswith("hf_"):
        return False, "Hugging Face token should start with 'hf_'"
    elif key_name == "PROVIDER_API_KEY":
        # Check for common API key prefixes
        valid_prefixes = ["sk-", "gsk_", "csk-"]
        if not any(key_value.startswith(prefix) for prefix in valid_prefixes):
            return False, "API key format may be invalid (expected prefixes: sk-, gsk_, csk-)"
    elif key_name == "PROVIDER_API_ENDPOINT" and not (key_value.startswith("http://") or key_value.startswith("https://")):
        return False, "API endpoint should start with http:// or https://"
    elif key_name == "TAVILY_API_KEY" and not key_value.startswith("tvly-"):
        return False, "Tavily API key should start with 'tvly-'"
    
    return True, "Valid format"

def apply_user_api_keys(api_keys_dict):
    """Apply user-provided API keys to environment"""
    for key, value in api_keys_dict.items():
        if value and value.strip():
            os.environ[key] = value.strip()
            logger.info(f"Applied user-provided API key: {key}")

def get_previous_notebooks():
    """Get list of previous notebook sessions (dev only)"""
    if not is_dev_environment():
        return []
    
    notebooks = []
    tmp_dir = Path(TMP_DIR)
    
    if not tmp_dir.exists():
        return notebooks
    
    for session_dir in tmp_dir.iterdir():
        if session_dir.is_dir() and session_dir.name != ".":
            notebook_file = session_dir / "jupyter-agent.ipynb"
            if notebook_file.exists():
                try:
                    # Get creation time and basic info
                    stat = notebook_file.stat()
                    size = stat.st_size
                    modified = stat.st_mtime
                    
                    # Try to read basic notebook info
                    with open(notebook_file, 'r') as f:
                        notebook_data = json.load(f)
                        cell_count = len(notebook_data.get('cells', []))
                    
                    # Format timestamp
                    formatted_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
                    
                    # Try to load session state for additional info
                    config_info = ""
                    try:
                        session_manager = SessionStateManager(session_dir.name, TMP_DIR)
                        session_state = session_manager.load_state()
                        if session_state:
                            hardware = session_state.get("hardware_config", {})
                            gpu = hardware.get("gpu_type", "unknown")
                            config_info = f", {gpu}"
                    except Exception:
                        pass
                    
                    notebooks.append({
                        'session_id': session_dir.name,
                        'path': str(notebook_file),
                        'modified': modified,
                        'size': size,
                        'cell_count': cell_count,
                        'display_name': f"{session_dir.name} ({cell_count} cells{config_info}, {formatted_time})"
                    })
                except Exception as e:
                    logger.warning(f"Failed to read notebook info for {session_dir.name}: {e}")
    
    # Sort by modification time (newest first)
    notebooks.sort(key=lambda x: x['modified'], reverse=True)
    return notebooks

def parse_environment_variables(env_vars_text):
    """
    Parse environment variables from text input
    
    Args:
        env_vars_text: String containing environment variables in KEY=value format, one per line
        
    Returns:
        dict: Dictionary of parsed environment variables
    """
    env_dict = {}
    if not env_vars_text or not env_vars_text.strip():
        return env_dict
        
    for line in env_vars_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue
            
        if '=' in line:
            key, value = line.split('=', 1)  # Split only on first =
            key = key.strip()
            value = value.strip()
            if key:  # Only add if key is not empty
                env_dict[key] = value
        else:
            logger.warning(f"Skipping invalid environment variable format: {line}")
    
    return env_dict

def create_notification_html(message, notification_type="info", show_spinner=False):
    """
    Create HTML for notification messages
    
    Args:
        message: The notification message
        notification_type: Type of notification ('info', 'success', 'warning', 'error')
        show_spinner: Whether to show a loading spinner
    """
    colors = {
        'info': '#3498db',
        'success': '#27ae60', 
        'warning': '#f39c12',
        'error': '#e74c3c',
        'loading': '#6c5ce7'
    }
    
    icons = {
        'info': '🔄',
        'success': '✅',
        'warning': '⚠️', 
        'error': '❌',
        'loading': '⏳'
    }
    
    color = colors.get(notification_type, colors['info'])
    icon = icons.get(notification_type, icons['info'])
    
    spinner_html = ""
    if show_spinner or notification_type == 'loading':
        spinner_html = """
        <div style="
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid {color};
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        "></div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """.format(color=color)
    
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
        display: flex;
        align-items: center;
    ">
        {spinner_html}
        <strong>{icon} {message}</strong>
    </div>
    """

def create_progress_notification(message, progress_percent=None):
    """Create a progress notification with optional progress bar"""
    progress_html = ""
    if progress_percent is not None:
        progress_html = f"""
        <div style="
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 5px;
            margin-top: 8px;
            height: 8px;
        ">
            <div style="
                width: {progress_percent}%;
                background-color: #3498db;
                height: 8px;
                border-radius: 5px;
                transition: width 0.3s ease;
            "></div>
        </div>
        <small style="color: #666; margin-top: 4px; display: block;">{progress_percent}% complete</small>
        """
    
    return create_notification_html(message, "loading", show_spinner=True) + progress_html


def initialize_phoenix_tracing():
    """Initialize Phoenix tracing with proper error handling and session support"""
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
            
        logger.info("Initializing Phoenix tracing with session support...")
        
        # Set required environment variables
        os.environ["PHOENIX_API_KEY"] = phoenix_api_key
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = collector_endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"

        # Configure the Phoenix tracer with OpenAI instrumentation enabled
        tracer_provider = register(
            project_name="eureka-agent",
            auto_instrument=True,  # Keep auto-instrument enabled for OpenAI tracing
            set_global_tracer_provider=True
        )
        
        # Additional instrumentation setup for session tracking
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            
            # Ensure OpenAI instrumentation is properly configured
            if not OpenAIInstrumentor().is_instrumented_by_opentelemetry:
                OpenAIInstrumentor().instrument()
                logger.info("OpenAI instrumentation configured for Phoenix session tracking")
            else:
                logger.info("OpenAI instrumentation already active")
                
        except ImportError:
            logger.warning("OpenAI instrumentation not available - session grouping may not work optimally")
        except Exception as e:
            logger.warning(f"Failed to configure OpenAI instrumentation: {str(e)}")
        
        logger.info("Phoenix tracing initialized successfully with session support")
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


def initialize_openai_client():
    """Initialize OpenAI client with proper error handling and fallbacks"""
    client = None
    model_name = None
    
    # Check if we have any API keys configured
    has_azure = os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY")
    has_provider = os.environ.get("PROVIDER_API_ENDPOINT") and os.environ.get("PROVIDER_API_KEY") 
    has_openai = os.environ.get("OPENAI_API_KEY")
    
    if not (has_azure or has_provider or has_openai):
        logger.warning("No API keys found in environment - client will be initialized later when user provides keys")
        return None, None
    
    try:
        # Option 1: Azure OpenAI
        if has_azure:
            logger.info("Initializing Azure OpenAI client")
            client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            )
            model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default fallback
            logger.info(f"Azure OpenAI client initialized with model: {model_name}")
            
        # Option 2: Custom Provider (Cerebras, etc.)  
        elif has_provider:
            logger.info("Initializing custom provider OpenAI client")
            client = OpenAI(
                base_url=os.environ.get("PROVIDER_API_ENDPOINT"),
                api_key=os.environ.get("PROVIDER_API_KEY")
            )
            model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default fallback
            logger.info(f"Custom provider client initialized with model: {model_name}")
            
        # Option 3: Standard OpenAI
        elif has_openai:
            logger.info("Initializing standard OpenAI client")
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default fallback
            logger.info(f"OpenAI client initialized with model: {model_name}")
            
        # Test the client with a simple request (optional - skip if client initialization should be fast)
        if client:
            logger.info("Testing client connection...")
            try:
                # Simple test to verify the client works
                _ = client.chat.completions.create(
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
        logger.warning("Client will be initialized later when user provides valid API keys")
        return None, None

client, model_name = initialize_openai_client()

# If no client was initialized, it means no API keys are available
if client is None:
    logger.info("No OpenAI client initialized - waiting for user to provide API keys through UI")



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
    user_input, files, message_history, gpu_type, cpu_cores, memory_gb, timeout_sec, env_vars_text, 
    modal_token_id, modal_token_secret, hf_token, provider_api_key, provider_api_endpoint, user_model_name,
    tavily_api_key, enable_web_search, request: gr.Request
):
    session_id = request.session_hash
    logger.info(f"Starting execution for session {session_id}")
    logger.info(f"Hardware config: GPU={gpu_type}, CPU={cpu_cores}, Memory={memory_gb}GB, Timeout={timeout_sec}s")
    logger.info(f"User input length: {len(user_input)} characters")
    
    # Check if execution is already running for this session
    if session_id in EXECUTION_STATES and EXECUTION_STATES[session_id].get("running", False):
        error_message = "❌ Execution already in progress for this session. Please wait for it to complete or stop it first."
        error_notification = create_notification_html(error_message, "warning")
        
        # Return current state without starting new execution
        session_dir = os.path.join(TMP_DIR, session_id)
        save_dir = os.path.join(session_dir, 'jupyter-agent.ipynb')
        if os.path.exists(save_dir):
            yield error_notification, message_history, save_dir
        else:
            yield error_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
        return
    
    # Initialize session state manager
    session_manager = SessionStateManager(session_id, TMP_DIR)
    
    # Check if this is a continuing session
    existing_session_state = session_manager.load_state()
    is_continuing_session = existing_session_state is not None
    
    if is_continuing_session:
        logger.info(f"Found existing session state for {session_id} - continuing from previous state")
    else:
        logger.info(f"No existing session state found for {session_id} - starting new session")
    
    # Apply user-provided API keys if any are provided
    user_api_keys = {}
    if modal_token_id:
        user_api_keys["MODAL_TOKEN_ID"] = modal_token_id
    if modal_token_secret:
        user_api_keys["MODAL_TOKEN_SECRET"] = modal_token_secret
    if hf_token:
        user_api_keys["HF_TOKEN"] = hf_token
    if provider_api_key:
        user_api_keys["PROVIDER_API_KEY"] = provider_api_key
    if provider_api_endpoint:
        user_api_keys["PROVIDER_API_ENDPOINT"] = provider_api_endpoint
    if user_model_name:
        user_api_keys["MODEL_NAME"] = user_model_name
    if tavily_api_key:
        user_api_keys["TAVILY_API_KEY"] = tavily_api_key
    
    # Check if we have a client or need to initialize one with user keys
    global client, model_name
    if client is None and not user_api_keys:
        missing_keys = get_missing_api_keys()
        if missing_keys:
            error_message = f"""❌ Missing Required API Keys

Please provide the following API keys to continue:
{chr(10).join([f"• {key}: {config['description']}" for key, config in missing_keys.items()])}

You can either:
1. Add them to your .env file, or 
2. Enter them in the API Keys section above"""
            error_notification = create_notification_html(error_message, "error")
            yield error_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
            return
    
    # Validate user-provided API keys
    if user_api_keys:
        validation_message = "🔍 Validating API keys..."
        validation_notification = create_progress_notification(validation_message)
        yield validation_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
        
        validation_errors = []
        for key, value in user_api_keys.items():
            is_valid, message = validate_api_key_format(key, value)
            if not is_valid:
                validation_errors.append(f"{key}: {message}")
        
        if validation_errors:
            error_message = f"❌ API Key Validation Failed:\n" + "\n".join(f"• {error}" for error in validation_errors)
            error_notification = create_notification_html(error_message, "error")
            yield error_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
            return
        
        logger.info(f"Applying user-provided API keys: {list(user_api_keys.keys())}")
        apply_user_api_keys(user_api_keys)
        
        # Reinitialize OpenAI client with new keys if provider keys were updated
        if any(key in user_api_keys for key in ["PROVIDER_API_KEY", "PROVIDER_API_ENDPOINT", "MODEL_NAME"]):
            try:
                reinit_message = "🔄 Reinitializing AI client with new credentials..."
                reinit_notification = create_progress_notification(reinit_message)
                yield reinit_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
                
                client, model_name = initialize_openai_client()
                if client is None:
                    error_message = "Failed to initialize client with provided API keys. Please check your credentials."
                    logger.error(error_message)
                    error_notification = create_notification_html(error_message, "error")
                    yield error_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
                    return
                logger.info("Reinitialized OpenAI client with user-provided keys")
                
                success_message = "✅ API credentials validated and applied successfully!"
                success_notification = create_notification_html(success_message, "success")
                yield success_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
            except Exception as e:
                error_message = f"Failed to initialize client with provided API keys: {str(e)}"
                logger.error(error_message)
                error_notification = create_notification_html(error_message, "error")
                yield error_notification, message_history, TMP_DIR + "jupyter-agent.ipynb"
                return
    
    # Initialize or reset stop event for this session
    STOP_EVENTS[session_id] = threading.Event()
    EXECUTION_STATES[session_id] = {"running": True, "paused": False, "current_phase": "initializing"}

    # Set up save directory early for notifications
    session_dir = os.path.join(TMP_DIR, request.session_hash)
    os.makedirs(session_dir, exist_ok=True)
    save_dir = os.path.join(session_dir, 'jupyter-agent.ipynb')
    
    # Create initial notebook file so it exists for Gradio
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(init_notebook.data, f, indent=2)
    logger.info(f"Initialized notebook for session {session_id}")
    
    # Session configuration is now handled by SessionStateManager

    if request.session_hash not in SANDBOXES:
        logger.info(f"Creating new Modal sandbox for session {session_id}")
        
        # Show initialization notification with spinner
        gpu_info = gpu_type.upper() if gpu_type != "cpu" else "CPU Only"
        if gpu_type in ["T4", "L4", "A100-40GB", "A100-80GB", "H100"]:
            gpu_info = f"NVIDIA {gpu_type}"
            
        init_message = f"Initializing {gpu_info} sandbox with {cpu_cores} CPU cores and {memory_gb}GB RAM..."
        notification_html = create_progress_notification(init_message)
        yield notification_html, message_history, save_dir
        
        # Create Modal sandbox with user-specified configuration
        environment_vars = {}
        if MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
            environment_vars.update({
                "MODAL_TOKEN_ID": MODAL_TOKEN_ID,
                "MODAL_TOKEN_SECRET": MODAL_TOKEN_SECRET
            })
            logger.debug(f"Modal credentials configured for session {session_id}")
        
        # Parse and add user-provided environment variables
        user_env_vars = parse_environment_variables(env_vars_text)
        if user_env_vars:
            environment_vars.update(user_env_vars)
            logger.info(f"Added {len(user_env_vars)} custom environment variables for session {session_id}")
            logger.debug(f"Custom environment variables: {list(user_env_vars.keys())}")
        
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

    # Initialize or continue session state
    if is_continuing_session:
        # Load existing session state
        session_state = existing_session_state
        
        # Validate and repair conversation history to prevent API errors
        session_manager.validate_and_repair_conversation(session_state)
        
        message_history = session_manager.get_conversation_history(session_state)
        logger.info(f"Continuing session {session_id} with {len(message_history)} existing messages")
        
        # Add new user input if provided
        if user_input and user_input.strip():
            # Check if this input was already added by comparing with the last message
            last_message = message_history[-1] if message_history else None
            should_add_input = True
            
            if last_message and last_message.get("role") == "user":
                # If the last message is from user and has the same content, don't add duplicate
                if last_message.get("content") == user_input:
                    should_add_input = False
                    logger.debug(f"User input already present in session {session_id}")
            
            if should_add_input:
                session_manager.add_message(session_state, "user", user_input)
                message_history = session_manager.get_conversation_history(session_state)
                logger.info(f"Added new user input to existing session {session_id}")
                
                # Show notification that we're continuing the conversation
                continue_message = "🔄 Continuing conversation with new input..."
                continue_notification = create_progress_notification(continue_message)
                yield continue_notification, message_history, save_dir
    else:
        # Create new session state
        logger.info(f"Initializing new session {session_id}")
        
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
        system_prompt = DEFAULT_SYSTEM_PROMPT.replace("{AVAILABLE_FILES}", files_section)
        system_prompt = system_prompt.replace("{GPU_TYPE}", gpu_info)
        system_prompt = system_prompt.replace("{CPU_CORES}", str(cpu_cores))
        system_prompt = system_prompt.replace("{MEMORY_GB}", str(memory_gb))
        system_prompt = system_prompt.replace("{TIMEOUT_SECONDS}", str(timeout_sec))
        system_prompt = system_prompt.replace("{AVAILABLE_PACKAGES}", packages_section)
        
        # Create session state with configuration
        hardware_config = {
            "gpu_type": gpu_type,
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
            "timeout_sec": timeout_sec
        }
        
        api_config = {
            "model_name": model_name or user_model_name or "unknown",
            "provider_endpoint": os.environ.get("PROVIDER_API_ENDPOINT") or provider_api_endpoint,
            "provider_type": "openai_compatible"
        }
        
        environment_config = {
            "variables": env_vars_text or "",
            "files_uploaded": filenames if filenames else []
        }
        
        # Create initial session state
        session_state = session_manager.create_initial_state(
            hardware_config, api_config, environment_config, system_prompt
        )
        
        # Add user input if provided
        if user_input and user_input.strip():
            session_manager.add_message(session_state, "user", user_input)
        
        # Get conversation history
        message_history = session_manager.get_conversation_history(session_state)
        
        # Save initial state
        session_manager.save_state(session_state)
        
        logger.info(f"Created new session {session_id} with {len(message_history)} messages")
    
    logger.debug(f"Session {session_id} ready with {len(message_history)} messages")

    # Determine which tools to use based on web search toggle
    from jupyter_agent import TOOLS
    if enable_web_search:
        # Check if Tavily API key is available
        tavily_key = os.environ.get("TAVILY_API_KEY") or tavily_api_key
        if tavily_key:
            selected_tools = TOOLS  # Use all tools (code + search)
            logger.info(f"Web search enabled for session {session_id} - using all tools")
        else:
            selected_tools = TOOLS[:1]  # Use only code execution tool
            logger.warning(f"Web search enabled but no Tavily API key found for session {session_id} - using code tool only")
    else:
        selected_tools = TOOLS[:1]  # Use only code execution tool  
        logger.info(f"Web search disabled for session {session_id} - using code tool only")

    logger.info(f"Starting interactive notebook execution for session {session_id}")
    
    # Import Phoenix session context if available
    try:
        from jupyter_agent import create_phoenix_session_context
        phoenix_available = True
    except ImportError:
        phoenix_available = False
    
    # Prepare session metadata for Phoenix tracing at the session level
    if phoenix_available:
        session_level_metadata = {
            "agent_type": "eureka-agent",
            "session_type": "jupyter_execution",
            "gpu_type": gpu_type,
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
            "timeout_sec": timeout_sec,
            "web_search_enabled": enable_web_search,
            "tools_available": len(selected_tools)
        }
        
        # Add API provider info if available
        if model_name:
            session_level_metadata["model"] = model_name
            
        session_context = create_phoenix_session_context(
            session_id=session_id,
            user_id=None,  # Could add user identification if available
            metadata=session_level_metadata
        )
    else:
        from contextlib import nullcontext
        session_context = nullcontext()
    
    # Wrap the entire execution in a Phoenix session context
    with session_context:
        logger.debug(f"Starting session-level Phoenix tracing for {session_id}")
        try:
            for notebook_html, notebook_data, messages in run_interactive_notebook_with_session_state(
                client, model_name, session_manager, session_state, sbx, STOP_EVENTS[session_id], selected_tools
            ):
                message_history = messages
                logger.debug(f"Interactive notebook yield for session {session_id}")
                # Update session state and yield with legacy notebook file for UI compatibility
                session_manager.update_notebook_data(session_state, notebook_data)
                session_manager.save_state(session_state)
                
                # Create legacy notebook file for UI download compatibility
                with open(save_dir, 'w', encoding='utf-8') as f:
                    json.dump(notebook_data, f, indent=2)
                    
                yield notebook_html, message_history, save_dir
                
        except Exception as e:
            logger.error(f"Error during interactive notebook execution for session {session_id}: {str(e)}")
            # Save error state
            session_manager.update_execution_state(session_state, is_running=False, last_execution_successful=False)
            session_manager.save_state(session_state)
            raise
    
    # Final save and cleanup
    try:
        session_manager.update_execution_state(session_state, is_running=False)
        session_manager.save_state(session_state)
        logger.info(f"Final session state saved for session {session_id}")
        
        # Create final legacy notebook file for UI
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save final session state for session {session_id}: {str(e)}")
        raise
    
    yield notebook_html, message_history, save_dir
    logger.info(f"Completed execution for session {session_id}")
    
    # Update legacy execution state for compatibility
    if session_id in EXECUTION_STATES:
        EXECUTION_STATES[session_id]["running"] = False

def clear(msg_state, request: gr.Request):
    """Clear notebook but keep session data (less destructive than shutdown)"""
    session_id = request.session_hash
    logger.info(f"Clearing notebook for session {session_id}")
    
    # Stop any running execution
    if session_id in STOP_EVENTS:
        STOP_EVENTS[session_id].set()
    
    # Clear execution states but keep session data
    if session_id in EXECUTION_STATES:
        EXECUTION_STATES[session_id]["running"] = False
        EXECUTION_STATES[session_id]["paused"] = False
        EXECUTION_STATES[session_id]["current_phase"] = "ready"
    
    # Reset message state for UI
    msg_state = []
    logger.info(f"Reset notebook display for session {session_id}")
        
    return init_notebook.render(), msg_state

def stop_execution(request: gr.Request):
    """Stop the current execution for this session"""
    session_id = request.session_hash
    logger.info(f"Stopping execution for session {session_id}")
    
    if session_id in STOP_EVENTS and session_id in EXECUTION_STATES:
        # Check if execution is actually running
        if EXECUTION_STATES[session_id].get("running", False):
            STOP_EVENTS[session_id].set()
            logger.info(f"Stop signal sent for session {session_id}")
            
            # Update execution state
            EXECUTION_STATES[session_id]["running"] = False
            EXECUTION_STATES[session_id]["paused"] = True
            EXECUTION_STATES[session_id]["current_phase"] = "stopping"
            
            # Also update session state if available
            session_manager = SessionStateManager(session_id, TMP_DIR)
            session_state = session_manager.load_state()
            if session_state:
                session_manager.update_execution_state(
                    session_state, is_running=False, is_paused=True, current_phase="stopping"
                )
                session_manager.save_state(session_state)
            
            return "⏸️ Execution stopped - click Run to resume with new input"
        else:
            logger.info(f"No active execution to stop for session {session_id}")
            return "⚪ No active execution to stop"
    else:
        logger.warning(f"No execution session found for {session_id}")
        return "❌ No execution session found"

def shutdown_sandbox(request: gr.Request):
    """Shutdown the sandbox and clear session state data while preserving user files"""
    session_id = request.session_hash
    logger.info(f"Shutting down sandbox and clearing session data for {session_id} (preserving user files)")
    
    try:
        # 1. Stop any running execution first
        if session_id in STOP_EVENTS:
            STOP_EVENTS[session_id].set()
            
        # 2. Shutdown Modal sandbox
        if session_id in SANDBOXES:
            logger.info(f"Killing Modal sandbox for session {session_id}")
            SANDBOXES[session_id].kill()
            SANDBOXES.pop(session_id)
            logger.info(f"Successfully shutdown sandbox for session {session_id}")
        
        # 3. Clear session state data but preserve user files
        session_manager = SessionStateManager(session_id, TMP_DIR)
        if session_manager.session_exists():
            logger.info(f"Clearing session state data for {session_id} (preserving user files)")
            
            # Load session state to show what's being cleared
            session_state = session_manager.load_state()
            if session_state:
                # Log what we're clearing
                stats = session_state.get("session_stats", {})
                llm_interactions = len(session_state.get("llm_interactions", []))
                tool_executions = len(session_state.get("tool_executions", []))
                
                logger.info(f"Clearing session {session_id}: "
                          f"{stats.get('total_messages', 0)} messages, "
                          f"{llm_interactions} LLM interactions, "
                          f"{tool_executions} tool executions, "
                          f"{stats.get('total_code_executions', 0)} code runs")
            
            # Only remove session state file, preserve other files
            if session_manager.state_file.exists():
                session_manager.state_file.unlink()
                logger.info(f"Removed session state file for {session_id}")
            
            # DON'T remove the session directory or user files - just log what's being preserved
            if session_manager.session_dir.exists():
                try:
                    # Count and log preserved files
                    preserved_files = []
                    for file_path in session_manager.session_dir.iterdir():
                        if file_path.is_file() and file_path.name != 'session_state.json':
                            preserved_files.append(file_path.name)
                    
                    if preserved_files:
                        logger.info(f"Preserving {len(preserved_files)} user files in {session_id}: {preserved_files}")
                    else:
                        logger.info(f"No user files to preserve in {session_id}")
                        
                except OSError as e:
                    logger.warning(f"Could not check session directory {session_id}: {e}")
        
        # 4. Clear global execution tracking
        if session_id in STOP_EVENTS:
            STOP_EVENTS.pop(session_id)
            logger.debug(f"Cleared stop event for {session_id}")
            
        if session_id in EXECUTION_STATES:
            EXECUTION_STATES.pop(session_id)
            logger.debug(f"Cleared execution state for {session_id}")
        
        # 5. Clear any legacy notebook files
        legacy_notebook_path = os.path.join(TMP_DIR, session_id, 'jupyter-agent.ipynb')
        if os.path.exists(legacy_notebook_path):
            os.remove(legacy_notebook_path)
            logger.debug(f"Removed legacy notebook file for {session_id}")
            
        logger.info(f"Complete shutdown finished for session {session_id} (user files preserved)")
        return gr.Button(visible=False)
        
    except Exception as e:
        logger.error(f"Error during shutdown for session {session_id}: {str(e)}")
        return f"❌ Error during shutdown: {str(e)}", gr.Button(visible=True)

# continue_execution function removed - functionality integrated into execute_jupyter_agent

def get_execution_status(request: gr.Request):
    """Get the current execution status for UI updates"""
    session_id = request.session_hash
    
    if session_id not in EXECUTION_STATES:
        return "⚪ Ready"
    
    state = EXECUTION_STATES[session_id]
    if state["running"]:
        if session_id in STOP_EVENTS and STOP_EVENTS[session_id].is_set():
            return "⏸️ Stopping..."
        else:
            # Check if we have more detailed phase information
            phase = state.get("current_phase", "running")
            if phase == "generating":
                return "🟢 Generating response..."
            elif phase == "executing_code":
                return "🟢 Executing code..."
            elif phase == "searching":
                return "🟢 Searching web..."
            else:
                return "🟢 Running"
    elif state.get("paused", False):
        return "⏸️ Paused - Click Run to continue"
    else:
        return "⚪ Ready"

def is_sandbox_active(request: gr.Request):
    """Check if sandbox is active for the current session"""
    session_id = request.session_hash
    return session_id in SANDBOXES

def get_sandbox_status_and_visibility(request: gr.Request):
    """Get sandbox status message and button visibility"""
    session_id = request.session_hash
    if session_id in SANDBOXES:
        return "🟢 Sandbox active", gr.Button(visible=True)
    else:
        return "⚪ No sandbox active", gr.Button(visible=False)

def update_sandbox_button_visibility(request: gr.Request):
    """Update only the button visibility based on sandbox status"""
    session_id = request.session_hash
    return gr.Button(visible=session_id in SANDBOXES)

def reset_ui_after_shutdown(request: gr.Request):
    """Reset UI components after complete shutdown"""
    session_id = request.session_hash
    
    # Check if session is truly cleared
    is_cleared = (session_id not in SANDBOXES and 
                 session_id not in EXECUTION_STATES and 
                 session_id not in STOP_EVENTS)
    
    if is_cleared:
        # Return reset state for all UI components
        return (
            init_notebook.render(),  # Reset notebook display
            [],  # Clear message state
            "⚪ Ready",  # Reset status
            "⚪ No sandbox active",  # Reset sandbox status
            gr.Button(visible=False)  # Hide shutdown button
        )
    else:
        # Return current state if not fully cleared
        status = get_execution_status(request)
        sandbox_status, button_vis = get_sandbox_status_and_visibility(request)
        return (
            init_notebook.render(),  # Still reset notebook display
            [],  # Still clear message state
            status,
            sandbox_status,
            button_vis
        )

def reconstruct_message_history_from_notebook(notebook_data):
    """Reconstruct message history from notebook cells"""
    message_history = []
    cells = notebook_data.get('cells', [])
    
    system_prompt = None
    current_conversation = []
    
    for cell in cells:
        cell_type = cell.get('cell_type', '')
        
        if cell_type == 'markdown':
            content = cell.get('source', '')
            if isinstance(content, list):
                content = ''.join(content)
            
            # Check if this is a system message
            if 'System' in content and 'IMPORTANT EXECUTION GUIDELINES' in content:
                # Extract the system prompt content
                system_content = content
                # Clean up the HTML and extract the actual content
                # Remove HTML tags and extract the text content
                clean_content = re.sub(r'<[^>]+>', '', system_content)
                clean_content = re.sub(r'\n+', '\n', clean_content).strip()
                system_prompt = clean_content
                
            elif 'User' in content and not any(word in content for word in ['Assistant', 'System']):
                # This is a user message
                # Extract the user content after the User header
                user_content = content.split('User')[1] if 'User' in content else content
                # Clean up HTML and formatting
                user_content = re.sub(r'<[^>]+>', '', user_content)
                user_content = re.sub(r'-{3,}', '', user_content)
                user_content = user_content.strip()
                
                if user_content:
                    current_conversation.append({
                        "role": "user", 
                        "content": user_content
                    })
                    
            elif 'Assistant' in content:
                # This is an assistant message
                assistant_content = content.split('Assistant')[1] if 'Assistant' in content else content
                # Clean up HTML and formatting
                assistant_content = re.sub(r'<[^>]+>', '', assistant_content)
                assistant_content = re.sub(r'-{3,}', '', assistant_content)
                assistant_content = assistant_content.strip()
                
                if assistant_content:
                    current_conversation.append({
                        "role": "assistant",
                        "content": assistant_content
                    })
    
    # Build the final message history
    if system_prompt:
        message_history.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add the conversation messages
    message_history.extend(current_conversation)
    
    return message_history

def load_previous_notebook(notebook_choice, request: gr.Request):
    """Load a previous notebook with complete session configuration (dev only)"""
    if not is_dev_environment():
        return (init_notebook.render(), [], "Load previous notebooks is only available in development mode",
                None, None, None, None, None, "", "", "", "", "", "", "", False)
    
    if not notebook_choice or notebook_choice == "None":
        return (init_notebook.render(), [], "Please select a notebook to load",
                None, None, None, None, None, "", "", "", "", "", "", "", False)
    
    try:
        # Parse the notebook choice to get the session ID
        session_id = notebook_choice.split(" ")[0]
        notebook_path = Path(TMP_DIR) / session_id / "jupyter-agent.ipynb"
        
        if not notebook_path.exists():
            return (init_notebook.render(), [], f"Notebook file not found: {notebook_path}",
                    None, None, None, None, None, "", "", "", "", "", "", "", False)
        
        # Load the notebook
        with open(notebook_path, 'r') as f:
            notebook_data = json.load(f)
        
        # Load session state 
        temp_session_manager = SessionStateManager(session_id, TMP_DIR)
        session_state = temp_session_manager.load_state()
        session_config = None  # For backward compatibility
        
        # Extract config from session state for UI restoration
        if session_state:
            session_config = {
                "hardware": session_state.get("hardware_config", {}),
                "environment_vars": session_state.get("environment", {}).get("variables", ""),
                "api_keys": {
                    "model_name": session_state.get("api_config", {}).get("model_name", "")
                }
            }
        
        # Create a new JupyterNotebook instance with the loaded data
        loaded_notebook = JupyterNotebook()
        loaded_notebook.data = notebook_data
        
        # Reconstruct message history from notebook cells
        message_history = reconstruct_message_history_from_notebook(notebook_data)
        
        # Store the loaded notebook info in session for continue functionality
        session_id_hash = request.session_hash
        if session_id_hash not in EXECUTION_STATES:
            EXECUTION_STATES[session_id_hash] = {}
        
        EXECUTION_STATES[session_id_hash]["loaded_notebook"] = {
            "notebook_data": notebook_data,
            "message_history": message_history,
            "original_session": session_id,
            "session_config": session_config
        }
        
        logger.info(f"Successfully loaded notebook from {notebook_path}")
        logger.info(f"Reconstructed message history with {len(message_history)} messages")
        
        # Prepare configuration values to restore UI state
        config_loaded = ""
        gpu_type = None
        cpu_cores = None
        memory_gb = None
        timeout_sec = None
        env_vars = ""
        modal_token_id = ""
        modal_token_secret = ""
        hf_token = ""
        provider_api_key = ""
        provider_api_endpoint = ""
        model_name = ""
        
        if session_config:
            hardware = session_config.get("hardware", {})
            gpu_type = hardware.get("gpu_type")
            cpu_cores = hardware.get("cpu_cores")
            memory_gb = hardware.get("memory_gb")
            timeout_sec = hardware.get("timeout_sec")
            env_vars = session_config.get("environment_vars", "")
            
            api_keys = session_config.get("api_keys", {})
            modal_token_id = api_keys.get("modal_token_id", "")
            modal_token_secret = api_keys.get("modal_token_secret", "")
            hf_token = api_keys.get("hf_token", "")
            provider_api_key = api_keys.get("provider_api_key", "")
            provider_api_endpoint = api_keys.get("provider_api_endpoint", "")
            model_name = api_keys.get("model_name", "")
            
            config_loaded = f"✅ Configuration restored: GPU={gpu_type}, CPU={cpu_cores}, Memory={memory_gb}GB, Timeout={timeout_sec}s"
        
        success_message = f"✅ Loaded notebook: {session_id} ({len(notebook_data.get('cells', []))} cells, {len(message_history)} messages)"
        if config_loaded:
            success_message += f"\n{config_loaded}"
        
        return (loaded_notebook.render(), message_history, success_message,
                gpu_type, cpu_cores, memory_gb, timeout_sec, env_vars,
                modal_token_id, modal_token_secret, hf_token, provider_api_key, provider_api_endpoint, model_name,
                "", False)  # Default empty tavily_api_key and False for enable_web_search
        
    except Exception as e:
        logger.error(f"Failed to load notebook {notebook_choice}: {str(e)}")
        error_message = f"❌ Failed to load notebook: {str(e)}"
        return (init_notebook.render(), [], error_message,
                None, None, None, None, None, "", "", "", "", "", "", "", False)

def get_notebook_options():
    """Get options for notebook dropdown (dev only)"""
    if not is_dev_environment():
        return ["Load previous notebooks is only available in development mode"]
    
    notebooks = get_previous_notebooks()
    if not notebooks:
        return ["No previous notebooks found"]
    
    options = ["None"] + [nb['display_name'] for nb in notebooks[:20]]  # Limit to 20 most recent
    return options

def refresh_notebook_options():
    """Refresh the notebook options dropdown"""
    return gr.Dropdown(choices=get_notebook_options(), value="None")

# Legacy session configuration functions removed - replaced by SessionStateManager
# All session data is now stored in a single comprehensive session_state.json file


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

/* Button states for execution control */
.button-executing {
    opacity: 0.6 !important;
    pointer-events: none !important;
    cursor: not-allowed !important;
}

.button-executing::after {
    content: " ⏳";
}

.status-running {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
"""


# Create the interface
with gr.Blocks() as demo:
    msg_state = gr.State(value=[])

    # Environment info display
    env_info = gr.Markdown(f"""
    **Environment**: {get_environment().upper()} | **Features**: {"Development features enabled" if is_dev_environment() else "Production mode"}
    """)

    html_output = gr.HTML(value=JupyterNotebook().render())
    
    user_input = gr.Textbox(
        # value="train a 5 neuron neural network to classify the iris dataset",
        value="can you finetune llama 3.2 1b on tiny stories dataset and using unsloth",
        lines=3,
        label="Agent task"
    )
    
    with gr.Accordion("Upload files ⬆ | Download notebook⬇", open=False):
        files = gr.File(label="Upload files to use", file_count="multiple")
        file = gr.File(TMP_DIR+"jupyter-agent.ipynb", label="Download Jupyter Notebook")


    with gr.Row():
        # Web Search Configuration
        with gr.Accordion("🔍 Web Search Settings", open=False):
            with gr.Row():
                enable_web_search = gr.Checkbox(
                    label="Enable Web Search",
                    value=bool(os.environ.get("TAVILY_API_KEY")),  # Default to True if API key is available
                    info="Allow the agent to search the web for current information and documentation"
                )
                
                # Show web search status with better formatting
                tavily_status = "✅ Available" if os.environ.get("TAVILY_API_KEY") else "❌ API Key Required"
                gr.Markdown(f"**Status:** {tavily_status}")
            
            gr.Markdown("""
            **Web Search Features:**
            - 🌐 Search for current tutorials, documentation, and best practices
            - 🐛 Find solutions to error messages and debugging help
            - 📚 Access up-to-date library documentation and examples
            - 💡 Get recent examples and code snippets from the web
            
            ⚠️ **Note**: Web search requires a Tavily API key. Get one free at [tavily.com](https://tavily.com)
            """)
        # Previous notebooks section (dev only)
        if is_dev_environment():
            with gr.Accordion("📂 Load Previous Notebook (Dev Only)", open=False):
                notebook_dropdown = gr.Dropdown(
                    choices=get_notebook_options(),
                    value="None",
                    label="Select Previous Notebook",
                    info="Load a previously created notebook session"
                )
                with gr.Row():
                    load_notebook_btn = gr.Button("📖 Load Selected", variant="secondary")
                    refresh_notebooks_btn = gr.Button("🔄 Refresh List", variant="secondary")
                
                load_status = gr.Textbox(
                    label="Load Status",
                    interactive=False,
                    visible=False
                )   
    # Check for missing API keys and show input fields conditionally
    missing_keys = get_missing_api_keys()
    
    # API Key Configuration (shown only if keys are missing)
    if missing_keys:
        with gr.Accordion("🔑 Required API Keys (Missing from .env)", open=True):
            gr.Markdown("""
            **⚠️ Some required API keys are missing from your .env file.**
            Please provide them below to use the application:
            """)
            
            api_key_components = {}
            
            if "MODAL_TOKEN_ID" in missing_keys:
                api_key_components["modal_token_id"] = gr.Textbox(
                    label="Modal Token ID",
                    placeholder="ak-...",
                    info="Modal Token ID for sandbox access",
                    type="password"
                )
            else:
                api_key_components["modal_token_id"] = gr.Textbox(visible=False)
                
            if "MODAL_TOKEN_SECRET" in missing_keys:
                api_key_components["modal_token_secret"] = gr.Textbox(
                    label="Modal Token Secret", 
                    placeholder="as-...",
                    info="Modal Token Secret for sandbox access",
                    type="password"
                )
            else:
                api_key_components["modal_token_secret"] = gr.Textbox(visible=False)
                
            if "HF_TOKEN" in missing_keys:
                api_key_components["hf_token"] = gr.Textbox(
                    label="Hugging Face Token (Optional)",
                    placeholder="hf_...",
                    info="Hugging Face Token for model access",
                    type="password"
                )
            else:
                api_key_components["hf_token"] = gr.Textbox(visible=False)
                
            if "PROVIDER_API_KEY" in missing_keys:
                api_key_components["provider_api_key"] = gr.Textbox(
                    label="AI Provider API Key",
                    placeholder="sk-, gsk_, or csk-...",
                    info="API Key for your AI provider (Anthropic, OpenAI, Cerebras, etc.)",
                    type="password"
                )
            else:
                api_key_components["provider_api_key"] = gr.Textbox(visible=False)
                
            if "PROVIDER_API_ENDPOINT" in missing_keys:
                api_key_components["provider_api_endpoint"] = gr.Textbox(
                    label="AI Provider API Endpoint",
                    placeholder="https://api.anthropic.com/v1/",
                    info="API endpoint for your AI provider"
                )
            else:
                api_key_components["provider_api_endpoint"] = gr.Textbox(visible=False)
                
            if "MODEL_NAME" in missing_keys:
                api_key_components["model_name"] = gr.Textbox(
                    label="Model Name",
                    placeholder="claude-sonnet-4-20250514",
                    info="Name of the model to use"
                )
            else:
                api_key_components["model_name"] = gr.Textbox(visible=False)
                
            if "TAVILY_API_KEY" in missing_keys:
                api_key_components["tavily_api_key"] = gr.Textbox(
                    label="Tavily API Key (Optional)",
                    placeholder="tvly-...",
                    info="Tavily API Key for web search functionality",
                    type="password"
                )
            else:
                api_key_components["tavily_api_key"] = gr.Textbox(visible=False)
    else:
        # Create hidden components when no keys are missing
        api_key_components = {
            "modal_token_id": gr.Textbox(visible=False),
            "modal_token_secret": gr.Textbox(visible=False),
            "hf_token": gr.Textbox(visible=False),
            "provider_api_key": gr.Textbox(visible=False),
            "provider_api_endpoint": gr.Textbox(visible=False),
            "model_name": gr.Textbox(visible=False),
            "tavily_api_key": gr.Textbox(visible=False)
        }
    

    

    
    with gr.Accordion("Hardware Configuration ⚙️", open=False):
        
        env_vars = gr.Textbox(
                label="Environment Variables",
                placeholder="Enter environment variables (one per line):\nAPI_KEY=your_key_here\nDATA_PATH=/path/to/data\nDEBUG=true",
                lines=5,
                info="Add custom environment variables for the sandbox. Format: KEY=value (one per line)"
            )
            
        env_info = gr.Markdown("""
            **Environment Variables Info:**
            - Variables will be available in the sandbox environment
            - Use KEY=value format, one per line
            - Common examples: API keys, data paths, configuration flags
            - Variables are session-specific and not persisted between sessions
            
            ⚠️ **Security**: Avoid sensitive credentials in shared environments
            """)
        
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
    
        # with gr.Accordion("Environment Variables 🔧", open=False):
            
            
    with gr.Row():
        generate_btn = gr.Button("Run!", variant="primary")
        stop_btn = gr.Button("⏸️ Stop", variant="secondary")
        # continue_btn removed - Run button handles continuation automatically
        clear_btn = gr.Button("Clear Notebook", variant="stop")
        shutdown_btn = gr.Button("🔴 Shutdown Sandbox", variant="stop", visible=False)
    
    # Status display
    status_display = gr.Textbox(
        value="⚪ Ready", 
        label="Execution Status", 
        interactive=False,
        max_lines=1
    )   

    generate_btn.click(
        fn=execute_jupyter_agent,
        inputs=[
            user_input, files, msg_state, gpu_type, cpu_cores, memory_gb, timeout_sec, env_vars,
            api_key_components["modal_token_id"], api_key_components["modal_token_secret"], 
            api_key_components["hf_token"], api_key_components["provider_api_key"], 
            api_key_components["provider_api_endpoint"], api_key_components["model_name"],
            api_key_components["tavily_api_key"], enable_web_search
        ],
        outputs=[html_output, msg_state, file],
        show_progress="hidden",
    )

    stop_btn.click(
        fn=stop_execution,
        outputs=[status_display],
        show_progress="hidden",
    )

    # continue_btn.click handler removed - Run button handles continuation automatically

    clear_btn.click(fn=clear, inputs=[msg_state], outputs=[html_output, msg_state])

    shutdown_btn.click(
        fn=shutdown_sandbox,
        outputs=[shutdown_btn],
        show_progress="hidden",
    )
    
    # Add event handlers for notebook loading (dev only)
    if is_dev_environment():
        load_notebook_btn.click(
            fn=load_previous_notebook,
            inputs=[notebook_dropdown],
            outputs=[
                html_output, msg_state, load_status,
                gpu_type, cpu_cores, memory_gb, timeout_sec, env_vars,
                api_key_components["modal_token_id"], api_key_components["modal_token_secret"], 
                api_key_components["hf_token"], api_key_components["provider_api_key"], 
                api_key_components["provider_api_endpoint"], api_key_components["model_name"],
                api_key_components["tavily_api_key"], enable_web_search
            ],
            show_progress="hidden"
        )
        
        refresh_notebooks_btn.click(
            fn=refresh_notebook_options,
            outputs=[notebook_dropdown],
            show_progress="hidden"
        )
        
        # Show/hide load status based on selection
        notebook_dropdown.change(
            fn=lambda choice: gr.Textbox(visible=choice != "None"),
            inputs=[notebook_dropdown],
            outputs=[load_status]
        )

    # Periodic status update using timer
    status_timer = gr.Timer(2.0)  # Update every 2 seconds
    status_timer.tick(
        fn=get_execution_status,
        outputs=[status_display],
        show_progress="hidden"
    )

    # Update button visibility periodically
    button_timer = gr.Timer(3.0)  # Check every 3 seconds
    button_timer.tick(
        fn=update_sandbox_button_visibility,
        outputs=[shutdown_btn],
        show_progress="hidden"
    )

    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=""" () => {
    if (document.querySelectorAll('.dark').length) {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    }
    
    // Add execution state management functions
    window.setExecutionState = function(isExecuting) {
        // Find Run button by text content since variant attribute might not be reliable
        const buttons = document.querySelectorAll('button');
        let runButton = null;
        let stopButton = null;
        
        buttons.forEach(button => {
            const text = button.textContent.trim().toLowerCase();
            if (text.includes('run') && !text.includes('stop')) {
                runButton = button;
            } else if (text.includes('stop') || text.includes('⏸️')) {
                stopButton = button;
            }
        });
        
        if (runButton) {
            if (isExecuting) {
                runButton.classList.add('button-executing');
                runButton.disabled = true;
                runButton.style.opacity = '0.6';
                runButton.style.cursor = 'not-allowed';
                runButton.style.pointerEvents = 'none';
                if (runButton.textContent.indexOf('⏳') === -1) {
                    runButton.textContent = runButton.textContent.replace('!', '! ⏳');
                }
            } else {
                runButton.classList.remove('button-executing');
                runButton.disabled = false;
                runButton.style.opacity = '1';
                runButton.style.cursor = 'pointer';
                runButton.style.pointerEvents = 'auto';
                runButton.textContent = runButton.textContent.replace(' ⏳', '');
            }
        }
        
        // Also update stop button visibility/state
        if (stopButton) {
            stopButton.style.display = isExecuting ? 'block' : 'inline-block';
        }
    };
    
    // Monitor for status changes and update button states
    window.monitorExecutionStatus = function() {
        // Try multiple ways to find the status element
        let statusElement = document.querySelector('input[label*="Execution Status"], input[label*="Status"], textarea[label*="Status"]');
        
        if (!statusElement) {
            // Fallback: look for any input that might contain status
            const allInputs = document.querySelectorAll('input, textarea');
            allInputs.forEach(input => {
                if (input.value && (input.value.includes('🟢') || input.value.includes('⚪') || input.value.includes('⏸️'))) {
                    statusElement = input;
                }
            });
        }
        
        if (statusElement) {
            const status = statusElement.value || '';
            const isRunning = status.includes('🟢') || status.includes('Running') || status.includes('Generating') || status.includes('Executing');
            const isReady = status.includes('⚪') || status.includes('Ready');
            
            window.setExecutionState(isRunning);
            
            // Add visual indicator to status element
            if (isRunning) {
                statusElement.style.background = '#e3f2fd';
                statusElement.style.borderColor = '#2196f3';
            } else if (isReady) {
                statusElement.style.background = '#f5f5f5';
                statusElement.style.borderColor = '#ccc';
            } else {
                statusElement.style.background = '#fff3e0';
                statusElement.style.borderColor = '#ff9800';
            }
        }
    };
    
    // Set up mutation observer to watch for status changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' || mutation.type === 'attributes') {
                setTimeout(window.monitorExecutionStatus, 100);
            }
        });
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true
    });
}
"""
    )

logger.info("Starting Gradio application")
demo.launch(ssr_mode=False)
