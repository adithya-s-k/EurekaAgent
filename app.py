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
from utils import (
    run_interactive_notebook,
)

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
                    
                    # Try to load session configuration for additional info
                    config_info = ""
                    try:
                        session_config = load_session_configuration(session_dir.name)
                        if session_config:
                            hardware = session_config.get("hardware", {})
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
TMP_DIR = './temp/'
# model="Qwen/Qwen3-Coder-480B-A35B-Instruct:cerebras"
# model="qwen-3-coder-480b"


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
    EXECUTION_STATES[session_id] = {"running": True, "paused": False}

    # Set up save directory early for notifications
    session_dir = os.path.join(TMP_DIR, request.session_hash)
    os.makedirs(session_dir, exist_ok=True)
    save_dir = os.path.join(session_dir, 'jupyter-agent.ipynb')
    
    # Create initial notebook file so it exists for Gradio
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(init_notebook.data, f, indent=2)
    logger.info(f"Initialized notebook for session {session_id}")
    
    # Save session configuration for future loading
    save_session_configuration(
        session_id=session_id,
        gpu_type=gpu_type,
        cpu_cores=cpu_cores, 
        memory_gb=memory_gb,
        timeout_sec=timeout_sec,
        env_vars_text=env_vars_text,
        modal_token_id=modal_token_id,
        modal_token_secret=modal_token_secret,
        hf_token=hf_token,
        provider_api_key=provider_api_key,
        provider_api_endpoint=provider_api_endpoint,
        model_name=user_model_name,
        files=files
    )

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

    # Check if this session has a loaded notebook with existing message history
    has_loaded_notebook = (session_id in EXECUTION_STATES and 
                           "loaded_notebook" in EXECUTION_STATES[session_id])
    
    # Initialize message_history if it doesn't exist and no loaded notebook
    if len(message_history) == 0 and not has_loaded_notebook:
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
        sytem_prompt = DEFAULT_SYSTEM_PROMPT.replace("{AVAILABLE_FILES}", files_section)
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
    elif has_loaded_notebook:
        logger.info(f"Using loaded notebook conversation for session {session_id} (history length: {len(message_history)})")
        # Clear the loaded notebook state after using it once
        if "loaded_notebook" in EXECUTION_STATES[session_id]:
            del EXECUTION_STATES[session_id]["loaded_notebook"]
    else:
        logger.info(f"Continuing existing conversation for session {session_id} (history length: {len(message_history)})")
    
    # Add user input if provided and not already added by continue_execution
    if user_input and user_input.strip():
        # Check if this input was already added by continue_execution
        if not (message_history and 
                message_history[-1].get("role") == "user" and 
                message_history[-1].get("content") == user_input):
            message_history.append({"role": "user", "content": user_input})
            logger.debug(f"Added user message to history for session {session_id}")
        else:
            logger.debug(f"User message already in history for session {session_id}")

    logger.debug(f"Message history for session {session_id}: {len(message_history)} messages")

    # Determine which tools to use based on web search toggle
    from utils import TOOLS
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
    try:
        for notebook_html, notebook_data, messages in run_interactive_notebook(
            client, model_name, message_history, sbx, STOP_EVENTS[session_id], selected_tools
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

def shutdown_sandbox(request: gr.Request):
    """Shutdown the sandbox for this session"""
    session_id = request.session_hash
    logger.info(f"Shutting down sandbox for session {session_id}")
    
    if session_id in SANDBOXES:
        try:
            logger.info(f"Killing Modal sandbox for session {session_id}")
            SANDBOXES[session_id].kill()
            SANDBOXES.pop(session_id)
            logger.info(f"Successfully shutdown sandbox for session {session_id}")
            
            # Clean up stop events and execution states
            if session_id in STOP_EVENTS:
                STOP_EVENTS.pop(session_id)
            if session_id in EXECUTION_STATES:
                EXECUTION_STATES.pop(session_id)
                
            return "🔴 Sandbox shutdown successfully", gr.Button(visible=False)
        except Exception as e:
            logger.error(f"Error shutting down sandbox for session {session_id}: {str(e)}")
            return f"❌ Error shutting down sandbox: {str(e)}", gr.Button(visible=True)
    else:
        logger.info(f"No active sandbox found for session {session_id}")
        return "⚪ No active sandbox to shutdown", gr.Button(visible=False)

def continue_execution(user_input, files, message_history, gpu_type, cpu_cores, memory_gb, timeout_sec, env_vars_text, 
                      modal_token_id, modal_token_secret, hf_token, provider_api_key, provider_api_endpoint, user_model_name,
                      tavily_api_key, enable_web_search, request: gr.Request):
    """Continue execution after it was stopped"""
    session_id = request.session_hash
    logger.info(f"Continuing execution for session {session_id}")
    
    # Check if there's a loaded notebook for this session
    if (session_id in EXECUTION_STATES and 
        "loaded_notebook" in EXECUTION_STATES[session_id]):
        
        loaded_info = EXECUTION_STATES[session_id]["loaded_notebook"]
        loaded_message_history = loaded_info["message_history"]
        original_session = loaded_info["original_session"]
        
        logger.info(f"Found loaded notebook from session {original_session} with {len(loaded_message_history)} messages")
        
        # Use the loaded message history instead of the current one
        message_history = loaded_message_history.copy()
        
        # If user provided new input, add it to the loaded history
        if user_input and user_input.strip():
            message_history.append({"role": "user", "content": user_input})
            logger.info(f"Added new user input to loaded message history")
    
    # Reset stop event and execution state
    if session_id in STOP_EVENTS:
        STOP_EVENTS[session_id].clear()
        logger.info(f"Cleared stop event for session {session_id}")
    
    if session_id in EXECUTION_STATES:
        EXECUTION_STATES[session_id]["running"] = True
        EXECUTION_STATES[session_id]["paused"] = False
        logger.info(f"Reset execution state for session {session_id}")
    
    # Continue with normal execution - yield from the generator
    yield from execute_jupyter_agent(user_input, files, message_history, gpu_type, cpu_cores, memory_gb, timeout_sec, env_vars_text,
                                    modal_token_id, modal_token_secret, hf_token, provider_api_key, provider_api_endpoint, user_model_name,
                                    tavily_api_key, enable_web_search, request)

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
        
        # Load session configuration
        session_config = load_session_configuration(session_id)
        
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

def save_session_configuration(session_id, gpu_type, cpu_cores, memory_gb, timeout_sec, 
                               env_vars_text, modal_token_id, modal_token_secret, hf_token,
                               provider_api_key, provider_api_endpoint, model_name, files):
    """Save the complete session configuration to a JSON file"""
    try:
        config = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "gpu_type": gpu_type,
                "cpu_cores": cpu_cores,
                "memory_gb": memory_gb,
                "timeout_sec": timeout_sec
            },
            "environment_vars": env_vars_text,
            "api_keys": {
                "modal_token_id": modal_token_id if modal_token_id else "",
                "modal_token_secret": modal_token_secret if modal_token_secret else "",
                "hf_token": hf_token if hf_token else "",
                "provider_api_key": provider_api_key if provider_api_key else "",
                "provider_api_endpoint": provider_api_endpoint if provider_api_endpoint else "",
                "model_name": model_name if model_name else ""
            },
            "files": [Path(f).name for f in files] if files else [],
            "environment": get_environment()
        }
        
        # Save to session directory
        session_dir = Path(TMP_DIR) / session_id
        config_file = session_dir / "session_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Saved session configuration for {session_id}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to save session configuration: {str(e)}")
        return None

def load_session_configuration(session_id):
    """Load the complete session configuration from a JSON file"""
    try:
        session_dir = Path(TMP_DIR) / session_id
        config_file = session_dir / "session_config.json"
        
        if not config_file.exists():
            logger.warning(f"No session configuration found for {session_id}")
            return None
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Loaded session configuration for {session_id}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load session configuration: {str(e)}")
        return None


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

    # Environment info display
    env_info = gr.Markdown(f"""
    **Environment**: {get_environment().upper()} | **Features**: {"Development features enabled" if is_dev_environment() else "Production mode"}
    """)

    html_output = gr.HTML(value=JupyterNotebook().render())
    
    user_input = gr.Textbox(
        value="train a 5 neuron neural network to classify the iris dataset",
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
                    placeholder="claude-sonnet-4",
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
        continue_btn = gr.Button("▶️ Continue", variant="secondary")
        clear_btn = gr.Button("Clear Notebook", variant="stop")
    
    with gr.Row():
        shutdown_btn = gr.Button("🔴 Shutdown Sandbox", variant="stop", visible=False)
        sandbox_status = gr.Textbox(
            show_label=False,
            value="⚪ No sandbox active", 
            # label="Sandbox Status", 
            interactive=False,
            max_lines=1
        )
    
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

    continue_btn.click(
        fn=continue_execution,
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

    clear_btn.click(fn=clear, inputs=[msg_state], outputs=[html_output, msg_state])

    shutdown_btn.click(
        fn=shutdown_sandbox,
        outputs=[sandbox_status, shutdown_btn],
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

    # Periodic status update
    # demo.load(
    #     fn=get_execution_status,
    #     inputs=None,
    #     outputs=[status_display],
    #     every=2,  # Update every 2 seconds
    #     show_progress="hidden"
    # )

    # # Update button visibility periodically
    # demo.load(
    #     fn=update_sandbox_button_visibility,
    #     outputs=[shutdown_btn],
    #     every=3,  # Check every 3 seconds
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
