"""
Modal Sandbox wrapper to provide E2B-compatible interface for the Jupyter Agent.
Simplified implementation using Modal's native API.
"""

import modal
import datetime
from typing import Optional, Dict, List
import json
import logging
import time

logger = logging.getLogger(__name__)


class ModalResult:
    """Mock E2B result structure for displaying outputs like plots"""
    
    def __init__(self, text: str = "", html: str = "", png: str = "", svg: str = "", 
                 jpeg: str = "", pdf: str = "", latex: str = "", json: str = "",
                 javascript: str = "", is_main_result: bool = True):
        self.text = text
        self.html = html
        self.png = png
        self.svg = svg
        self.jpeg = jpeg
        self.pdf = pdf
        self.latex = latex
        self.json = json
        self.javascript = javascript
        self.is_main_result = is_main_result

class ModalExecution:
    """Mock E2B execution result to maintain compatibility with existing code"""
    
    def __init__(self, stdout: str = "", stderr: str = "", error: Optional[Dict] = None, results: List[ModalResult] = None):
        self.logs = ModalLogs(stdout, stderr)
        self.error = ModalError(error) if error else None
        self.results = results or []
        self.execution_count = 1

class ModalLogs:
    """Mock E2B logs structure"""
    
    def __init__(self, stdout: str = "", stderr: str = ""):
        self.stdout = [stdout] if stdout else []
        self.stderr = [stderr] if stderr else []

class ModalError:
    """Mock E2B error structure"""
    
    def __init__(self, error_data: Dict):
        self.name = error_data.get('name', 'Error')
        self.value = error_data.get('value', 'Unknown error')
        self.traceback = error_data.get('traceback', f"{self.name}: {self.value}")

class ModalFiles:
    """Simplified Modal files interface using native Modal Sandbox API"""
    
    def __init__(self, modal_sandbox):
        self.modal_sandbox = modal_sandbox  # ModalSandbox wrapper
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
    
    @property
    def _sandbox(self):
        """Get the actual Modal sandbox instance"""
        return self.modal_sandbox._sandbox
    
    def write(self, path: str, content):
        """Write file to Modal sandbox using native Modal API"""
        try:
            # Handle file-like objects
            if hasattr(content, 'read'):
                file_content = content.read()
                # Reset file pointer if possible
                if hasattr(content, 'seek'):
                    content.seek(0)
            else:
                file_content = content
            
            # Check file size for bytes content
            content_size = len(file_content) if isinstance(file_content, (bytes, str)) else 0
            if content_size > self.max_file_size:
                raise ValueError(f"File size ({content_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)")
            
            # Use Modal's native file API
            if isinstance(file_content, bytes):
                # Write binary content
                with self._sandbox.open(path, "wb") as f:
                    f.write(file_content)
            else:
                # Write text content
                with self._sandbox.open(path, "w") as f:
                    f.write(str(file_content))
                
            logger.debug(f"Successfully wrote file {path} ({content_size} bytes) using Modal native API")
            
        except Exception as e:
            logger.error(f"Failed to write file {path}: {str(e)}")
            raise RuntimeError(f"Could not write file {path}: {str(e)}")
    
    def read(self, path: str, mode: str = "r"):
        """Read file from Modal sandbox using native API"""
        try:
            with self._sandbox.open(path, mode) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {path}: {str(e)}")
            raise
    
    def exists(self, path: str) -> bool:
        """Check if file exists in Modal sandbox"""
        try:
            # Try to open the file to check existence
            with self._sandbox.open(path, "r"):
                pass
            return True
        except Exception:
            return False
    
    def list_files(self, directory: str = ".") -> List[str]:
        """List files in directory using Modal's native ls method"""
        try:
            return self._sandbox.ls(directory)
        except Exception as e:
            logger.error(f"Failed to list files in {directory}: {str(e)}")
            return []
    
    def verify_file_upload(self, path: str, expected_size: Optional[int] = None) -> bool:
        """Verify that a file was uploaded correctly"""
        try:
            if not self.exists(path):
                logger.error(f"File {path} does not exist after upload")
                return False
            
            # Check file size if expected size is provided
            if expected_size is not None:
                # Use Modal's exec to get file size
                result = self._sandbox.exec("wc", "-c", path)
                result.wait()
                
                if result.returncode == 0:
                    output = result.stdout.read().strip()
                    actual_size = int(output.split()[0])
                    if actual_size != expected_size:
                        logger.error(f"File {path} size mismatch: expected {expected_size}, got {actual_size}")
                        return False
                    else:
                        logger.debug(f"File {path} size verified: {actual_size} bytes")
                else:
                    logger.warning(f"Could not verify file size for {path}")
            
            logger.debug(f"File {path} upload verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify file upload {path}: {str(e)}")
            return False

class ModalSandboxInfo:
    """Mock E2B sandbox info for countdown timer"""
    
    def __init__(self, timeout_seconds: int = 300):
        self.started_at = datetime.datetime.now(datetime.timezone.utc)
        self.end_at = self.started_at + datetime.timedelta(seconds=timeout_seconds)

class ModalSandbox:
    """Modal sandbox wrapper that provides E2B-compatible interface"""
    
    def __init__(self, gpu_config: str = "cpu", cpu_cores: float = 2.0, memory_mb: int = 8192, 
                 timeout: int = 300, environment_vars: Dict[str, str] = None):
        """
        Initialize Modal sandbox with hardware configuration
        
        Args:
            gpu_config: GPU configuration (e.g., "cpu", "T4", "A100-40GB", "H100")
            cpu_cores: Number of CPU cores
            memory_mb: Memory in MB
            timeout: Timeout in seconds
            environment_vars: Environment variables to set
        """
        self.gpu_config = gpu_config
        self.cpu_cores = cpu_cores
        self.memory_mb = memory_mb
        self.timeout = timeout
        self.environment_vars = environment_vars or {}
        self.files = ModalFiles(self)
        self._sandbox = None
        self._app = None
        self._sandbox_info = ModalSandboxInfo(timeout)
        self._persistent_session = None  # For maintaining state across executions
        
        # Define package lists for different hardware configurations
        CPU_PACKAGES = [
            "jupyter-server", "ipykernel", "ipython", "orjson", "pandas", 
            "matplotlib", "pillow", "numpy", "scipy", "scikit-learn", 
            "seaborn", "plotly", "requests", "beautifulsoup4", "opencv-python", 
            "nltk", "textblob", "librosa>=0.10.0", "soundfile", "sympy", "xarray"
        ]
        
        GPU_PACKAGES = [
            "jupyter-server", "ipykernel", "ipython", "orjson", "pandas", 
            "matplotlib", "pillow", "numpy", "scipy", "scikit-learn", 
            "seaborn", "plotly", "requests", "beautifulsoup4", "opencv-python", 
            "nltk", "textblob", "librosa>=0.10.0", "soundfile", "sympy", "xarray",
            # GPU-specific ML/AI packages
            "torch", "transformers", "datasets", "bitsandbytes", "hf_transfer", 
            "peft", "trl", "accelerate", "xformers", "wandb", "deepspeed", 
            "pyyaml", "packaging", "rouge_score", "bert_score", "jiwer", 
            "tqdm", "pyarrow", "sentencepiece", "protobuf", "huggingface_hub"
        ]
        
        # Store package lists for system prompt
        self.available_packages = GPU_PACKAGES if gpu_config != "cpu" else CPU_PACKAGES
        
        # Create appropriate image based on hardware configuration
        if gpu_config == "cpu" or gpu_config == "CPU-only":
            self.base_image = self._create_cpu_image(CPU_PACKAGES)
        else:
            self.base_image = self._create_gpu_image(GPU_PACKAGES)
        
        self._setup_modal()
        logger.info(f"Initialized Modal sandbox with {gpu_config} GPU, {cpu_cores} CPU cores, {memory_mb}MB RAM")
    
    def _create_cpu_image(self, packages):
        """Create CPU-optimized image with basic data science packages"""
        packages_string = " ".join(packages)
        return (modal.Image.debian_slim()
                .apt_install("git", "build-essential")
                .run_commands("pip install --upgrade pip")
                .run_commands("pip install uv")
                .run_commands("uv pip install 'numba>=0.58.0' --system")  # Ensure compatible numba version
                .run_commands(f"uv pip install {packages_string} --system"))
    
    def _create_gpu_image(self, packages):
        """Create GPU-optimized image with ML/AI packages including PyTorch and Transformers"""
        # CUDA Configuration for SGLang
        CUDA_VERSION = "12.8.1"
        CUDA_FLAVOR = "devel"
        CUDA_OS = "ubuntu24.04"
        CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"
        
        # Base packages that don't require special handling
        base_packages = [pkg for pkg in packages if pkg not in [
            "torch", "transformers", "bitsandbytes", "accelerate", "xformers", 
            "peft", "trl", "unsloth", "deepspeed"
        ]]
        base_packages_string = " ".join(base_packages)
        
        return (modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
                .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
                .run_commands("ln -fs /usr/share/zoneinfo/UTC /etc/localtime")
                .apt_install("git", "build-essential")
                .run_commands("pip install --upgrade pip")
                .run_commands("pip install uv")
                .run_commands("uv pip install 'numba>=0.58.0' --system")  # Ensure compatible numba version
                .run_commands("uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --system")
                .run_commands(f"uv pip install {base_packages_string} --system")
                .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
    
    def _setup_modal(self):
        """Setup Modal app and sandbox configuration"""
        try:
            # Initialize Modal app using lookup to create if missing
            self._app = modal.App.lookup("jupyter-agent", create_if_missing=True)
            
            # Configure hardware based on user selection
            sandbox_kwargs = {
                "image": self.base_image,
                "timeout": self.timeout,
                "cpu": self.cpu_cores,
                "memory": self.memory_mb,
                "app": self._app
            }
            
            # Add GPU configuration if not CPU-only
            if self.gpu_config != "cpu" and self.gpu_config != "CPU-only":
                if self.gpu_config == "T4":
                    sandbox_kwargs["gpu"] = modal.gpu.T4()
                elif self.gpu_config == "L4":
                    sandbox_kwargs["gpu"] = modal.gpu.L4()
                elif self.gpu_config == "A100-40GB":
                    sandbox_kwargs["gpu"] = modal.gpu.A100(size="40GB")
                elif self.gpu_config == "A100-80GB":
                    sandbox_kwargs["gpu"] = modal.gpu.A100(size="80GB")
                elif self.gpu_config == "H100":
                    sandbox_kwargs["gpu"] = modal.gpu.H100()
                else:
                    print(f"Warning: Unknown GPU config {self.gpu_config}, falling back to CPU")
            
            # Add environment variables
            if self.environment_vars:
                sandbox_kwargs["secrets"] = [
                    modal.Secret.from_dict(self.environment_vars)
                ]
            
            # Create sandbox
            self._sandbox = modal.Sandbox.create(**sandbox_kwargs)
            
        except Exception as e:
            print(f"Error setting up Modal sandbox: {e}")
            raise
    
    def _initialize_persistent_session(self):
        """Initialize a persistent Python session for stateful execution using file-based communication"""
        if self._persistent_session is not None:
            return  # Session already exists
        
        try:
            logger.debug("Initializing persistent Python session with file-based communication")
            
            # Create a persistent Python script that monitors for command files
            session_script = '''
import sys
import json
import traceback
import base64
import io
import time
import os
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

# Global namespace to maintain state - includes built-ins for better compatibility
_global_namespace = {
    '__builtins__': __builtins__,
    '__name__': '__main__',
    '__doc__': None,
    '__package__': None
}

# Store original show function and setup plot capture
_original_show = plt.show
_captured_figures = []

def _capture_show(*args, **kwargs):
    """Custom show function that captures figures as base64"""
    global _captured_figures
    try:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            _captured_figures.append(img_base64)
            buf.close()
            plt.close(fig)
    except Exception as e:
        print(f"Error capturing plot: {e}", file=sys.stderr)

# Replace plt.show with our capture function
plt.show = _capture_show

# Signal that session is ready
with open("/tmp/session_ready", "w") as f:
    f.write("READY")

print("Persistent Python session started", flush=True)

# Process commands by monitoring for command files
while True:
    try:
        if os.path.exists("/tmp/execute_command"):
            # Read and execute command
            with open("/tmp/execute_command", "r") as f:
                content = f.read().strip()
                if not content:
                    continue  # Skip empty files
                try:
                    command = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Invalid JSON in command file: {content[:100]}...", file=sys.stderr)
                    continue  # Skip malformed JSON
            
            # Remove command file
            os.remove("/tmp/execute_command")
            
            if command.get("action") == "execute":
                code = command.get("code", "")
                _captured_figures = []  # Reset for this execution
                
                try:
                    # Check if code contains shell commands (lines starting with !)
                    lines = code.strip().split('\\n')
                    shell_commands = []
                    python_code_lines = []
                    
                    for line in lines:
                        stripped_line = line.strip()
                        if stripped_line.startswith('!'):
                            # This is a shell command
                            shell_cmd = stripped_line[1:].strip()  # Remove the !
                            shell_commands.append(shell_cmd)
                        else:
                            # This is Python code
                            python_code_lines.append(line)
                    
                    stdout_parts = []
                    stderr_parts = []
                    
                    # Execute shell commands first
                    for shell_cmd in shell_commands:
                        try:
                            import subprocess
                            result = subprocess.run(
                                shell_cmd,
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=30  # 30 second timeout for shell commands
                            )
                            
                            if result.stdout:
                                stdout_parts.append(f"$ {shell_cmd}")
                                stdout_parts.append(result.stdout.rstrip())
                            
                            if result.stderr:
                                stderr_parts.append(f"$ {shell_cmd}")
                                stderr_parts.append(result.stderr.rstrip())
                            
                            # If command failed, add error info
                            if result.returncode != 0:
                                stderr_parts.append(f"Command exited with code {result.returncode}")
                                
                        except subprocess.TimeoutExpired:
                            stderr_parts.append(f"$ {shell_cmd}")
                            stderr_parts.append("Command timed out after 30 seconds")
                        except Exception as e:
                            stderr_parts.append(f"$ {shell_cmd}")
                            stderr_parts.append(f"Error executing shell command: {str(e)}")
                    
                    # Execute Python code if present
                    python_stdout = ""
                    if python_code_lines and any(line.strip() for line in python_code_lines):
                        python_code = '\\n'.join(python_code_lines)
                        
                        # Capture stdout during Python execution
                        import io
                        from contextlib import redirect_stdout
                        
                        stdout_buffer = io.StringIO()
                        
                        with redirect_stdout(stdout_buffer):
                            # Execute code in the persistent namespace
                            exec(python_code, _global_namespace)
                        
                        python_stdout = stdout_buffer.getvalue()
                    
                    # Combine all stdout
                    all_stdout_parts = stdout_parts.copy()
                    if python_stdout:
                        all_stdout_parts.append(python_stdout.rstrip())
                    
                    stdout_output = '\\n'.join(all_stdout_parts) if all_stdout_parts else ""
                    stderr_output = '\\n'.join(stderr_parts) if stderr_parts else ""
                    
                    # Send results back
                    result = {
                        "status": "success",
                        "stdout": stdout_output,
                        "stderr": stderr_output,
                        "plots": _captured_figures.copy()
                    }
                    
                    with open("/tmp/execute_result", "w") as f:
                        f.write(json.dumps(result))
                    
                except Exception as e:
                    error_result = {
                        "status": "error",
                        "error": {
                            "name": type(e).__name__,
                            "value": str(e),
                            "traceback": traceback.format_exc()
                        }
                    }
                    
                    with open("/tmp/execute_result", "w") as f:
                        f.write(json.dumps(error_result))
            
            elif command.get("action") == "terminate":
                break
        
        else:
            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Session error: {e}", file=sys.stderr)
        # Write error to result file
        error_result = {
            "status": "error",
            "error": {
                "name": type(e).__name__,
                "value": str(e),
                "traceback": traceback.format_exc()
            }
        }
        with open("/tmp/execute_result", "w") as f:
            f.write(json.dumps(error_result))
'''
            
            # Start the persistent Python session (no stdin needed)
            self._persistent_session = self._sandbox.exec(
                "python3", "-c", session_script,
                timeout=None  # No timeout for persistent session
            )
            
            # Wait for the session to be ready by checking for the ready file
            max_wait = 10  # Wait up to 10 seconds
            for _ in range(max_wait * 10):  # Check every 0.1 seconds
                try:
                    with self._sandbox.open("/tmp/session_ready", "r") as f:
                        if f.read().strip() == "READY":
                            logger.info("Persistent Python session initialized successfully")
                            return
                except Exception:
                    pass
                time.sleep(0.1)
            
            raise RuntimeError("Failed to initialize persistent session: timeout waiting for ready signal")
            
        except Exception as e:
            logger.error(f"Failed to initialize persistent session: {e}")
            self._persistent_session = None
            raise
    
    def run_code(self, code: str, on_stdout=None) -> ModalExecution:
        """
        Execute Python code or shell commands in persistent Modal sandbox session using file-based communication
        
        Args:
            code: Python code to execute (lines starting with '!' are treated as shell commands)
            on_stdout: Callback for stdout (for compatibility, not fully implemented)
            
        Returns:
            ModalExecution object compatible with E2B execution results
        """
        try:
            if not self._sandbox:
                raise RuntimeError("Sandbox not initialized")
            
            # Initialize persistent session if not already done
            if self._persistent_session is None:
                self._initialize_persistent_session()
            
            logger.debug(f"Executing code in persistent session ({len(code)} chars)")
            
            # Clean up any existing command/result files
            try:
                self._sandbox.exec("rm", "-f", "/tmp/execute_command", "/tmp/execute_result").wait()
            except Exception:
                pass  # Ignore cleanup errors
            
            # Send execution command via file
            command = {
                "action": "execute",
                "code": code
            }
            
            with self._sandbox.open("/tmp/execute_command", "w") as f:
                f.write(json.dumps(command))
            
            # Small delay to ensure file is fully written
            time.sleep(0.01)
            
            # Wait for result file to appear
            max_wait = 30  # Wait up to 30 seconds for code execution
            result = None
            
            for _ in range(max_wait * 10):  # Check every 0.1 seconds
                try:
                    with self._sandbox.open("/tmp/execute_result", "r") as f:
                        result_json = f.read().strip()
                        if result_json:  # Make sure file has content
                            try:
                                result = json.loads(result_json)
                                break
                            except json.JSONDecodeError as e:
                                logger.debug(f"Invalid JSON in result file: {e}")
                                continue  # Try again
                except Exception:
                    pass
                time.sleep(0.1)
            
            if result is None:
                raise RuntimeError("Timeout waiting for code execution result")
            
            # Clean up result file
            try:
                self._sandbox.exec("rm", "-f", "/tmp/execute_result").wait()
            except Exception:
                pass
            
            if result["status"] == "success":
                # Create results for plots only - don't duplicate stdout as execute_result
                results = []
                
                # Add plots
                for i, base64_img in enumerate(result.get("plots", [])):
                    results.append(ModalResult(
                        png=base64_img,
                        is_main_result=(i == 0)  # First plot is main result
                    ))
                
                # Get stdout and stderr output for logs
                stdout_output = result.get("stdout", "")
                stderr_output = result.get("stderr", "")
                
                # Return execution with stdout/stderr in logs, plots in results
                # Don't add stdout to results to avoid duplication
                return ModalExecution(stdout=stdout_output, stderr=stderr_output, error=None, results=results)
            
            elif result["status"] == "error":
                # Execution had an error
                error_info = result["error"]
                error_data = {
                    "name": error_info["name"],
                    "value": error_info["value"], 
                    "traceback": error_info["traceback"]
                }
                return ModalExecution(stdout="", stderr="", error=error_data, results=[])
            
            else:
                raise RuntimeError(f"Unknown status from persistent session: {result['status']}")
            
        except Exception as e:
            # Handle session errors and other exceptions
            logger.error(f"Error executing code in persistent session: {str(e)}")
            
            # Reset persistent session on error
            if self._persistent_session:
                try:
                    self._persistent_session.terminate()
                except Exception:
                    pass
                self._persistent_session = None
            
            error_data = {
                "name": type(e).__name__,
                "value": str(e),
                "traceback": f"Traceback: {type(e).__name__}: {str(e)}"
            }
            return ModalExecution(error=error_data)
    
    def run_shell(self, command: str, timeout: int = 30) -> ModalExecution:
        """
        Execute raw shell commands directly in the sandbox without Python wrapper
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (default 30)
            
        Returns:
            ModalExecution object with shell output
        """
        try:
            if not self._sandbox:
                raise RuntimeError("Sandbox not initialized")
                
            logger.debug(f"Executing raw shell command: {command}")
            
            # Use Modal's exec to run shell command directly
            # Split command into parts for exec (simple approach for common commands)
            if ' ' in command:
                # For complex commands, use sh -c
                result = self._sandbox.exec("sh", "-c", command, timeout=timeout)
            else:
                # For simple commands, run directly
                result = self._sandbox.exec(command, timeout=timeout)
            
            # Wait for completion
            result.wait()
            
            # Get output
            stdout_output = ""
            stderr_output = ""
            
            try:
                stdout_output = result.stdout.read() if result.stdout else ""
            except Exception:
                pass
                
            try:
                stderr_output = result.stderr.read() if result.stderr else ""
            except Exception:
                pass
            
            # Check for errors based on return code
            error_data = None
            if result.returncode != 0:
                error_data = {
                    "name": "ShellCommandError",
                    "value": f"Command '{command}' exited with code {result.returncode}",
                    "traceback": f"Command: {command}\nExit Code: {result.returncode}\nSTDERR: {stderr_output}"
                }
            
            logger.debug(f"Shell command completed with exit code: {result.returncode}")
            
            return ModalExecution(
                stdout=stdout_output,
                stderr=stderr_output,
                error=error_data,
                results=[]
            )
            
        except Exception as e:
            logger.error(f"Error executing shell command '{command}': {str(e)}")
            
            # Return error execution
            error_data = {
                "name": type(e).__name__,
                "value": str(e),
                "traceback": f"Shell command failed: {command}\nError: {str(e)}"
            }
            
            return ModalExecution(
                stdout="",
                stderr="",
                error=error_data,
                results=[]
            )

    def get_info(self) -> ModalSandboxInfo:
        """Get sandbox info for countdown timer"""
        return self._sandbox_info
    
    def kill(self):
        """Terminate the sandbox and persistent session"""
        try:
            # Terminate persistent session first
            if self._persistent_session:
                try:
                    # Send terminate command via file
                    terminate_command = {"action": "terminate"}
                    with self._sandbox.open("/tmp/execute_command", "w") as f:
                        f.write(json.dumps(terminate_command))
                except Exception:
                    pass  # Ignore errors during graceful shutdown
                
                try:
                    self._persistent_session.terminate()
                except Exception:
                    pass  # Ignore errors during forced termination
                
                self._persistent_session = None
                logger.info("Persistent session terminated")
            
            # Terminate sandbox
            if self._sandbox:
                self._sandbox.terminate()
                self._sandbox = None
                logger.info("Modal sandbox terminated")
                
        except Exception as e:
            logger.error(f"Error terminating Modal sandbox: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.kill()


def create_modal_sandbox(gpu_config: str = "cpu", gpu_count: int = 1, cpu_cores: float = 2.0, 
                        memory_gb: float = 8.0, timeout: int = 300, 
                        environment_vars: Dict[str, str] = None) -> ModalSandbox:
    """
    Factory function to create Modal sandbox with specified configuration
    
    Args:
        gpu_config: GPU type ("cpu", "T4", "L4", "A100-40GB", "A100-80GB", "H100")
        gpu_count: Number of GPUs (for future implementation)
        cpu_cores: Number of CPU cores
        memory_gb: Memory in GB
        timeout: Timeout in seconds
        environment_vars: Environment variables
        
    Returns:
        ModalSandbox instance
    """
    memory_mb = int(memory_gb * 1024)
    
    # For multi-GPU support (future implementation)
    if gpu_count > 1:
        print(f"Warning: Multi-GPU ({gpu_count}) not yet implemented, using single GPU")
    
    return ModalSandbox(
        gpu_config=gpu_config,
        cpu_cores=cpu_cores,
        memory_mb=memory_mb,
        timeout=timeout,
        environment_vars=environment_vars
    )