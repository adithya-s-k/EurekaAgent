# Eureka Agent

An AI-powered research automation system that can execute Python code, analyze data, and generate insights through an interactive Jupyter-like interface.

## 🎯 What it does

Eureka Agent automates research workflows by:

- **Executing Python code** in a secure containerized environment
- **Analyzing data** with full context awareness across conversations
- **Generating visualizations** and interactive outputs
- **Iterative development** - builds upon previous code and results
- **Error recovery** - learns from execution failures and improves

## ⚡ Key Features

- **Stateful Jupyter Environment**: Variables and imports persist across all code executions
- **GPU/CPU Support**: Configurable hardware (CPU, T4, L4, A100, H100)
- **Interactive Development**: Build complex solutions incrementally
- **Rich Output Support**: Plots, tables, HTML, and multimedia content
- **Error Handling**: Intelligent error recovery and debugging assistance
- **File Upload**: Process your own datasets and documents

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Modal account (for containerized execution)
- OpenAI API key or compatible LLM provider

### Installation

1. Clone the repository:

```bash
git clone https://github.com/adithya-s-k/EurekaAgent
cd EurekaAgent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export MODAL_TOKEN_ID="your-modal-token-id"
export MODAL_TOKEN_SECRET="your-modal-token-secret"
```

### Running the Application

```bash
python app.py
```

The application will launch a Gradio interface accessible via your web browser.

## 🔧 Configuration

### Environment Variables

| Variable                     | Description                   | Required | Format/Example                  |
| ---------------------------- | ----------------------------- | -------- | ------------------------------- |
| `MODAL_TOKEN_ID`             | Modal token ID                | Yes      | `ak-...`                        |
| `MODAL_TOKEN_SECRET`         | Modal token secret            | Yes      | `as-...`                        |
| `PROVIDER_API_KEY`           | AI Provider API key           | Yes\*    | `sk-...`, `gsk_...`, `csk-...`  |
| `PROVIDER_API_ENDPOINT`      | AI Provider API endpoint      | Yes\*    | `https://api.anthropic.com/v1/` |
| `MODEL_NAME`                 | Model to use                  | Yes\*    | `claude-sonnet-4-20250514`      |
| `HF_TOKEN`                   | Hugging Face token (optional) | No       | `hf_...`                        |
| `TAVILY_API_KEY`             | Tavily API key for web search | No       | `tvly-...`                      |
| `PHOENIX_API_KEY`            | Phoenix tracing API key       | No       | -                               |
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix collector endpoint    | No       | -                               |
| `ENVIRONMENT`                | Environment mode              | No       | `dev`/`prod`                    |

\*At least one complete AI provider configuration must be provided

**Legacy OpenAI Support:**
| Variable | Description | Required |
| ----------------------- | ----------------------------- | -------- |
| `OPENAI_API_KEY` | OpenAI API key | No |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | No |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | No |

### Hardware Options

- **CPU Only**: Free, suitable for basic tasks
- **NVIDIA T4**: Low-cost GPU for small models
- **NVIDIA L4**: Mid-range GPU for better performance
- **NVIDIA A100**: High-end GPU for large models (40GB/80GB variants)
- **NVIDIA H100**: Latest flagship GPU for maximum performance

## 💡 Usage Examples

### Basic Data Analysis

```
"Analyze the uploaded CSV file and create visualizations showing key trends"
```

### Machine Learning

```
"Train a neural network to classify the iris dataset and evaluate its performance"
```

### Research Tasks

```
"Download stock price data for the last year and perform technical analysis"
```

## 🏗️ Architecture

- **Frontend**: Gradio web interface with real-time status updates
- **Backend**: Python application with multi-provider AI integration
- **Execution Environment**: Modal containerized sandboxes with GPU support
- **Code Execution**: Persistent Jupyter-like stateful environment
- **Session Management**: Comprehensive session state tracking with Phoenix tracing
- **Storage**: File-based session persistence with notebook compatibility
- **Web Search**: Integrated Tavily search for current information
- **Hardware Support**: CPU, T4, L4, A100, H100 configurations

## 📁 Project Structure

```
EurekaAgent/
├── app.py              # Main Gradio application with API key management
├── jupyter_handler.py  # Jupyter notebook management and rendering
├── jupyter_agent.py           # Utility functions, execution logic, and session management
├── modal_sandbox.py   # Modal sandbox configuration with GPU support
├── system_prompt.txt  # System prompt for the AI agent
├── requirements.txt   # Python dependencies
└── temp/              # Temporary files, notebooks, and session states
    ├── <session_id>/
    │   ├── session_state.json    # Complete session state and history
    │   └── jupyter-agent.ipynb   # Legacy notebook file for UI compatibility
    └── jupyter-agent.ipynb       # Default notebook template
```

## 🤝 Contributing

This project is a fork of [Jupyter Agent 2](https://huggingface.co/spaces/lvwerra/jupyter-agent-2) by Hugging Face. Contributions are welcome!

## 📄 License

See LICENSE file for details.
