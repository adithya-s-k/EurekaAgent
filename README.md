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
git clone <repository-url>
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

| Variable                | Description                   | Required |
| ----------------------- | ----------------------------- | -------- |
| `OPENAI_API_KEY`        | OpenAI API key                | Yes\*    |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint         | No       |
| `AZURE_OPENAI_API_KEY`  | Azure OpenAI API key          | No       |
| `PROVIDER_API_ENDPOINT` | Custom provider endpoint      | No       |
| `PROVIDER_API_KEY`      | Custom provider API key       | No       |
| `MODEL_NAME`            | Model to use (default: gpt-4) | No       |
| `MODAL_TOKEN_ID`        | Modal token ID                | Yes      |
| `MODAL_TOKEN_SECRET`    | Modal token secret            | Yes      |

\*At least one API provider must be configured

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

- **Frontend**: Gradio web interface
- **Backend**: Python application with OpenAI integration
- **Execution Environment**: Modal containerized sandboxes
- **Code Execution**: Jupyter-like stateful environment
- **Storage**: Temporary file storage for notebooks and outputs

## 📁 Project Structure

```
jupyter-agent-2/
├── app.py              # Main Gradio application
├── jupyter_handler.py  # Jupyter notebook management
├── utils.py           # Utility functions and execution logic
├── modal_sandbox.py   # Modal sandbox configuration
├── ds-system-prompt-v1.txt  # System prompt for the AI agent
├── requirements.txt   # Python dependencies
└── tmp/              # Temporary files and notebooks
```

## 🤝 Contributing

This project is a fork of [Jupyter Agent 2](https://huggingface.co/spaces/lvwerra/jupyter-agent-2) by Hugging Face. Contributions are welcome!

## 📄 License

See LICENSE file for details.
