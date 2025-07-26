# vLLM Inference

A Python package for efficient large language model serving and inference operations using vLLM.

## Description

vLLM Inference is a high-performance library for serving and running inference on large language models. It provides optimized inference capabilities with support for various model architectures and deployment scenarios.

## Features

- **High Performance**: Optimized inference engine for fast model serving
- **Model Support**: Compatible with various transformer-based language models
- **Scalable**: Designed for production deployment with horizontal scaling
- **Easy Integration**: Simple API for model loading and inference
- **Memory Efficient**: Optimized memory usage for large models

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for GPU acceleration)
- CUDA Toolkit 11.8 or higher

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/vllm-inference.git
cd vllm-inference

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Install via pip

```bash
pip install vllm-inference
```

## Quick Start

### Basic Usage

```python
from vllm_inference import LLM, SamplingParams

# Initialize the model
llm = LLM(model="microsoft/DialoGPT-medium")

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# Generate text
outputs = llm.generate("Hello, how are you?", sampling_params)
print(outputs[0].outputs[0].text)
```

### Server Mode

```python
from vllm_inference import LLMEngine, SamplingParams

# Start the engine
engine = LLMEngine.from_pretrained("microsoft/DialoGPT-medium")

# Generate responses
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = engine.generate(["Hello world"], sampling_params)
```

## API Reference

### LLM Class

The main class for model inference.

```python
class LLM:
    def __init__(self, model: str, **kwargs)
    def generate(self, prompts: List[str], sampling_params: SamplingParams)
```

### SamplingParams

Configuration for text generation.

```python
class SamplingParams:
    def __init__(self, 
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 max_tokens: int = 100,
                 **kwargs)
```

## Configuration

### Environment Variables

- `VLLM_USE_CUDA`: Set to "1" to enable CUDA acceleration
- `VLLM_MODEL_CACHE_DIR`: Directory for caching downloaded models
- `VLLM_MAX_MODEL_LEN`: Maximum sequence length for models

### Model Configuration

```python
llm = LLM(
    model="microsoft/DialoGPT-medium",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=2048
)
```

## Examples

### Text Generation

```python
from vllm_inference import LLM, SamplingParams

llm = LLM(model="gpt2")
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    max_tokens=50
)

prompts = [
    "The future of artificial intelligence is",
    "In a world where technology advances rapidly,",
    "The best way to learn programming is"
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### Chat Completion

```python
from vllm_inference import LLM, SamplingParams

llm = LLM(model="microsoft/DialoGPT-medium")

messages = [
    {"role": "user", "content": "What is machine learning?"}
]

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate([messages], sampling_params)
```

## Performance Optimization

### GPU Memory Management

```python
llm = LLM(
    model="gpt2",
    gpu_memory_utilization=0.8,  # Use 80% of GPU memory
    tensor_parallel_size=2,       # Use 2 GPUs
    max_model_len=1024           # Limit sequence length
)
```

### Batch Processing

```python
# Process multiple prompts efficiently
prompts = ["Prompt 1", "Prompt 2", "Prompt 3", ...]
outputs = llm.generate(prompts, sampling_params)
```

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 8000
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Setup

```python
from vllm_inference import LLMEngine

# Initialize engine for production
engine = LLMEngine.from_pretrained(
    model="gpt2",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    max_model_len=2048
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/vllm-inference.git
cd vllm-inference

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Run tests
pytest tests/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the vLLM framework
- Inspired by the Hugging Face Transformers library
- Community contributors and maintainers

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/samaitra/vllm-inference/issues)
- **Discussions**: [GitHub Discussions](https://github.com/samaitra/vllm-inference/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.

## How the Server Gets Created:

### 1. **Server Module Structure**
I created `server.py` which contains:
- **FastAPI application** with proper middleware and CORS support
- **Model initialization** on startup using vLLM's `LLMEngine`
- **REST API endpoints** for text generation, health checks, and model listing
- **Command-line interface** for running the server

### 2. **Key Components**:

#### **FastAPI App Setup**:
```python
app = FastAPI(
    title="vLLM Inference Server",
    description="High-performance LLM inference server using vLLM",
    version="0.1.0"
)
```

#### **Model Loading**:
```python
@app.on_event("startup")
async def startup_event():
    llm_engine = LLMEngine.from_pretrained(
        model="microsoft/DialoGPT-medium",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )
```

#### **API Endpoints**:
- `GET /health` - Health check with model status
- `POST /generate` - Text generation endpoint
- `GET /models` - List available models
- `GET /` - Root endpoint with API info

### 3. **Docker Integration**:
The Dockerfile now correctly runs:
```bash
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. **Dependencies**:
Created `requirements.txt` with all necessary packages:
- `vllm` - Core inference engine
- `fastapi` & `uvicorn` - Web server framework
- `torch` & `transformers` - PyTorch and model support

### 5. **Usage**:
```bash
# Run directly
python server.py --host 0.0.0.0 --port 8000

# Run with Docker
docker build -t vllm-inference .
docker run --gpus all -p 8000:8000 vllm-inference
```

The server is now fully functional and ready to serve LLM inference requests via HTTP API!