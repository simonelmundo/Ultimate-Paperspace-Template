# Ollama API Complete Guide

Complete reference for using the Ollama API with examples, endpoints, and best practices.

## Table of Contents
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [JSON Output Guide](#json-output-guide)
- [Performance & GPU Usage](#performance--gpu-usage)
- [Best Practices](#best-practices)
- [Examples](#examples)

---

## Quick Start

### Base URL
```
https://your-domain.com/textgen/api
```
or locally:
```
http://localhost:7009/api
```

### Authentication
No authentication required for local instances. For remote instances, check your server configuration.

---

## API Endpoints

### 1. Generate Text (`/api/generate`)

Generate a completion for a given prompt.

**Endpoint:** `POST /api/generate`

**Request:**
```json
{
  "model": "llama3.2:3b",
  "prompt": "Write a short story about...",
  "stream": false,
  "format": "json"  // Optional: force JSON output
}
```

**Response:**
```json
{
  "model": "llama3.2:3b",
  "created_at": "2025-12-09T13:40:01.904272457Z",
  "response": "Generated text here...",
  "done": true,
  "done_reason": "stop",
  "context": [...],
  "total_duration": 4617959715,
  "load_duration": 121994973,
  "prompt_eval_count": 49,
  "prompt_eval_duration": 16816188,
  "eval_count": 547,
  "eval_duration": 3910924660
}
```

**cURL Example:**
```bash
curl -X POST "https://your-domain.com/textgen/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Explain quantum computing in simple terms",
    "stream": false
  }'
```

---

### 2. Chat Completion (`/api/chat`)

Generate a response for a conversational prompt with context.

**Endpoint:** `POST /api/chat`

**Request:**
```json
{
  "model": "llama3.2:3b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Tell me a joke"
    }
  ],
  "stream": false,
  "format": "json"  // Optional: force JSON output
}
```

**Response:**
```json
{
  "model": "llama3.2:3b",
  "created_at": "2025-12-09T13:46:03.760114373Z",
  "message": {
    "role": "assistant",
    "content": "Response text here..."
  },
  "done": true,
  "total_duration": 511450036
}
```

**cURL Example:**
```bash
curl -X POST "https://your-domain.com/textgen/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

---

### 3. List Models (`/api/tags`)

Get a list of all locally available models.

**Endpoint:** `GET /api/tags`

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.2:3b",
      "model": "llama3.2:3b",
      "modified_at": "2025-12-09T13:38:04.320804966Z",
      "size": 2019393189,
      "digest": "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "3.2B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

**cURL Example:**
```bash
curl "https://your-domain.com/textgen/api/tags"
```

---

### 4. Pull/Download Model (`/api/pull`)

Download a model from the Ollama library.

**Endpoint:** `POST /api/pull`

**Request:**
```json
{
  "name": "llama3.2:3b"
}
```

**Response:** Streaming JSON with download progress
```json
{"status":"pulling manifest"}
{"status":"pulling dde5aa3fc5ff","digest":"sha256:...","total":2019377376,"completed":32642480}
{"status":"success"}
```

**cURL Example:**
```bash
curl -X POST "https://your-domain.com/textgen/api/pull" \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:3b"}' \
  --no-buffer
```

---

### 5. Show Model Information (`/api/show`)

Get detailed information about a specific model.

**Endpoint:** `POST /api/show`

**Request:**
```json
{
  "name": "llama3.2:3b"
}
```

**cURL Example:**
```bash
curl -X POST "https://your-domain.com/textgen/api/show" \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:3b"}'
```

---

### 6. Check Running Models & GPU Usage (`/api/ps`)

List models currently loaded in memory and their resource usage.

**Endpoint:** `GET /api/ps`

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.2:3b",
      "model": "llama3.2:3b",
      "size": 2750261248,
      "size_vram": 2750261248,  // VRAM usage indicates GPU
      "digest": "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "3.2B",
        "quantization_level": "Q4_K_M"
      },
      "expires_at": "2025-12-09T13:45:32.616981377Z",
      "context_length": 4096
    }
  ]
}
```

**cURL Example:**
```bash
curl "https://your-domain.com/textgen/api/ps"
```

**GPU Detection:**
- If `size_vram` is present and > 0, the model is using GPU
- Check logs for: `"library=CUDA"` or `"processor": "100% GPU"`

---

### 7. Generate Embeddings (`/api/embeddings`)

Generate embeddings for text.

**Endpoint:** `POST /api/embeddings`

**Request:**
```json
{
  "model": "llama3.2:3b",
  "prompt": "text to embed"
}
```

**cURL Example:**
```bash
curl -X POST "https://your-domain.com/textgen/api/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Hello world"
  }'
```

---

### 8. Copy Model (`/api/copy`)

Create a copy of a model with a new name.

**Endpoint:** `POST /api/copy`

**Request:**
```json
{
  "source": "llama3.2:3b",
  "destination": "my-custom-name"
}
```

---

### 9. Delete Model (`/api/delete`)

Remove a model from local storage.

**Endpoint:** `DELETE /api/delete`

**Request:**
```json
{
  "name": "model-name"
}
```

---

## JSON Output Guide

### Method 1: Using `format: "json"` (Recommended)

The simplest way to get structured JSON output. Works with both `/api/generate` and `/api/chat`.

#### Basic Example

**Request:**
```json
{
  "model": "llama3.2:3b",
  "format": "json",
  "prompt": "List the top 3 programming languages with rank, name, and popularity",
  "stream": false
}
```

**Response:**
```json
{
  "response": "{\"rank\":1,\"name\":\"JavaScript\",\"popularity\":85.2}"
}
```

#### With System Prompt (Best Practice)

**Request:**
```json
{
  "model": "llama3.2:3b",
  "format": "json",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that answers in JSON only. Use this exact schema:\n{\n  \"name\": string,\n  \"age\": number,\n  \"is_student\": boolean,\n  \"hobbies\": string[]\n}"
    },
    {
      "role": "user",
      "content": "Give me info about Alice who is 25 years old, studies computer science and loves hiking and photography."
    }
  ],
  "stream": false
}
```

**Response:**
```json
{
  "message": {
    "content": "{\n  \"name\": \"Alice\",\n  \"age\": 25,\n  \"is_student\": true,\n  \"hobbies\": [\"hiking\", \"photography\"]\n}"
  }
}
```

### Method 2: JSON Schema with Tool Calling (Advanced)

For models that support tool calling (Llama 3.1+, Qwen2.5+, Phi-4+, etc.):

**Request:**
```json
{
  "model": "llama3.2:8b",
  "messages": [
    {
      "role": "user",
      "content": "Extract person info: John Doe is 30 years old and works as a chef."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "record_person",
        "description": "Record a person",
        "parameters": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "job": {"type": "string"}
          },
          "required": ["name", "age", "job"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

---

## Best Practices for JSON Output

### 1. Always Use System Prompts

Put JSON schema instructions in the **system message**, not the user prompt. Models pay more attention to system prompts.

✅ **Good:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You MUST respond with valid JSON only. Schema: {...}"
    },
    {
      "role": "user",
      "content": "Extract the data"
    }
  ]
}
```

❌ **Bad:**
```json
{
  "prompt": "Return JSON with schema: {...}"
}
```

### 2. Include Explicit Instructions

Add this to your system prompt for stubborn models:
```
Important: You MUST answer with valid JSON only and nothing else. 
No explanations, no markdown, no code blocks, no ```json tags.
```

### 3. Provide Examples

Show the model exactly what you want:

```json
{
  "role": "system",
  "content": "Return JSON array. Example: [{\"rank\":1,\"name\":\"Python\",\"popularity\":85}]"
}
```

### 4. Use `/api/chat` for Complex Structures

`/api/chat` with system prompts is more reliable than `/api/generate` for structured output.

### 5. Disable Streaming for Clean JSON

Always set `"stream": false` when you need clean, parseable JSON.

---

## Performance & GPU Usage

### Performance Metrics

From our testing with `llama3.2:3b` on NVIDIA RTX A4000:

- **Model Size:** 2.75 GB VRAM
- **Generation Speed:** ~140 tokens/second
- **Load Time (first time):** ~2 seconds
- **Load Time (cached):** ~120ms
- **Average Response Time:** 0.5-4.6 seconds (depending on length)

### Checking GPU Usage

#### Method 1: Check `/api/ps` Response
```bash
curl "https://your-domain.com/textgen/api/ps"
```

Look for `size_vram` field - if present and > 0, GPU is being used.

#### Method 2: Check Server Logs
Look for these indicators in Ollama logs:
- `"library=CUDA"` - Using CUDA
- `"processor": "100% GPU"` - GPU usage
- `"name": "CUDA0"` - GPU device name

#### Method 3: nvidia-smi (if you have SSH access)
```bash
nvidia-smi
```

---

## Best Practices

### 1. Model Selection
- **For JSON:** `llama3.2:3b/8b`, `qwen2.5:7b/14b`, `gemma2:9b/27b`
- **For Speed:** Smaller models (3B parameters)
- **For Quality:** Larger models (8B+ parameters)

### 2. Error Handling
Always check the `done` field and handle errors:
```json
{
  "done": true,
  "done_reason": "stop"  // or "length", "error"
}
```

### 3. Context Management
- Models have context limits (e.g., 4096 tokens for llama3.2:3b)
- Keep prompts concise
- Use streaming for long responses

### 4. Rate Limiting
- Ollama processes requests sequentially by default
- For concurrent requests, run multiple Ollama instances or use a queue

---

## Examples

### Example 1: Simple Text Generation

```bash
curl -X POST "https://your-domain.com/textgen/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Write a haiku about AI",
    "stream": false
  }'
```

### Example 2: Structured JSON Output

```bash
curl -X POST "https://your-domain.com/textgen/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "format": "json",
    "messages": [
      {
        "role": "system",
        "content": "You are a JSON-only assistant. Return valid JSON only."
      },
      {
        "role": "user",
        "content": "Extract: Name=John, Age=30, Job=Engineer"
      }
    ],
    "stream": false
  }'
```

### Example 3: Pull and Test a Model

```bash
# Pull model
curl -X POST "https://your-domain.com/textgen/api/pull" \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:3b"}' \
  --no-buffer

# Test it
curl -X POST "https://your-domain.com/textgen/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Hello!",
    "stream": false
  }'
```

### Example 4: Check GPU Status

```bash
# Check running models
curl "https://your-domain.com/textgen/api/ps"

# List all models
curl "https://your-domain.com/textgen/api/tags"
```

---

## Troubleshooting

### Issue: Model returns invalid JSON
**Solution:** 
- Add `"format": "json"` to request
- Use system prompt with explicit JSON instructions
- Include example in system prompt

### Issue: Model returns markdown code blocks
**Solution:**
- Add to system prompt: "No markdown, no ```json tags, just raw JSON"
- Use `"format": "json"`

### Issue: Slow response times
**Solution:**
- Check if GPU is being used (`/api/ps`)
- Use smaller model (3B instead of 8B)
- Reduce `context_length` if possible

### Issue: Model not found
**Solution:**
- Pull the model first: `POST /api/pull`
- Check available models: `GET /api/tags`

---

## API Response Fields Explained

- `model`: Model name used
- `created_at`: Timestamp of response
- `response` / `message.content`: Generated text
- `done`: Whether generation is complete
- `done_reason`: Why generation stopped ("stop", "length", "error")
- `total_duration`: Total time in nanoseconds
- `load_duration`: Time to load model into memory
- `prompt_eval_count`: Number of tokens in prompt
- `prompt_eval_duration`: Time to evaluate prompt
- `eval_count`: Number of tokens generated
- `eval_duration`: Time to generate response
- `context`: Token context array

---

## Additional Resources

- [Ollama Official Documentation](https://docs.ollama.com)
- [Ollama API Reference](https://docs.ollama.com/api)
- [Model Library](https://ollama.com/library)

---

**Last Updated:** December 2025  
**Tested With:** Ollama 0.13.2, llama3.2:3b, NVIDIA RTX A4000, CUDA 12.8

