#!/bin/bash

# Script to pull and test Ollama model
# Usage: bash test_ollama_model.sh

MODEL="llama3.2:3b"
API_BASE="https://nus8idfzog.clg07azjl.paperspacegradient.com/textgen"

echo "=================================================="
echo "  PULLING AND TESTING OLLAMA MODEL: $MODEL"
echo "=================================================="
echo ""

# Step 1: Pull the model via API
echo "📥 Step 1: Pulling model $MODEL..."
echo "This may take several minutes depending on your connection speed..."
echo ""

curl -X POST "$API_BASE/api/pull" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$MODEL\"}" \
  --no-buffer

echo ""
echo ""

# Step 2: Verify model is installed
echo "📋 Step 2: Verifying model installation..."
MODELS=$(curl -s "$API_BASE/api/tags" | grep -o "\"$MODEL\"" || echo "")
if [[ -n "$MODELS" ]]; then
  echo "✅ Model $MODEL successfully installed!"
else
  echo "⚠️  Model may still be downloading. Checking again..."
  sleep 2
  curl -s "$API_BASE/api/tags" | jq .
fi

echo ""
echo ""

# Step 3: Test the model
echo "🧪 Step 3: Testing model with a simple prompt..."
echo ""

RESPONSE=$(curl -s -X POST "$API_BASE/api/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"Write a short haiku about artificial intelligence.\",
    \"stream\": false
  }")

echo "Response:"
echo "$RESPONSE" | jq -r '.response' 2>/dev/null || echo "$RESPONSE"

echo ""
echo ""

# Step 4: Test with chat endpoint
echo "💬 Step 4: Testing chat endpoint..."
echo ""

CHAT_RESPONSE=$(curl -s -X POST "$API_BASE/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Hello! Can you introduce yourself in one sentence?\"}
    ],
    \"stream\": false
  }")

echo "Chat Response:"
echo "$CHAT_RESPONSE" | jq -r '.message.content' 2>/dev/null || echo "$CHAT_RESPONSE"

echo ""
echo "=================================================="
echo "  TESTING COMPLETE!"
echo "=================================================="


