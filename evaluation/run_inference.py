#!/usr/bin/env python3
"""
Run Qwen3-8B inference on home robot test cases.
Outputs responses to JSON for evaluation.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
MODEL_PATH = Path(__file__).parent.parent / "models" / "Qwen_Qwen3-8B"
TEST_CASES_PATH = Path(__file__).parent / "test_cases_home_robot_en.json"
OUTPUT_PATH = Path(__file__).parent / "inference_results.json"

# System prompt for home robot
SYSTEM_PROMPT = """You are a friendly and helpful home assistant robot. You help the family with daily tasks, conversations, emotional support, education, and practical advice. 

Guidelines:
- Respond naturally like a caring family member
- Keep responses concise and conversational
- Be warm and empathetic when needed
- Give practical, actionable advice
- Adapt your tone to the situation (playful with kids, supportive with adults)"""


def load_model():
    """Load Qwen3-8B model and tokenizer."""
    print(f"Loading model from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """Generate response for a single question."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking mode for conversational responses
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (only new tokens)
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def run_inference():
    """Run inference on all test cases."""
    
    # Load test cases
    with open(TEST_CASES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = [tc for tc in data['test_cases'] if 'id' in tc]  # Filter out section headers
    print(f"Loaded {len(test_cases)} test cases")
    
    # Load model
    model, tokenizer = load_model()
    
    # Run inference
    results = []
    start_time = datetime.now()
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Category: {tc['category']} | Level: {tc['level']}")
        print(f"Q: {tc['question']}")
        
        try:
            response = generate_response(model, tokenizer, tc['question'])
            print(f"A: {response[:100]}..." if len(response) > 100 else f"A: {response}")
            
            results.append({
                "id": tc['id'],
                "category": tc['category'],
                "level": tc['level'],
                "question": tc['question'],
                "response": response,
                "response_length": len(response),
                "status": "success"
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": tc['id'],
                "category": tc['category'],
                "level": tc['level'],
                "question": tc['question'],
                "response": f"ERROR: {str(e)}",
                "response_length": 0,
                "status": "error"
            })
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save results
    output_data = {
        "metadata": {
            "model": "Qwen/Qwen3-8B",
            "model_path": str(MODEL_PATH),
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(results),
            "successful": sum(1 for r in results if r['status'] == 'success'),
            "duration_seconds": duration,
            "avg_time_per_case": duration / len(results) if results else 0
        },
        "results": results
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average per case: {duration/len(results):.2f} seconds")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_inference()
