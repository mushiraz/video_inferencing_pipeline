#!/usr/bin/env python3

import sys
import json
import base64
import requests
import os

def encode_image_base64(image_path):
    """Encode an image file to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_images(image_paths, prompt):
    """Analyze images using Ollama API"""
    
    # Encode images
    images = []
    for path in image_paths:
        if os.path.exists(path):
            encoded = encode_image_base64(path)
            images.append(encoded)
        else:
            print(f"Warning: Image file not found: {path}", file=sys.stderr)
    
    # API request payload
    payload = {
        "model": "llama3.2-vision:11b",
        "prompt": prompt,
        "images": images,
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        return {
            "success": True,
            "description": result.get("response", ""),
            "model": result.get("model", ""),
            "done": result.get("done", False),
            "total_duration": result.get("total_duration", 0),
            "eval_count": result.get("eval_count", 0)
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "description": ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "description": ""
        }

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 ollama_bridge.py <prompt> <image1> [image2] ...")
        print("Example: python3 ollama_bridge.py 'Describe this image' test.jpg")
        sys.exit(1)
    
    prompt = sys.argv[1]
    image_paths = sys.argv[2:]
    
    # Validate image files
    valid_images = []
    for path in image_paths:
        if os.path.exists(path):
            valid_images.append(path)
        else:
            print(f"Error: Image file not found: {path}", file=sys.stderr)
    
    if not valid_images:
        print("Error: No valid image files provided", file=sys.stderr)
        sys.exit(1)
    
    # Analyze images
    result = analyze_images(valid_images, prompt)
    
    # Output JSON result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 