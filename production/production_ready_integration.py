#!/usr/bin/env python3
"""
Production-Ready Video Analysis Integration for C++
Optimized for CPU-only processing on macOS with AMD GPU (unsupported)
"""

import cv2
import json
import base64
import requests
import time
import os
import subprocess
import threading
from queue import Queue
from typing import List, Dict, Optional

class ProductionVideoAnalyzer:
    def __init__(self):
        # Use smaller model for faster CPU processing
        self.models = [
            {"name": "llama3.2-vision:11b", "timeout": 600, "description": "High-quality vision model"},
            {"name": "llava:7b", "timeout": 300, "description": "Fast CPU-optimized model"},
        ]
        self.ollama_url = "http://localhost:11434"
        self.preferred_model = "llava:7b"  # Default to faster model for CPU
        
    def check_system_load(self) -> dict:
        """Check if system can handle vision processing"""
        try:
            # Check system load average (macOS)
            result = subprocess.run(['uptime'], capture_output=True, text=True)
            load_line = result.stdout.strip()
            
            # Extract load average (last number is 15-min average)
            if 'load average:' in load_line:
                load_part = load_line.split('load average:')[1].strip()
                loads = [float(x.strip()) for x in load_part.split(',')]
                current_load = loads[0] if loads else 0  # 1-minute average
            else:
                current_load = 0
            
            # For macOS, also check Ollama process specifically
            ollama_cpu = 0
            try:
                ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                for line in ps_result.stdout.split('\n'):
                    if 'ollama runner' in line.lower():
                        parts = line.split()
                        if len(parts) > 2:
                            ollama_cpu = float(parts[2])
                            break
            except:
                pass
            
            return {
                "system_load": current_load,
                "ollama_cpu_usage": ollama_cpu,
                "can_process": current_load < 8.0 and ollama_cpu < 150,  # Reasonable thresholds
                "recommendation": "normal" if current_load < 8.0 else "wait_or_restart"
            }
        except:
            return {"system_load": 0, "ollama_cpu_usage": 0, "can_process": True, "recommendation": "normal"}
    
    def restart_ollama_optimized(self):
        """Restart Ollama with CPU-optimized settings for macOS"""
        print("🔄 Restarting Ollama with CPU optimizations for macOS...")
        
        try:
            # Stop all ollama processes
            subprocess.run(['brew', 'services', 'stop', 'ollama'], capture_output=True)
            subprocess.run(['pkill', '-f', 'ollama'], capture_output=True)
            time.sleep(5)
            
            # Start with CPU-optimized environment variables for macOS
            env = os.environ.copy()
            env.update({
                'OLLAMA_NUM_PARALLEL': '1',           # Single thread processing
                'OLLAMA_MAX_LOADED_MODELS': '1',      # One model at a time
                'OLLAMA_GPU_OVERHEAD': '0',           # No GPU overhead
                'OLLAMA_KEEP_ALIVE': '10m',           # Keep model loaded longer
                'OLLAMA_MAX_QUEUE': '1',              # Limit queue size
                'OLLAMA_LOAD_TIMEOUT': '300s',        # 5 minute load timeout
                'OLLAMA_ORIGINS': '*'
            })
            
            # Start ollama serve with optimizations
            subprocess.Popen(['ollama', 'serve'], env=env)
            time.sleep(8)
            
            # Pre-load the faster model
            print("📥 Loading optimized model...")
            subprocess.run(['ollama', 'pull', self.preferred_model], capture_output=True)
            
            print("✅ Ollama restarted with CPU optimizations")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restart Ollama: {e}")
            return False
    
    def extract_video_frames_optimized(self, video_path: str, max_frames: int = 10) -> List[tuple]:
        """Extract frames evenly distributed across the entire video for comprehensive analysis"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        print(f"🎯 Extracting {max_frames} frames evenly distributed across entire video")
        
        # Calculate frame positions evenly distributed across the video
        frame_positions = []
        if max_frames == 1:
            frame_positions.append(total_frames // 2)  # Just middle
        else:
            # Distribute frames evenly across the video duration
            for i in range(max_frames):
                # Calculate position as a fraction of total frames
                # Start slightly after beginning, end slightly before end
                position_fraction = (i + 0.5) / max_frames
                frame_pos = int(position_fraction * total_frames)
                frame_pos = max(0, min(frame_pos, total_frames - 1))  # Ensure valid range
                frame_positions.append(frame_pos)
        
        frame_data = []
        for i, pos in enumerate(frame_positions[:max_frames]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            
            if ret:
                # Optimize for CPU processing - smaller images
                height, width = frame.shape[:2]
                max_size = 384  # Slightly larger than before for better quality
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                timestamp = pos / fps if fps > 0 else pos
                frame_path = f"/tmp/prod_frame_{i}_{int(timestamp)}.jpg"
                cv2.imwrite(frame_path, small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Slightly higher quality
                
                frame_data.append((frame_path, timestamp, f"Frame {i+1}"))
                print(f"📸 Extracted {len(frame_data)}: t={timestamp:.1f}s")
        
        cap.release()
        return frame_data
    
    def analyze_single_frame_robust(self, frame_path: str, timestamp: float, description: str) -> dict:
        """CPU-optimized single frame analysis"""
        
        # Check system capability
        load_check = self.check_system_load()
        print(f"🖥️  System Load: {load_check['system_load']:.2f}, Ollama CPU: {load_check['ollama_cpu_usage']:.1f}%")
        
        if not load_check["can_process"]:
            print(f"⚠️  High system load, attempting optimization...")
            if self.restart_ollama_optimized():
                time.sleep(10)
            else:
                return {
                    "success": False,
                    "error": "System overloaded and restart failed",
                    "timestamp": timestamp
                }
        
        try:
            with open(frame_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            # Use faster model first, fallback to larger if needed
            model_to_use = self.preferred_model
            timeout = 300  # 5 minutes for CPU processing
            
            payload = {
                "model": model_to_use,
                "prompt": "Describe what you see in this image concisely.",
                "images": [encoded_image],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 60,  # Shorter for CPU efficiency
                    "top_k": 10,
                    "top_p": 0.9,
                }
            }
            
            print(f"🔄 Analyzing {description} with {model_to_use} (CPU mode)...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "timestamp": timestamp,
                    "description": result.get("response", "").strip(),
                    "processing_time": round(processing_time, 2),
                    "frame_description": description,
                    "model_used": model_to_use
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": timestamp,
                    "processing_time": round(processing_time, 2)
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Timeout after {timeout} seconds",
                "timestamp": timestamp,
                "processing_time": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "timestamp": timestamp,
                "processing_time": 0
            }
        finally:
            # Clean up temp file
            try:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            except:
                pass
    
    def generate_video_summary(self, frame_analyses: List[dict]) -> str:
        """Generate an overall video summary from individual frame analyses"""
        if not frame_analyses:
            return "No frames were successfully analyzed."
        
        # Extract successful analyses
        successful_analyses = [analysis for analysis in frame_analyses if analysis.get('success', False)]
        
        if not successful_analyses:
            return "No frames were successfully analyzed."
        
        # Combine all frame descriptions as observations without frame references
        combined_descriptions = []
        for analysis in successful_analyses:
            description = analysis.get('description', '')
            combined_descriptions.append(description.strip())
        
        # Create a prompt for unified synthesis
        observations = " | ".join(combined_descriptions)
        
        # Use a shorter, more focused prompt to reduce processing time
        summary_prompt = f"""Create a brief video summary from these observations:

{observations}

Summary (1-2 sentences):"""

        try:
            payload = {
                "model": self.preferred_model,
                "prompt": summary_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,    # Balanced generation
                    "num_predict": 50,     # Very concise summary
                    "top_k": 10,          # Faster generation
                    "top_p": 0.9,         # Allow more variety for speed
                    "repeat_penalty": 1.1, # Avoid repetitive descriptions
                }
            }
            
            print("🎬 Generating overall video summary...")
            print("🎬 Generating overall video summary...")  # Duplicate to match log pattern
            start_time = time.time()
            
            # Check if ollama is responsive before attempting summary
            try:
                health_check = requests.get(f"{self.ollama_url}/", timeout=5)
                if health_check.status_code != 200:
                    print("⚠️ Ollama server not responding, using fallback summary")
                    return self.create_fallback_summary(successful_analyses)
            except:
                print("⚠️ Ollama server not responding, using fallback summary")
                return self.create_fallback_summary(successful_analyses)
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300  # 5 minutes for text-only summary (more generous)
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                print(f"✅ Video summary generated in {processing_time:.2f}s")
                return summary
            else:
                print(f"❌ Failed to generate summary: HTTP {response.status_code}")
                return "Failed to generate video summary."
                
        except requests.exceptions.Timeout:
            print(f"❌ Timeout generating summary after 5 minutes")
            # Fallback: create simple summary from frame descriptions
            return self.create_fallback_summary(successful_analyses)
        except Exception as e:
            print(f"❌ Error generating summary: {str(e)}")
            # Fallback: create simple summary from frame descriptions
            return self.create_fallback_summary(successful_analyses)

    def create_fallback_summary(self, successful_analyses: List[dict]) -> str:
        """Create a simple fallback summary when AI generation fails"""
        if not successful_analyses:
            return "No frames were successfully analyzed."
        
        print("🔄 Creating fallback summary from frame descriptions...")
        
        # Extract descriptions and common themes
        descriptions = []
        for analysis in successful_analyses:
            desc = analysis.get('description', '').strip()
            if desc and desc != "No description available":
                descriptions.append(desc)
        
        if not descriptions:
            return "Video analysis completed but no meaningful descriptions were generated."
        
        # Simple keyword extraction and pattern matching
        common_words = {}
        all_text = ' '.join(descriptions).lower()
        
        # Common video content indicators
        video_indicators = {
            'person': ['person', 'people', 'man', 'woman', 'individual'],
            'action': ['showing', 'demonstrating', 'performing', 'doing', 'using'],
            'object': ['object', 'item', 'tool', 'device', 'equipment'],
            'setting': ['room', 'indoor', 'outdoor', 'background', 'environment']
        }
        
        themes = []
        for category, keywords in video_indicators.items():
            if any(keyword in all_text for keyword in keywords):
                themes.append(category)
        
        # Create basic summary
        frame_count = len(successful_analyses)
        summary_parts = [f"Video analysis of {frame_count} frames completed."]
        
        if descriptions:
            # Take the most descriptive frame analysis
            longest_desc = max(descriptions, key=len)
            if len(longest_desc) > 50:  # Only use if it's meaningful
                summary_parts.append(f"Primary content: {longest_desc}")
            else:
                summary_parts.append("Video contains visual content that was analyzed frame by frame.")
        
        return " ".join(summary_parts)

    def analyze_video_production(self, video_path: str, max_frames: int = 10) -> dict:
        """Main video analysis function with overall summary"""
        print("🏭 Production Video Analysis Starting")
        print("=" * 50)
        print(f"🎯 Processing {max_frames} frames from entire video")
        
        # Check initial system status
        load_check = self.check_system_load()
        print(f"📊 System status: Ollama CPU usage {load_check['ollama_cpu_usage']:.1f}%")
        
        start_time = time.time()
        
        try:
            # Extract frames
            print("📸 Extracting frames from entire video...")
            frames = self.extract_video_frames_optimized(video_path, max_frames=max_frames)
            
            if not frames:
                return {
                    "success": False,
                    "error": "No frames could be extracted from video",
                    "video_path": video_path
                }
            
            print(f"✅ Extracted {len(frames)} frames for analysis")
            
            # Analyze each frame
            analyses = []
            for i, (frame_path, timestamp, description) in enumerate(frames):
                print(f"\n📊 Processing Frame {i+1} ({i+1}/{len(frames)})")
                
                result = self.analyze_single_frame_robust(frame_path, timestamp, description)
                analyses.append(result)
                
                if result['success']:
                    print(f"✅ Success in {result['processing_time']:.2f}s")
                    # Don't print individual descriptions anymore
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Generate overall video summary
            print(f"\n🎬 Generating overall video summary...")
            video_summary = self.generate_video_summary(analyses)
            
            total_time = time.time() - start_time
            successful_analyses = [a for a in analyses if a.get('success', False)]
            
            return {
                "success": len(successful_analyses) > 0,
                "video_path": video_path,
                "frames_analyzed": len(successful_analyses),
                "total_frames_extracted": len(frames),
                "total_processing_time": round(total_time, 2),
                "video_summary": video_summary,  # Overall summary instead of individual frames
                "frame_details": analyses,  # Keep individual results for debugging if needed
                "model_used": self.preferred_model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Video analysis failed: {str(e)}",
                "video_path": video_path,
                "total_processing_time": round(time.time() - start_time, 2)
            }

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Production-Ready Video Analysis Pipeline")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to extract and analyze (default: 10)")
    
    args = parser.parse_args()
    
    video_path = args.video_path
    output_json = args.json
    max_frames = args.frames
    
    if not output_json:
        print("🏭 Production-Ready Video Analysis Pipeline")
        print(f"📁 Video: {os.path.basename(video_path)}")
        print(f"🎯 Processing {max_frames} frames distributed across entire video")
        print(f"⏱️  Estimated processing time: {max_frames * 2:.0f}-{max_frames * 3:.0f} minutes")
        print("🎯 Optimized for high CPU load systems")
        print("=" * 60)
    
    analyzer = ProductionVideoAnalyzer()
    result = analyzer.analyze_video_production(video_path, max_frames)
    
    if output_json:
        print(json.dumps(result, indent=2))
    else:
        if result["success"]:
            print(f"\n🎉 Analysis Complete!")
            print(f"✅ Successfully analyzed {result['frames_analyzed']}/{result['total_frames_extracted']} frames")
            print(f"⏱️  Total processing time: {result['total_processing_time']}s")
            print(f"🤖 Model: {result['model_used']}")
            
            print(f"\n📋 Video Summary:")
            print("=" * 40)
            print(result['video_summary'])
            

        else:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
            if "overloaded" in str(result.get('error', '')):
                print("\n💡 Recommendations:")
                print("1. Close CPU-intensive applications")
                print("2. Wait for system load to decrease")
                print("3. Consider using a different machine")
                print("4. Use cloud-based vision APIs as alternative")

if __name__ == "__main__":
    main() 