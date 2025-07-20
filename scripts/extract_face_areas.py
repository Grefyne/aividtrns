#!/usr/bin/env python3
"""
Face area extraction script.
Extracts face area videos and images for each speaker based on face area analysis.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
import subprocess


def load_face_area_results(results_path):
    """Load face area analysis results from JSON file."""
    if not os.path.exists(results_path):
        print(f"Error: Face area results not found: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_face_areas(video_path, face_area_results, output_dir="extracted_faces"):
    """Extract face area videos and images for each speaker."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting face areas from video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Process each speaker cluster
    clusters = face_area_results['face_clusters']
    
    if not clusters:
        print("No speaker clusters found in face area results")
        return False
    
    print(f"Processing {len(clusters)} speaker clusters...")
    
    extraction_results = []
    
    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        print(f"\nProcessing {cluster_id}...")
        
        # Get cluster boundaries
        x1 = int(cluster['x_range'][0])
        y1 = int(cluster['y_range'][0])
        x2 = int(cluster['x_range'][1])
        y2 = int(cluster['y_range'][1])
        
        # Ensure coordinates are within video bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Calculate dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        
        print(f"  Face area: ({x1}, {y1}) to ({x2}, {y2}) - {face_width}x{face_height}")
        
        # Create output filenames
        video_filename = f"{cluster_id}_face_area.mp4"
        image_filename = f"{cluster_id}_face_area.png"
        video_path_out = os.path.join(output_dir, video_filename)
        image_path_out = os.path.join(output_dir, image_filename)
        
        # Extract face area video
        success = extract_face_area_video(cap, x1, y1, x2, y2, video_path_out, fps)
        
        if success:
            # Extract a sample frame as image
            extract_face_area_image(cap, x1, y1, x2, y2, image_path_out)
            
            extraction_results.append({
                "cluster_id": cluster_id,
                "video_file": video_filename,
                "image_file": image_filename,
                "face_area": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": face_width,
                    "height": face_height
                },
                "detection_count": cluster['face_count'],
                "time_range": cluster['time_range']
            })
            
            print(f"  Extracted: {video_filename} and {image_filename}")
        else:
            print(f"  Failed to extract face area for {cluster_id}")
    
    cap.release()
    
    # Save extraction summary
    summary_data = {
        "video_path": video_path,
        "total_clusters": len(clusters),
        "successful_extractions": len(extraction_results),
        "extractions": extraction_results
    }
    
    summary_path = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nFace area extraction completed!")
    print(f"Successful extractions: {len(extraction_results)}/{len(clusters)}")
    print(f"Summary saved to: {summary_path}")
    
    return True


def extract_face_area_video(cap, x1, y1, x2, y2, output_path, fps):
    """Extract face area video using ffmpeg."""
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # FFmpeg command to extract face area video
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",  # Read from stdin
        "-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-y",
        output_path
    ]
    
    try:
        # Start ffmpeg process
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Read frames and pipe to ffmpeg
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame to video format and pipe to ffmpeg
            success, encoded_frame = cv2.imencode('.mp4', frame)
            if success:
                process.stdin.write(encoded_frame.tobytes())
        
        # Close stdin and wait for completion
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {process.stderr.read().decode()}")
            return False
            
    except Exception as e:
        print(f"Error extracting face area video: {e}")
        return False


def extract_face_area_image(cap, x1, y1, x2, y2, output_path):
    """Extract a sample frame as face area image."""
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Read a frame from the middle of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    
    if ret:
        # Crop the face area
        face_area = frame[y1:y2, x1:x2]
        
        # Save the image
        cv2.imwrite(output_path, face_area)
        return True
    else:
        print(f"Could not read frame for image extraction")
        return False


def create_face_area_visualization(face_area_results, output_dir):
    """Create a visualization of extracted face areas."""
    
    print("Creating face area visualization...")
    
    # Load a sample frame from the video
    video_path = face_area_results['video_path']
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Could not open video for visualization")
        return
    
    # Read a frame from the middle
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frame for visualization")
        return
    
    # Draw face areas on the frame
    clusters = face_area_results['face_clusters']
    
    for i, cluster in enumerate(clusters):
        x1 = int(cluster['x_range'][0])
        y1 = int(cluster['y_range'][0])
        x2 = int(cluster['x_range'][1])
        y2 = int(cluster['y_range'][1])
        
        # Draw rectangle
        color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for first, red for others
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = cluster['cluster_id']
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save visualization
    vis_path = os.path.join(output_dir, "face_areas_visualization.png")
    cv2.imwrite(vis_path, frame)
    
    print(f"Face area visualization saved to: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract face area videos and images")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("face_area_results", help="Path to face area results JSON file")
    parser.add_argument("--output-dir", default="extracted_faces",
                       help="Output directory for extracted face areas")
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(args.face_area_results):
        print(f"Error: Face area results not found: {args.face_area_results}")
        sys.exit(1)
    
    # Load face area results
    face_area_results = load_face_area_results(args.face_area_results)
    if not face_area_results:
        sys.exit(1)
    
    # Extract face areas
    success = extract_face_areas(args.video_path, face_area_results, args.output_dir)
    
    if not success:
        print("Face area extraction failed!")
        sys.exit(1)
    
    # Create visualization
    create_face_area_visualization(face_area_results, args.output_dir)
    
    print("Face area extraction completed successfully!")


if __name__ == "__main__":
    main() 