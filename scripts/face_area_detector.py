#!/usr/bin/env python3
"""
Face area detection script.
Analyzes face areas in video for speaker positioning and visualization.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from gpu_utils import get_device, setup_multi_gpu_processing, clear_gpu_cache


def detect_faces_in_video(video_path, output_dir, padding_ratio=0.3, min_face_size=30, gpu_id=None):
    """Detect faces in video and analyze face areas."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing face areas in video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Setup GPU processing for OpenCV
    available_gpus = setup_multi_gpu_processing()
    if available_gpus:
        if gpu_id is None:
            gpu_id = available_gpus[0]  # Use first available GPU
        print(f"Using GPU {gpu_id} for face detection")
        # Set OpenCV to use GPU if available
        cv2.cuda.setDevice(gpu_id)
    else:
        print("Using CPU for face detection")
    
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
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Analyze frames
    face_areas = []
    frame_count = 0
    sample_interval = max(1, total_frames // 100)  # Sample every 1% of frames
    
    print(f"Analyzing {total_frames // sample_interval} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample frames
        if frame_count % sample_interval != 0:
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )
        
        # Process detected faces
        for (x, y, w, h) in faces:
            # Add padding
            padding_x = int(w * padding_ratio)
            padding_y = int(h * padding_ratio)
            
            # Calculate padded coordinates
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(width, x + w + padding_x)
            y2 = min(height, y + h + padding_y)
            
            # Calculate center and area
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            face_areas.append({
                "frame": frame_count,
                "time": frame_count / fps,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "center_x": center_x,
                "center_y": center_y,
                "area": area
            })
    
    cap.release()
    
    if not face_areas:
        print("No faces detected in video")
        return False
    
    # Analyze face areas
    print(f"Detected {len(face_areas)} face instances")
    
    # Group faces by position (simple clustering)
    face_clusters = cluster_faces_by_position(face_areas, width, height)
    
    # Create analysis results
    analysis_results = {
        "video_path": video_path,
        "video_properties": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames
        },
        "detection_settings": {
            "padding_ratio": padding_ratio,
            "min_face_size": min_face_size,
            "sample_interval": sample_interval
        },
        "total_face_instances": len(face_areas),
        "face_clusters": face_clusters,
        "face_areas": face_areas
    }
    
    # Save results
    results_path = os.path.join(output_dir, "face_area_results.json")
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Analysis results saved to: {results_path}")
    
    # Create visualizations
    create_face_area_visualizations(analysis_results, output_dir)
    
    # Clear GPU cache
    if available_gpus:
        clear_gpu_cache(gpu_id)
    
    return True


def cluster_faces_by_position(face_areas, video_width, video_height):
    """Cluster faces by position to identify different speakers."""
    
    if not face_areas:
        return []
    
    # Simple clustering based on center position
    clusters = []
    tolerance = min(video_width, video_height) * 0.1  # 10% of smaller dimension
    
    for face in face_areas:
        assigned = False
        
        for cluster in clusters:
            # Check if face is close to cluster center
            cluster_center_x = sum(f['center_x'] for f in cluster) / len(cluster)
            cluster_center_y = sum(f['center_y'] for f in cluster) / len(cluster)
            
            distance = np.sqrt((face['center_x'] - cluster_center_x)**2 + (face['center_y'] - cluster_center_y)**2)
            
            if distance < tolerance:
                cluster.append(face)
                assigned = True
                break
        
        if not assigned:
            clusters.append([face])
    
    # Convert clusters to analysis format
    cluster_analysis = []
    for i, cluster in enumerate(clusters):
        if len(cluster) < 3:  # Skip small clusters
            continue
            
        center_x = sum(f['center_x'] for f in cluster) / len(cluster)
        center_y = sum(f['center_y'] for f in cluster) / len(cluster)
        avg_area = sum(f['area'] for f in cluster) / len(cluster)
        
        cluster_analysis.append({
            "cluster_id": f"SPEAKER_{i:02d}",
            "face_count": len(cluster),
            "center_x": center_x,
            "center_y": center_y,
            "avg_area": avg_area,
            "x_range": (min(f['x1'] for f in cluster), max(f['x2'] for f in cluster)),
            "y_range": (min(f['y1'] for f in cluster), max(f['y2'] for f in cluster)),
            "time_range": (min(f['time'] for f in cluster), max(f['time'] for f in cluster))
        })
    
    return cluster_analysis


def create_face_area_visualizations(analysis_results, output_dir):
    """Create visualizations of face area analysis."""
    
    print("Creating face area visualizations...")
    
    # Extract data
    face_areas = analysis_results['face_areas']
    clusters = analysis_results['face_clusters']
    video_width = analysis_results['video_properties']['width']
    video_height = analysis_results['video_properties']['height']
    
    # Create face area analysis plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Face positions over time
    plt.subplot(2, 2, 1)
    for cluster in clusters:
        cluster_faces = [f for f in face_areas if any(f['center_x'] >= cluster['x_range'][0] and 
                                                    f['center_x'] <= cluster['x_range'][1] and
                                                    f['center_y'] >= cluster['y_range'][0] and 
                                                    f['center_y'] <= cluster['y_range'][1])]
        
        times = [f['time'] for f in cluster_faces]
        x_positions = [f['center_x'] for f in cluster_faces]
        
        plt.scatter(times, x_positions, label=cluster['cluster_id'], alpha=0.6)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('X Position (pixels)')
    plt.title('Face Positions Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Face area distribution
    plt.subplot(2, 2, 2)
    areas = [f['area'] for f in face_areas]
    plt.hist(areas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Face Area (pixels²)')
    plt.ylabel('Frequency')
    plt.title('Face Area Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Face positions in video frame
    plt.subplot(2, 2, 3)
    for cluster in clusters:
        cluster_faces = [f for f in face_areas if any(f['center_x'] >= cluster['x_range'][0] and 
                                                    f['center_x'] <= cluster['x_range'][1] and
                                                    f['center_y'] >= cluster['y_range'][0] and 
                                                    f['center_y'] <= cluster['y_range'][1])]
        
        x_positions = [f['center_x'] for f in cluster_faces]
        y_positions = [f['center_y'] for f in cluster_faces]
        
        plt.scatter(x_positions, y_positions, label=cluster['cluster_id'], alpha=0.6)
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Face Positions in Video Frame')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, video_width)
    plt.ylim(video_height, 0)  # Invert Y axis for video coordinates
    
    # Plot 4: Cluster statistics
    plt.subplot(2, 2, 4)
    if clusters:
        cluster_ids = [c['cluster_id'] for c in clusters]
        face_counts = [c['face_count'] for c in clusters]
        
        plt.bar(cluster_ids, face_counts, alpha=0.7)
        plt.xlabel('Speaker Cluster')
        plt.ylabel('Face Detection Count')
        plt.title('Face Detection Count by Speaker')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "face_area_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Face area analysis plot saved to: {plot_path}")
    
    # Create total face areas visualization
    create_total_face_areas_plot(analysis_results, output_dir)


def create_total_face_areas_plot(analysis_results, output_dir):
    """Create a comprehensive visualization of all face areas."""
    
    face_areas = analysis_results['face_areas']
    clusters = analysis_results['face_clusters']
    video_width = analysis_results['video_properties']['width']
    video_height = analysis_results['video_properties']['height']
    
    plt.figure(figsize=(12, 8))
    
    # Create a heatmap-like visualization
    for cluster in clusters:
        cluster_faces = [f for f in face_areas if any(f['center_x'] >= cluster['x_range'][0] and 
                                                    f['center_x'] <= cluster['x_range'][1] and
                                                    f['center_y'] >= cluster['y_range'][0] and 
                                                    f['center_y'] <= cluster['y_range'][1])]
        
        if cluster_faces:
            # Create bounding box for this cluster
            x1 = min(f['x1'] for f in cluster_faces)
            y1 = min(f['y1'] for f in cluster_faces)
            x2 = max(f['x2'] for f in cluster_faces)
            y2 = max(f['y2'] for f in cluster_faces)
            
            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2, 
                               label=f"{cluster['cluster_id']} ({len(cluster_faces)} detections)")
            plt.gca().add_patch(rect)
            
            # Add center point
            center_x = cluster['center_x']
            center_y = cluster['center_y']
            plt.scatter(center_x, center_y, c='red', s=100, marker='x', linewidths=3)
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Total Face Areas - Speaker Positioning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, video_width)
    plt.ylim(video_height, 0)  # Invert Y axis for video coordinates
    
    # Save plot
    plot_path = os.path.join(output_dir, "total_face_areas.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Total face areas plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect and analyze face areas in video")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", default="face_area", 
                       help="Output directory for analysis results")
    parser.add_argument("--padding-ratio", type=float, default=0.3,
                       help="Padding ratio around detected faces")
    parser.add_argument("--min-face-size", type=int, default=30,
                       help="Minimum face size for detection")
    parser.add_argument("--gpu-id", type=int, default=None,
                       help="Specific GPU ID to use (default: auto-select)")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Input video file not found: {args.video_path}")
        sys.exit(1)
    
    # Perform face area detection
    success = detect_faces_in_video(args.video_path, args.output_dir, 
                                  args.padding_ratio, args.min_face_size, args.gpu_id)
    
    if not success:
        print("Face area detection failed!")
        sys.exit(1)
    else:
        print("Face area detection completed successfully!")


if __name__ == "__main__":
    main() 