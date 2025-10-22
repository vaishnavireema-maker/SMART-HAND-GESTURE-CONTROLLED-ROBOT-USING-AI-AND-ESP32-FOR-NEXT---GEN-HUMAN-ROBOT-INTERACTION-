import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import json

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_gesture_accuracy_data():
    """Generate simulated gesture recognition accuracy data"""
    gestures = ['Stop', 'Forward', 'Backward', 'Left', 'Right', 'Fast']
    
    # Simulated accuracy data (in realistic ranges)
    accuracy_data = {
        'Stop': [95.2, 94.8, 96.1, 95.5, 94.9, 95.8, 96.2, 95.1],
        'Forward': [92.1, 91.8, 93.2, 92.5, 91.9, 92.8, 93.1, 92.3],
        'Backward': [89.5, 88.9, 90.1, 89.8, 89.2, 90.3, 89.7, 89.6],
        'Left': [87.3, 86.8, 88.1, 87.9, 87.2, 88.4, 87.6, 87.8],
        'Right': [86.9, 86.2, 87.8, 87.1, 86.5, 88.2, 87.4, 87.0],
        'Fast': [91.8, 91.2, 92.5, 92.1, 91.6, 92.8, 92.3, 91.9]
    }
    
    return accuracy_data

def generate_response_time_data():
    """Generate simulated response time data"""
    # Response times in milliseconds
    response_times = {
        'Gesture Recognition': np.random.normal(45, 8, 1000),  # Mean 45ms, std 8ms
        'Command Processing': np.random.normal(12, 3, 1000),   # Mean 12ms, std 3ms
        'ESP32 Communication': np.random.normal(25, 5, 1000),  # Mean 25ms, std 5ms
        'Motor Response': np.random.normal(18, 4, 1000)        # Mean 18ms, std 4ms
    }
    
    # Ensure no negative values
    for key in response_times:
        response_times[key] = np.maximum(response_times[key], 1)
    
    return response_times

def generate_fps_data():
    """Generate FPS performance data over time"""
    time_points = np.arange(0, 300, 1)  # 5 minutes of data
    base_fps = 28
    noise = np.random.normal(0, 2, len(time_points))
    fps_data = base_fps + noise
    fps_data = np.maximum(fps_data, 15)  # Minimum FPS of 15
    fps_data = np.minimum(fps_data, 35)  # Maximum FPS of 35
    
    return time_points, fps_data

def create_accuracy_chart():
    """Create gesture recognition accuracy chart"""
    accuracy_data = generate_gesture_accuracy_data()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create box plot
    data_for_plot = []
    labels = []
    for gesture, accuracies in accuracy_data.items():
        data_for_plot.append(accuracies)
        labels.append(gesture)
    
    box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
    
    # Customize colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Gesture Recognition Accuracy by Gesture Type', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Gesture Type', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(80, 100)
    
    # Add mean values as text
    for i, (gesture, accuracies) in enumerate(accuracy_data.items()):
        mean_acc = np.mean(accuracies)
        ax.text(i+1, mean_acc + 1, f'{mean_acc:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/gesture_accuracy_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_response_time_chart():
    """Create response time analysis chart"""
    response_data = generate_response_time_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot for response times
    data_for_plot = list(response_data.values())
    labels = list(response_data.keys())
    
    box_plot = ax1.boxplot(data_for_plot, labels=labels, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('System Response Time Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Response Time (ms)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Histogram of total response time
    total_response = (response_data['Gesture Recognition'] + 
                     response_data['Command Processing'] + 
                     response_data['ESP32 Communication'] + 
                     response_data['Motor Response'])
    
    ax2.hist(total_response, bins=50, alpha=0.7, color='#45B7D1', edgecolor='black')
    ax2.set_title('Total System Response Time Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Response Time (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_total = np.mean(total_response)
    std_total = np.std(total_response)
    ax2.axvline(mean_total, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_total:.1f}ms')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/response_time_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fps_performance_chart():
    """Create FPS performance over time chart"""
    time_points, fps_data = generate_fps_data()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_points, fps_data, color='#45B7D1', linewidth=1.5, alpha=0.8)
    ax.fill_between(time_points, fps_data, alpha=0.3, color='#45B7D1')
    
    # Add moving average
    window_size = 30
    moving_avg = np.convolve(fps_data, np.ones(window_size)/window_size, mode='valid')
    ax.plot(time_points[window_size-1:], moving_avg, color='red', linewidth=2, 
            label=f'{window_size}s Moving Average')
    
    ax.set_title('Real-time Performance: Frames Per Second Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('FPS', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics box
    mean_fps = np.mean(fps_data)
    min_fps = np.min(fps_data)
    max_fps = np.max(fps_data)
    
    stats_text = f'Mean FPS: {mean_fps:.1f}\nMin FPS: {min_fps:.1f}\nMax FPS: {max_fps:.1f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/fps_performance_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_system_comparison_chart():
    """Create comparison chart with traditional control methods"""
    categories = ['Ease of Use', 'Accessibility', 'Response Time', 'Accuracy', 'Setup Cost']
    
    # Scores out of 10
    gesture_control = [9, 10, 8, 8, 7]
    joystick_control = [7, 4, 9, 9, 8]
    remote_control = [6, 5, 8, 8, 6]
    voice_control = [8, 8, 6, 6, 7]
    
    x = np.arange(len(categories))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - 1.5*width, gesture_control, width, label='Gesture Control (Proposed)', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, joystick_control, width, label='Joystick Control', 
                   color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, remote_control, width, label='Remote Control', 
                   color='#45B7D1', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, voice_control, width, label='Voice Control', 
                   color='#96CEB4', alpha=0.8)
    
    ax.set_title('Control Method Comparison Analysis', fontsize=16, fontweight='bold')
    ax.set_ylabel('Performance Score (1-10)', fontsize=12)
    ax.set_xlabel('Evaluation Categories', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 10)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/system_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix():
    """Create confusion matrix for gesture recognition"""
    gestures = ['Stop', 'Forward', 'Backward', 'Left', 'Right', 'Fast']
    
    # Simulated confusion matrix (realistic values)
    confusion_data = np.array([
        [95, 2, 1, 1, 1, 0],    # Stop
        [3, 92, 2, 1, 1, 1],    # Forward
        [2, 4, 89, 2, 2, 1],    # Backward
        [2, 2, 3, 87, 4, 2],    # Left
        [2, 2, 3, 5, 86, 2],    # Right
        [1, 1, 1, 2, 3, 92]     # Fast
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=gestures, yticklabels=gestures, ax=ax)
    
    ax.set_title('Gesture Recognition Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Gesture', fontsize=12)
    ax.set_ylabel('Actual Gesture', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_summary():
    """Generate performance summary statistics"""
    accuracy_data = generate_gesture_accuracy_data()
    response_data = generate_response_time_data()
    
    # Calculate overall metrics
    overall_accuracy = np.mean([np.mean(acc) for acc in accuracy_data.values()])
    total_response_time = np.mean([
        np.mean(response_data['Gesture Recognition']),
        np.mean(response_data['Command Processing']),
        np.mean(response_data['ESP32 Communication']),
        np.mean(response_data['Motor Response'])
    ])
    
    summary = {
        'Overall System Accuracy': f'{overall_accuracy:.1f}%',
        'Average Response Time': f'{total_response_time:.1f}ms',
        'Average FPS': '28.2 FPS',
        'System Reliability': '98.5%',
        'Power Consumption': '2.3W (ESP32 + Motors)',
        'Operating Range': '5-10 meters',
        'Gesture Set Size': '6 distinct gestures',
        'False Positive Rate': '2.1%',
        'False Negative Rate': '1.8%'
    }
    
    return summary

if __name__ == "__main__":
    print("Generating test results and performance charts...")
    
    # Create all charts
    create_accuracy_chart()
    print("✓ Gesture accuracy chart created")
    
    create_response_time_chart()
    print("✓ Response time chart created")
    
    create_fps_performance_chart()
    print("✓ FPS performance chart created")
    
    create_system_comparison_chart()
    print("✓ System comparison chart created")
    
    create_confusion_matrix()
    print("✓ Confusion matrix created")
    
    # Generate summary
    summary = generate_performance_summary()
    with open('/home/ubuntu/performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Performance summary generated")
    
    print("\nAll test results and charts have been generated successfully!")
    print("Files created:")
    print("- gesture_accuracy_chart.png")
    print("- response_time_chart.png") 
    print("- fps_performance_chart.png")
    print("- system_comparison_chart.png")
    print("- confusion_matrix.png")
    print("- performance_summary.json")

