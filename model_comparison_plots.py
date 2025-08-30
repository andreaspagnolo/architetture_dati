import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the style for the plots
plt.style.use('ggplot')
sns.set_palette('Set2')

# Define the models and their metrics
models = ['Decision Tree', 'SVM', 'Neural Network']

# Example metrics - replace these with actual values from your notebook
# Format: [original_model_value, optimized_model_value]
accuracy = {
    'Decision Tree': [0.85, 0.89],
    'SVM': [0.87, 0.91],
    'Neural Network': [0.86, 0.90]
}

precision = {
    'Decision Tree': [0.84, 0.88],
    'SVM': [0.86, 0.90],
    'Neural Network': [0.85, 0.89]
}

recall = {
    'Decision Tree': [0.83, 0.87],
    'SVM': [0.85, 0.89],
    'Neural Network': [0.84, 0.88]
}

f1_score = {
    'Decision Tree': [0.83, 0.87],
    'SVM': [0.85, 0.90],
    'Neural Network': [0.84, 0.88]
}

training_time = {
    'Decision Tree': [0.05, 0.08],
    'SVM': [0.15, 0.20],
    'Neural Network': [2.5, 3.0]
}

# Function to create comparison bar plots
def create_comparison_plot(metric_dict, metric_name, ylabel):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    original_values = [metric_dict[model][0] for model in models]
    optimized_values = [metric_dict[model][1] for model in models]
    
    rects1 = ax.bar(x - width/2, original_values, width, label='Original')
    rects2 = ax.bar(x + width/2, optimized_values, width, label='Optimized')
    
    ax.set_title(f'Comparison of {metric_name} across Models', fontsize=16)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f'{metric_name.lower().replace(" ", "_")}_comparison.png', dpi=300)
    plt.show()

# Create all comparison plots
create_comparison_plot(accuracy, 'Accuracy', 'Accuracy Score')
create_comparison_plot(precision, 'Precision', 'Precision Score')
create_comparison_plot(recall, 'Recall', 'Recall Score')
create_comparison_plot(f1_score, 'F1 Score', 'F1 Score')

# Create training time comparison plot (different scale)
def create_time_comparison_plot():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    original_values = [training_time[model][0] for model in models]
    optimized_values = [training_time[model][1] for model in models]
    
    rects1 = ax.bar(x - width/2, original_values, width, label='Original')
    rects2 = ax.bar(x + width/2, optimized_values, width, label='Optimized')
    
    ax.set_title('Comparison of Training Time across Models', fontsize=16)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Training Time (seconds)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('training_time_comparison.png', dpi=300)
    plt.show()

create_time_comparison_plot()

# Create a radar chart to compare all metrics across models
def create_radar_chart():
    # Prepare the data
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    
    # Set up the radar chart parameters
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add subplots for each model
    for i, model in enumerate(models):
        ax = fig.add_subplot(2, 2, i+1, polar=True)
        
        # Get values for original and optimized models
        original_values = [accuracy[model][0], precision[model][0], recall[model][0], f1_score[model][0]]
        optimized_values = [accuracy[model][1], precision[model][1], recall[model][1], f1_score[model][1]]
        
        # Close the loop for plotting
        original_values += original_values[:1]
        optimized_values += optimized_values[:1]
        
        # Plot the values
        ax.plot(angles, original_values, 'o-', linewidth=2, label='Original')
        ax.plot(angles, optimized_values, 'o-', linewidth=2, label='Optimized')
        ax.fill(angles, original_values, alpha=0.1)
        ax.fill(angles, optimized_values, alpha=0.1)
        
        # Set the labels
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        
        # Set the y-axis limits
        ax.set_ylim(0.8, 0.95)
        
        # Add title and legend
        ax.set_title(model, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('radar_chart_comparison.png', dpi=300)
    plt.show()

create_radar_chart()

# Create a combined bar chart for all models and metrics
def create_combined_bar_chart():
    # Prepare the data
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_dicts = [accuracy, precision, recall, f1_score]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set up the bar chart parameters
    x = np.arange(len(models))
    width = 0.1
    offsets = [-0.15, -0.05, 0.05, 0.15]
    
    # Plot bars for each metric
    for i, (metric, metric_dict) in enumerate(zip(metrics, metric_dicts)):
        original_values = [metric_dict[model][0] for model in models]
        optimized_values = [metric_dict[model][1] for model in models]
        
        ax.bar(x + offsets[i] - width/2, original_values, width, label=f'Original {metric}')
        ax.bar(x + offsets[i] + width/2, optimized_values, width, label=f'Optimized {metric}')
    
    # Set labels and title
    ax.set_title('Comparison of All Metrics across Models', fontsize=16)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    
    plt.tight_layout()
    plt.savefig('combined_metrics_comparison.png', dpi=300)
    plt.show()

create_combined_bar_chart()

print("All comparison plots have been created and saved as PNG files.")