import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create DataFrames from the data
# Standard Vector data
std_vector_data = {
    'Operation': ['Multiplication', 'Dot', 'Addition'],
    'Time_0_100': [63176, 291216, 29909],
    'Time_0_1000': [64279, 294096, 27121],
    'Time_0_2^16': [64582, 316584, 27588],
    'Memory_0_100': [9.5, 9.5, 9.5],
    'Memory_0_1000': [19, 19, 19],
    'Memory_0_2^16': [76.3, 76.3, 76.3]
}

# BSI-10M Vector data (Vector length-10M, Data structure - BSI)
bsi_vector_data = {
    'Operation': ['Multiplication', 'Dot', 'Addition'],
    'Time_0_100': [296296, 154063, 12787],
    'Time_0_1000': [537651, 316956, 16604],
    'Time_0_2^16': [1232678, 866003, 28390],
    'Memory_0_100': [8.3, 8.3, 8.3],
    'Memory_0_1000': [12, 12, 12],
    'Memory_0_2^16': [19, 19, 19]
}

df_std = pd.DataFrame(std_vector_data)
df_bsi = pd.DataFrame(bsi_vector_data)

# Set up the plot
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Performance Comparison: Time and Memory Usage', fontsize=18, y=0.98)

# Define ranges and their column names
ranges = ['[0, 100]', '[0, 1000]', '[0, 2^16]']
col_suffixes = ['_0_100', '_0_1000', '_0_2^16']

# Define colors for consistent appearance
std_color = '#1f77b4'  # Blue
bsi_color = '#ff7f0e'  # Orange

# Plot data for each operation
for i, op in enumerate(df_std['Operation']):
    # Time plot (left column)
    ax_time = axes[i, 0]
    ax_time.set_title(f'{op} - Execution Time', fontsize=14)
    ax_time.set_ylabel('Time (microseconds)', fontsize=12)
    ax_time.grid(axis='y', linestyle='--', alpha=0.7)

    # Memory plot (right column)
    ax_mem = axes[i, 1]
    ax_mem.set_title(f'{op} - Memory Usage', fontsize=14)
    ax_mem.set_ylabel('Memory (MB)', fontsize=12)
    ax_mem.grid(axis='y', linestyle='--', alpha=0.7)

    # Position for the bars
    x = np.arange(len(ranges))
    width = 0.35

    # Get data for the current operation
    std_times = [df_std.loc[i, f'Time{suffix}'] for suffix in col_suffixes]
    bsi_times = [df_bsi.loc[i, f'Time{suffix}'] for suffix in col_suffixes]
    std_mems = [df_std.loc[i, f'Memory{suffix}'] for suffix in col_suffixes]
    bsi_mems = [df_bsi.loc[i, f'Memory{suffix}'] for suffix in col_suffixes]

    # Plot the time bars
    time_bars1 = ax_time.bar(x - width/2, std_times, width, label='Standard Vector', color=std_color)
    time_bars2 = ax_time.bar(x + width/2, bsi_times, width, label='BSI-10M Vector', color=bsi_color)

    # Plot the memory bars
    mem_bars1 = ax_mem.bar(x - width/2, std_mems, width, label='Standard Vector', color=std_color)
    mem_bars2 = ax_mem.bar(x + width/2, bsi_mems, width, label='BSI-10M Vector', color=bsi_color)

    # Format large numbers for better readability
    def format_number(num):
        if num >= 1000000:
            return f'{num/1000000:.1f}M'
        elif num >= 1000:
            return f'{num/1000:.1f}K'
        else:
            return f'{num}'

    # Add value labels to bars
    for bars in [time_bars1, time_bars2]:
        for rect in bars:
            height = rect.get_height()
            rotation = 90 if height > 1000000 else 0
            y_pos = height + 0.02*height
            ax_time.text(rect.get_x() + rect.get_width()/2., y_pos,
                         format_number(height),
                         ha='center', va='bottom', rotation=rotation, fontsize=9)

    for bars in [mem_bars1, mem_bars2]:
        for rect in bars:
            height = rect.get_height()
            ax_mem.text(rect.get_x() + rect.get_width()/2., height + 0.02*height,
                        f'{height}',
                        ha='center', va='bottom', fontsize=9)

    # Set x-ticks
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(ranges)
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(ranges)

    # Only add legend to the first row
    if i == 0:
        ax_time.legend()
        ax_mem.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('performance_comparison.png', dpi=300)
plt.show()