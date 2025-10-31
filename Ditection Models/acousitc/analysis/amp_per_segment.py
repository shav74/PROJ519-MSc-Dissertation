import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV starting from row 152
file_path = 'bbl/long shot.csv'

df = pd.read_csv(file_path, skiprows=151, on_bad_lines='skip', low_memory=False)
df = df[:-1500] # landing imbalance

# Find current column
amperage_cols = [col for col in df.columns if 'amp' in col.lower() or 'current' in col.lower()]
if not amperage_cols:
    print("No current column found!")
    exit()

current_col = amperage_cols[0]
current = pd.to_numeric(df[current_col], errors='coerce').dropna()

# Convert from centi-amps if needed
if current.max() > 100:
    current = current / 100

# Create time in seconds
time = np.arange(len(current)) / 2000  # 1000 Hz logging
total_time = time[-1]

print(f"Total flight time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
print(f"Total data points: {len(current)}")

# Define flight segments (ADJUST THESE based on your actual flight!)
# You can identify segments by looking at the current draw plot
segments = {
    'Takeoff': (3, 5),
    'Hover': (5,9),
    'Approach': (9, 14),
    'Fly-by': (14, 19),
    'Landing': (19, 20)
}

# Or auto-detect based on current thresholds (simple method)
# Low current = hover, High current = maneuver
# Uncomment if you want auto-detection:
"""
hover_threshold = current.quantile(0.33)  # Bottom 33% = hover
maneuver_threshold = current.quantile(0.66)  # Top 33% = maneuvers
"""

print("\n=== Segment Analysis ===")
segment_stats = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot full timeline with segments highlighted
ax1.plot(time, current, linewidth=0.8, alpha=0.7, color='blue', label='Current Draw')
ax1.axhline(current.mean(), color='red', linestyle='--',
            linewidth=1.5, label=f'Overall Mean: {current.mean():.2f}A')

colors_seg = ['yellow', 'lightgreen', 'orange', 'purple', 'pink']

for i, (segment_name, (start, end)) in enumerate(segments.items()):
    # Get segment data
    mask = (time >= start) & (time <= end)
    segment_current = current[mask]
    segment_time = time[mask]

    if len(segment_current) > 0:
        seg_mean = segment_current.mean()
        seg_median = segment_current.median()
        seg_max = segment_current.max()
        seg_std = segment_current.std()

        segment_stats.append({
            'Segment': segment_name,
            'Mean (A)': seg_mean,
            'Median (A)': seg_median,
            'Max (A)': seg_max,
            'Std (A)': seg_std,
            'Duration (s)': end - start
        })

        # Highlight segment
        ax1.axvspan(start, end, alpha=0.2, color=colors_seg[i % 5],
                    label=f'{segment_name} ({seg_mean:.2f}A)')

        print(
            f"{segment_name:12s}: Mean = {seg_mean:.2f} A, Median = {seg_median:.2f} A, Max = {seg_max:.2f} A, Std = {seg_std:.2f} A")

ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Current Draw (A)', fontsize=12)
ax1.set_title('Current Draw by Flight Segment', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# Segment comparison bar chart
seg_df = pd.DataFrame(segment_stats)
x = np.arange(len(seg_df))
width = 0.35

ax2.bar(x - width / 2, seg_df['Mean (A)'], width,
        label='Mean Current', color='skyblue', edgecolor='black')
ax2.bar(x + width / 2, seg_df['Max (A)'], width,
        label='Max Current', color='salmon', edgecolor='black')

# Add error bars for standard deviation
ax2.errorbar(x - width / 2, seg_df['Mean (A)'], yerr=seg_df['Std (A)'],
             fmt='none', ecolor='black', capsize=5, alpha=0.5)

ax2.set_xlabel('Flight Segment', fontsize=12)
ax2.set_ylabel('Current Draw (A)', fontsize=12)
ax2.set_title('Mean and Max Current by Flight Segment', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(seg_df['Segment'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('segment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Plot saved as 'segment_analysis.png'")

# Save statistics to CSV
seg_df.to_csv('segment_statistics.csv', index=False)
print("✓ Statistics saved as 'segment_statistics.csv'")
