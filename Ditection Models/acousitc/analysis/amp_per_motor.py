import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV starting from row 152
file_path = 'bbl/long shot.csv'

df = pd.read_csv(file_path, skiprows=151, on_bad_lines='skip', low_memory=False)
df = df[:-1500]

print(f"Loaded {len(df)} rows")
print("\nAvailable columns:")
print(df.columns.tolist())

# Find motor current columns (usually named motor[0], motor[1], etc.)
motor_columns = [col for col in df.columns if 'motor' in col.lower() and 'current' in col.lower()]

# If not found, try just 'motor' or 'esc'
if not motor_columns:
    motor_columns = [col for col in df.columns if 'motor' in col.lower() or 'esc' in col.lower()]

print(f"\nFound motor columns: {motor_columns}")

if motor_columns:
    # Create time axis
    time = np.arange(len(df)) / 2000  # Assuming 1000 Hz

    # Analyze each motor
    print("\n=== Per-Motor Current Analysis ===")
    fig, ax = plt.subplots(figsize=(14, 7))

    motor_stats = []
    colors = ['blue', 'green', 'red', 'orange']

    for i, motor_col in enumerate(motor_columns[:4]):  # Max 4 motors
        motor_current = pd.to_numeric(df[motor_col], errors='coerce').dropna()

        # Convert from centi-amps if needed
        if motor_current.max() > 100:
            motor_current = motor_current / 100

        mean_current = motor_current.mean()
        max_current = motor_current.max()

        motor_stats.append({
            'Motor': motor_col,
            'Mean (A)': mean_current,
            'Max (A)': max_current,
            'Std (A)': motor_current.std()
        })

        # Plot
        ax.plot(time[:len(motor_current)], motor_current,
                label=f'{motor_col} (mean: {mean_current:.2f}A)',
                linewidth=0.8, alpha=0.7, color=colors[i % 4])

        print(
            f"{motor_col:15s}: Mean = {mean_current:.2f} A, Max = {max_current:.2f} A, Std = {motor_current.std():.2f} A")

    # Add total current if available
    if 'amperage' in [c.lower() for c in df.columns] or 'current' in [c.lower() for c in df.columns]:
        total_col = [c for c in df.columns if 'amperage' in c.lower() or 'current' in c.lower()][0]
        total_current = pd.to_numeric(df[total_col], errors='coerce').dropna()
        if total_current.max() > 100:
            total_current = total_current / 100
        ax.plot(time[:len(total_current)], total_current,
                label=f'Total (mean: {total_current.mean():.2f}A)',
                linewidth=1.5, alpha=0.9, color='black', linestyle='--')
        print(f"\nTotal current:   Mean = {total_current.mean():.2f} A, Max = {total_current.max():.2f} A")

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Current per Motor (A)', fontsize=12)
    ax.set_title('Individual Motor Current Draw Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('motor_current_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✓ Plot saved as 'motor_current_analysis.png'")

    # Create comparison bar chart
    stats_df = pd.DataFrame(motor_stats)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stats_df))
    width = 0.35

    ax.bar(x - width / 2, stats_df['Mean (A)'], width,
           label='Mean Current', color='skyblue', edgecolor='black')
    ax.bar(x + width / 2, stats_df['Max (A)'], width,
           label='Max Current', color='salmon', edgecolor='black')

    ax.set_xlabel('Motor', fontsize=12)
    ax.set_ylabel('Current (A)', fontsize=12)
    ax.set_title('Mean and Max Current by Motor', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['Motor'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('motor_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Comparison plot saved as 'motor_comparison.png'")

else:
    print("\n✗ No motor current columns found!")
    print("Available columns:", df.columns.tolist())
