import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV starting from row 152 (skiprows skips 0-151)
file_path = 'bbl/long shot.csv'

df = pd.read_csv(file_path, skiprows=151, on_bad_lines='skip', low_memory=False)
df = df[:-1500]

print(f"Loaded {len(df)} rows")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Find amperage/current column
amperage_cols = [col for col in df.columns if 'amp' in col.lower() or 'current' in col.lower()]
print(f"\nFound current columns: {amperage_cols}")

if amperage_cols:
    current_col = amperage_cols[0]
    current = pd.to_numeric(df[current_col], errors='coerce').dropna()

    # Convert from centi-amps if needed
    if current.max() > 100:
        current = current / 100
        print("Converted from centi-amps to Amps")

    # Statistics
    print("\n=== Current Draw Statistics ===")
    print(f"Mean:      {current.mean():.2f} A")
    print(f"Median:    {current.median():.2f} A")
    print(f"Max:       {current.max():.2f} A")
    print(f"Min:       {current.min():.2f} A")
    print(f"Std Dev:   {current.std():.2f} A")
    print(f"Data points: {len(current)}")

    # Flight time estimate
    battery_ah = 5.5
    flight_time = (battery_ah * 0.8 / current.mean()) * 60
    print(f"\nEstimated flight time: {flight_time:.1f} minutes")

    # Create time axis (assuming 1000 Hz logging)
    time = np.arange(len(current)) / 2000

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time series
    ax1.plot(time, current, linewidth=0.8, alpha=0.7, color='blue')
    ax1.axhline(current.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {current.mean():.2f} A')
    ax1.axhline(current.median(), color='green', linestyle='--',
                linewidth=2, label=f'Median: {current.median():.2f} A')
    ax1.fill_between(time, 0, current, alpha=0.2, color='blue')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Current Draw (A)', fontsize=12)
    ax1.set_title('Current Draw Over Flight Duration', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Histogram
    ax2.hist(current, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(current.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {current.mean():.2f} A')
    ax2.axvline(current.median(), color='green', linestyle='--',
                linewidth=2, label=f'Median: {current.median():.2f} A')
    ax2.set_xlabel('Current Draw (A)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Current Draw Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('current_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✓ Plot saved as 'current_analysis.png'")

else:
    print("\n✗ No current column found. Available columns:")
    print(df.columns.tolist())
