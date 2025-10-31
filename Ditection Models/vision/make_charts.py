import pandas as pd
from pathlib import Path

# Load files
manifest = pd.read_csv('data/meta/manifest_own.csv')
yoloworld = pd.read_csv('results/yoloworld_scores_own.csv')
drone_yolo = pd.read_csv('results/drone_yolo_scores_own.cs')

print("Normalizing paths...\n")

# Fix manifest paths (remove ../ prefix)
manifest['path'] = manifest['path'].astype(str).apply(
    lambda x: x.replace('\\', '/').lstrip('./').lstrip('../').lstrip('/')
)

print("Manifest path examples (first 3):")
print(manifest['path'].head(3).tolist())

# Fix YOLO paths (same treatment)
yolo_col = yoloworld.columns[0]
yoloworld['path'] = yoloworld[yolo_col].astype(str).apply(
    lambda x: x.replace('\\', '/').lstrip('./').lstrip('../').lstrip('/')
)

drone_col = drone_yolo.columns[0]
drone_yolo['path'] = drone_yolo[drone_col].astype(str).apply(
    lambda x: x.replace('\\', '/').lstrip('./').lstrip('../').lstrip('/')
)

print("\nYOLO world path examples (first 3):")
print(yoloworld['path'].head(3).tolist())

# Verify matches now
manifest_paths = set(manifest['path'].unique())
yolo_paths = set(yoloworld['path'].unique())
matches = manifest_paths.intersection(yolo_paths)

print(f"\nMatching results:")
print(f"  Manifest paths: {len(manifest_paths)}")
print(f"  YOLO paths: {len(yolo_paths)}")
print(f"  Matching: {len(matches)}")

if len(matches) > 0:
    print(f"\n✓ SUCCESS! Sample matches (first 3):")
    for p in list(matches)[:3]:
        print(f"  {p}")

    # Save corrected CSVs
    manifest.to_csv('data/meta/manifest_own.csv', index=False)
    yoloworld.to_csv('results/yoloworld_scores_own.csv', index=False)
    drone_yolo.to_csv('results/drone_yolo_scores_own.csv', index=False)

    print("\n✓ Corrected CSVs saved!")
    print("\nNow run:")
    print(
        "python make_charts.py --manifest data/meta/manifest_own.csv --inputs results/yoloworld_scores_own.csv results/drone_yolo_scores_own.csv --outdir charts")
else:
    print("\n✗ Still no matches. Let me check individual components...")

    # Break down paths
    m_sample = list(manifest_paths)[0]
    y_sample = list(yolo_paths)[0]

    print(f"\nManifest: {m_sample}")
    print(f"YOLO:     {y_sample}")

    # Extract video folder and frame name
    m_parts = m_sample.split('/')
    y_parts = y_sample.split('/')

    print(f"\nManifest parts: {m_parts}")
    print(f"YOLO parts:     {y_parts}")
