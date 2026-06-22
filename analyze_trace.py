# Reads trace.csv, categorizes vertices into low/medium/high degree buckets,
# and analyzes access patterns for pointer vs CSR layouts to understand spatial locality.

import pandas as pd
import numpy as np
import os

if not os.path.exists("trace.csv"):
    print("Error: trace.csv not found! Run the C++ executable first.")
    exit(1)

print("Loading trace data... (This might take a few seconds)")
df = pd.read_csv("trace.csv")

total_vertices = 1000
hdv_threshold = np.sqrt(total_vertices)  # High-degree: >= 31.6
ldv_threshold = 10                        # Low-degree: <= 10

print(f"Bucket Boundaries -> Low: <= {ldv_threshold} | High (Hubs): >= {hdv_threshold:.1f}\n")

def assign_bucket(deg):
    if deg <= ldv_threshold:
        return "Low-Degree"
    elif deg >= hdv_threshold:
        return "High-Degree (Hub)"
    else:
        return "Medium-Degree"

df['bucket'] = df['degree'].apply(assign_bucket)

print("--- Total Edge Accesses by Bucket ---")
summary = df.groupby(['layout', 'bucket']).size().unstack(fill_value=0)
print(summary)
print("-" * 40)

print("\nCalculating Spatial Locality Metrics...")
for layout in ['POINTER', 'CSR']:
    layout_df = df[df['layout'] == layout].copy()
    for bucket in ['Low-Degree', 'Medium-Degree', 'High-Degree (Hub)']:
        bucket_df = layout_df[layout_df['bucket'] == bucket]
        if len(bucket_df) > 1:
            address_jumps = np.abs(np.diff(bucket_df['address']))
            avg_stride = np.mean(address_jumps)
            print(f"[{layout}] {bucket:18} -> Avg Memory Jump: {avg_stride:12.2f} bytes")