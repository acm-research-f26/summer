"""
generate_mock_data.py
Generates synthetic HAI-like CSVs so the pipeline runs immediately.
When you download the real HAI 21.03 dataset from github.com/icsdataset/hai,
replace data/train1.csv and data/test1.csv with the real files and re-run.

HAI columns follow a pattern of:
  - Sensor readings (P1_FT01, P1_PT01, etc.) -- continuous float values
  - Actuator/command columns (P1_PCV01Z, P1_B2004, etc.) -- often 0/1 or setpoints
  - attack: 0 = normal, nonzero = attack scenario ID
"""

import numpy as np
import pandas as pd
import os

SENSOR_COLS = ["P1_FT01", "P1_PT01", "P1_FT02", "P2_FT01", "P2_PT01",
               "P3_FT01", "P3_PT01", "P4_FT01", "P4_PT01", "P4_ST01"]

ACTUATOR_COLS = ["P1_PCV01Z", "P1_PCV02Z", "P2_MCP01", "P2_MCP02",
                 "P3_LCV01Z", "P3_PCV01Z", "P4_HV01", "P4_HV02"]

ALL_COLS = SENSOR_COLS + ACTUATOR_COLS + ["attack"]

np.random.seed(42)


def make_normal_row(t):
    sensors = np.sin(np.arange(len(SENSOR_COLS)) * 0.3 + t * 0.01) * 5 + 10
    sensors += np.random.randn(len(SENSOR_COLS)) * 0.1
    actuators = (np.sin(np.arange(len(ACTUATOR_COLS)) * 0.5 + t * 0.02) > 0).astype(float)
    actuators += np.random.randn(len(ACTUATOR_COLS)) * 0.02
    actuators = np.clip(actuators, 0, 1)
    return list(sensors) + list(actuators) + [0]


def make_attack_row(t, attack_id):
    row = make_normal_row(t)
    # Attacks manipulate actuator commands -- spike or zero out setpoints
    for i in range(len(ACTUATOR_COLS)):
        if np.random.rand() < 0.6:
            row[len(SENSOR_COLS) + i] = np.random.choice([0.0, 1.0, 0.95, 0.05])
    row[-1] = attack_id
    return row


def generate(n_normal, n_attack_episodes=5, attack_len=80):
    rows = []
    for t in range(n_normal):
        rows.append(make_normal_row(t))

    for ep in range(n_attack_episodes):
        start = n_normal + ep * (attack_len + 40)
        for t in range(attack_len):
            rows.append(make_attack_row(start + t, ep + 1))
        # brief normal recovery after each attack
        for t in range(40):
            rows.append(make_normal_row(start + attack_len + t))

    return pd.DataFrame(rows, columns=ALL_COLS)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_df = generate(n_normal=3000, n_attack_episodes=0)
    train_df.to_csv(os.path.join(DATA_DIR, "train1.csv"), index=False)
    print(f"train1.csv: {len(train_df)} rows, {(train_df.attack != 0).sum()} attack rows")

    test_df = generate(n_normal=1000, n_attack_episodes=8, attack_len=60)
    test_df.to_csv(os.path.join(DATA_DIR, "test1.csv"), index=False)
    print(f"test1.csv:  {len(test_df)} rows, {(test_df.attack != 0).sum()} attack rows")
    print("Done. Replace with real HAI 21.03 CSVs for actual results.")


if __name__ == "__main__":
    main()
