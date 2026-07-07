import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("⏳ HealthTrace AI Engine — Processing real Tracy dataset...")

# ── Load real CSV files ────────────────────────────────────────────────
df       = pd.read_csv('vitals.csv')
contacts = pd.read_csv('contact_tracing.csv')
mobility = pd.read_csv('mobility.csv')

# ── Rename columns to unified internal names ───────────────────────────
df['temperature_c']   = pd.to_numeric(df['temperature'], errors='coerce')
df['heart_rate_bpm']  = pd.to_numeric(df['heartbeat'],   errors='coerce')
df['movement_status'] = pd.to_numeric(df['movement'],    errors='coerce').fillna(0).astype(int)

# ── Drop rows with missing vitals ──────────────────────────────────────
df = df.dropna(subset=['temperature_c', 'heart_rate_bpm'])

# ── Assign states from GPS ─────────────────────────────────────────────
def assign_state(lat, lon):
    if   7.20 <= lat <= 7.65   and 4.70 <= lon <= 5.25:  return 'Ondo'
    elif 7.75 <= lat <= 7.95   and 9.75 <= lon <= 9.90:  return 'Benue'
    elif 4.90 <= lat <= 5.10   and 8.30 <= lon <= 8.42:  return 'Cross River'
    elif 12.70 <= lat <= 13.20 and 5.00 <= lon <= 5.30:  return 'Sokoto'
    elif 7.60 <= lat <= 7.72   and 6.38 <= lon <= 6.46:  return 'Kogi'
    elif 5.35 <= lat <= 5.42   and 6.95 <= lon <= 7.05:  return 'Imo'
    elif 7.25 <= lat <= 7.95   and 9.84 <= lon <= 11.00: return 'Taraba'
    else:                                                 return 'Unknown'

df['state'] = df.apply(lambda r: assign_state(r['latitude'], r['longitude']), axis=1)

# ── AI Anomaly Detection — Isolation Forest ────────────────────────────
features = ['temperature_c', 'heart_rate_bpm', 'movement_status']
X        = df[features]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

model            = IsolationForest(contamination=0.04, random_state=42, n_jobs=-1)
df['anomaly_score'] = model.fit_predict(X_scaled)

# ── Save flagged devices ───────────────────────────────────────────────
infected_nodes = df[df['anomaly_score'] == -1]['device_id'].unique()

with open('infected_nodes.txt', 'w') as f:
    for node in infected_nodes:
        f.write(f"{node}\n")

print(f"✅ AI Engine Complete: {len(infected_nodes)} high-risk devices flagged.")
print(f"   Flagged devices: {list(infected_nodes)}")
print(f"   State breakdown:")
for state, count in df[df['anomaly_score']==-1]['state'].value_counts().items():
    print(f"   - {state}: {count} anomalous readings")
print(f"\n✅ infected_nodes.txt saved successfully.")
