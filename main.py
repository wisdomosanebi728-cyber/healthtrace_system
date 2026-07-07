from fastapi import FastAPI, HTTPException, Query, Body, APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import math
import sqlite3
import requests as req
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = FastAPI(title="HealthTrace AI Core Service", version="3.0.0")

# Create an APIRouter with the /api prefix to cleanly handle all frontend requests
api_router = APIRouter(prefix="/api")

ENV_FILE = ".env"
DB_FILE = "healthtrace.db"

# =====================================================================
# DATABASE PERMANENT STORAGE INITIALISATION
# =====================================================================
def init_db():
    """Initialises a local SQLite database file to permanently store worker profiles."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            zone TEXT NOT NULL,
            name TEXT NOT NULL
        )
    ''')
    # Insert default users if the database table is completely fresh and empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
            [
                ("admin", "admin123", "Admin", "All States", "Chief Medical Officer"),
                ("drtunde", "ondo1", "Health Worker", "Ondo", "Dr. Tunde Adeyemi")
            ]
        )
    conn.commit()
    conn.close()

# Execute database setup immediately on startup
init_db()

RUNTIME_INGEST_CACHE = []

# =====================================================================
# CORE UTILITIES & PERSISTENT KEY MANAGEMENT
# =====================================================================
def load_env_keys():
    keys = {"provider": "groq", "groq_key": "", "claude_key": "", "termii_key": ""}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    keys[k.strip()] = v.strip()
    return keys

def save_env_keys(provider, groq_key, claude_key, termii_key):
    with open(ENV_FILE, "w") as f:
        f.write(f"provider={provider}\n")
        f.write(f"groq_key={groq_key}\n")
        f.write(f"claude_key={claude_key}\n")
        f.write(f"termii_key={termii_key}\n")

STATES_GEO_MARKERS = {
    'Ondo':         {"lat": 7.31,  "lon": 5.13,  "zoom": 11.0},
    'Benue':        {"lat": 7.85,  "lon": 9.78,  "zoom": 11.0},
    'Cross River':  {"lat": 4.97,  "lon": 8.34,  "zoom": 11.0},
    'Sokoto':       {"lat": 13.05, "lon": 5.25,  "zoom": 11.0},
    'Kogi':         {"lat": 7.68,  "lon": 6.42,  "zoom": 11.5},
    'Imo':          {"lat": 5.38,  "lon": 6.99,  "zoom": 11.5},
    'Taraba':       {"lat": 7.85,  "lon": 10.50, "zoom": 10.5},
}

# =====================================================================
# DATA INGESTION & PIPELINE SYNTHESIS ENGINE
# =====================================================================
def process_healthtrace_layers():
    vitals = pd.read_csv('vitals.csv').rename(
        columns={
            'temperature': 'temperature_c',
            'heartbeat': 'heart_rate_bpm',
            'movement': 'movement_status'
        }
    )
    contacts = pd.read_csv('contact_tracing.csv').rename(
        columns={
            'user_id': 'source_device',
            'mac': 'target_device'
        }
    )
    mobility = pd.read_csv('mobility.csv')

    if RUNTIME_INGEST_CACHE:
        cache_df = pd.DataFrame(RUNTIME_INGEST_CACHE)
        vitals = pd.concat([vitals, cache_df], ignore_index=True)

    vitals['timestamp'] = pd.to_datetime(vitals['date'] + ' ' + vitals['time'])
    contacts['timestamp'] = pd.to_datetime(contacts['date'] + ' ' + contacts['time'])
    mobility['timestamp'] = pd.to_datetime(mobility['date'] + ' ' + mobility['time'])

    def assign_state(lat, lon):
        if   7.20 <= lat <= 7.65   and 4.70 <= lon <= 5.25:  return 'Ondo'
        elif 7.75 <= lat <= 7.95   and 9.75 <= lon <= 9.90:  return 'Benue'
        elif 4.90 <= lat <= 5.10   and 8.30 <= lon <= 8.42:  return 'Cross River'
        elif 12.70 <= lat <= 13.20 and 5.00 <= lon <= 5.30:  return 'Sokoto'
        elif 7.60 <= lat <= 7.72   and 6.38 <= lon <= 6.46:  return 'Kogi'
        elif 5.35 <= lat <= 5.42   and 6.95 <= lon <= 7.05:  return 'Imo'
        elif 7.20 <= lat <= 7.95   and 9.84 <= lon <= 11.00: return 'Taraba'
        else:                                                  return 'Unknown'

    vitals['state'] = vitals.apply(lambda r: assign_state(r['latitude'], r['longitude']), axis=1)
    mobility['state'] = mobility.apply(lambda r: assign_state(r['latitude'], r['longitude']), axis=1)
    contacts['state'] = contacts.apply(lambda r: assign_state(r['latitude'], r['longitude']), axis=1)

    features = ['temperature_c', 'heart_rate_bpm', 'movement_status']
    vitals_clean = vitals.dropna(subset=features).copy()
    
    if not vitals_clean.empty:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(vitals_clean[features])
        model = IsolationForest(contamination=0.04, random_state=42, n_jobs=-1)
        vitals_clean['anomaly_score'] = model.fit_predict(X_scaled)
        
        vitals = vitals.merge(
            vitals_clean[['device_id', 'date', 'time', 'anomaly_score']],
            on=['device_id', 'date', 'time'],
            how='left'
        ).fillna({'anomaly_score': 1})
    else:
        vitals['anomaly_score'] = 1

    vitals['Status'] = vitals['anomaly_score'].apply(
        lambda s: '🚨 Anomaly Flagged' if s == -1 else '✅ Normal Baseline'
    )

    mob_cols = ['user_id', 'date', 'time', 'latitude', 'longitude', 'state', 'exposure_score', 'has_contact']
    mob_map = mobility[mob_cols].copy().rename(columns={'user_id': 'device_id'})
    
    mob_map['Status'] = mob_map.apply(
        lambda r: '🚨 Anomaly Flagged' if (r['has_contact'] and r['exposure_score'] > 50) else '✅ Normal Baseline',
        axis=1
    )
    
    vitals_cols = ['device_id', 'date', 'time', 'latitude', 'longitude', 'state', 'temperature_c', 'heart_rate_bpm', 'anomaly_score', 'Status']
    vitals_map = vitals[vitals_cols].copy()
    
    unified = pd.concat([vitals_map, mob_map[mob_map['state'] != 'Ondo']], ignore_index=True)
    unified = unified[unified['state'] != 'Unknown']

    try:
        with open('infected_nodes.txt', 'r') as f:
            infected_devices = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        infected_devices = vitals[vitals['Status'] == '🚨 Anomaly Flagged']['device_id'].unique().tolist()

    high_exp = mob_map[mob_map['Status'] == '🚨 Anomaly Flagged']['device_id'].unique().tolist()
    infected_list = list(set(infected_devices + high_exp))

    return unified, contacts, vitals, infected_list

# =====================================================================
# PAIRED REQUEST VALIDATION PYDANTIC SCHEMAS
# =====================================================================
class ConfigKeys(BaseModel):
    provider: str
    groq_key: str
    claude_key: str
    termii_key: str

class SmsAlertRequest(BaseModel):
    zone: str
    phone: str
    device: str

class ChatMessage(BaseModel):
    role: str
    content: str

class UserRegister(BaseModel):
    username: str
    password: str
    name: str
    zone: str

class TelemetryIngest(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    temperature: float
    heartbeat: float
    movement: float
    date: str
    time: str

# =====================================================================
# API ROUTER ENDPOINTS
# =====================================================================
@api_router.get("/config")
def get_config():
    return load_env_keys()

@api_router.post("/config")
def update_config(cfg: ConfigKeys):
    save_env_keys(cfg.provider, cfg.groq_key, cfg.claude_key, cfg.termii_key)
    return {"status": "success", "message": "Settings saved to backend environment"}

@api_router.post("/auth/register")
def register(data: UserRegister):
    uname = data.username.strip().lower()
    if not uname or not data.password.strip():
        raise HTTPException(status_code=400, detail="Credentials cannot be empty spaces.")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if user already exists in the permanent database file
    cursor.execute("SELECT username FROM users WHERE username = ?", (uname,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Identity mapping already exists for this username.")
    
    # Save the new account permanently to the database file on disk
    cursor.execute(
        "INSERT INTO users (username, password, role, zone, name) VALUES (?, ?, ?, ?, ?)",
        (uname, data.password.strip(), "Health Worker", data.zone, data.name.strip())
    )
    conn.commit()
    conn.close()
    
    return {"success": True, "message": f"Account permanently saved for {data.name}"}

@api_router.get("/debug/users")
def get_all_registered_users():
    """Operational helper endpoint to inspect active database credentials."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password, role, zone, name FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [{"username": r[0], "password": r[1], "role": r[2], "zone": r[3], "name": r[4]} for r in rows]

@api_router.post("/telemetry/ingest")
def ingest_device_telemetry(payload: TelemetryIngest):
    if not (3.0 <= payload.latitude <= 15.0) or not (3.0 <= payload.longitude <= 15.0):
        raise HTTPException(status_code=422, detail="Geographic telemetry boundaries fall outside Nigeria.")
    
    data_dict = {
        "device_id": payload.device_id,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "temperature_c": payload.temperature,
        "heart_rate_bpm": payload.heartbeat,
        "movement_status": payload.movement,
        "date": payload.date,
        "time": payload.time
    }
    RUNTIME_INGEST_CACHE.append(data_dict)
    return {"status": "Injected Successfully", "node_registered": payload.device_id, "live_queue_depth": len(RUNTIME_INGEST_CACHE)}

@api_router.get("/reports/brief")
def generate_tactical_brief(zone: str = "All States"):
    unified, _, _, _ = process_healthtrace_layers()
    target_df = unified if zone == "All States" else unified[unified['state'] == zone]
    
    if target_df.empty:
        return {"compiled_report": "No active operational traces found for the specified zone."}
        
    total_nodes = int(target_df['device_id'].nunique())
    anomaly_nodes = int(target_df[target_df['Status'] == '🚨 Anomaly Flagged'].groupby('device_id').ngroups)
    anomaly_rate = round((anomaly_nodes / total_nodes) * 100, 2) if total_nodes > 0 else 0.0
    
    alert_level = "🟢 LEVEL 1 - MONITORING BASELINE"
    if anomaly_rate > 8.0:    alert_level = "🔴 LEVEL 3 - CRITICAL OUTBREAK ESCALATION"
    elif anomaly_rate > 4.0:  alert_level = "🟡 LEVEL 2 - HEIGHTENED COMMUNITY SPREAD"

    brief_markdown = f"""### 🏥 HEALTHTRACE EPIDEMIOLOGICAL SITUATION REPORT
**Operational Scope Focus:** {zone} Region System Snapshot  
**System Evaluation Alert Status:** {alert_level}

---
#### 📊 TELEMETRY COUNTS & DENSITY METRICS
- **Total Monitored Active Endpoints:** {total_nodes}
- **Confirmed Biometric Anomalies (Isolation Forest Engine):** {anomaly_nodes}
- **Current Cluster Proximity Spread Ratio:** {anomaly_rate}% 

#### 🛡️ CLINICAL PROTOCOL ACTION DIRECTIVES
"""
    if anomaly_rate > 4.0:
        brief_markdown += f"- 🚨 **Immediate Intervention Required:** Anomaly concentrations exceed safe limits ({anomaly_rate}% > 4.0%). Dispatch targeted mobile health squads.\n- Enforce NCDC Section-A ring containment fences immediately across zone {zone} coordinates."
    else:
        brief_markdown += "- Maintain normal diagnostic telemetry streaming channels.\n- Continue routine contact interaction pathway mapping logs."
        
    return {"zone": zone, "alert_status": alert_level, "compiled_report": brief_markdown}

@api_router.get("/telemetry")
def get_telemetry(zone: str = "All States"):
    unified, contacts, vitals, infected_list = process_healthtrace_layers()
    filtered_vitals = unified if zone == "All States" else unified[unified['state'] == zone]
    return {
        "metrics": {
            "total_vitals": len(unified),
            "total_contacts": len(contacts),
            "high_risk_count": len(infected_list)
        },
        "infected_list": infected_list,
        "records": filtered_vitals.replace({np.nan: None}).to_dict(orient="records")
    }

@api_router.get("/forecast")
def get_forecast(zone: str = "All States"):
    unified, contacts, vitals, infected_list = process_healthtrace_layers()
    
    if zone in ['Ondo', 'All States']:
        vitals['date_only'] = pd.to_datetime(vitals['date']).dt.date
        daily_counts = vitals[vitals['Status'] == '🚨 Anomaly Flagged'].groupby('date_only')['device_id'].nunique().reset_index()
    else:
        con_s = contacts[contacts['state'] == zone].copy()
        con_s['date_only'] = pd.to_datetime(con_s['date']).dt.date
        daily_counts = con_s[con_s['proximity'].isin(['close','very close'])].groupby('date_only')['source_device'].nunique().reset_index()

    daily_counts.columns = ['date', 'cases']
    daily_counts = daily_counts.sort_values('date').reset_index(drop=True)

    if len(daily_counts) < 3:
        return {"data_available": False}

    daily_counts['day_num'] = range(len(daily_counts))
    X_train = daily_counts[['day_num']].values
    y_train = daily_counts['cases'].values

    model = LinearRegression().fit(X_train, y_train)
    last_date = pd.to_datetime(daily_counts['date'].max())

    future_days = np.array([[daily_counts['day_num'].max() + i] for i in range(1, 8)])
    future_preds = model.predict(future_days).clip(min=0).round().astype(int).tolist()
    future_labels = [(last_date + pd.Timedelta(days=i)).strftime("%b %d") for i in range(1, 8)]

    return {
        "data_available": True,
        "historical_dates": [str(d) for d in daily_counts['date']],
        "historical_cases": daily_counts['cases'].tolist(),
        "future_dates": future_labels,
        "future_preds": future_preds,
        "slope": float(model.coef_[0])
    }

@api_router.get("/network")
def get_network_edges(device_id: str):
    if device_id == "None Specified":
        return []
    contacts = pd.read_csv('contact_tracing.csv').rename(columns={'user_id': 'source_device', 'mac': 'target_device'})
    net_slice = contacts[(contacts['source_device'] == device_id) | (contacts['target_device'] == device_id)].head(25)
    return net_slice[['source_device', 'target_device']].to_dict(orient="records")

@api_router.post("/alert")
def trigger_termii_alert(payload: SmsAlertRequest):
    keys = load_env_keys()
    if not keys["termii_key"]:
        raise HTTPException(status_code=400, detail="Backend Termii Gateway key unconfigured")
        
    unified, contacts, _, infected_list = process_healthtrace_layers()
    zone_vitals = unified if payload.zone == "All States" else unified[unified['state'] == payload.zone]
    
    anomalies = len(zone_vitals[zone_vitals['Status'] == '🚨 Anomaly Flagged']['device_id'].unique())
    exposed = len(contacts[contacts['source_device'].isin(infected_list) | contacts['target_device'].isin(infected_list)])

    sms_text = f"[HealthTrace Alert] OUTBREAK RISK\nZone: {payload.zone}\nFlagged: {payload.device}\nAnomalies: {anomalies}\nExposed Contacts: {exposed}\nAction: View healthtrace.ai"
    
    url = "https://api.ng.termii.com/api/sms/send"
    termii_payload = {"to": payload.phone, "from": "HlthTrace", "sms": sms_text, "type": "plain", "channel": "generic", "api_key": keys["termii_key"]}
    
    res = req.post(url, json=termii_payload, timeout=10).json()
    if res.get("code") == 404 or "SenderID" in str(res.get("message", "")):
        termii_payload["from"] = "Termii"
        res = req.post(url, json=termii_payload, timeout=10).json()
    return res

@api_router.post("/chat")
def handle_ai_assistant(history: list[ChatMessage] = Body(...), zone: str = "All States", device: str = "None Specified"):
    keys = load_env_keys()
    provider = keys["provider"]
    api_key = keys["groq_key"] if provider == "groq" else keys["claude_key"]
    
    if not api_key:
        return {"response": "⚙️ AI Gateway API keys are missing on the core microservice configuration environment."}

    user_query = history[-1].content if history else ""

    guardrail_prompt = f"Classify the user prompt into exactly 'Epidemiology' or 'Irrelevant'. Prompt: '{user_query}'. Respond with only the single word."
    try:
        if provider == "groq":
            guard_res = req.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json={
                "model": "llama3-8b-8192", "max_tokens": 5, "temperature": 0.0,
                "messages": [{"role": "user", "content": guardrail_prompt}]
            }, timeout=5).json()
            intent = guard_res["choices"][0]["message"]["content"].strip()
        else:
            intent = "Epidemiology"
    except Exception:
        intent = "Epidemiology"

    if "Irrelevant" in intent:
        return {"response": "🛑 Security Guardrail: This AI assistant gateway is locked down strictly to epidemiological parameters, dataset inquiries, and NCDC guidelines containment strategies."}

    ncdc_protocols = """
    NCDC PROTOCOL COMPLIANCE RULES:
    - Level-2 Quarantine Ring routing must be initiated whenever regional anomaly rates cross a 4% threshold.
    - Tracked vector nodes require close path analytics up to 2 degrees of separation matrices.
    """

    unified, _, _, infected_list = process_healthtrace_layers()
    state_breakdown = ""
    for st_name in STATES_GEO_MARKERS.keys():
        sl = unified[unified['state'] == st_name]
        sa = len(sl[sl['Status'] == '🚨 Anomaly Flagged']['device_id'].unique())
        state_breakdown += f"  - {st_name}: {sa} anomalies / {len(sl['device_id'].unique())} devices\n"

    sys_prompt = f"""You are HealthTrace AI, a specialized medical agent running within an optimized production telemetry microservice loop.

NCDC GUIDELINE PARAMETERS:
{ncdc_protocols}

REAL-TIME TRACKING CORE CONTEXT:
- Global Telemetry Records Evaluated: {len(unified):,}
- Unique High-Risk Carrier Footprints: {len(infected_list)}
- Field Operator Session Area: {zone}
- Targeted Analysis Anchor Node: {device}

CURRENT STATE OUTBREAK TRACKS:
{state_breakdown}

Ground all recommendations purely within the tracking context and NCDC rules above."""

    payload_messages = [{"role": m.role, "content": m.content} for m in history]

    if provider == "groq":
        try:
            res = req.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json={
                "model": "llama-3.3-70b-versatile", "max_tokens": 500, "temperature": 0.2,
                "messages": [{"role": "system", "content": sys_prompt}] + payload_messages
            }, timeout=20).json()
            return {"response": res["choices"][0]["message"]["content"]}
        except Exception as e:
            return {"response": f"❌ Groq API Gateway Error: {str(e)}"}
            
    elif provider == "claude":
        try:
            res = req.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json={
                "model": "claude-haiku-4-5-20251001", "max_tokens": 500, "system": sys_prompt, "messages": payload_messages
            }, timeout=20).json()
            return {"response": res["content"][0]["text"]}
        except Exception as e:
            return {"response": f"❌ Claude API Gateway Error: {str(e)}"}

# Mount the router cleanly back to the global application wrapper
app.include_router(api_router)