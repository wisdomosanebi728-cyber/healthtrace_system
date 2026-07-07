import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
import streamlit.components.v1 as components

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="HealthTrace AI Deployment", page_icon="🏥", layout="wide")

# =====================================================================
# IDENTITY MANAGEMENT ACCESS CONTROL PORTAL
# =====================================================================
if "logged_in" not in st.session_state:
    st.markdown("<h2 style='text-align:center;'>🏥 HealthTrace AI Operational Portal</h2>", unsafe_allow_html=True)
    auth_tab, reg_tab, guest_tab = st.tabs(["🔒 Secure Login", "📝 Register Worker Account", "👁️ Guest Access"])
    
    with auth_tab:
        username = st.text_input("Username", key="login_uname")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Access Command Centre", use_container_width=True, type="primary"):
            uname_low = username.strip().lower()
            # FIX: Query backend database directly for authentication
            try:
                users_resp = requests.get(f"{BACKEND_URL}/api/debug/users").json()
                user_match = next((u for u in users_resp if u["username"] == uname_low), None)
                
                # Check credentials against the database record
                if user_match and user_match.get("password") == password:
                    st.session_state.logged_in = True
                    st.session_state.user = user_match
                    st.rerun()
                else:
                    st.error("Authentication rejected. Check your credentials.")
            except Exception as e:
                st.error(f"Cannot connect to backend: {e}")
                
    with reg_tab:
        st.markdown("#### Create Field Health Worker Credentials")
        new_name = st.text_input("Full Name", key="reg_name")
        new_uname = st.text_input("Desired Username", key="reg_uname")
        new_pass = st.text_input("Secure Password", type="password", key="reg_pass")
        selected_zone = st.selectbox("Assigned Zone", options=['Ondo', 'Benue', 'Cross River', 'Sokoto', 'Kogi', 'Imo', 'Taraba'], key="reg_zone")
        
        # CORRECTED: Properly indented registration logic
        if st.button("Register System Account", use_container_width=True):
            if not new_name or not new_uname or not new_pass: 
                st.warning("All fields are required.")
            else:
                payload = {"username": new_uname.lower().strip(), "password": new_pass.strip(), "name": new_name, "zone": selected_zone}
                response = requests.post(f"{BACKEND_URL}/api/auth/register", json=payload)
                if response.status_code == 200:
                    st.success("✅ Account permanently saved to Database!")
                else:
                    st.error(f"❌ Registration failed: {response.json().get('detail', 'Unknown error')}")

    with guest_tab:
        if st.button("Explore as Guest", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.user = {"name": "Anonymous Observer", "role": "Guest", "zone": "All States"}
            st.rerun()
    st.stop()

# =====================================================================
# POST-LOGIN ACCESS & DASHBOARD
# =====================================================================
# Initialize session user securely
CURRENT_USER = st.session_state.get("user")
IS_ADMIN = CURRENT_USER["role"] == "Admin"
IS_GUEST = CURRENT_USER["role"] == "Guest"

# Logout
st.sidebar.markdown(f"User: **{CURRENT_USER['name']}**\n\nRole: **{CURRENT_USER['role']}**\n\nZone: **{CURRENT_USER['zone']}**")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# Fetch baseline telemetry
try:
    telemetry = requests.get(f"{BACKEND_URL}/api/telemetry", params={"zone": CURRENT_USER["zone"]}).json()
except Exception:
    st.error("⚠️ Backend server unreachable.")
    st.stop()

st.title("🏥 HealthTrace AI Outbreak Command Centre")

# =====================================================================
# MAIN RUNTIME NAVIGATION INTERFACE TABS
# =====================================================================
tab_dashboard, tab_spider, tab_recovery, tab_ai = st.tabs([
    "📊 Command Centre Dashboard", 
    "🕸️ Spider Contact Dynamic Tracing", 
    "🩹 Node Recovery Simulation", 
    "🧬 Context AI Core Assistant"
])

# ---------------------------------------------------------------------
# TAB 1: OPERATIONAL COMMAND CENTRE DASHBOARD
# ---------------------------------------------------------------------
with tab_dashboard:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ingested Records", f"{telemetry['metrics']['total_vitals']:,}")
    col2.metric("Total Contacts Tracked", f"{telemetry['metrics']['total_contacts']:,}")
    col3.metric("High Risk Carriers Flagged", f"{telemetry['metrics']['high_risk_count']}")
    st.markdown("---")
    
    # ---- UPGRADE 4: REAL-TIME TIERED DEVICE ANOMALY ALERTS ----
    st.subheader("🚨 Real-Time Tiered Device Anomaly Alerts")
    df = pd.DataFrame(telemetry["records"])
    
    if not df.empty:
        df['temp_val'] = pd.to_numeric(df['temperature_c'], errors='coerce').fillna(37.0)
        df['heart_val'] = pd.to_numeric(df['heart_rate_bpm'], errors='coerce').fillna(80)
        
        crit_ids = df[(df['Status'] == '🚨 Anomaly Flagged') & (df['temp_val'] >= 39.5)]['device_id'].unique()
        high_ids = df[(df['Status'] == '🚨 Anomaly Flagged') & (df['temp_val'] < 39.5)]['device_id'].unique()
        med_ids = df[(df['Status'] == '✅ Normal Baseline') & (df['heart_val'] > 100)]['device_id'].unique()
        low_ids = df[(df['Status'] == '✅ Normal Baseline') & (df['heart_val'] <= 100)]['device_id'].unique()
        
        c_critical, c_high, c_med, c_low = st.columns(4)
        with c_critical:
            st.error(f"🔴 Critical (Fever ≥ 39.5°C)\n\n**{len(crit_ids)} Devices**")
            if len(crit_ids) > 0: st.caption(f", ".join(list(crit_ids)[:8]))
        with c_high:
            st.warning(f"🟠 High Risk (Biometric Anomaly)\n\n**{len(high_ids)} Devices**")
            if len(high_ids) > 0: st.caption(f", ".join(list(high_ids)[:8]))
        with c_med:
            st.info(f"🟡 Medium Risk (Elevated Pulse)\n\n**{len(med_ids)} Devices**")
            if len(med_ids) > 0: st.caption(f", ".join(list(med_ids)[:8]))
        with c_low:
            st.success(f"🟢 Low Risk (Normal Baseline)\n\n**{len(low_ids)} Devices**")
            if len(low_ids) > 0: st.caption(f", ".join(list(low_ids)[:8]))
    st.markdown("---")

    if IS_ADMIN or IS_GUEST:
        selected_state = st.selectbox("📍 Select View Focus:", options=["All States"] + ['Ondo', 'Benue', 'Cross River', 'Sokoto', 'Kogi', 'Imo', 'Taraba'])
    else:
        selected_state = CURRENT_USER["zone"]
        st.info(f"📍 Viewing assigned zone: **{selected_state} State**")

    if selected_state != "All States":
        df = df[df['state'] == selected_state]

    left, right = st.columns(2)
    with left:
        st.markdown("#### 📍 Real-Time Location Mapping")
        if not df.empty:
            fig_map = px.scatter_mapbox(df.head(2000), lat="latitude", lon="longitude", color="Status",
                                        color_discrete_map={'✅ Normal Baseline': '#2ECC71', '🚨 Anomaly Flagged': '#E74C3C'},
                                        mapbox_style="open-street-map", zoom=5, height=350)
            st.plotly_chart(fig_map, use_container_width=True)

    with right:
        st.markdown("#### 📈 AI Predictive Modeling Forecast")
        fc_res = requests.get(f"{BACKEND_URL}/api/forecast", params={"zone": selected_state}).json()
        if fc_res.get("data_available"):
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=fc_res["historical_dates"], y=fc_res["historical_cases"], name="Actual"))
            fig_fc.add_trace(go.Scatter(x=fc_res["future_dates"], y=fc_res["future_preds"], name="AI Forecast", line=dict(dash='dash', color='red')))
            fig_fc.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fc, use_container_width=True)

    # --- INGESTION AND DISPATCH TOOLS ---
    st.markdown("---")
    report_col, ingest_col = st.columns(2)
    with report_col:
        st.markdown("#### 📄 Automated NCDC Tactical Briefing Compiler")
        if st.button("Generate Regional Sit-Rep Report", use_container_width=True):
            rep_res = requests.get(f"{BACKEND_URL}/api/reports/brief", params={"zone": selected_state}).json()
            if "compiled_report" in rep_res: st.markdown(rep_res["compiled_report"])
    
    with ingest_col:
        st.markdown("#### 📲 Live Field Device Telemetry Ingest")
        with st.form("ingest_sim_form"):
            sim_device = st.text_input("Hardware ID (MAC)", value="DEV-99X-8821")
            sim_temp = st.slider("Core Temperature (°C)", 35.0, 42.0, 38.9, step=0.1)
            sim_heart = st.slider("Heartbeat Speed (BPM)", 50, 160, 115)
            if st.form_submit_button("Transmit Live Telemetry Packet"):
                if IS_GUEST: st.error("Guest profile restricted.")
                else:
                    payload = {"device_id": sim_device, "latitude": 7.245, "longitude": 5.181, "temperature": sim_temp, "heartbeat": sim_heart, "movement": 1.0, "date": pd.Timestamp.now().strftime("%Y-%m-%d"), "time": pd.Timestamp.now().strftime("%H:%M:%S")}
                    requests.post(f"{BACKEND_URL}/api/telemetry/ingest", json=payload)
                    st.success("📡 Webhook Packet Routed!")

# ---------------------------------------------------------------------
# TAB 2: UPGRADE 1: DEDICATED SPIDER CONTACT TRACING & 3D SIMULATOR
# ---------------------------------------------------------------------
with tab_spider:
    st.subheader("🕸️ Vector Node Proximity Transmission Matrix")
    infected_options = telemetry["infected_list"] if telemetry["infected_list"] else ["None Specified"]
    selected_device = st.selectbox("Select high-risk anchor node matrix:", options=infected_options, key="spider_device")
    
    l_graph, r_sim = st.columns([1, 1])
    
    with l_graph:
        st.markdown("#### Topology Proximity Map")
        if selected_device != "None Specified":
            edges = requests.get(f"{BACKEND_URL}/api/network", params={"device_id": selected_device}).json()
            if edges:
                G = nx.Graph()
                for edge in edges: G.add_edge(edge['source_device'], edge['target_device'])
                pos = nx.spring_layout(G)
                edge_x, edge_y = [], []
                for e in G.edges():
                    edge_x.extend([pos[e[0]][0], pos[e[1]][0], None])
                    edge_y.extend([pos[e[0]][1], pos[e[1]][1], None])
                fig_net = go.Figure(data=[
                    go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#555', width=1)),
                    go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], mode='markers+text', text=list(G.nodes()), marker=dict(size=14, color='#E74C3C'))
                ], layout=go.Layout(height=400, showlegend=False, xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False)))
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("No network contact links extracted for this node.")

    with r_sim:
        st.markdown("#### 🏃 Live 3D Moving Transmission Particle Simulation")
        st.caption("Real-time HTML5 3D spatial projection rendering viral transmission waves extending outward from the infected node cluster.")
        
        # HTML5 Canvas rendering engine for moving 3D matrix particle lines
        three_d_canvas = """
        <div style="background-color:#111; padding:10px; border-radius:10px; text-align:center;">
            <canvas id="spreadCanvas" width="500" height="350" style="background:#0a0a0c; border:1px solid #333;"></canvas>
        </div>
        <script>
            const canvas = document.getElementById('spreadCanvas');
            const ctx = canvas.getContext('2d');
            let nodes = [];
            let angle = 0;
            
            // Central Red Infected Root Index Node
            nodes.push({x: 0, y: 0, z: 0, size: 9, color: '#E74C3C'});
            
            // Generate surrounding peripheral device points in 3D space
            for(let i=0; i<22; i++){
                let theta = Math.random() * Math.PI * 2;
                let phi = Math.acos((Math.random() * 2) - 1);
                let dist = 100 + Math.random() * 70;
                nodes.push({
                    x: dist * Math.sin(phi) * Math.cos(theta),
                    y: dist * Math.sin(phi) * Math.sin(theta),
                    z: dist * Math.cos(phi),
                    size: 4,
                    color: '#2ECC71'
                });
            }
            
            function draw() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                let cx = canvas.width / 2;
                let cy = canvas.height / 2;
                angle += 0.008; // Orbit rotational speed
                
                let cosA = Math.cos(angle);
                let sinA = Math.sin(angle);
                
                // Draw contact tracing lines and flying infection waves
                nodes.forEach((node, idx) => {
                    if(idx === 0) return;
                    let rotX = node.x * cosA - node.z * sinA;
                    let rotZ = node.x * sinA + node.z * cosA;
                    let scale = 250 / (250 + rotZ);
                    let x2d = cx + rotX * scale;
                    let y2d = cy + node.y * scale;
                    
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(x2d, y2d);
                    ctx.strokeStyle = 'rgba(231, 76, 60, 0.2)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    
                    // Moving transmission wave particle
                    let speedFactor = (Date.now() * 0.0015 + (idx * 0.1)) % 1;
                    let pX = cx + (x2d - cx) * speedFactor;
                    let pY = cy + (y2d - cy) * speedFactor;
                    ctx.beginPath();
                    ctx.arc(pX, pY, 3, 0, Math.PI*2);
                    ctx.fillStyle = '#F1C40F';
                    ctx.fill();
                });
                
                // Draw actual interactive structural elements
                nodes.forEach((node) => {
                    let rotX = node.x * cosA - node.z * sinA;
                    let rotZ = node.x * sinA + node.z * cosA;
                    let scale = 250 / (250 + rotZ);
                    let x2d = cx + rotX * scale;
                    let y2d = cy + node.y * scale;
                    
                    ctx.beginPath();
                    ctx.arc(x2d, y2d, node.size * scale * 1.5, 0, Math.PI*2);
                    ctx.fillStyle = node.color;
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = node.color;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                });
                requestAnimationFrame(draw);
            }
            draw();
        </script>
        """
        components.html(three_d_canvas, height=380)

# ---------------------------------------------------------------------
# TAB 3: UPGRADE 3: NODE RECOVERY SIMULATION INTERFACE
# ---------------------------------------------------------------------
with tab_recovery:
    st.subheader("🩹 Predictive Kinetic Node Recovery Simulator (SIR Framework)")
    st.caption("Forecast transmission clearing timelines and view when infected nodes return safely back to baseline health.")
    
    rs_col1, rs_col2 = st.columns([1, 2])
    with rs_col1:
        st.markdown("#### Model Configurations")
        param_beta = st.slider("Transmission Velocity Coefficient (β)", 0.1, 1.0, 0.45, step=0.05)
        param_gamma = st.slider("Recovery Rate Clearance (γ)", 0.05, 0.5, 0.12, step=0.01)
        horizon_days = st.slider("Mathematical Evaluation Window (Days)", 30, 120, 75)
        
        r_0 = round(param_beta / param_gamma, 2)
        st.metric("Calculated Outbreak Reproduction Number (R₀)", f"{r_0}")
        if r_0 > 1.0:
            st.error("🚨 Outbreak Warning: R₀ > 1. Pathogen elements will expand through the node cluster network.")
        else:
            st.success("🟢 Contained Cluster: Active anomaly carriers will clear without network safety degradation.")
            
    with rs_col2:
        # Run standard SIR calculus tracking loop matrices
        S, I, R = [0.99], [0.01], [0.0]
        dt = 0.1
        steps = int(horizon_days / dt)
        
        for _ in range(steps):
            dS = -param_beta * S[-1] * I[-1] * dt
            dI = (param_beta * S[-1] * I[-1] - param_gamma * I[-1]) * dt
            dR = (param_gamma * I[-1]) * dt
            S.append(S[-1] + dS)
            I.append(I[-1] + dI)
            R.append(R[-1] + dR)
            
        time_axis = np.linspace(0, horizon_days, len(S))
        fig_sir = go.Figure()
        fig_sir.add_trace(go.Scatter(x=time_axis, y=S, name="Susceptible", line=dict(color='#3498DB', width=2.5)))
        fig_sir.add_trace(go.Scatter(x=time_axis, y=I, name="Active Carrier Flagged", line=dict(color='#E74C3C', width=3)))
        fig_sir.add_trace(go.Scatter(x=time_axis, y=R, name="Cleared / Recovered", line=dict(color='#2ECC71', width=2.5)))
        fig_sir.update_layout(title="Outbreak Phase Space Metrics Over Evaluation Timeline", xaxis_title="Days Elapsed", yaxis_title="Ratio Percentage", plot_bgcolor='rgba(0,0,0,0)', height=380)
        st.plotly_chart(fig_sir, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: UPGRADE 2: SEPARATE DEDICATED AI ASSISTANT TAB
# ---------------------------------------------------------------------
with tab_ai:
    st.subheader("🧬 Fullscreen Contextual Epidemiological AI Core Assistant")
    st.caption("Query vector data layers or generate intervention checklists securely inside a dedicated space.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Microservice AI engine active. Ask clear epidemiological tracing vectors."}]

    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("Query real-time automated system vector reports..."):
        if IS_GUEST: st.error("Access restriction applied.")
        else:
            with chat_container:
                st.chat_message("user").write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            try:
                res = requests.post(f"{BACKEND_URL}/api/chat", json=st.session_state.chat_history, params={"zone": selected_state, "device": selected_device}).json()
                ai_reply = res.get("response", "No microservice payload returned.")
            except Exception as e:
                ai_reply = f"❌ Error executing backend connectivity chain: {str(e)}"
                
            with chat_container:
                st.chat_message("assistant").write(ai_reply)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})