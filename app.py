"""Streamlit entry point: Enterprise Face Recognition Attendance System.

Tabs:
1. 📷 Kiosk Scanner (Instant snapshot scanner & continuous live stream)
2. 🧑 Face Enrollment & Members Directory
3. 📊 Reports & Logs (Present vs Absent, Analytics, CSV Export)
4. ⚙️ Settings & Diagnostics
"""
import datetime
import os
from pathlib import Path

import altair as alt
import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

from database.db_setup import init_db
from services.attendance_service import AttendanceService
from services.enrollment_service import EnrollmentService
from services.face_service import FaceService, active_backend, is_fr_available

# -----------------------------------------------------------------------------
# Page Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database
init_db()


def load_custom_css():
    css_path = Path(config.ASSETS_DIR) / "custom.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css()


# -----------------------------------------------------------------------------
# Service Instantiation
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_services(_backend: str, _threshold: float, _model: str):
    fs = FaceService(model=_model, backend=_backend)
    es = EnrollmentService(face_service=fs, threshold=_threshold)
    ats = AttendanceService()
    return fs, es, ats


# Sidebar Settings & Configuration
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    current_backend = active_backend()
    backend_choice = st.selectbox(
        "Face Backend",
        ["auto", "face_recognition", "opencv"],
        index=["auto", "face_recognition", "opencv"].index(config.FACE_BACKEND),
        help="• face_recognition: dlib 128-d (High Accuracy)\n• opencv: Haar + 128-d multi-feature embedding (Universal Fallback)",
    )
    
    resolved_backend = active_backend(backend_choice)
    backend_color = "#10B981" if resolved_backend == "face_recognition" else "#3B82F6"
    st.markdown(
        f"<div style='font-size: 13px; color: #475569; margin-top:-6px; margin-bottom: 12px;'>"
        f"Active: <span style='font-weight:700; color:{backend_color};'>{resolved_backend.upper()}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "Matching Threshold",
        min_value=0.20,
        max_value=1.00,
        value=float(config.RECOGNITION_THRESHOLD),
        step=0.05,
        help="Max distance for a match. Lower = stricter verification.",
    )

    det_model = st.selectbox(
        "Detection Model",
        ["hog", "cnn"],
        index=0 if config.DETECTION_MODEL == "hog" else 1,
        help="HOG: CPU-fast; CNN: GPU-accurate.",
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Status")
    
    # Pre-instantiate services to get stats
    face_svc, enroll_svc, attend_svc = get_services(backend_choice, threshold, det_model)
    enrolled_count = enroll_svc.count()
    today_count = attend_svc.today_count()
    
    st.markdown(
        f"""
        <div style="background:#F1F5F9; border:1px solid #CBD5E1; border-radius:8px; padding:10px 12px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:12px; color:#475569; font-weight:600;">👥 Enrolled Total:</span>
                <span style="font-weight:800; color:#0F172A;">{enrolled_count}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:#475569; font-weight:600;">✅ Present Today:</span>
                <span style="font-weight:800; color:#10B981;">{today_count}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.caption("Face Attendance Portal v2.0")


# -----------------------------------------------------------------------------
# Compact Top Navigation Bar
# -----------------------------------------------------------------------------
today_str = datetime.date.today().strftime("%A, %B %d, %Y")
st.markdown(
    f"""
    <div class="top-navbar">
        <div style="display:flex; align-items:center;">
            <span style="font-size:20px; margin-right:8px;">🛡️</span>
            <span class="top-navbar-title">Face Attendance Portal</span>
            <span class="top-navbar-sub">• {today_str}</span>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
            <span class="badge badge-primary">{resolved_backend.upper()}</span>
            <span class="badge badge-success">● Online</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_live, tab_enroll, tab_report, tab_settings = st.tabs([
    "Kiosk Scanner",
    "Face Enrollment",
    "Reports & Logs",
    "Settings",
])



# =============================================================================
# TAB 1: LIVE ATTENDANCE KIOSK
# =============================================================================
with tab_live:
    kiosk_col, feed_col = st.columns([1.55, 1.0], gap="large")

    with kiosk_col:
        st.markdown("#### 📷 Biometric Face Scanner")
        
        scan_mode = st.radio(
            "Scanner Input Mode",
            ["Webcam Snapshot", "Upload Photo", "Live Camera Feed"],
            horizontal=True,
        )

        # ---------------- Mode 1: Webcam Snapshot ----------------
        if scan_mode == "Webcam Snapshot":
            shot = st.camera_input("Look at the camera to check in", key="kiosk_camera")
            if shot is not None:
                frame = cv2.imdecode(np.frombuffer(shot.read(), np.uint8), cv2.IMREAD_COLOR)
                with st.spinner("Analyzing facial encodings..."):
                    results = enroll_svc.recognize_frame(frame)

                marked_names = []
                already_names = []
                unknown_count = 0
                marked_status = {}

                for who, _box, dist in results:
                    if who != "Unknown":
                        is_new, msg = attend_svc.mark_attendance(who, confidence=dist)
                        if is_new:
                            marked_names.append(who)
                            marked_status[who] = True
                        else:
                            already_names.append((who, msg))
                            marked_status[who] = True
                    else:
                        unknown_count += 1

                # Visual overlay
                annotated = face_svc.draw_boxes(frame, results, marked_status=marked_status)
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Scan Result: {len(results)} face(s) found",
                    use_container_width=True,
                )

                # Feedback Alerts
                if marked_names:
                    st.success(f"🎉 **Marked Present**: {', '.join(marked_names)} ✅")
                if already_names:
                    for w, m in already_names:
                        st.info(f"ℹ️ **{w}**: {m}")
                if unknown_count > 0 and not marked_names and not already_names:
                    st.warning("⚠️ Unrecognized face(s). Please enroll in the **Face Enrollment** tab first.")
                elif not results:
                    st.warning("⚠️ No face detected in frame. Please face the camera with adequate lighting.")

        # ---------------- Mode 2: Upload Photo / Group Photo ----------------
        elif scan_mode == "Upload Photo":
            up_file = st.file_uploader(
                "Upload a single or group photo",
                type=["jpg", "jpeg", "png"],
                key="kiosk_upload",
            )
            if up_file:
                frame = cv2.imdecode(np.frombuffer(up_file.read(), np.uint8), cv2.IMREAD_COLOR)
                with st.spinner("Processing photo and matching faces..."):
                    results = enroll_svc.recognize_frame(frame)

                marked_names = []
                already_names = []
                marked_status = {}

                for who, _box, dist in results:
                    if who != "Unknown":
                        is_new, msg = attend_svc.mark_attendance(who, confidence=dist)
                        if is_new:
                            marked_names.append(who)
                            marked_status[who] = True
                        else:
                            already_names.append((who, msg))
                            marked_status[who] = True

                annotated = face_svc.draw_boxes(frame, results, marked_status=marked_status)
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Scan Result: {len(results)} face(s) found",
                    use_container_width=True,
                )

                if marked_names:
                    st.success(f"✅ Marked Present: **{', '.join(marked_names)}**")
                if already_names:
                    for w, m in already_names:
                        st.info(f"ℹ️ {w}: {m}")
                if results and all(n == "Unknown" for n, _b, _d in results):
                    st.warning("⚠️ None of the detected faces are enrolled in the system.")

        # ---------------- Mode 3: Continuous Live Stream Loop ----------------
        else:
            st.caption("Continuous video feed scanner. Toggle Activate to begin live recognition.")
            c_ctrl1, c_ctrl2 = st.columns([1, 1])
            cam_idx = c_ctrl1.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
            live_active = c_ctrl2.toggle("🔴 Activate Live Camera", value=False)

            frame_spot = st.empty()
            status_spot = st.empty()

            if live_active:
                cap = cv2.VideoCapture(int(cam_idx))
                if not cap.isOpened():
                    st.error(f"Unable to open Camera index {cam_idx}. Please check your webcam connection.")
                else:
                    frame_count = 0
                    current_results = []
                    marked_status = {}
                    
                    # Live loop
                    while live_active:
                        ret, frame = cap.read()
                        if not ret:
                            status_spot.error("Camera stream interrupted.")
                            break
                        
                        frame_count += 1
                        # Process every Nth frame for smooth performance
                        if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                            current_results = enroll_svc.recognize_frame(frame)
                            for who, _b, dist in current_results:
                                if who != "Unknown":
                                    is_new, _ = attend_svc.mark_attendance(who, confidence=dist)
                                    marked_status[who] = True
                        
                        annotated = face_svc.draw_boxes(frame, current_results, marked_status=marked_status)
                        frame_spot.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                    cap.release()

    # ---------------- Right Column: Today's Live Check-in Feed ----------------
    with feed_col:
        st.markdown("#### 📋 Today's Check-ins")
        
        today_report = attend_svc.get_report(date=datetime.date.today().isoformat())
        
        if not today_report.empty:
            st.markdown(f"<span class='badge badge-success'>{len(today_report)} Total Today</span>", unsafe_allow_html=True)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
            # Display recent 10 check-ins
            for _, row in today_report.head(10).iterrows():
                u_name = row["name"]
                u_time = row["time"]
                conf = row["confidence"]
                conf_text = f"• {int((1.0 - min(conf or 0.0, 1.0))*100)}% match" if conf is not None else ""
                initials = "".join([part[0].upper() for part in u_name.split()[:2]]) if u_name else "?"
                
                st.markdown(
                    f"""
                    <div class="feed-item">
                        <div class="feed-user-info">
                            <div class="user-initials-avatar">{initials}</div>
                            <div>
                                <div class="feed-user-name">{u_name}</div>
                                <div class="feed-user-time">{u_time} {conf_text}</div>
                            </div>
                        </div>
                        <span class="badge badge-success">Present</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div style="background:#FFFFFF; border:1px dashed #CBD5E1; border-radius:10px; padding:24px 16px; text-align:center; color:#64748B;">
                    <div style="font-size:28px; margin-bottom:6px;">⏳</div>
                    <div style="font-weight:600; color:#1E293B; font-size:13.5px;">No Check-ins Yet Today</div>
                    <div style="font-size:12px; margin-top:2px;">Faces recognized via scanner will instantly appear here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# TAB 2: FACE ENROLLMENT & ROSTER
# =============================================================================
with tab_enroll:
    col_en_form, col_en_list = st.columns([1.1, 1.3], gap="large")

    with col_en_form:
        st.markdown("#### 🧑 Enroll New Member")
        st.caption("Register a new person by capturing or uploading reference facial photos.")

        with st.container():
            en_name = st.text_input("Full Name", placeholder="e.g. Sarah Jenkins")
            en_src = st.radio(
                "Photo Source",
                ["Webcam Capture", "Upload Image File"],
                horizontal=True,
            )

            en_image = None
            if en_src == "Webcam Capture":
                cam_shot = st.camera_input("Capture Enrollment Photo", key="enroll_cam")
                if cam_shot is not None:
                    en_image = cv2.imdecode(np.frombuffer(cam_shot.read(), np.uint8), cv2.IMREAD_COLOR)
            else:
                up_shot = st.file_uploader(
                    "Select Photo",
                    type=["jpg", "jpeg", "png"],
                    key="enroll_upload",
                )
                if up_shot is not None:
                    en_image = cv2.imdecode(np.frombuffer(up_shot.read(), np.uint8), cv2.IMREAD_COLOR)

            # Live preview and face quality check
            detected_faces = []
            if en_image is not None:
                detected_faces = face_svc.detect_and_encode(en_image)
                if len(detected_faces) == 1:
                    st.success("✅ Exactly 1 face detected with clear visibility.")
                    crop = face_svc.crop_face(en_image, detected_faces[0][0])
                    if crop is not None:
                        st.image(
                            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                            caption="Face Reference Preview",
                            width=130,
                        )
                elif len(detected_faces) > 1:
                    st.warning(f"⚠️ Found {len(detected_faces)} faces in frame. Primary face will be used, but single-person photo is recommended.")
                else:
                    st.error("⚠️ No face detected in this photo yet. Please face the camera directly with good lighting.")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # Interactive Enroll Button with clear contextual feedback
            if st.button("✨ Complete Enrollment", type="primary", use_container_width=True):
                clean_name = en_name.strip()
                if not clean_name:
                    st.warning("⚠️ **Step 1 Incomplete**: Please enter a Full Name above.")
                elif en_image is None:
                    st.warning("⚠️ **Step 2 Incomplete**: Please take a webcam photo or upload an image above.")
                elif not detected_faces:
                    st.error("❌ **No face detected**: Please look directly at the camera with clear lighting and take a new photo.")
                else:
                    try:
                        with st.spinner("Analyzing biometric landmarks & storing encodings..."):
                            enroll_svc.enroll(clean_name, en_image, allow_multi_face=True)
                        st.success(f"🎉 **Successfully Enrolled {clean_name}** into the system!")
                        st.balloons()
                        st.rerun()
                    except ValueError as err:
                        st.error(f"Enrollment Error: {err}")


    # ---------------- Right Column: Enrolled Members Directory ----------------
    with col_en_list:
        st.markdown("#### 👥 Enrolled Members Directory")
        users_list = enroll_svc.list_users()

        if users_list:
            search_query = st.text_input("🔍 Search Members", placeholder="Search by name...")
            filtered_users = [
                u for u in users_list
                if not search_query or search_query.lower() in u["name"].lower()
            ]

            st.markdown(f"<span class='badge badge-primary'>{len(filtered_users)} Registered Member(s)</span>", unsafe_allow_html=True)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # Display table of users
            df_users = pd.DataFrame(filtered_users)
            display_cols = ["name", "photos", "enrolled_on"]
            df_display = df_users[[c for c in display_cols if c in df_users.columns]].copy()
            df_display.columns = ["Member Name", "Photos", "Enrolled Since"]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Management Expander
            with st.expander("🛠️ Add Additional Photos / Delete Member", expanded=False):
                sel_user = st.selectbox("Select Member", [u["name"] for u in users_list], key="manage_user_select")
                
                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    st.markdown("**Add Extra Photo**")
                    st.caption("Improves recognition across different lighting.")
                    add_up = st.file_uploader("Upload additional photo", type=["jpg", "jpeg", "png"], key="add_photo_up")
                    if st.button("➕ Add Photo", use_container_width=True):
                        if add_up:
                            arr = cv2.imdecode(np.frombuffer(add_up.read(), np.uint8), cv2.IMREAD_COLOR)
                            try:
                                enroll_svc.enroll(sel_user, arr)
                                st.success(f"Added photo for {sel_user}!")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))
                        else:
                            st.warning("Please choose a photo file first.")

                with act_col2:
                    st.markdown("**Delete Member**")
                    st.caption("Permanently removes data.")
                    st.markdown(f"Selected: **{sel_user}**")
                    if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                        enroll_svc.delete_user(sel_user)
                        st.warning(f"Deleted {sel_user}.")
                        st.rerun()
        else:
            st.info("No members enrolled yet. Use the form on the left to enroll your first person! 👈")


# =============================================================================
# TAB 3: ATTENDANCE REPORTS & INTELLIGENCE
# =============================================================================
with tab_report:
    st.markdown("#### 📊 Attendance Intelligence & Records")
    
    # Filter Controls
    all_users = [u["name"] for u in enroll_svc.list_users()]
    
    fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with fc1:
        preset = st.selectbox(
            "Time Preset",
            ["Today", "Yesterday", "Last 7 Days", "This Month", "All Time", "Custom Range"],
        )
    
    # Compute date ranges based on preset
    today_dt = datetime.date.today()
    if preset == "Today":
        start_d, end_d = today_dt, today_dt
    elif preset == "Yesterday":
        start_d, end_d = today_dt - datetime.timedelta(days=1), today_dt - datetime.timedelta(days=1)
    elif preset == "Last 7 Days":
        start_d, end_d = today_dt - datetime.timedelta(days=6), today_dt
    elif preset == "This Month":
        start_d, end_d = today_dt.replace(day=1), today_dt
    elif preset == "All Time":
        start_d, end_d = None, None
    else:
        start_d = fc2.date_input("From Date", value=today_dt - datetime.timedelta(days=7))
        end_d = fc3.date_input("To Date", value=today_dt)

    with fc4:
        user_filter = st.selectbox("Filter by Member", ["All Members"] + all_users)

    # Fetch Data
    start_str = start_d.isoformat() if start_d else None
    end_str = end_d.isoformat() if end_d else None
    selected_name = None if user_filter == "All Members" else user_filter

    report_df = attend_svc.get_report(
        start=start_str,
        end=end_str,
        name=selected_name,
    )
    
    # Calculate Roster for the target period
    target_roster_date = (start_d or today_dt).isoformat()
    roster_info = attend_svc.get_daily_roster(enrolled_names=all_users, date=target_roster_date)

    # Top KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Enrolled Total</div>
                <div class="metric-value">{len(all_users)}</div>
                <div class="metric-subtext">Registered identities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card success">
                <div class="metric-label">Present Count</div>
                <div class="metric-value">{roster_info['present_count']}</div>
                <div class="metric-subtext">On {target_roster_date}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card danger">
                <div class="metric-label">Absent Count</div>
                <div class="metric-value">{roster_info['absent_count']}</div>
                <div class="metric-subtext">Unrecorded on {target_roster_date}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card warning">
                <div class="metric-label">Attendance Rate</div>
                <div class="metric-value">{roster_info['attendance_rate']}%</div>
                <div class="metric-subtext">Daily participation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Breakdown Tabs: Present List, Absent List, Analytics, Manual Mark
    subtab_present, subtab_absent, subtab_analytics, subtab_override = st.tabs([
        "📋 Present Log",
        "❌ Absent Members",
        "📈 Charts",
        "✍️ Manual Entry",
    ])

    # 1. Present Table
    with subtab_present:
        if not report_df.empty:
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            csv_data = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Report (CSV)",
                data=csv_data,
                file_name=f"attendance_report_{datetime.date.today()}.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.info("No attendance records match the selected filter criteria.")

    # 2. Absent Members Roster
    with subtab_absent:
        if roster_info["absent_names"]:
            st.markdown(f"**Absent Members on {target_roster_date}:**")
            absent_df = pd.DataFrame([{"Absent Member": name, "Status": "Unrecorded"} for name in roster_info["absent_names"]])
            st.dataframe(absent_df, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 All enrolled members are marked present for this date!")

    # 3. Analytics & Charts
    with subtab_analytics:
        summary_data = attend_svc.summary(start=start_str, end=end_str)
        
        c_ch1, c_ch2 = st.columns(2, gap="large")
        with c_ch1:
            st.markdown("**📈 Daily Attendance Volume**")
            if summary_data["per_day"]:
                trend_df = pd.DataFrame([
                    {"Date": str(d), "Check-ins": int(c)}
                    for d, c in summary_data["per_day"].items()
                ])
                chart_trend = (
                    alt.Chart(trend_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=32, color="#4F46E5")
                    .encode(
                        x=alt.X("Date:N", axis=alt.Axis(title="Date", labelAngle=0, labelColor="#64748B", titleColor="#475569")),
                        y=alt.Y("Check-ins:Q", axis=alt.Axis(title="Present Count", tickMinStep=1, format="d", labelColor="#64748B", titleColor="#475569")),
                        tooltip=["Date", "Check-ins"]
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_trend, use_container_width=True)
            else:
                st.caption("No trend data available for this range.")

        with c_ch2:
            st.markdown("**👥 Check-in Frequency by Member**")
            if summary_data["per_person"]:
                freq_df = pd.DataFrame([
                    {"Member": str(m), "Days Present": int(c)}
                    for m, c in summary_data["per_person"].items()
                ]).sort_values(by="Days Present", ascending=True)
                
                chart_freq = (
                    alt.Chart(freq_df)
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=24, color="#10B981")
                    .encode(
                        y=alt.Y("Member:N", sort="-x", axis=alt.Axis(title=None, labelColor="#1E293B", labelFontWeight="bold")),
                        x=alt.X("Days Present:Q", axis=alt.Axis(title="Days Present", tickMinStep=1, format="d", labelColor="#64748B", titleColor="#475569")),
                        tooltip=["Member", "Days Present"]
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_freq, use_container_width=True)
            else:
                st.caption("No frequency data available.")


    # 4. Manual Override
    with subtab_override:
        st.markdown("**Manual Attendance Entry (Admin Override)**")
        st.caption("Manually mark an attendance record if camera lighting was unavailable.")
        
        mo_c1, mo_c2, mo_c3 = st.columns(3)
        with mo_c1:
            m_who = st.selectbox("Select Member", all_users, key="manual_who")
        with mo_c2:
            m_date = st.date_input("Date", value=today_dt, key="manual_date")
        with mo_c3:
            m_time = st.time_input("Check-in Time", value=datetime.datetime.now().time(), key="manual_time")

        if st.button("Mark Attendance Manually", type="primary"):
            if m_who:
                ok, msg = attend_svc.manual_mark(
                    m_who,
                    date=m_date.isoformat(),
                    time=m_time.strftime("%H:%M:%S"),
                )
                if ok:
                    st.success(f"✅ Manually recorded attendance for {m_who} on {m_date}!")
                    st.rerun()
                else:
                    st.warning(f"Notice: {msg}")


# =============================================================================
# TAB 4: SYSTEM DIAGNOSTICS & SETTINGS
# =============================================================================
with tab_settings:
    st.markdown("#### ⚙️ System Diagnostics & Logs")
    
    subtab_storage, subtab_logs = st.tabs([
        "💾 Database & Storage",
        "📜 Application Logs",
    ])

    with subtab_storage:
        st.markdown("**Database & Storage Architecture**")
        db_exists = os.path.exists(config.DB_PATH)
        db_size_kb = round(os.path.getsize(config.DB_PATH) / 1024, 2) if db_exists else 0
        total_photos = sum(len(files) for _, _, files in os.walk(config.KNOWN_FACES_IMAGE_DIR))

        st.markdown(
            f"""
            <div class="clean-card">
                <table style="width:100%; font-size:13.5px; line-height:2.2;">
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color:#64748B; width:260px;">SQLite Database Location:</td>
                        <td style="font-family:monospace; color:#0F172A;">{config.DB_PATH}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color:#64748B;">Database Size on Disk:</td>
                        <td><b>{db_size_kb} KB</b></td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color:#64748B;">Face Image Store:</td>
                        <td style="font-family:monospace; color:#0F172A;">{config.KNOWN_FACES_IMAGE_DIR}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color:#64748B;">Saved Reference Photos:</td>
                        <td><b>{total_photos} photos stored</b></td>
                    </tr>
                    <tr>
                        <td style="color:#64748B;">face_recognition (dlib) Status:</td>
                        <td><b>{'Installed & Active ✅' if is_fr_available() else 'OpenCV Fallback Active (Standard) ℹ️'}</b></td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with subtab_logs:
        st.markdown("**Application Audit Logs (`logs/app.log`)**")
        
        log_ctrl1, log_ctrl2, log_ctrl3 = st.columns([2, 1, 1])
        with log_ctrl1:
            log_search = st.text_input("🔍 Filter logs by keyword", placeholder="Search in logs (e.g. INFO, enroll, Alice)...")
        with log_ctrl2:
            log_level = st.selectbox("Log Level", ["All Levels", "INFO", "WARNING", "ERROR"])
        with log_ctrl3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            refresh_clicked = st.button("🔄 Refresh Logs", use_container_width=True)

        if os.path.exists(config.LOG_FILE):
            with open(config.LOG_FILE, encoding="utf-8") as f:
                raw_lines = f.readlines()
            
            # Apply filters
            filtered_lines = []
            for line in raw_lines:
                if log_level != "All Levels" and log_level not in line:
                    continue
                if log_search and log_search.lower() not in line.lower():
                    continue
                filtered_lines.append(line)
            
            log_display_text = "".join(filtered_lines[-50:]) if filtered_lines else "No matching log entries found."
            
            st.code(log_display_text, language="log")
            
            # Download full logs button
            with open(config.LOG_FILE, encoding="utf-8") as f:
                full_log_data = f.read()
            st.download_button(
                "⬇️ Download Full Log File",
                data=full_log_data,
                file_name=f"app_log_{datetime.date.today()}.log",
                mime="text/plain",
            )
        else:
            st.info("Log file will be created automatically on first action.")

