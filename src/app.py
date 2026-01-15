import streamlit as st
import cv2
import time
import os
import pandas as pd
import altair as alt
import numpy as np
import base64
import matplotlib.pyplot as plt
import tempfile
from ultralytics import YOLO
from datetime import datetime
from fpdf import FPDF

# ==========================================
# [설정] 페이지 기본 설정
# ==========================================
BASE_DIR = r"C:\GitHub_Project\AICV_03\PCB_object_detection\src"
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="PCB AI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [UI 스타일링] 
st.markdown("""
    <style>
        /* 헤더 숨김 & 여백 최적화 */
        header { visibility: hidden; }
        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 2rem; 
        }
        
        /* 1. 이미지 컨테이너 (높이 400px 고정) */
        .fixed-img {
            height: 400px !important;
            object-fit: contain;
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f8f9fa;
        }
        
        /* 2. 로그 컨테이너 (높이 240px 고정) -> 좌측 총 높이 확보용 */
        .log-container {
            height: 240px; 
            overflow-y: auto;
            border: 1px solid #eee;
            background-color: #fafafa;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.85rem;
        }

        /* 3. 통계 카드 (높이 고정) */
        .metric-container {
            background-color: white;
            padding: 5px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            text-align: center;
            margin-bottom: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            height: 85px; /* 높이 고정 */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-value { font-size: 1.4rem; font-weight: 800; color: #333; margin: 0; }
        .metric-label { font-size: 0.8rem; color: #666; font-weight: 600; margin-top: 2px; }
        
        /* 4. PDF 버튼 (빨간색, 높이 및 마진 조정) */
        div.stDownloadButton > button {
            background-color: #FF4B4B !important;
            color: white !important;
            width: 100%;
            border: none;
            font-weight: bold;
            padding: 0.8rem;
            margin-top: 12px; /* 그래프와의 간격 */
        }
        div.stDownloadButton > button:hover {
            background-color: #d93434 !important;
        }
        
        /* 5. 이력 카드 스타일 */
        .history-card {
            border: 1px solid #eee;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            background-color: #fff;
        }
        
        h3 { margin-bottom: 0.5rem; }
        h4, h5 { color: #444; margin-bottom: 0.5rem; margin-top: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [상태 관리]
# ==========================================
DEFECT_TYPES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# 색상 매핑 
COLOR_MAP_HEX = {
    'missing_hole': '#FF0000',    # Red
    'mouse_bite': '#FFA500',      # Orange
    'open_circuit': '#FFFF00',    # Yellow
    'short': '#0000FF',           # Blue
    'spur': '#800080',            # Purple
    'spurious_copper': '#FF00FF'  # Magenta
}

if 'total_inspections' not in st.session_state: st.session_state.total_inspections = 0
if 'ng_inspections' not in st.session_state: st.session_state.ng_inspections = 0
if 'class_counts' not in st.session_state: 
    st.session_state.class_counts = {k: 0 for k in DEFECT_TYPES}
if 'detection_log' not in st.session_state: st.session_state.detection_log = []
if 'last_inference_time' not in st.session_state: st.session_state.last_inference_time = 0.0
if 'inspection_history' not in st.session_state: st.session_state.inspection_history = []

def reset_statistics():
    st.session_state.total_inspections = 0
    st.session_state.ng_inspections = 0
    st.session_state.class_counts = {k: 0 for k in DEFECT_TYPES}
    st.session_state.detection_log = []
    st.session_state.last_inference_time = 0.0
    st.session_state.inspection_history = []
    st.toast("시스템 초기화 완료", icon="🔄")

# [PDF 리포트 생성]
def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="PCB Inspection Report", ln=1, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='R')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)

    # Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="1. Summary", ln=1)
    
    total = st.session_state.total_inspections
    ng = st.session_state.ng_inspections
    ok = total - ng
    ng_rate = (ng / total * 100) if total > 0 else 0
    
    # Pie Chart
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.pie([ok, ng], labels=['OK', 'NG'], colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
        pie_path = tmp.name
    plt.close(fig1)

    pdf.set_font("Arial", size=11)
    pdf.cell(100, 8, txt=f"Total: {total}", ln=1)
    pdf.cell(100, 8, txt=f"NG Count: {ng}", ln=1)
    pdf.cell(100, 8, txt=f"NG Rate: {ng_rate:.1f}%", ln=1)
    pdf.image(pie_path, x=120, y=40, w=60)
    os.remove(pie_path)
    pdf.ln(40)

    # Defect Breakdown
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="2. Defect Analysis", ln=1)
    
    # Bar Chart (색상 반영)
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    types = DEFECT_TYPES
    counts = [st.session_state.class_counts.get(t, 0) for t in types]
    colors = [COLOR_MAP_HEX.get(t, '#3498db') for t in types]
    
    ax2.bar(types, counts, color=colors)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.savefig(tmp.name, dpi=100)
        bar_path = tmp.name
    plt.close(fig2)
    
    pdf.image(bar_path, x=10, w=190)
    os.remove(bar_path)
    pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin-1')

@st.cache_resource
def load_model(path):
    return YOLO(path)

def draw_bbox(image, results):
    img_draw = image.copy()
    COLORS_BGR = {
        'missing_hole': (0, 0, 255),    # Red
        'mouse_bite': (0, 165, 255),    # Orange
        'open_circuit': (0, 255, 255),  # Yellow
        'short': (255, 0, 0),           # Blue
        'spur': (128, 0, 128),          # Purple
        'spurious_copper': (255, 0, 255)# Magenta
    }
    DEFAULT_COLOR = (0, 255, 0)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls_id = int(box.cls)
        cls_name = results.names[cls_id]
        label = f"{cls_name} {conf:.2f}"
        color = COLORS_BGR.get(cls_name, DEFAULT_COLOR)

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        font_scale, thickness = 0.6, 1
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y_label = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.rectangle(img_draw, (x1, y_label - h - 5), (x1 + w, y_label + 5), color, -1)
        cv2.putText(img_draw, label, (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness+1)
    return img_draw

# ==========================================
# [사이드바]
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    input_mode = st.radio("입력 모드", ["📷 이미지 업로드", "🎥 웹캠 연결"])
    st.divider()
    
    uploaded_file = None
    if input_mode == "📷 이미지 업로드":
        uploaded_file = st.file_uploader("이미지 파일", type=['jpg', 'png'], label_visibility="collapsed")
    
    st.divider()
    model_key = st.selectbox("AI 모델", ["Baseline (기본)", "Recall+ (미검출 방지)", "High-Res (1280px)"], index=2)
    model_path = os.path.join(MODEL_DIR, {
        "Baseline (기본)": "best(yolov11m_baseline).pt",
        "Recall+ (미검출 방지)": "best(yolov11m_imgz640).pt",
        "High-Res (1280px)": "best(yolov11m+imgsz1280).pt"
    }[model_key])
    conf_thresh = st.slider("민감도", 0.0, 1.0, 0.25, 0.05)
    
    st.divider()
    st.button("🔄 시스템 초기화", on_click=reset_statistics, use_container_width=True)

# ==========================================
# [메인 화면]
# ==========================================
st.markdown("### 🏭 PCB Defect Inspection System")

tab_dashboard, tab_history = st.tabs(["🔍 대시보드 (Dashboard)", "🗂️ 검사 이력 (History)"])

# --- [탭 1] 대시보드 ---
with tab_dashboard:
    col_visual, col_monitor = st.columns([2.0, 1.4], gap="large")

    # --- 좌측: 이미지 + 로그 ---
    with col_visual:
        status_placeholder = st.empty()
        img_html_orig = '<div class="fixed-img" style="display:flex;align-items:center;justify-content:center;color:#aaa;">Original Image</div>'
        img_html_res = '<div class="fixed-img" style="display:flex;align-items:center;justify-content:center;color:#aaa;">Result Image</div>'

        # 로직 실행
        if input_mode == "📷 이미지 업로드" and uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            model = load_model(model_path)
            imgsz = 1280 if "High-Res" in model_key else 640
            
            t0 = time.time()
            results = model.predict(img_bgr, imgsz=imgsz, conf=conf_thresh, verbose=False)[0]
            st.session_state.last_inference_time = time.time() - t0
            
            defect_count = len(results.boxes)
            
            if defect_count > 0:
                status_placeholder.error(f"🚨 **불량 판정 (NG)** - {defect_count}개 결함 발견", icon="⚠️")
            else:
                status_placeholder.success("✅ **정상 판정 (OK)**", icon="🛡️")
            
            # 통계 업데이트
            file_id = uploaded_file.name
            if 'last_processed_file' not in st.session_state or st.session_state.last_processed_file != file_id:
                st.session_state.total_inspections += 1
                if defect_count > 0:
                    st.session_state.ng_inspections += 1
                
                ts = datetime.now().strftime("%H:%M:%S")
                if defect_count > 0:
                    for box in results.boxes:
                        cls = results.names[int(box.cls)]
                        st.session_state.class_counts[cls] += 1
                        st.session_state.detection_log.insert(0, f"[{ts}] NG: {cls}")
                else:
                    st.session_state.detection_log.insert(0, f"[{ts}] OK: Passed")
                
                img_res_save = draw_bbox(img_bgr, results)
                st.session_state.inspection_history.insert(0, {
                    'id': file_id, 'time': ts, 'img_orig': img_bgr, 'img_res': img_res_save,
                    'is_ng': defect_count > 0
                })
                
                st.session_state.last_processed_file = file_id
                st.rerun()

            img_res = draw_bbox(img_bgr, results)
            _, buf = cv2.imencode('.png', img_bgr)
            b64_orig = base64.b64encode(buf).decode()
            img_html_orig = f'<img src="data:image/png;base64,{b64_orig}" class="fixed-img">'
            _, buf_res = cv2.imencode('.png', img_res)
            b64_res = base64.b64encode(buf_res).decode()
            img_html_res = f'<img src="data:image/png;base64,{b64_res}" class="fixed-img">'

        elif input_mode == "🎥 웹캠 연결":
            run = st.checkbox("🚀 웹캠 검사 시작", value=False)
            if run:
                cam_id = st.number_input("ID", 0, label_visibility="collapsed")
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                cap.set(3, 1280); cap.set(4, 720)
                webcam_spot = st.empty()
                model = load_model(model_path)
                
                while run:
                    ret, frame = cap.read()
                    if not ret: break
                    t0 = time.time()
                    results = model.predict(frame, imgsz=640, conf=conf_thresh, verbose=False)[0]
                    st.session_state.last_inference_time = time.time() - t0
                    
                    img_res = draw_bbox(frame, results)
                    _, buf = cv2.imencode('.png', img_res)
                    b64_web = base64.b64encode(buf).decode()
                    webcam_spot.markdown(f'<img src="data:image/png;base64,{b64_web}" class="fixed-img">', unsafe_allow_html=True)
                    
                    if len(results.boxes) > 0: status_placeholder.error("🚨 결함 탐지중")
                    else: status_placeholder.success("✅ 정상")
                    time.sleep(0.01)
                cap.release()

        # [이미지]
        if input_mode == "📷 이미지 업로드":
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"**📄 Original**\n{img_html_orig}", unsafe_allow_html=True)
            with c2: st.markdown(f"**🔍 Result**\n{img_html_res}", unsafe_allow_html=True)

        # [로그]
        st.markdown("---")
        st.markdown("**📜 Recent Logs**")
        log_html = ""
        for log in st.session_state.detection_log:
            color = "#e74c3c" if "NG" in log else "#2ecc71"
            log_html += f"<div style='margin-bottom:4px;'><span style='color:{color};'>●</span> {log}</div>"
        
        st.markdown(f'<div class="log-container">{log_html if log_html else "No logs yet."}</div>', unsafe_allow_html=True)

    # --- 우측: 통계 + 그래프 ---
    with col_monitor:
        st.markdown("#### 📊 Monitoring Board")
        
        # 1. 지표
        ng_rate = 0.0
        if st.session_state.total_inspections > 0:
            ng_rate = (st.session_state.ng_inspections / st.session_state.total_inspections) * 100
            
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1: st.markdown(f'<div class="metric-container"><p class="metric-value">{st.session_state.total_inspections}</p><p class="metric-label">Total</p></div>', unsafe_allow_html=True)
        with r1_c2: st.markdown(f'<div class="metric-container"><p class="metric-value" style="color:#FF4B4B;">{st.session_state.ng_inspections}</p><p class="metric-label">NG</p></div>', unsafe_allow_html=True)
        
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1: st.markdown(f'<div class="metric-container"><p class="metric-value">{ng_rate:.1f}%</p><p class="metric-label">Rate</p></div>', unsafe_allow_html=True)
        with r2_c2: st.markdown(f'<div class="metric-container"><p class="metric-value" style="color:#2980b9;">{st.session_state.last_inference_time:.3f}s</p><p class="metric-label">Speed</p></div>', unsafe_allow_html=True)

        # 2. 그래프 (높이 430px로 확대하여 하단 라인 맞춤)
        st.markdown("##### 📉 Defect Types")
        
        chart_data = [{'Type': dt, 'Count': st.session_state.class_counts.get(dt, 0)} for dt in DEFECT_TYPES]
        df = pd.DataFrame(chart_data)
        
        # 색상 적용
        domain = DEFECT_TYPES
        range_ = [COLOR_MAP_HEX.get(t, '#333') for t in DEFECT_TYPES]

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Type', 
                    scale=alt.Scale(domain=DEFECT_TYPES), # X축 도메인 고정
                    axis=alt.Axis(labelAngle=-45, title=None)), 
            y=alt.Y('Count', axis=alt.Axis(tickMinStep=1, title=None)),
            color=alt.Color('Type', scale=alt.Scale(domain=domain, range=range_), legend=None),
            tooltip=['Type', 'Count']
        ).properties(
            height=430 # ✅ 중요: 그래프 높이를 430px로 늘려 PDF 버튼을 아래로 밀어냄 (좌측 로그 하단과 정렬)
        ).configure_axis(grid=False).configure_view(strokeWidth=0)
        
        st.altair_chart(chart, use_container_width=True)

        # 3. PDF 버튼
        if st.session_state.total_inspections > 0:
            report_data = create_pdf_report()
            st.download_button(
                label="📄 PDF 리포트 다운로드",
                data=report_data,
                file_name=f"PCB_Report_{datetime.now().strftime('%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.button("📄 PDF 리포트 (대기 중)", disabled=True, use_container_width=True)

# --- [탭 2] 이력 ---
with tab_history:
    st.markdown("### 🗂️ Inspection History")
    if not st.session_state.inspection_history:
        st.info("아직 검사 기록이 없습니다.")
    else:
        for item in st.session_state.inspection_history:
            with st.container():
                status_icon = "🔴 NG" if item['is_ng'] else "🟢 OK"
                st.markdown(f"""<div class="history-card"><h4>📂 {item['id']} <span style="font-size:0.8em; color:#888;">({item['time']})</span> - {status_icon}</h4></div>""", unsafe_allow_html=True)
                hc1, hc2 = st.columns(2)
                with hc1: st.image(cv2.cvtColor(item['img_orig'], cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                with hc2: st.image(cv2.cvtColor(item['img_res'], cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
                st.divider()