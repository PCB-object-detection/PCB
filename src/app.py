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

# ✅ SAHI 라이브러리
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

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

# [UI 스타일링] - 기존 디자인 100% 유지
st.markdown("""
    <style>
        header { visibility: hidden; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        
        .fixed-img {
            height: 400px !important;
            object-fit: contain;
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f8f9fa;
        }
        
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

        .metric-container {
            background-color: white; padding: 5px; border-radius: 8px; border: 1px solid #e0e0e0;
            text-align: center; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            height: 85px; display: flex; flex-direction: column; justify-content: center;
        }
        .metric-value { font-size: 1.4rem; font-weight: 800; color: #333; margin: 0; }
        .metric-label { font-size: 0.8rem; color: #666; font-weight: 600; margin-top: 2px; }
        
        div.stDownloadButton > button {
            background-color: #FF4B4B !important; color: white !important; width: 100%;
            border: none; font-weight: bold; padding: 0.8rem; margin-top: 12px;
        }
        div.stDownloadButton > button:hover { background-color: #d93434 !important; }
        
        .history-card { border: 1px solid #eee; padding: 10px; border-radius: 8px; margin-bottom: 10px; background-color: #fff; }
        h3 { margin-bottom: 0.5rem; }
        h4, h5 { color: #444; margin-bottom: 0.5rem; margin-top: 0.5rem; }
        
        /* 스캐닝 애니메이션 */
        .scan-container { position: relative; width: 100%; height: 400px; background-color: #000; border-radius: 8px; overflow: hidden; border: 2px solid #00ff00; display: flex; justify-content: center; align-items: center; }
        .scan-img { height: 100%; object-fit: contain; opacity: 0.6; }
        .scan-line { position: absolute; top: 0; left: 0; width: 100%; height: 4px; background-color: #00ff00; box-shadow: 0 0 15px #00ff00; animation: scanning 1.5s linear infinite; }
        .scan-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #00ff00; font-family: monospace; font-size: 1.5rem; font-weight: bold; background: rgba(0,0,0,0.6); padding: 10px; border-radius: 5px; }
        @keyframes scanning { 0% { top: 0%; } 50% { top: 100%; } 100% { top: 0%; } }
    </style>
""", unsafe_allow_html=True)

# [상태 관리]
DEFECT_TYPES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
COLOR_MAP_HEX = {'missing_hole': '#FF0000', 'mouse_bite': '#FFA500', 'open_circuit': '#FFFF00', 'short': '#0000FF', 'spur': '#800080', 'spurious_copper': '#FF00FF'}
COLORS_BGR = {'missing_hole': (0, 0, 255), 'mouse_bite': (0, 165, 255), 'open_circuit': (0, 255, 255), 'short': (255, 0, 0), 'spur': (128, 0, 128), 'spurious_copper': (255, 0, 255)}

if 'total_inspections' not in st.session_state: st.session_state.total_inspections = 0
if 'ng_inspections' not in st.session_state: st.session_state.ng_inspections = 0
if 'class_counts' not in st.session_state: st.session_state.class_counts = {k: 0 for k in DEFECT_TYPES}
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

# ✂️ [배경 제거 함수]
def crop_pcb(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40]); upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 100 and h > 100: return image[y:y+h, x:x+w], True
    return image, False

# 🛠️ [PDF 관련 함수 - WinError 32 해결]
def save_temp_image(image):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    t.close(); cv2.imwrite(t.name, image); return t.name

def create_pdf_report():
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    pdf.set_font("Arial", 'B', 20); pdf.cell(0, 15, txt="PCB AI Inspection Report", ln=1, align='C'); pdf.ln(5)
    pdf.set_font("Arial", 'I', 10); pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='R'); pdf.line(10, 35, 200, 35); pdf.ln(10)
    
    total = st.session_state.total_inspections; ng = st.session_state.ng_inspections; ok = total - ng; ng_rate = (ng/total*100) if total > 0 else 0
    pdf.set_fill_color(240, 240, 240); pdf.rect(10, 40, 190, 35, 'F'); pdf.set_y(45)
    pdf.set_font("Arial", 'B', 12); pdf.cell(45, 10, "Total", align='C'); pdf.cell(45, 10, "OK", align='C'); pdf.cell(45, 10, "NG", align='C'); pdf.cell(45, 10, "Rate", align='C', ln=1)
    pdf.set_font("Arial", 'B', 14); pdf.cell(45, 10, str(total), align='C'); pdf.set_text_color(46, 204, 113); pdf.cell(45, 10, str(ok), align='C'); pdf.set_text_color(231, 76, 60); pdf.cell(45, 10, str(ng), align='C'); pdf.set_text_color(0,0,0); pdf.cell(45, 10, f"{ng_rate:.1f}%", align='C', ln=1); pdf.ln(15)

    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "1. Defect Analysis", ln=1)
    fig, ax = plt.subplots(figsize=(7, 3))
    types = DEFECT_TYPES; counts = [st.session_state.class_counts.get(t, 0) for t in types]; colors = [COLOR_MAP_HEX.get(t, '#3498db') for t in types]
    ax.bar(types, counts, color=colors); plt.xticks(rotation=45, ha='right', fontsize=8); plt.tight_layout()
    tmp_chart = tempfile.NamedTemporaryFile(delete=False, suffix=".png"); tmp_chart.close()
    plt.savefig(tmp_chart.name, dpi=100); plt.close(fig); pdf.image(tmp_chart.name, x=10, w=190); os.remove(tmp_chart.name); pdf.ln(5)

    if any(i['is_ng'] for i in st.session_state.inspection_history):
        pdf.add_page(); pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "2. NG Visual Evidence", ln=1)
        for item in [i for i in st.session_state.inspection_history if i['is_ng']][:3]:
            pdf.set_font("Arial", 'B', 10); pdf.cell(0, 8, f"File: {item['id']}", ln=1)
            p1 = save_temp_image(item['img_orig']); p2 = save_temp_image(item['img_res'])
            try: y = pdf.get_y(); pdf.image(p1, x=10, y=y, w=90); pdf.image(p2, x=105, y=y, w=90); pdf.ln(65)
            finally: 
                if os.path.exists(p1): os.remove(p1)
                if os.path.exists(p2): os.remove(p2)
            pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

@st.cache_resource
def load_base_model(path): 
    # ONNX 모델이면 task='detect' 명시
    if path.endswith(".onnx"):
        return YOLO(path, task='detect')
    return YOLO(path)

@st.cache_resource
def load_sahi_model(model_path, conf):
    return AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=model_path, confidence_threshold=conf, device="cpu")

def draw_bbox_common(image, boxes, labels):
    img_draw = image.copy()
    for box, label_info in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box); cls_name, conf = label_info; color = COLORS_BGR.get(cls_name, (0, 255, 0))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_draw, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_draw

def get_img_html(img_bgr, label="Image"):
    _, buf = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buf).decode()
    return f'<img src="data:image/png;base64,{b64_str}" class="fixed-img">'

# ==========================================
# [사이드바]
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    input_mode = st.radio("입력 모드", ["📷 이미지 업로드", "🎞️ 동영상/웹캠 (실시간)"])
    st.divider()
    
    uploaded_file = None
    use_sahi = False
    
    if input_mode == "📷 이미지 업로드":
        uploaded_file = st.file_uploader("이미지 파일", type=['jpg', 'png'], label_visibility="collapsed")
        auto_crop = st.checkbox("✂️ 배경 자동 제거", value=True)
        with st.expander("🛠️ 정밀 검사 옵션 (SAHI)", expanded=True):
            use_sahi = st.checkbox("🔍 정밀 검사 (SAHI)", value=False)
            if use_sahi: st.caption("ℹ️ 미세 결함을 찾기 위해 이미지를 조각내어 분석합니다.")
            
    elif input_mode == "🎞️ 동영상/웹캠 (실시간)":
        source_type = st.selectbox("소스 선택", ["동영상 파일 업로드", "웹캠 (Camera 0)"])
        if source_type == "동영상 파일 업로드":
            uploaded_file = st.file_uploader("동영상 (MP4)", type=['mp4', 'avi'], label_visibility="collapsed")
        
        # ✅ [신규] 스마트 트리거 설정
        st.markdown("---")
        st.markdown("**⚡ 스마트 트리거 (Smart Trigger)**")
        trigger_mode = st.checkbox("센서 모드 (중앙 감지 시 촬영)", value=True, help="PCB가 화면 중앙에 올 때만 검사합니다.")
        cooldown_time = st.slider("검사 간격 (초)", 1.0, 5.0, 2.0, 0.5)

    st.divider()
    
    # ✅ [신규] ONNX 모델 추가됨
    MODEL_FILES = {
        "YOLOv8n (Nano)": "best(yolov8n).pt",
        "Baseline (v11m)": "best(yolov11m_baseline).pt",
        "Recall+ (v11m)": "best(yolov11m_imgz640).pt",
        "High-Res (v11m)": "best(yolov11m+imgsz1280).pt",
        "ONNX (FP16)": "yolov11m_baseline_fp16.onnx" # 추가됨
    }
    model_key = st.selectbox("AI 모델", list(MODEL_FILES.keys()), index=1)
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_key])
    conf_thresh = st.slider("민감도", 0.0, 1.0, 0.25, 0.05)
    
    st.divider()
    st.button("🔄 시스템 초기화", on_click=reset_statistics, width="stretch")

# ==========================================
# [메인 화면]
# ==========================================
st.markdown("### 🏭 PCB Defect Inspection System")
tab_dashboard, tab_history = st.tabs(["🔍 대시보드 (Dashboard)", "🗂️ 검사 이력 (History)"])

# --- [탭 1] 대시보드 ---
with tab_dashboard:
    col_visual, col_monitor = st.columns([2.0, 1.4], gap="large")

    # [우측] 통계 패널 (먼저 렌더링해서 자리 잡음)
    with col_monitor:
        st.markdown("#### 📊 Monitoring Board")
        
        # Placeholder (나중에 실시간 업데이트용)
        ph_metrics = st.empty()
        ph_chart = st.empty()
        
        # 초기 렌더링 함수
        def render_dashboard():
            total = st.session_state.total_inspections
            ng = st.session_state.ng_inspections
            rate = (ng / total * 100) if total > 0 else 0.0
            
            with ph_metrics.container():
                r1_c1, r1_c2 = st.columns(2)
                with r1_c1: st.markdown(f'<div class="metric-container"><p class="metric-value">{total}</p><p class="metric-label">Total</p></div>', unsafe_allow_html=True)
                with r1_c2: st.markdown(f'<div class="metric-container"><p class="metric-value" style="color:#FF4B4B;">{ng}</p><p class="metric-label">NG</p></div>', unsafe_allow_html=True)
                r2_c1, r2_c2 = st.columns(2)
                with r2_c1: st.markdown(f'<div class="metric-container"><p class="metric-value">{rate:.1f}%</p><p class="metric-label">Rate</p></div>', unsafe_allow_html=True)
                with r2_c2: st.markdown(f'<div class="metric-container"><p class="metric-value" style="color:#2980b9;">{st.session_state.last_inference_time:.3f}s</p><p class="metric-label">Speed</p></div>', unsafe_allow_html=True)

            with ph_chart.container():
                st.markdown("##### 📉 Defect Types")
                df = pd.DataFrame([{'Type': k, 'Count': v} for k, v in st.session_state.class_counts.items()])
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Type', scale=alt.Scale(domain=DEFECT_TYPES), axis=alt.Axis(labelAngle=-45, title=None)), 
                    y=alt.Y('Count', axis=alt.Axis(tickMinStep=1, title=None)),
                    color=alt.Color('Type', scale=alt.Scale(domain=DEFECT_TYPES, range=[COLOR_MAP_HEX.get(t, '#333') for t in DEFECT_TYPES]), legend=None)
                ).properties(height=430).configure_axis(grid=False).configure_view(strokeWidth=0)
                st.altair_chart(chart, width="stretch")

        render_dashboard() # 최초 1회 실행

        if st.session_state.total_inspections > 0:
            st.download_button("📄 PDF 리포트 다운로드", create_pdf_report(), file_name=f"Report_{datetime.now().strftime('%H%M')}.pdf", mime="application/pdf", width="stretch")
        else: st.button("📄 PDF 리포트 (대기 중)", disabled=True, width="stretch")

    # [좌측] 영상/이미지 패널
    with col_visual:
        video_spot = st.empty() # 동영상용 자리
        
        # 🚀 [동영상/웹캠 + 트리거 로직]
        if input_mode == "🎞️ 동영상/웹캠 (실시간)":
            start_btn = st.checkbox("▶️ 시스템 가동 (Start)", value=False)
            
            if start_btn:
                cap = None
                if "동영상" in source_type and uploaded_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name)
                elif "Camera" in source_type: cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
                if cap and cap.isOpened():
                    cap.set(3, 1280); cap.set(4, 720)
                    model = load_base_model(model_path) # ONNX도 자동 로드됨
                    
                    last_trigger_time = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break

                        # ✅ 트리거 로직 (중앙 초록색 감지)
                        h, w, _ = frame.shape
                        center_crop = frame[h//2-50:h//2+50, w//2-50:w//2+50]
                        avg_color = np.mean(center_crop, axis=(0, 1))
                        # Green > Red+20 and Green > Blue+20
                        is_pcb = (avg_color[1] > avg_color[0] + 20) and (avg_color[1] > avg_color[2] + 20)
                        
                        current_time = time.time()
                        is_cooldown = (current_time - last_trigger_time) < cooldown_time
                        
                        # 검사 조건: (트리거 꺼짐 OR (트리거 켜짐 AND PCB있음 AND 쿨다운 끝남))
                        should_detect = (not trigger_mode) or (trigger_mode and is_pcb and not is_cooldown)
                        
                        img_display = frame.copy()
                        status_text = "WAITING..."; status_color = (0, 255, 255) # Yellow

                        if should_detect:
                            status_text = "DETECTING..."; status_color = (0, 0, 255) # Red
                            t0 = time.time()
                            res = model.predict(frame, imgsz=640, conf=conf_thresh, verbose=False)[0]
                            st.session_state.last_inference_time = time.time() - t0
                            
                            defect_cnt = len(res.boxes)
                            boxes = [box.xyxy[0].tolist() for box in res.boxes]
                            labels = [[res.names[int(box.cls)], float(box.conf)] for box in res.boxes]
                            
                            # 통계 갱신
                            st.session_state.total_inspections += 1
                            if defect_cnt > 0: st.session_state.ng_inspections += 1
                            
                            # 이력 저장
                            ts = datetime.now().strftime("%H:%M:%S")
                            img_res = draw_bbox_common(frame, boxes, labels)
                            st.session_state.inspection_history.insert(0, {'id': f"Stream_{ts}", 'time': ts, 'img_orig': frame, 'img_res': img_res, 'is_ng': defect_cnt > 0})
                            
                            last_trigger_time = time.time()
                            render_dashboard() # 🚀 대시보드 갱신
                            
                            # 결과 잠깐 보여주기 위해 img_display 교체
                            img_display = img_res
                            status_text = f"RESULT: {defect_cnt} EA"; status_color = (0, 255, 0)

                        elif is_cooldown:
                            status_text = f"COOLDOWN ({cooldown_time - (current_time - last_trigger_time):.1f}s)"
                            status_color = (128, 128, 128)

                        # UI 그리기
                        if trigger_mode: cv2.line(img_display, (w//2, 0), (w//2, h), (0, 255, 255), 2)
                        cv2.rectangle(img_display, (10, 10), (320, 60), (0,0,0), -1)
                        cv2.putText(img_display, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                        _, buf = cv2.imencode('.png', img_display)
                        b64_web = base64.b64encode(buf).decode()
                        video_spot.markdown(f'<img src="data:image/png;base64,{b64_web}" class="fixed-img" style="height:500px !important;">', unsafe_allow_html=True)

                    cap.release()
                    if "동영상" in source_type: os.remove(tfile.name)
            else:
                video_spot.info("좌측 패널에서 '시스템 가동'을 체크해주세요.")

        # [모드 2] 이미지 업로드 (기존 로직 유지 - 생략 없이 포함)
        elif input_mode == "📷 이미지 업로드":
            status_placeholder = st.empty()
            c1, c2 = st.columns(2)
            with c1: st.markdown("**📄 Original**"); orig_spot = st.empty(); orig_spot.markdown('<div class="fixed-img" style="display:flex;align-items:center;justify-content:center;color:#aaa;">Original Image</div>', unsafe_allow_html=True)
            with c2: st.markdown("**🔍 Result**"); res_spot = st.empty(); res_spot.markdown('<div class="fixed-img" style="display:flex;align-items:center;justify-content:center;color:#aaa;">Result Image</div>', unsafe_allow_html=True)

            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_raw = cv2.imdecode(file_bytes, 1)
                
                if auto_crop: img_bgr, cropped = crop_pcb(img_raw)
                else: img_bgr = img_raw

                _, buf = cv2.imencode('.png', img_bgr); b64_orig = base64.b64encode(buf).decode()
                orig_spot.markdown(f'<img src="data:image/png;base64,{b64_orig}" class="fixed-img">', unsafe_allow_html=True)
                
                if st.button("검사 실행", type="primary", width="stretch"):
                    if use_sahi:
                        res_spot.markdown(f"""<div class="scan-container"><img src="data:image/png;base64,{b64_orig}" class="scan-img"><div class="scan-line"></div><div class="scan-text">SCANNING...</div></div>""", unsafe_allow_html=True)
                        time.sleep(0.1)
                        sahi_model = load_sahi_model(model_path, conf_thresh)
                        t0 = time.time()
                        res = get_sliced_prediction(img_bgr, sahi_model, slice_height=1024, slice_width=1024, overlap_height_ratio=0.1, overlap_width_ratio=0.1)
                        st.session_state.last_inference_time = time.time() - t0
                        boxes = [[p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy] for p in res.object_prediction_list]
                        labels = [[p.category.name, p.score.value] for p in res.object_prediction_list]
                        defect_cnt = len(res.object_prediction_list)
                    else:
                        model = load_base_model(model_path)
                        t0 = time.time()
                        res = model.predict(img_bgr, imgsz=640, conf=conf_thresh, verbose=False)[0]
                        st.session_state.last_inference_time = time.time() - t0
                        boxes = [box.xyxy[0].tolist() for box in res.boxes]
                        labels = [[res.names[int(box.cls)], float(box.conf)] for box in res.boxes]
                        defect_cnt = len(res.boxes)
                    
                    st.session_state.total_inspections += 1
                    if defect_cnt > 0: st.session_state.ng_inspections += 1
                    
                    img_res = draw_bbox_common(img_bgr, boxes, labels)
                    _, buf = cv2.imencode('.png', img_res); b64_res = base64.b64encode(buf).decode()
                    res_spot.markdown(f'<img src="data:image/png;base64,{b64_res}" class="fixed-img">', unsafe_allow_html=True)
                    
                    st.session_state.inspection_history.insert(0, {'id': uploaded_file.name, 'time': datetime.now().strftime("%H:%M:%S"), 'img_orig': img_bgr, 'img_res': img_res, 'is_ng': defect_cnt > 0})
                    render_dashboard() # 갱신

        st.markdown("---"); st.markdown("**📜 Recent Logs**")
        log_html = "".join([f"<div style='margin-bottom:4px;'><span style='color:{'#e74c3c' if 'NG' in log else '#2ecc71'};'>●</span> {log}</div>" for log in st.session_state.detection_log])
        st.markdown(f'<div class="log-container">{log_html if log_html else "No logs yet."}</div>', unsafe_allow_html=True)

# --- [탭 2] 이력 ---
with tab_history:
    st.markdown("### 🗂️ Inspection History")
    if not st.session_state.inspection_history: st.info("아직 검사 기록이 없습니다.")
    else:
        with st.container(height=800):
            for item in st.session_state.inspection_history:
                status_color = "#FF4B4B" if item['is_ng'] else "#2ecc71"; status_text = "🔴 NG" if item['is_ng'] else "🟢 OK"
                st.markdown(f"""<div class="history-header" style="border-left: 5px solid {status_color};"><span style="font-size:1.1em; font-weight:bold;">📂 {item['id']}</span><span style="float:right; color:{status_color}; font-weight:bold;">{status_text} <span style="color:#888; font-size:0.8em; font-weight:normal;">({item['time']})</span></span></div>""", unsafe_allow_html=True)
                hc1, hc2 = st.columns(2)
                with hc1: st.markdown("**Original**"); st.markdown(get_img_html(item['img_orig']), unsafe_allow_html=True)
                with hc2: st.markdown("**Result**"); st.markdown(get_img_html(item['img_res']), unsafe_allow_html=True)
                st.divider()