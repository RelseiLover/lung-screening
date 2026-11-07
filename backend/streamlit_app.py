from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from medical223_langgraph import run_medical223

st.set_page_config(
    page_title="肺部异常辅助筛查系统",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def build_structured_symptom(
    symptom: str,
    duration_value: str,
    fever_value: str,
    sputum_value: str,
    sputum_color: str,
    chest_value: str,
    tb_value: str,
) -> str:
    details = [symptom.strip()]
    if duration_value and duration_value != "未说明":
        details.append(f"病程约{duration_value}")
    if fever_value == "有":
        details.append("伴发热")
    elif fever_value == "无":
        details.append("无发热")
    if sputum_value == "有":
        details.append("有咳痰")
        if sputum_color and sputum_color != "未说明":
            details.append(sputum_color)
    elif sputum_value == "无":
        details.append("无咳痰")
    if chest_value == "有":
        details.append("伴胸痛或气促")
    elif chest_value == "无":
        details.append("无明显胸痛或气促")
    if tb_value == "有":
        details.append("伴盗汗、消瘦或咯血等慢性异常线索")
    elif tb_value == "无":
        details.append("无盗汗、消瘦或咯血")
    return "；".join(item for item in details if item)


def result_tone(result: str) -> tuple[str, str, str]:
    if "正常" in result or "低风险" in result:
        return "#1f8a5b", "#ecfdf5", "低风险"
    if "感染" in result or "肺炎" in result:
        return "#d97706", "#fff7ed", "感染性异常"
    if "慢性" in result or "结核" in result:
        return "#0f766e", "#ecfeff", "慢性异常"
    return "#64748b", "#f8fafc", "需复核"


def metric_card(title: str, value: str, note: str = "", accent: str = "#45b786") -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color:{accent};">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, color: str = "#45b786") -> str:
    return f"<span class='pill' style='background:{color};'>{text}</span>"


def display_model_name(raw: str) -> str:
    """Convert internal model identifiers into competition-friendly names."""
    value = str(raw or "").strip()
    exact_names = {
        "no-image": "未上传影像",
        "missing-file": "影像文件缺失",
        "filename-heuristic": "影像启发式回退模块",
        "rule-specialist": "规则复核模块",
        "rule-based": "本地规则决策模块",
        "none": "未触发",
        "huggingface:radiovision/Pneumonia-Detection": "肺炎专病复核模型",
        "huggingface:sukhmani1303/tuberculosis-vit-model": "结核/慢性异常专病复核模型",
    }
    if value in exact_names:
        return exact_names[value]

    lower = value.lower()
    if "efficientnetv2s" in lower or "efficientnetv2_s" in lower:
        return "EfficientNetV2-S 肺部三分类模型"
    if "densenet121" in lower:
        return "DenseNet121 胸片三分类模型"
    if "torchxrayvision" in lower:
        return "TorchXRayVision 通用胸片模型"
    if "pneumonia-detection" in lower:
        return "肺炎专病复核模型"
    if "tuberculosis-vit" in lower:
        return "结核/慢性异常专病复核模型"
    return value or "未触发"


def display_route_name(raw: str) -> str:
    route_names = {
        "infectious": "急性感染方向复核",
        "chronic": "结核/慢性方向复核",
        "dual_review": "双专病联合复核",
        "low_risk": "低风险观察分支",
        "uncertain": "不确定复核分支",
    }
    return route_names.get(str(raw or "").strip(), str(raw or "未触发"))


def sanitize_trace(text: str) -> str:
    cleaned = str(text or "")
    if "LLM 调用失败" in cleaned or "LLM调用失败" in cleaned or "curlopen error" in cleaned:
        return "DiagnosisAgent: DeepSeek 联网增强暂不可用，已自动切换为本地规则决策。"

    replacements = {
        "ImageAgent223": "ImageAgent",
        "医疗223三分类主模型": "EfficientNetV2-S 肺部三分类模型",
        "huggingface:radiovision/Pneumonia-Detection": "肺炎专病复核模型",
        "huggingface:sukhmani1303/tuberculosis-vit-model": "结核/慢性异常专病复核模型",
        "rule-specialist": "规则复核模块",
        "filename-heuristic": "影像启发式回退模块",
        "normal=": "低风险=",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=Source+Sans+3:wght@400;600;700&display=swap');

    :root {
        --mint-50: #f0fdf4;
        --mint-100: #dcfce7;
        --mint-200: #bbf7d0;
        --mint-600: #16a34a;
        --mint-700: #15803d;
        --teal-700: #0f766e;
        --ink: #15372d;
        --muted: #66756f;
        --paper: rgba(255,255,255,0.86);
    }

    html, body, [class*="css"] {
        font-family: "Source Sans 3", "Microsoft YaHei", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(69,183,134,0.24), transparent 28%),
            radial-gradient(circle at 92% 12%, rgba(15,118,110,0.14), transparent 30%),
            linear-gradient(180deg, #f4fbf7 0%, #eaf7ef 48%, #f8fbf8 100%);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e6f8ee 0%, #f7fcf9 100%);
        border-right: 1px solid rgba(21,128,61,0.12);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 36px 28px 36px;
        border-radius: 34px;
        color: #f8fffb;
        background:
            radial-gradient(circle at 86% 18%, rgba(187,247,208,0.34), transparent 20%),
            linear-gradient(135deg, #0c3f34 0%, #0f766e 48%, #22a06b 100%);
        box-shadow: 0 24px 76px rgba(15,76,66,0.28);
        margin-bottom: 18px;
    }

    .hero:after {
        content: "";
        position: absolute;
        right: -70px;
        top: -70px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(255,255,255,0.12);
    }

    .hero h1 {
        font-family: "Noto Serif SC", serif;
        font-size: 42px;
        line-height: 1.16;
        margin: 0 0 12px 0;
        letter-spacing: -0.03em;
    }

    .hero p {
        max-width: 850px;
        font-size: 17px;
        color: #0b3d32 !important;
        background: rgba(247,255,249,0.90);
        border: 1px solid rgba(255,255,255,0.72);
        border-radius: 16px;
        padding: 12px 16px;
        opacity: 1;
        margin: 0;
        font-weight: 600;
        text-shadow: none;
        box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        position: relative;
        z-index: 2;
    }

    .capability-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin-top: 22px;
        max-width: 1120px;
        position: relative;
        z-index: 2;
    }

    .capability-card {
        background: rgba(247,255,249,0.90);
        border: 1px solid rgba(255,255,255,0.72);
        border-radius: 18px;
        padding: 12px 14px;
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.10);
    }

    .capability-card b {
        display: block;
        color: #0f4f40;
        font-size: 14px;
        margin-bottom: 3px;
    }

    .capability-card span {
        color: #2f6658;
        font-size: 12px;
        font-weight: 600;
        text-shadow: none;
    }

    .pill {
        display: inline-block;
        color: white;
        font-size: 13px;
        font-weight: 700;
        border-radius: 999px;
        padding: 7px 12px;
        box-shadow: 0 8px 18px rgba(21,128,61,0.18);
    }

    .panel {
        background: var(--paper);
        border: 1px solid rgba(21,128,61,0.12);
        border-radius: 26px;
        padding: 22px;
        box-shadow: 0 18px 48px rgba(21,55,45,0.08);
        backdrop-filter: blur(14px);
        margin-bottom: 18px;
    }

    .section-title {
        font-family: "Noto Serif SC", serif;
        color: var(--ink);
        font-size: 23px;
        font-weight: 700;
        margin: 0 0 14px 0;
    }

    .subtle {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f7fcfa 100%);
        border: 1px solid rgba(21,128,61,0.10);
        border-top: 5px solid #45b786;
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 16px 36px rgba(15,118,110,0.08);
        min-height: 122px;
    }

    .metric-title {
        color: #5f716a;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .metric-value {
        color: var(--ink);
        font-size: 29px;
        font-weight: 800;
        line-height: 1.15;
    }

    .metric-note {
        color: var(--muted);
        font-size: 13px;
        margin-top: 8px;
    }

    .timeline {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
    }

    .node {
        position: relative;
        min-height: 94px;
        padding: 16px 15px 15px 54px;
        border-radius: 20px;
        background: linear-gradient(180deg, #ffffff, #f1fbf6);
        border: 1px solid rgba(21,128,61,0.12);
        box-shadow: 0 12px 28px rgba(21,128,61,0.06);
    }

    .node:before {
        content: attr(data-step);
        position: absolute;
        left: 15px;
        top: 16px;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: #dff7ea;
        color: #0f766e;
        font-weight: 900;
        font-size: 13px;
    }

    .node b {
        color: #126143;
        font-size: 15px;
    }

    .node span {
        display: block;
        color: #65746f;
        font-size: 13px;
        margin-top: 5px;
    }

    .result-banner {
        border-radius: 24px;
        padding: 22px;
        border: 1px solid rgba(21,128,61,0.12);
        margin-bottom: 16px;
    }

    .result-label {
        color: #64746e;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-main {
        font-size: 34px;
        font-weight: 900;
        line-height: 1.15;
    }

    div[data-testid="stButton"] > button {
        height: 3rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #16a34a, #0f766e) !important;
        border: 0 !important;
        color: white !important;
        font-weight: 800;
        box-shadow: 0 14px 32px rgba(22,163,74,0.24);
    }

    div[data-testid="stButton"] > button:hover {
        border: 0;
        color: white;
        filter: brightness(1.03);
    }

    /* Make Streamlit widgets readable on the light mint background. */
    .stMarkdown, .stMarkdown p, .stText, label, [data-testid="stWidgetLabel"] p {
        color: #173d31 !important;
    }

    textarea, input {
        background: #ffffff !important;
        color: #173d31 !important;
        border-color: rgba(21,128,61,0.22) !important;
        border-radius: 14px !important;
    }

    textarea {
        border: 1.5px solid rgba(21,128,61,0.34) !important;
        outline: none !important;
        box-shadow: 0 10px 24px rgba(15,118,110,0.05) !important;
    }

    div[data-baseweb="textarea"] > div {
        border: 1.5px solid rgba(21,128,61,0.34) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 24px rgba(15,118,110,0.05) !important;
    }

    div[data-baseweb="select"] > div {
        background: linear-gradient(90deg, #16a34a, #0f766e) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 24px rgba(15,118,110,0.10) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    div[role="listbox"] {
        background: #15803d !important;
        border: 1px solid rgba(236,253,245,0.20) !important;
        border-radius: 14px !important;
        box-shadow: 0 18px 42px rgba(15,61,52,0.28) !important;
        overflow: hidden !important;
    }

    div[role="option"] {
        background: #15803d !important;
        color: #ffffff !important;
    }

    div[role="option"]:hover,
    div[aria-selected="true"] {
        background: #16a34a !important;
        color: #ffffff !important;
    }

    textarea:focus, input:focus {
        border-color: #22a06b !important;
        outline: 0 !important;
        box-shadow: 0 0 0 3px rgba(34,160,107,0.14) !important;
    }

    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] div,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="textarea"] div:focus-within {
        border-color: rgba(21,128,61,0.34) !important;
        outline: 0 !important;
        box-shadow: none !important;
    }

    div[data-baseweb="textarea"]:focus-within {
        border-radius: 16px !important;
        box-shadow: 0 0 0 3px rgba(34,160,107,0.14) !important;
    }

    div[data-testid="stRadio"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stTextArea"] label p,
    div[data-testid="stFileUploader"] label p {
        color: #315247 !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 1px dashed rgba(21,128,61,0.34) !important;
        border-radius: 18px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #173d31 !important;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button[kind],
    [data-testid="stFileUploader"] section button {
        background: linear-gradient(90deg, #16a34a, #0f766e) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 24px rgba(15,118,110,0.18) !important;
    }

    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploader"] section button * {
        color: #ffffff !important;
    }

    .question-box {
        position: relative;
        background: linear-gradient(180deg, #ffffff 0%, #f8fdfb 100%);
        border: 1px solid rgba(21,128,61,0.10) !important;
        border-radius: 22px !important;
        padding: 20px 22px 18px 24px;
        box-shadow: 0 14px 36px rgba(15,118,110,0.07);
        margin-top: 12px;
        margin-bottom: 14px;
    }

    .question-box:before {
        content: "";
        position: absolute;
        left: 0;
        top: 20px;
        bottom: 20px;
        width: 5px;
        border-radius: 0 6px 6px 0;
        background: linear-gradient(180deg, #22c55e, #0f766e);
    }

    .question-box-title {
        color: #123b31 !important;
        font-family: "Noto Serif SC", serif;
        font-weight: 800;
        font-size: 19px;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }

    .question-box-subtitle {
        color: #5d7169;
        font-size: 13px;
        margin-bottom: 0;
    }

    [data-testid="stPills"] label,
    [data-testid="stPills"] label p {
        color: #173d31 !important;
        font-weight: 900 !important;
    }

    [data-testid="stPills"] button {
        border-radius: 999px !important;
        border: 1px solid rgba(15,118,110,0.18) !important;
        background: rgba(255,255,255,0.82) !important;
        color: #173d31 !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 12px rgba(15,118,110,0.04);
    }

    [data-testid="stPills"] button *,
    [data-testid="stPills"] button p,
    [data-testid="stPills"] button div {
        color: #173d31 !important;
    }

    [data-testid="stPills"] button[aria-pressed="true"],
    [data-testid="stPills"] button[data-selected="true"] {
        background: linear-gradient(90deg, #16a34a, #0f766e) !important;
        border-color: #16a34a !important;
        color: #ffffff !important;
        box-shadow: 0 10px 22px rgba(22,163,74,0.22);
    }

    [data-testid="stPills"] button[aria-pressed="true"] *,
    [data-testid="stPills"] button[aria-pressed="true"] p,
    [data-testid="stPills"] button[aria-pressed="true"] div,
    [data-testid="stPills"] button[data-selected="true"] *,
    [data-testid="stPills"] button[data-selected="true"] p,
    [data-testid="stPills"] button[data-selected="true"] div {
        color: #ffffff !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(21,128,61,0.10);
        border-radius: 18px;
        padding: 12px 14px;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div {
        color: #173d31 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 999px;
        border: 1px solid rgba(21,128,61,0.12);
        color: #173d31;
        padding: 8px 14px;
    }

    .stTabs [aria-selected="true"] {
        background: #dcfce7 !important;
        color: #0f5132 !important;
    }

    @media (max-width: 900px) {
        .capability-grid, .timeline {
            grid-template-columns: 1fr;
        }
        .hero h1 {
            font-size: 32px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
      <h1>肺部异常辅助筛查与分诊系统</h1>
      <p>
        面向基层医疗场景的多模态智能体原型，融合胸片影像模型、症状问诊、RAG 医学指南、
        LangGraph 多智能体分支和 DeepSeek 诊断整合，输出可解释的肺部异常筛查结果。
      </p>
      <div class="capability-grid">
        <div class="capability-card"><b>胸片影像</b><span>EfficientNetV2-S 三分类基础模型</span></div>
        <div class="capability-card"><b>专病复核</b><span>肺炎 / 结核方向二次评估</span></div>
        <div class="capability-card"><b>RAG 指南</b><span>ATS/IDSA 与 WHO 医学证据</span></div>
        <div class="capability-card"><b>LangGraph</b><span>条件分支与自反馈闭环</span></div>
        <div class="capability-card"><b>DeepSeek</b><span>诊断整合与报告解释</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### 系统定位")
    st.write("肺部异常辅助筛查，不替代医生诊断。")
    st.markdown("### 核心能力")
    st.markdown("- 胸片三分类基础模型")
    st.markdown("- 肺炎与结核专病复核")
    st.markdown("- RAG 指南证据召回")
    st.markdown("- DeepSeek 诊断整合")
    st.markdown("- 自反馈与一致性校验")
    st.markdown("---")
    st.markdown("### 演示建议")
    st.markdown("1. 输入症状或使用问诊补全")
    st.markdown("2. 上传胸部 X 光图像")
    st.markdown("3. 点击开始筛查")
    st.markdown("4. 展示 RAG 与 Agent 轨迹")


left, right = st.columns([0.94, 1.36], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">病例输入</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">先用快速问诊完成结构化选择，再补充患者原话，适合现场演示和基层问诊。</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="question-box">
          <div class="question-box-title">快速结构化问诊</div>
          <div class="question-box-subtitle">先选择关键症状，系统会自动组合为多智能体推理输入；必要时可选择“自定义”。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    duration_choice = st.pills(
        "病程持续多久？",
        ["未说明", "1天", "2天", "3天", "1周", "2周", "1个月", "更久", "自定义"],
        default="未说明",
    )
    duration_value = duration_choice
    if duration_choice == "自定义":
        duration_value = st.text_input("请输入病程", placeholder="例如：5天、半个月、2个月")

    q1, q2 = st.columns(2)
    with q1:
        fever_value = st.pills("是否发热？", ["未说明", "有", "无"], default="未说明")
        sputum_value = st.pills("是否咳痰？", ["未说明", "有", "无"], default="未说明")
    with q2:
        chest_value = st.pills("是否胸痛/气促？", ["未说明", "有", "无"], default="未说明")
        tb_value = st.pills("是否盗汗/消瘦/咯血？", ["未说明", "有", "无"], default="未说明")

    sputum_choice = st.pills(
        "痰液特点", ["未说明", "白痰", "黄痰", "血痰", "自定义"], default="未说明"
    )
    sputum_color = sputum_choice
    if sputum_choice == "自定义":
        sputum_color = st.text_input(
            "请输入痰液特点", placeholder="例如：少量黏痰、铁锈色痰、无明显痰"
        )

    symptom = st.text_area(
        "补充症状描述（可选）",
        value="",
        placeholder="例如：发热3天，咳嗽咳痰，伴胸痛",
        height=118,
        help="可以输入患者原话，例如：前几天嗓子痛，后来咳嗽、黄痰、发烧。",
    )

    uploaded = st.file_uploader("上传胸部影像", type=["png", "jpg", "jpeg"])
    st.caption(
        "请仅使用脱敏且已获授权的演示资料。影像在本地推理；若启用云端大模型，症状及派生证据将发送至所配置的服务。处理完成后删除本次上传的临时文件。"
    )
    analyze = st.button("开始辅助筛查", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">多智能体工作流</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="timeline">
          <div class="node" data-step="01"><b>SymptomAgent</b><span>抽取症状、病程和否定语境</span></div>
          <div class="node" data-step="02"><b>ImageAgent</b><span>第一阶段影像分诊给出三类风险</span></div>
          <div class="node" data-step="03"><b>KnowledgeAgent</b><span>召回肺炎与结核指南证据</span></div>
          <div class="node" data-step="04"><b>RouteAgent</b><span>影像触发 + RAG 文本触发</span></div>
          <div class="node" data-step="05"><b>SpecialistAgents</b><span>肺炎/结核专病模型复核</span></div>
          <div class="node" data-step="06"><b>DiagnosisAgent</b><span>DeepSeek 或本地规则综合诊断</span></div>
          <div class="node" data-step="07"><b>ReviewAgent</b><span>图文冲突与一致性校验</span></div>
          <div class="node" data-step="08"><b>SelfFeedbackAgent</b><span>低置信度时追加保守规则</span></div>
          <div class="node" data-step="09"><b>CheckAgent</b><span>生成下一步检查建议</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">筛查结果</div>', unsafe_allow_html=True)

    if analyze:
        enriched_symptom = build_structured_symptom(
            symptom,
            duration_value,
            fever_value,
            sputum_value,
            sputum_color,
            chest_value,
            tb_value,
        )
        if not enriched_symptom.strip():
            st.error("请先选择问诊信息或补充症状描述。")
        else:
            image_path = save_uploaded_file(uploaded) if uploaded is not None else ""

            try:
                with st.spinner("正在执行 LangGraph 多智能体筛查流程..."):
                    state = run_medical223(enriched_symptom, image_path)
            except Exception:
                st.error("筛查未完成，请检查配置与输入后重试。")
                st.stop()
            finally:
                if image_path:
                    Path(image_path).unlink(missing_ok=True)

            color, bg, tag = result_tone(state.screening_result)
            st.markdown(
                f"""
                <div class="result-banner" style="background:{bg};">
                  <div class="result-label">Screening Result · {tag}</div>
                  <div class="result-main" style="color:{color};">{state.screening_result}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if uploaded is not None:
                st.image(uploaded, caption="上传的胸部影像", use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card(
                    "急性感染方向",
                    f"{state.image_scores.get('感染性异常', 0.0):.2f}",
                    "肺炎/急性感染性病变风险",
                    "#d97706",
                )
            with c2:
                metric_card(
                    "结核慢性方向",
                    f"{state.image_scores.get('慢性异常', 0.0):.2f}",
                    "结核样/慢性异常风险",
                    "#0f766e",
                )
            with c3:
                metric_card(
                    "综合置信度", f"{state.confidence:.2f}", "多源证据综合置信度", "#16a34a"
                )

            st.markdown("#### 结构化问诊")
            st.info(enriched_symptom)

            a, b = st.columns(2)
            with a:
                st.markdown("#### 主要依据")
                st.write(state.rationale or "系统未返回主要依据。")
                st.markdown("#### 鉴别诊断")
                st.write(state.differential or "系统未返回鉴别诊断。")
            with b:
                st.markdown("#### 建议检查")
                st.write(state.check_suggestion or "系统未返回检查建议。")
                st.markdown("#### 复核意见")
                st.write(state.review_note or "系统未返回复核意见。")

            st.markdown("#### 模型与路由")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("运行模式", "LLM增强" if state.used_llm else "本地Agent")
            with r2:
                st.metric("路由分支", display_route_name(state.route_decision))
            with r3:
                st.metric("自反馈轮次", state.iteration)
            st.caption(sanitize_trace(state.route_reason))

            tab1, tab2, tab3, tab4 = st.tabs(["RAG 证据", "Agent 轨迹", "专病复核", "结构化数据"])
            with tab1:
                st.write(f"RAG感染性证据：{state.rag_infectious_score:.2f}")
                st.write(f"RAG慢性异常证据：{state.rag_chronic_score:.2f}")
                st.write(state.rag_route_note)
                for idx, item in enumerate(state.retrieved_knowledge, start=1):
                    st.markdown(f"**{idx}.** {item}")
            with tab2:
                for idx, item in enumerate(state.agent_trace, start=1):
                    st.markdown(f"**Step {idx}**  {sanitize_trace(item)}")
            with tab3:
                c4, c5 = st.columns(2)
                with c4:
                    metric_card(
                        "肺炎专病复核",
                        f"{state.infectious_specialist_score:.2f}",
                        display_model_name(state.infectious_specialist_source),
                        "#d97706",
                    )
                    st.write(state.infectious_specialist_note)
                with c5:
                    metric_card(
                        "结核/慢性复核",
                        f"{state.chronic_specialist_score:.2f}",
                        display_model_name(state.chronic_specialist_source),
                        "#0f766e",
                    )
                    st.write(state.chronic_specialist_note)
            with tab4:
                st.json(
                    {
                        "screening_result": state.screening_result,
                        "confidence": state.confidence,
                        "image_model": display_model_name(state.image_model_name),
                        "image_scores": state.image_scores,
                        "route_decision": display_route_name(state.route_decision),
                        "rag_infectious_score": state.rag_infectious_score,
                        "rag_chronic_score": state.rag_chronic_score,
                        "infectious_specialist_score": state.infectious_specialist_score,
                        "infectious_specialist_model": display_model_name(
                            state.infectious_specialist_source
                        ),
                        "chronic_specialist_score": state.chronic_specialist_score,
                        "chronic_specialist_model": display_model_name(
                            state.chronic_specialist_source
                        ),
                        "used_llm": state.used_llm,
                        "iteration": state.iteration,
                    }
                )
    else:
        st.markdown(
            """
            <div class="result-banner" style="background:#f0fdf4;">
              <div class="result-label">Waiting For Input</div>
              <div class="result-main" style="color:#15803d;">请在左侧输入症状并上传胸片</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("系统将展示筛查结论、三类风险分数、RAG 召回证据、Agent 执行轨迹和下一步检查建议。")

    st.markdown("</div>", unsafe_allow_html=True)
