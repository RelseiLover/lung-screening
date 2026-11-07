"""Presentation functions copied unchanged from the original Streamlit page."""


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
