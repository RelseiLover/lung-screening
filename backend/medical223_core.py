from __future__ import annotations

import json
import os
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from runtime_config import load_environment

load_environment()

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


HF_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}
HF_TB_CACHE: Dict[str, object] = {}
RAG_CACHE: Dict[str, object] = {}
MEDICAL223_MODEL_CACHE: Dict[Tuple[str, str], Tuple[object, object, list[str]]] = {}
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

HF_TB_REPO = os.getenv("HF_TB_REPO", "sukhmani1303/tuberculosis-vit-model").strip()
HF_PNEUMONIA_REPO = os.getenv("HF_PNEUMONIA_REPO", "radiovision/Pneumonia-Detection").strip()
MEDICAL223_REPO_DIR = os.getenv(
    "MEDICAL223_REPO_DIR",
    os.path.dirname(PROJECT_DIR),
).strip()
MEDICAL223_MODEL_PATH = os.getenv("MEDICAL223_MODEL_PATH", "").strip()
MEDICAL223_MODEL_ARCH = os.getenv(
    "MEDICAL223_MODEL_ARCH", "tf_efficientnetv2_s.in21k_ft_in1k"
).strip()
MEDICAL223_IMAGE_SIZE = int(os.getenv("MEDICAL223_IMAGE_SIZE", "512").strip() or 512)
MEDICAL223_CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]
LLM_LAST_ERROR = ""


def load_local_deepseek_config() -> Dict[str, str]:
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "deepseek_local_config.json"
    )
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v).strip() for k, v in data.items() if str(v).strip()}
    except Exception:
        return {}


@dataclass
class MedicalState:
    symptom: str
    image_path: str = ""
    symptom_features: Dict[str, object] = field(default_factory=dict)
    image_result: str = ""
    image_scores: Dict[str, float] = field(default_factory=dict)
    image_model_name: str = "rule-based"
    retrieved_knowledge: List[str] = field(default_factory=list)
    rag_infectious_score: float = 0.0
    rag_chronic_score: float = 0.0
    rag_route_note: str = ""
    screening_result: str = ""
    confidence: float = 0.0
    rationale: str = ""
    differential: str = ""
    check_suggestion: str = ""
    review_note: str = ""
    chronic_specialist_score: float = 0.0
    chronic_specialist_note: str = ""
    chronic_specialist_source: str = "none"
    infectious_specialist_score: float = 0.0
    infectious_specialist_note: str = ""
    infectious_specialist_source: str = "none"
    iteration: int = 0
    used_llm: bool = False
    agent_trace: List[str] = field(default_factory=list)
    route_decision: str = "undecided"
    route_reason: str = ""


RESPIRATORY_FEATURES = {
    "发热": ["发热", "高热", "低热", "发烧", "体温", "fever"],
    "咳嗽": ["咳嗽", "cough"],
    "咳痰": ["咳痰", "白痰", "黄痰", "浓痰", "sputum", "phlegm"],
    "胸痛": ["胸痛", "胸闷", "chest pain", "tightness"],
    "气促": ["气促", "呼吸困难", "憋气", "dyspnea", "shortness of breath"],
    "盗汗": ["盗汗", "night sweat"],
    "消瘦": ["消瘦", "体重下降", "weight loss"],
    "咯血": ["咯血", "血痰", "hemoptysis", "bloody sputum"],
    "乏力": ["乏力", "疲倦", "无力", "fatigue", "weakness"],
    "病程长": [
        "1个月",
        "两个月",
        "数周",
        "长期",
        "反复",
        "1 month",
        "2 months",
        "weeks",
        "chronic",
    ],
    "急性起病": [
        "1天",
        "2天",
        "3天",
        "1周内",
        "急性",
        "突然",
        "1 day",
        "2 days",
        "3 days",
        "acute",
        "sudden",
    ],
}

NEGATIVE_PATTERNS = {
    "发热": ["无发热", "没有发热", "未发热", "否认发热", "no fever", "without fever"],
    "咳嗽": ["无咳嗽", "没有咳嗽", "未咳嗽", "否认咳嗽", "no cough", "without cough"],
    "咳痰": ["无痰", "无咳痰", "没有咳痰", "未咳痰", "否认咳痰", "no sputum", "without sputum"],
    "胸痛": ["无胸痛", "没有胸痛", "未胸痛", "否认胸痛", "no chest pain", "without chest pain"],
    "气促": [
        "无气促",
        "没有气促",
        "未气促",
        "否认气促",
        "no dyspnea",
        "without shortness of breath",
    ],
    "盗汗": ["无盗汗", "没有盗汗", "未盗汗", "否认盗汗", "no night sweat", "without night sweat"],
    "消瘦": ["无消瘦", "没有消瘦", "未消瘦", "否认消瘦", "no weight loss", "without weight loss"],
    "咯血": ["无咯血", "没有咯血", "未咯血", "否认咯血", "no hemoptysis", "without hemoptysis"],
}

NO_SYMPTOM_PATTERNS = [
    "无不适",
    "没有不适",
    "未诉不适",
    "无明显不适",
    "体检",
    "复查",
    "no symptom",
    "asymptomatic",
]

RAG_DOCUMENTS = [
    {
        "title": "ATS/IDSA 成人社区获得性肺炎指南",
        "source": "ATS/IDSA CAP Guideline (2019)",
        "content": "成人社区获得性肺炎常见急性起病，表现为发热、咳嗽、咳痰、胸痛或气促，影像学出现新的肺部浸润影有助于支持诊断。",
    },
    {
        "title": "WHO 结核病事实与症状要点",
        "source": "WHO Tuberculosis Fact Sheet",
        "content": "肺结核常见症状包括持续咳嗽、胸痛、乏力、体重下降、发热和夜间盗汗。对有症状者建议结合快速分子检测与痰标本进一步检查。",
    },
    {
        "title": "WHO 胸片在结核检测中的作用",
        "source": "WHO Chest Radiography in TB Detection",
        "content": "胸部X线是结核早期发现的重要工具，可作为分诊工具和诊断辅助工具，但不能单独完成最终确诊。",
    },
    {
        "title": "肺部异常分诊要点",
        "source": "Local triage summary",
        "content": "感染性异常更常见急性病程、发热和咳痰。慢性异常风险更常见盗汗、消瘦、咯血和较长病程。影像异常但方向不清时应建议进一步检查。",
    },
]


def state_to_dict(state: MedicalState) -> Dict[str, object]:
    return asdict(state)


def dict_to_state(data: Dict[str, object] | MedicalState) -> MedicalState:
    if isinstance(data, MedicalState):
        return data
    return MedicalState(**data)


def call_openai_compatible(prompt: str) -> Optional[str]:
    global LLM_LAST_ERROR
    LLM_LAST_ERROR = ""
    local_llm_config = load_local_deepseek_config()
    api_key = (
        os.getenv("DEEPSEEK_API_KEY", "")
        or local_llm_config.get("api_key", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()
    base_url = (
        (
            os.getenv("DEEPSEEK_BASE_URL", "")
            or local_llm_config.get("base_url", "")
            or os.getenv("OPENAI_BASE_URL", "")
            or ("https://api.deepseek.com/v1" if api_key else "https://api.openai.com/v1")
        )
        .strip()
        .rstrip("/")
    )
    model = (
        os.getenv("DEEPSEEK_MODEL", "")
        or local_llm_config.get("model", "")
        or os.getenv("OPENAI_MODEL", "")
        or ("deepseek-chat" if api_key else "gpt-4o-mini")
    ).strip()
    if not api_key:
        LLM_LAST_ERROR = "未配置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，已使用本地规则决策。"
        return None
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "你是医疗辅助分诊模型，只能基于提供的证据输出初步筛查结论。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")[:300]
        except Exception:
            body = ""
        LLM_LAST_ERROR = f"LLM HTTP {exc.code}: {body or exc.reason}"
        return None
    except (OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
        LLM_LAST_ERROR = f"LLM 调用失败: {exc}"
        return None


def extract_duration(symptom: str) -> str:
    patterns = [
        (r"(\d+)\s*个?月", "个月"),
        (r"(\d+)\s*周", "周"),
        (r"(\d+)\s*天", "天"),
        (r"(\d+)\s*month", "个月"),
        (r"(\d+)\s*week", "周"),
        (r"(\d+)\s*day", "天"),
    ]
    for pattern, unit in patterns:
        match = re.search(pattern, symptom, re.I)
        if match:
            return f"{match.group(1)}{unit}"
    return "未提及"


def symptom_agent(state: MedicalState) -> MedicalState:
    text = state.symptom.lower()
    signals: List[str] = []
    for feature, aliases in RESPIRATORY_FEATURES.items():
        negatives = NEGATIVE_PATTERNS.get(feature, [])
        if any(token.lower() in text for token in negatives):
            continue
        if any(token.lower() in text for token in aliases):
            signals.append(feature)

    no_symptom_context = any(token.lower() in text for token in NO_SYMPTOM_PATTERNS)
    broad_negation_context = no_symptom_context and any(
        token in state.symptom for token in ["没有", "无", "未", "否认"]
    )
    if no_symptom_context:
        positive_terms = ["发热", "咳嗽", "咳痰", "黄痰", "胸痛", "气促", "盗汗", "消瘦", "咯血"]
        negated_terms = [
            term
            for term in positive_terms
            if any(prefix + term in state.symptom for prefix in ["无", "没有", "未", "否认"])
        ]
        if broad_negation_context or len(negated_terms) >= 2 or not signals:
            signals = []

    state.symptom_features = {
        "duration": extract_duration(state.symptom),
        "signals": signals,
        "infectious_score": sum(
            1 for x in ["发热", "咳嗽", "咳痰", "胸痛", "气促", "急性起病"] if x in signals
        ),
        "chronic_score": sum(
            1 for x in ["盗汗", "消瘦", "咯血", "乏力", "病程长", "低热"] if x in signals
        ),
        "no_symptom_context": no_symptom_context and not signals,
    }
    signal_text = ", ".join(signals) if signals else "无明显高特异性特征"
    state.agent_trace.append(
        f"SymptomAgent: 提取到特征 {signal_text}；病程 {state.symptom_features['duration']}。"
    )
    return state


def run_torchxrayvision_inference(image_path: str) -> Optional[Dict[str, object]]:
    try:
        import skimage.io
        import torch
        import torchvision
        import torchxrayvision as xrv
    except ImportError:
        return None
    if np is None:
        return None
    try:
        image = skimage.io.imread(image_path)
        image = xrv.datasets.normalize(image, 255)
        if image.ndim == 3:
            image = image.mean(2)
        image = image[None, ...]
        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )
        image = transform(image)
        image_tensor = torch.from_numpy(np.asarray(image)).float()
        weights = os.getenv("TORCHXRAYVISION_WEIGHTS", "densenet121-res224-all").strip()
        model = xrv.models.DenseNet(weights=weights)
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor[None, ...])[0].detach().cpu().numpy()
        scores = dict(zip(model.pathologies, outputs))
        infectious = max(
            float(scores.get("Pneumonia", 0.0)),
            0.65 * float(scores.get("Infiltration", 0.0))
            + 0.35 * float(scores.get("Lung Opacity", 0.0)),
        )
        chronic = max(
            0.05,
            0.45 * float(scores.get("Fibrosis", 0.0))
            + 0.35 * float(scores.get("Lung Opacity", 0.0))
            + 0.20 * float(scores.get("Infiltration", 0.0)),
        )
        return {
            "summary": "TorchXRayVision 提示肺部异常存在，但对感染性异常与慢性异常的区分度有限。",
            "scores": {
                "感染性异常": round(min(0.98, infectious), 2),
                "慢性异常": round(min(0.98, chronic), 2),
            },
            "model_name": f"torchxrayvision:{weights}",
        }
    except Exception:
        return None


def load_hf_tb(repo_id: str) -> Optional[Tuple[object, dict]]:
    if repo_id in HF_TB_CACHE:
        return HF_TB_CACHE[repo_id]
    try:
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    for local_only in (True, False):
        try:
            model_path = hf_hub_download(repo_id, "model.pt", local_files_only=local_only)
            config_path = hf_hub_download(repo_id, "config.json", local_files_only=local_only)
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model = torch.jit.load(model_path, map_location="cpu")
            model.eval()
            HF_TB_CACHE[repo_id] = (model, config)
            return model, config
        except Exception:
            continue
    return None


def run_hf_tb(image_path: str) -> Optional[float]:
    loaded = load_hf_tb(HF_TB_REPO)
    if not loaded or np is None:
        return None
    try:
        import torch
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return None
    model, config = loaded
    try:
        image = Image.open(image_path).convert("L")
        image = ImageOps.equalize(image)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        size = int(config.get("input_size", 224))
        image = image.resize((size, size))
        arr = np.asarray(image).astype("float32")
        arr = np.stack([arr, arr, arr], axis=0)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        tensor = torch.tensor(arr).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            if len(output.shape) > 1:
                output = output.squeeze(-1)
            return float(torch.sigmoid(output).item())
    except Exception:
        return None


def load_hf_classifier(repo_id: str) -> Optional[Tuple[object, object]]:
    if repo_id in HF_MODEL_CACHE:
        return HF_MODEL_CACHE[repo_id]
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        return None
    for local_only in (True, False):
        try:
            processor = AutoImageProcessor.from_pretrained(repo_id, local_files_only=local_only)
            model = AutoModelForImageClassification.from_pretrained(
                repo_id, local_files_only=local_only, use_safetensors=False
            )
            HF_MODEL_CACHE[repo_id] = (processor, model)
            return processor, model
        except Exception:
            continue
    return None


def run_hf_pneumonia(image_path: str) -> Optional[float]:
    try:
        import torch
        from PIL import Image
    except ImportError:
        return None
    loaded = load_hf_classifier(HF_PNEUMONIA_REPO)
    if not loaded:
        return None
    processor, model = loaded
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=0).detach().cpu().tolist()
        labels = getattr(model.config, "id2label", {}) or {}
        for idx, prob in enumerate(probs):
            if "pneumonia" in str(labels.get(idx, "")).lower():
                return float(prob)
        return max(probs) if probs else None
    except Exception:
        return None


def _guess_medical223_model_path(repo_dir: str) -> Optional[str]:
    if not repo_dir:
        return None
    candidates = [
        os.path.join(repo_dir, "models", "efficientnetv2s_best.pth"),
        os.path.join(repo_dir, "models", "densenet121_best.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _resolve_medical223_target() -> Tuple[str, str]:
    model_path = MEDICAL223_MODEL_PATH or _guess_medical223_model_path(MEDICAL223_REPO_DIR) or ""
    model_arch = MEDICAL223_MODEL_ARCH
    if (
        model_path
        and "densenet121" in os.path.basename(model_path).lower()
        and MEDICAL223_MODEL_ARCH == "tf_efficientnetv2_s.in21k_ft_in1k"
    ):
        model_arch = "densenet121"
    return model_path, model_arch


def _load_medical223_state_dict(model_path: str):
    import torch

    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_medical223_model(model_path: str, model_arch: str):
    cache_key = (model_path, model_arch)
    if cache_key in MEDICAL223_MODEL_CACHE:
        return MEDICAL223_MODEL_CACHE[cache_key]

    import timm
    from torchvision import transforms

    model = timm.create_model(model_arch, pretrained=False, num_classes=len(MEDICAL223_CLASS_NAMES))
    state_dict = _load_medical223_state_dict(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((MEDICAL223_IMAGE_SIZE, MEDICAL223_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    MEDICAL223_MODEL_CACHE[cache_key] = (model, transform, MEDICAL223_CLASS_NAMES)
    return model, transform, MEDICAL223_CLASS_NAMES


def run_medical223_primary_model(image_path: str) -> Optional[Dict[str, object]]:
    model_path, model_arch = _resolve_medical223_target()
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        import torch
        from PIL import Image
    except ImportError:
        return None

    try:
        model, transform, class_names = load_medical223_model(model_path, model_arch)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits[0], dim=0).detach().cpu().tolist()
        scores = {class_names[idx]: float(prob) for idx, prob in enumerate(probs)}
        return {
            "scores": scores,
            "model_name": f"medical223:{model_arch}:{os.path.basename(model_path)}",
            "summary": (
                f"医疗223三分类主模型预测："
                f"normal={scores.get('normal', 0.0):.2f}, "
                f"pneumonia={scores.get('pneumonia', 0.0):.2f}, "
                f"tuberculosis={scores.get('tuberculosis', 0.0):.2f}"
            ),
        }
    except Exception:
        return None


def image_agent(state: MedicalState) -> MedicalState:
    if not state.image_path:
        state.image_result = "未提供影像，影像证据缺失。"
        state.image_scores = {"感染性异常": 0.35, "慢性异常": 0.35}
        state.image_model_name = "no-image"
        state.agent_trace.append("ImageAgent: 未输入影像，使用缺省中性影像分数。")
        return state
    if not os.path.exists(state.image_path):
        state.image_result = "影像路径不存在，未能读取影像文件。"
        state.image_scores = {"感染性异常": 0.30, "慢性异常": 0.30}
        state.image_model_name = "missing-file"
        state.agent_trace.append("ImageAgent: 影像路径不存在，回退到保守影像分数。")
        return state

    medical223 = run_medical223_primary_model(state.image_path)
    if medical223:
        normal_score = round(medical223["scores"].get("normal", 0.0), 2)
        infectious_score = round(medical223["scores"].get("pneumonia", 0.0), 2)
        chronic_score = round(medical223["scores"].get("tuberculosis", 0.0), 2)
        state.image_result = (
            f"{medical223['summary']}。"
            f"影像主判断为 normal={normal_score:.2f}, 感染性异常={infectious_score:.2f}, 慢性异常={chronic_score:.2f}。"
        )
        state.image_scores = {
            "感染性异常": infectious_score,
            "慢性异常": chronic_score,
            "正常倾向": normal_score,
        }
        state.image_model_name = medical223["model_name"]
        state.agent_trace.append(
            "ImageAgent223: 已切换为医疗223三分类主模型，"
            f"normal={normal_score:.2f}, 感染性异常={infectious_score:.2f}, 慢性异常={chronic_score:.2f}。"
        )
        return state

    xrv = run_torchxrayvision_inference(state.image_path)
    if xrv:
        state.image_result = xrv["summary"]
        state.image_scores = xrv["scores"]
        state.image_model_name = xrv["model_name"]
        state.agent_trace.append(
            f"ImageAgent: 已接入 TorchXRayVision，影像评分=感染性异常 {state.image_scores['感染性异常']:.2f}, 慢性异常 {state.image_scores['慢性异常']:.2f}。"
        )
        state.agent_trace.append(
            "ImageAgent223: 未检测到医疗223主模型权重，已自动回退到原始通用影像模型。"
        )
        return state

    state.image_result = "影像存在非特异性肺部异常阴影，需要结合症状与检查进一步判断。"
    state.image_scores = {"感染性异常": 0.55, "慢性异常": 0.48}
    state.image_model_name = "filename-heuristic"
    state.agent_trace.append(
        f"ImageAgent: 回退到启发式影像评分=感染性异常 {state.image_scores['感染性异常']:.2f}, 慢性异常 {state.image_scores['慢性异常']:.2f}。"
    )
    state.agent_trace.append(
        "ImageAgent223: 当前未成功加载医疗223三分类主模型，仍使用原有影像回退逻辑。"
    )
    return state


def build_rag_index() -> Optional[Tuple[object, object]]:
    if "tfidf" in RAG_CACHE:
        return RAG_CACHE["tfidf"]
    if TfidfVectorizer is None or cosine_similarity is None:
        return None
    corpus = [f"{doc['title']} {doc['source']} {doc['content']}" for doc in RAG_DOCUMENTS]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    RAG_CACHE["tfidf"] = (vectorizer, matrix)
    return vectorizer, matrix


def retrieve_medical_knowledge(query: str, top_k: int = 3) -> List[Dict[str, object]]:
    built = build_rag_index()
    if not built:
        return []
    vectorizer, matrix = built
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    ranked = sorted(
        [
            {"score": float(scores[idx]), "doc": RAG_DOCUMENTS[idx]}
            for idx in range(len(RAG_DOCUMENTS))
        ],
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked[:top_k]


def score_rag_route_evidence(
    state: MedicalState, hits: List[Dict[str, object]]
) -> Tuple[float, float, str]:
    signals = set(state.symptom_features.get("signals", []))
    text = f"{state.symptom} {' '.join(state.retrieved_knowledge)}"
    if state.symptom_features.get("no_symptom_context"):
        return 0.0, 0.0, "症状文本为体检/无不适场景，RAG不单独触发专病复核。"
    infectious_symptoms = {"发热", "咳嗽", "咳痰", "胸痛", "气促", "急性起病"}
    chronic_symptoms = {"盗汗", "消瘦", "咯血", "乏力", "病程长", "低热"}

    infectious_match = len(signals & infectious_symptoms)
    chronic_match = len(signals & chronic_symptoms)
    infectious_doc = 0.0
    chronic_doc = 0.0
    for hit in hits:
        doc = hit.get("doc", {})
        title = str(doc.get("title", ""))
        content = str(doc.get("content", ""))
        source = str(doc.get("source", ""))
        score = float(hit.get("score", 0.0))
        blob = f"{title} {source} {content}"
        if any(token in blob for token in ["肺炎", "CAP", "ATS/IDSA", "感染", "浸润"]):
            infectious_doc = max(infectious_doc, score)
        if any(token in blob for token in ["结核", "Tuberculosis", "WHO", "盗汗", "消瘦", "咯血"]):
            chronic_doc = max(chronic_doc, score)

    infectious_score = 0.0
    chronic_score = 0.0
    if infectious_match:
        infectious_score += min(0.14, infectious_match * 0.035)
    if chronic_match:
        chronic_score += min(0.16, chronic_match * 0.04)
    if infectious_doc >= 0.05 and infectious_match >= 2:
        infectious_score += min(0.08, infectious_doc * 0.20)
    if chronic_doc >= 0.05 and chronic_match >= 1:
        chronic_score += min(0.09, chronic_doc * 0.22)

    if "无发热" in text or "否认发热" in text:
        infectious_score = max(0.0, infectious_score - 0.03)
    if "无盗汗" in text or "否认盗汗" in text or "无咯血" in text:
        chronic_score = max(0.0, chronic_score - 0.04)

    infectious_score = round(min(0.22, infectious_score), 2)
    chronic_score = round(min(0.24, chronic_score), 2)
    if infectious_score and chronic_score:
        note = f"RAG同时发现感染性与慢性异常指南线索，感染性证据={infectious_score:.2f}，慢性证据={chronic_score:.2f}。"
    elif infectious_score:
        note = f"RAG/症状文本符合肺炎指南常见特征，感染性复核触发证据={infectious_score:.2f}。"
    elif chronic_score:
        note = f"RAG/症状文本符合结核或慢性异常指南特征，慢性复核触发证据={chronic_score:.2f}。"
    else:
        note = "RAG未发现足以单独触发专病复核的指南特征。"
    return infectious_score, chronic_score, note


def knowledge_agent(state: MedicalState) -> MedicalState:
    signals = state.symptom_features.get("signals", [])
    query = (
        f"症状:{state.symptom}; 特征:{','.join(signals)}; 影像:{state.image_result}; "
        f"感染性异常分数:{state.image_scores.get('感染性异常', 0.0):.2f}; "
        f"慢性异常分数:{state.image_scores.get('慢性异常', 0.0):.2f}"
    )
    hits = retrieve_medical_knowledge(query)
    state.retrieved_knowledge = [
        f"RAG召回[{hit['doc']['title']}] 相似度={hit['score']:.2f} 来源={hit['doc']['source']}: {hit['doc']['content']}"
        for hit in hits
    ] or ["通用知识: 单一症状无法直接确诊，需要结合影像和实验室检查。"]
    state.rag_infectious_score, state.rag_chronic_score, state.rag_route_note = (
        score_rag_route_evidence(state, hits)
    )
    state.agent_trace.append(
        f"KnowledgeAgent: 使用轻量RAG召回 {len(state.retrieved_knowledge)} 条知识证据；"
        f"感染性RAG证据 {state.rag_infectious_score:.2f}，慢性RAG证据 {state.rag_chronic_score:.2f}。"
    )
    return state


def route_after_image(state: MedicalState) -> MedicalState:
    infectious_base = state.image_scores.get("感染性异常", 0.35)
    chronic_base = state.image_scores.get("慢性异常", 0.35)
    normal_base = state.image_scores.get("正常倾向", 0.0)
    infectious_hint = state.symptom_features.get("infectious_score", 0) * 0.03
    chronic_hint = state.symptom_features.get("chronic_score", 0) * 0.03
    infectious_signal = infectious_base + infectious_hint + state.rag_infectious_score
    chronic_signal = chronic_base + chronic_hint + state.rag_chronic_score
    gap = infectious_signal - chronic_signal
    no_symptom_context = bool(state.symptom_features.get("no_symptom_context"))
    image_trigger_infectious = infectious_base >= 0.58 or (
        infectious_base >= 0.48 and abs(gap) < 0.18
    )
    image_trigger_chronic = chronic_base >= 0.52 or (chronic_base >= 0.43 and abs(gap) < 0.18)
    text_trigger_infectious = state.rag_infectious_score >= 0.10 and not no_symptom_context
    text_trigger_chronic = state.rag_chronic_score >= 0.10 and not no_symptom_context
    if no_symptom_context and normal_base >= 0.70:
        image_trigger_infectious = False
        image_trigger_chronic = False
    trigger_notes: List[str] = []
    if image_trigger_infectious:
        trigger_notes.append("影像触发感染性复核")
    if image_trigger_chronic:
        trigger_notes.append("影像触发慢性异常复核")
    if text_trigger_infectious:
        trigger_notes.append("RAG/症状触发感染性复核")
    if text_trigger_chronic:
        trigger_notes.append("RAG/症状触发慢性异常复核")

    if not trigger_notes and infectious_signal < 0.48 and chronic_signal < 0.45:
        state.route_decision = "low_risk"
        state.route_reason = "通用模型未提示明显高风险异常，先进入综合判断。"
    elif (image_trigger_infectious or text_trigger_infectious) and (
        image_trigger_chronic or text_trigger_chronic
    ):
        state.route_decision = "dual"
        state.route_reason = "影像或RAG文本同时触发两类专病复核，启动双分支复核。"
    elif image_trigger_infectious or text_trigger_infectious or gap >= 0.12:
        state.route_decision = "infectious"
        state.route_reason = "影像或RAG文本提示感染性异常风险，优先进入感染性专病分支。"
    elif image_trigger_chronic or text_trigger_chronic or gap <= -0.12:
        state.route_decision = "chronic"
        state.route_reason = "影像或RAG文本提示慢性异常风险，优先进入慢性异常专病分支。"
    else:
        state.route_decision = "dual"
        state.route_reason = "通用模型方向接近，启动双分支复核。"

    state.agent_trace.append(
        "RouteAgent: "
        f"感染性信号 {infectious_signal:.2f}，慢性信号 {chronic_signal:.2f}；"
        f"触发依据={'; '.join(trigger_notes) if trigger_notes else '未触发专病阈值'}；"
        f"{state.rag_route_note} 路由决策={state.route_decision}。{state.route_reason}"
    )
    return state


def chronic_specialist_agent(state: MedicalState) -> MedicalState:
    signals = state.symptom_features.get("signals", [])
    score = state.image_scores.get("慢性异常", 0.35) * 0.35
    source = "rule-specialist"
    if state.image_path and os.path.exists(state.image_path):
        model_score = run_hf_tb(state.image_path)
        if model_score is not None:
            score += model_score * 0.45
            source = f"huggingface:{HF_TB_REPO}"
    if "盗汗" in signals:
        score += 0.08
    if "消瘦" in signals:
        score += 0.08
    if "咯血" in signals:
        score += 0.10
    if "病程长" in signals:
        score += 0.10
    if "乏力" in signals:
        score += 0.04
    if "急性起病" in signals:
        score -= 0.06
    state.chronic_specialist_score = round(max(0.0, min(0.98, score)), 2)
    state.chronic_specialist_source = source
    state.chronic_specialist_note = (
        "慢性异常评估模块提示存在一定慢性异常风险，需要进一步鉴别。"
        if state.chronic_specialist_score >= 0.55
        else "慢性异常评估模块未发现强慢性异常支持证据。"
    )
    state.agent_trace.append(
        f"ChronicSpecialistAgent: 慢性异常评分 {state.chronic_specialist_score:.2f}，来源 {state.chronic_specialist_source}。"
    )
    return state


def infectious_specialist_agent(state: MedicalState) -> MedicalState:
    signals = state.symptom_features.get("signals", [])
    score = state.image_scores.get("感染性异常", 0.35) * 0.35
    source = "rule-specialist"
    if state.image_path and os.path.exists(state.image_path):
        model_score = run_hf_pneumonia(state.image_path)
        if model_score is not None:
            score += model_score * 0.45
            source = f"huggingface:{HF_PNEUMONIA_REPO}"
    if "发热" in signals:
        score += 0.08
    if "咳嗽" in signals:
        score += 0.05
    if "咳痰" in signals:
        score += 0.08
    if "胸痛" in signals:
        score += 0.04
    if "急性起病" in signals:
        score += 0.08
    if "病程长" in signals:
        score -= 0.05
    state.infectious_specialist_score = round(max(0.0, min(0.98, score)), 2)
    state.infectious_specialist_source = source
    state.infectious_specialist_note = (
        "感染性异常评估模块提示较高风险。"
        if state.infectious_specialist_score >= 0.75
        else "感染性异常评估模块提示存在一定风险，需要结合临床信息。"
    )
    state.agent_trace.append(
        f"InfectiousSpecialistAgent: 感染性异常评分 {state.infectious_specialist_score:.2f}，来源 {state.infectious_specialist_source}。"
    )
    return state


def local_screening_decision(state: MedicalState) -> Dict[str, object]:
    signals = state.symptom_features.get("signals", [])
    infectious_image_score = state.image_scores.get("感染性异常", 0.0)
    chronic_image_score = state.image_scores.get("慢性异常", 0.0)
    normal_image_score = state.image_scores.get("正常倾向", 0.0)
    weak_infectious_symptom = (
        state.symptom_features.get("infectious_score", 0) == 0 and len(signals) <= 1
    )
    strong_text_infectious = (
        state.symptom_features.get("infectious_score", 0) >= 3 or state.rag_infectious_score >= 0.16
    )
    strong_text_chronic = (
        state.symptom_features.get("chronic_score", 0) >= 3 or state.rag_chronic_score >= 0.16
    )
    strong_text_signal = strong_text_infectious or strong_text_chronic
    infectious_total = (
        state.symptom_features.get("infectious_score", 0) * (0.05 if len(signals) <= 1 else 0.10)
        + infectious_image_score * (0.45 if len(signals) <= 1 else 0.35)
        + state.infectious_specialist_score * (0.35 if len(signals) <= 1 else 0.30)
        + state.rag_infectious_score * 0.10
    )
    chronic_total = (
        state.symptom_features.get("chronic_score", 0) * (0.05 if len(signals) <= 1 else 0.10)
        + chronic_image_score * (0.40 if len(signals) <= 1 else 0.30)
        + state.chronic_specialist_score * (0.40 if len(signals) <= 1 else 0.35)
        + state.rag_chronic_score * 0.10
    )
    gap = abs(infectious_total - chronic_total)

    if (
        strong_text_signal
        and normal_image_score >= 0.70
        and infectious_image_score < 0.40
        and chronic_image_score < 0.35
        and state.infectious_specialist_score < 0.45
        and state.chronic_specialist_score < 0.45
    ):
        return {
            "screening_result": "需进一步检查",
            "confidence": 0.66,
            "rationale": "症状/RAG提示存在较强肺部异常线索，但影像主模型和专病复核均偏低风险，属于图文证据不一致场景。",
            "differential": "建议结合体温、血常规、CRP、痰检/TB-DNA及胸部CT复核，避免仅凭症状或单张胸片直接定性。",
        }

    if (
        weak_infectious_symptom
        and 0.50 <= infectious_image_score <= 0.88
        and normal_image_score >= 0.08
        and 0 < state.infectious_specialist_score < 0.40
    ):
        if infectious_image_score < 0.70 and normal_image_score >= 0.25:
            return {
                "screening_result": "正常/低风险",
                "confidence": round(max(0.76, min(0.84, 0.70 + normal_image_score * 0.25)), 2),
                "rationale": "主模型提示轻中度感染性风险，但肺炎专病复核分数较低，且症状未提供感染性支持证据，综合判断暂按低风险随访。",
                "differential": "若后续出现发热、黄痰、胸痛或影像进展，应复查胸片/胸部CT并重新评估感染性肺部异常。",
            }
        return {
            "screening_result": "需进一步检查",
            "confidence": 0.68,
            "rationale": "主模型提示感染性异常，但肺炎专病复核和症状证据支持不足，系统将其标记为影像-专病证据不一致病例。",
            "differential": "建议结合体温、血常规、CRP、病原学检查或胸部CT进一步区分早期感染、影像假阳性及其他肺部改变。",
        }

    if (
        infectious_image_score < 0.50
        and chronic_image_score < 0.35
        and state.infectious_specialist_score < 0.30
        and state.chronic_specialist_score < 0.25
        and len(signals) <= 1
    ):
        return {
            "screening_result": "正常/低风险",
            "confidence": 0.82,
            "rationale": "影像模型与专病模型均未提示明显高风险异常，且当前症状特异性较弱。",
            "differential": "建议结合随访观察，若症状持续或加重再进一步检查。",
        }

    if gap < 0.12:
        return {
            "screening_result": "需进一步检查",
            "confidence": round(0.58 + gap * 0.4, 2),
            "rationale": "症状和影像证据存在交叉，当前更适合作为辅助分诊结果，需进一步检查确认。",
            "differential": "需重点鉴别感染性异常、慢性异常及其他肺部病变。",
        }

    if infectious_total > chronic_total:
        return {
            "screening_result": "感染性肺部异常倾向",
            "confidence": round(min(0.96, 0.60 + min(0.35, gap * 0.5)), 2),
            "rationale": "症状更符合急性感染表现，且感染性异常评估模块给出较高风险。",
            "differential": "需与慢性异常风险、病毒性肺部感染和支气管炎进行鉴别。",
        }

    return {
        "screening_result": "慢性异常风险倾向",
        "confidence": round(min(0.96, 0.60 + min(0.35, gap * 0.5)), 2),
        "rationale": "当前影像和专病模型更偏向慢性异常风险，需进一步检查确认。",
        "differential": "需与感染性病变、肺脓肿及其他慢性肺部病变进行鉴别。",
    }


def apply_consistency_guard(state: MedicalState, source: str) -> MedicalState:
    signals = state.symptom_features.get("signals", [])
    infectious_image_score = state.image_scores.get("感染性异常", 0.0)
    normal_image_score = state.image_scores.get("正常倾向", 0.0)
    weak_infectious_symptom = (
        state.symptom_features.get("infectious_score", 0) == 0 and len(signals) <= 1
    )

    guarded_results = {"感染性肺部异常倾向", "需进一步检查"}
    if not (
        weak_infectious_symptom
        and state.screening_result in guarded_results
        and 0.45 <= infectious_image_score <= 0.88
        and normal_image_score >= 0.08
        and 0 < state.infectious_specialist_score < 0.40
    ):
        return state

    if (
        infectious_image_score < 0.76
        and normal_image_score >= 0.20
        and state.infectious_specialist_score < 0.36
    ):
        state.screening_result = "正常/低风险"
        state.confidence = round(max(0.76, min(0.84, 0.70 + normal_image_score * 0.25)), 2)
        state.rationale = "经一致性复核，主模型仅提示轻中度感染风险，肺炎专病模型和症状证据均不支持高风险感染，暂按低风险随访。"
        state.differential = "若后续出现发热、黄痰、胸痛或影像进展，应复查胸片/胸部CT并重新评估。"
    else:
        state.screening_result = "需进一步检查"
        state.confidence = min(state.confidence, 0.68)
        state.rationale = "经一致性复核，主模型提示感染性异常，但肺炎专病复核和症状证据支持不足，需进一步检查确认。"
        state.differential = "建议结合体温、血常规、CRP、病原学检查或胸部CT进一步区分早期感染、影像假阳性及其他肺部改变。"
    state.agent_trace.append(f"ReviewGuard: {source} 结论触发影像-专病一致性校验，已进行保守修正。")
    return state


def diagnosis_agent(state: MedicalState) -> MedicalState:
    prompt = textwrap.dedent(
        f"""
        请根据以下证据输出肺部异常辅助筛查结果，并严格返回 JSON：
        {{
          "screening_result": "感染性肺部异常倾向/慢性异常风险倾向/需进一步检查/正常低风险",
          "confidence": 0.00,
          "rationale": "一句话说明主要依据",
          "differential": "一句话说明鉴别与分诊建议"
        }}
        症状: {state.symptom}
        症状特征: {state.symptom_features}
        通用胸片模型: {state.image_model_name}, 影像结果: {state.image_result}, 分数: {state.image_scores}
        复核路由: {state.route_decision}, 原因: {state.route_reason}
        慢性异常模块: {state.chronic_specialist_source}, 分数 {state.chronic_specialist_score}, 说明 {state.chronic_specialist_note}
        感染性异常模块: {state.infectious_specialist_source}, 分数 {state.infectious_specialist_score}, 说明 {state.infectious_specialist_note}
        RAG复核触发证据: 感染性={state.rag_infectious_score}, 慢性={state.rag_chronic_score}, 说明={state.rag_route_note}
        RAG知识: {" | ".join(state.retrieved_knowledge)}
        注意: RAG只作为指南证据和复核触发依据，不能单独覆盖影像模型与专病模型。
        """
    ).strip()
    llm_text = call_openai_compatible(prompt)
    if llm_text:
        match = re.search(r"\{.*\}", llm_text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                state.screening_result = str(data.get("screening_result", "需进一步检查")).strip()
                state.confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
                state.rationale = str(data.get("rationale", "")).strip()
                state.differential = str(data.get("differential", "")).strip()
                state.used_llm = True
                state = apply_consistency_guard(state, "LLM")
                state.agent_trace.append(
                    f"DiagnosisAgent: 使用 LLM 输出筛查结论 {state.screening_result}，置信度 {state.confidence:.2f}。"
                )
                return state
            except Exception:
                pass
        state.agent_trace.append(
            "DiagnosisAgent: LLM 已返回内容，但未解析出合规 JSON，改用本地决策器。"
        )
    elif LLM_LAST_ERROR:
        state.agent_trace.append(f"DiagnosisAgent: {LLM_LAST_ERROR}")
    result = local_screening_decision(state)
    state.screening_result = str(result["screening_result"])
    state.confidence = float(result["confidence"])
    state.rationale = str(result["rationale"])
    state.differential = str(result["differential"])
    state = apply_consistency_guard(state, "本地决策器")
    state.agent_trace.append(
        f"DiagnosisAgent: 使用本地决策器输出 {state.screening_result}，置信度 {state.confidence:.2f}。"
    )
    return state


def review_agent(state: MedicalState) -> MedicalState:
    signals = state.symptom_features.get("signals", [])
    notes: List[str] = []
    if state.image_model_name in {"no-image", "missing-file"}:
        state.confidence = min(state.confidence, 0.72)
        notes.append("当前缺少有效影像证据，结论主要依据症状与知识检索，建议补充胸片或CT。")
    if state.screening_result == "感染性肺部异常倾向" and any(
        x in signals for x in ["盗汗", "消瘦", "咯血", "病程长"]
    ):
        state.confidence = max(0.45, round(state.confidence - 0.10, 2))
        notes.append("存在慢性异常线索，降低感染性异常结论置信度。")
    if state.screening_result == "慢性异常风险倾向" and any(
        x in signals for x in ["急性起病", "咳痰", "胸痛"]
    ):
        state.confidence = max(0.45, round(state.confidence - 0.10, 2))
        notes.append("存在急性感染线索，降低慢性异常结论置信度。")
    if state.screening_result == "需进一步检查":
        state.confidence = min(state.confidence, 0.72)
        notes.append("证据冲突较明显，建议进一步检查。")
    if state.screening_result == "正常/低风险":
        state.confidence = max(state.confidence, 0.78)
        notes.append("当前影像与专病模型未提示明显高风险异常。")
    if not notes:
        notes.append("当前筛查结论与症状和影像证据基本一致。")
    state.review_note = " ".join(notes)
    state.agent_trace.append(
        f"ReviewAgent: {state.review_note} 调整后置信度 {state.confidence:.2f}。"
    )
    return state


def check_agent(state: MedicalState) -> MedicalState:
    if state.screening_result == "感染性肺部异常倾向":
        state.check_suggestion = "建议完善血常规、C反应蛋白、降钙素原及胸部CT，必要时做病原学检查。"
    elif state.screening_result == "慢性异常风险倾向":
        state.check_suggestion = "建议完善痰涂片、TB-DNA、T-SPOT 和胸部CT，必要时专科进一步评估。"
    elif state.screening_result == "正常/低风险":
        state.check_suggestion = (
            "建议结合症状变化进行观察，若咳嗽持续、发热或影像异常再进一步复查。"
        )
    else:
        state.check_suggestion = "建议优先补充胸部CT、血常规及相关实验室检查后再做进一步分诊。"
    state.agent_trace.append("CheckAgent: 输出下一步检查建议。")
    return state


def self_feedback_agent(state: MedicalState) -> MedicalState:
    if state.iteration == 1:
        state.retrieved_knowledge.append("补充规则: 若影像证据冲突，则优先参考专病模型与症状病程。")
    elif state.iteration == 2:
        state.retrieved_knowledge.append(
            "补充规则: 若感染性异常与慢性异常评分接近，则输出需进一步检查并强化分诊建议。"
        )
    state.agent_trace.append("SelfFeedbackAgent: 追加保守推理规则，触发下一轮诊断。")
    return state


def should_continue(state: MedicalState) -> bool:
    if state.screening_result == "正常/低风险":
        return False
    return state.confidence < 0.85 and state.iteration < 3


def build_graph_description() -> str:
    return "SymptomAgent -> ImageAgent223 -> KnowledgeAgent -> RouteAgent -> [InfectiousSpecialistAgent / ChronicSpecialistAgent / Dual Review] -> DiagnosisAgent -> ReviewAgent -> CheckAgent -> [低置信度时进入 SelfFeedbackAgent 再回到 DiagnosisAgent]"


def render_report(state: MedicalState) -> str:
    trace = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(state.agent_trace))
    rag = "\n".join(f"{idx}. {item}" for idx, item in enumerate(state.retrieved_knowledge, start=1))
    signals = ", ".join(state.symptom_features.get("signals", [])) or "未提取到高特异性特征"
    mode = "LLM增强模式" if state.used_llm else "本地多Agent模式"
    return textwrap.dedent(
        f"""
        ===== 肺部异常辅助筛查与分诊报告 =====
        运行模式: {mode}
        LangGraph流程: {build_graph_description()}

        症状描述: {state.symptom}
        提取特征: {signals}
        病程信息: {state.symptom_features.get("duration", "未提及")}
        通用影像模型: {state.image_model_name}
        影像分析结论: {state.image_result}
        分支路由: {state.route_decision} / {state.route_reason}
        RAG复核触发: 感染性证据 {state.rag_infectious_score:.2f}，慢性证据 {state.rag_chronic_score:.2f}；{state.rag_route_note}
        慢性异常评估模块: {state.chronic_specialist_source} / 分数 {state.chronic_specialist_score:.2f}
        慢性异常评估意见: {state.chronic_specialist_note}
        感染性异常评估模块: {state.infectious_specialist_source} / 分数 {state.infectious_specialist_score:.2f}
        感染性异常评估意见: {state.infectious_specialist_note}

        筛查结论: {state.screening_result}
        风险置信度: {state.confidence:.2f}
        主要依据: {state.rationale}
        鉴别诊断: {state.differential}
        分诊建议: {state.review_note}
        建议检查: {state.check_suggestion}
        自反馈轮次: {state.iteration}

        RAG召回结果:
        {rag}

        Agent执行轨迹:
        {trace}
        """
    ).strip()
