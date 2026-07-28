"""
Streamlit Web Application
"""
import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime
import time
import os
import sys
import re
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.rag_engine import RAGEngine
from src.core.logger import log
from config.settings import settings

from src.layer_alterator_agent.workflow import generate_layer_alterator_inputs
from src.layer_alterator_agent.vector_writer import (
    load_vector,
    get_polygon_ids,
    resolve_id_column,
)
from src.layer_alterator_agent.matcher import match_urban_type
from src.layer_alterator_agent.reference_loader import (
    load_reference_table,
    get_predictor_values,
)
from src.agent.schemas import AgentStage, AgentState, Intent
from src.agent.intent_router import route_intent
from src.agent.state_machine import (
    clarify_proposal,
    confirm_proposal,
    register_goal,
    register_vector,
    revise_proposal,
    set_proposal,
)
from src.agent.typology_resolver import propose_typology
from src.layer_alterator.service import run_c1_layer_alterator


LAYER_AGENT_SYSTEM_PROMPT = """
You are a Conversational Urban Simulation Assistant for a thesis prototype called
AI-assisted Urban Spatial Simulation System.

Your role:
- Help non-GIS users prepare Layer Alterator simulation inputs.
- Translate natural language planning goals into urban typology choices.
- Guide users to assign target urban types to polygon areas.
- Explain LCZ-based predictor values in simple language.
- Generate structured Layer Alterator inputs only when vector and polygon descriptions are ready.

Important constraints:
- Do not invent numerical predictor values.
- Predictor values must come from the LCZ reference table.
- Do not pretend that the simulation has been run if only input files were generated.
- If the user has not uploaded a vector, you can still answer questions, but you cannot generate Layer Alterator inputs.
- If some polygons do not have target urban types, explain what is missing.
- Keep answers concise and practical.
- Prefer clear guidance over long theoretical explanations.

Available user operations:
- describe an overall goal
- ask GIS / GeoAI / urban simulation questions
- upload a polygon vector
- assign all polygons to one urban type
- assign one polygon to a target type
- ask why a recommendation was made
- generate Layer Alterator input files

Supported urban types:
- Compact midrise
- Compact low-rise
- Open midrise
- Open low-rise
- Large low-rise
- Dense trees
- Scattered trees
- Low Plants
- Bare rock or paved
- Bare Soil or sand
- Water
"""

def build_intent_router_prompt(user_text: str,has_vector: bool,polygon_ids: list,polygon_descriptions: dict):
    return f"""
{LAYER_AGENT_SYSTEM_PROMPT}

You are now acting ONLY as an intent router.

Your task is to classify the user's message into exactly one intent.

Available intents:

1. chat
General conversation, GIS question, GeoAI question, help request, or casual discussion.

2. set_goal
User describes an overall urban simulation goal, such as:
- reducing heat island effect
- increasing green space
- improving cooling
- adding water
- changing urban morphology

3. apply_all
User wants to apply one urban type or transformation to all polygons.

Examples:
- make all polygons dense trees
- all areas should become water
- 全部改成 low plants

4. set_polygon
User wants to set or modify one specific polygon.

Examples:
- Polygon 2 should become water
- change polygon 1 to dense trees
- 第二个区域改成水体

5. ask_explanation
User asks why, how, explanation, reasoning, or interpretation.

Examples:
- why dense trees?
- explain this recommendation
- 为什么推荐这个

6. generate
User wants to generate Layer Alterator input files.

Examples:
- generate
- run simulation input generation
- 创建文件

7. upload_help
User asks about uploading vector files or required data.

Examples:
- what file should I upload
- how to upload geojson
- 支持什么格式

System state:
- has_vector: {has_vector}
- polygon_ids: {polygon_ids}
- polygon_descriptions: {polygon_descriptions}

Important rules:
- If user asks to generate, always return "generate".
- If user mentions a polygon number/area number, prefer "set_polygon".
- If user says all/every/全部/所有 polygons or areas, prefer "apply_all".
- If user asks why/how/explain/为什么/解释, prefer "ask_explanation".
- If message is a normal GIS or GeoAI question, return "chat".
- Do NOT answer the user.
- Do NOT explain.
- Return ONLY valid JSON.

User message:
{user_text}

Return JSON format:
{{
  "intent": "chat",
  "polygon_id": null,
  "target_description": null,
  "confidence": 0.0
}}
"""

def classify_user_intent_with_llm(user_text: str):
    has_vector = st.session_state.get("la_vector_path") is not None
    polygon_ids = st.session_state.get("la_polygon_ids", [])
    polygon_descriptions = st.session_state.get("la_polygon_descriptions", {})

    prompt = build_intent_router_prompt(
        user_text=user_text,
        has_vector=has_vector,
        polygon_ids=polygon_ids,
        polygon_descriptions=polygon_descriptions,
    )

    raw = run_llm_once(prompt)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start == -1 or end <= start:
            raise ValueError("No JSON found in LLM output")

        data = json.loads(raw[start:end])

        intent = data.get("intent", "chat")

        allowed_intents = {
            "chat",
            "set_goal",
            "apply_all",
            "set_polygon",
            "ask_explanation",
            "generate",
            "upload_help",
        }

        if intent not in allowed_intents:
            intent = "chat"

        return {
            "intent": intent,
            "polygon_id": data.get("polygon_id"),
            "target_description": data.get("target_description"),
            "confidence": float(data.get("confidence", 0.0)),
        }

    except Exception:
        return {
            "intent": classify_user_intent(user_text),
            "polygon_id": None,
            "target_description": None,
            "confidence": 0.3,
        }

def normalize_intent_result(user_text: str, intent_result: dict):
    """
    Validate LLM intent result.
    If LLM output is unreasonable, fallback to rule-based intent.
    """
    rule_intent = classify_user_intent(user_text)

    intent = intent_result.get("intent", "chat")
    pid = intent_result.get("polygon_id")
    desc = intent_result.get("target_description")

    # If LLM says set_polygon but no polygon_id, it is probably wrong
    if intent == "set_polygon" and pid is None:
        return {
            "intent": rule_intent,
            "polygon_id": None,
            "target_description": None,
            "confidence": 0.3,
        }

    # If rule clearly detects goal, trust rule more
    if rule_intent == "set_goal":
        return {
            "intent": "set_goal",
            "polygon_id": None,
            "target_description": None,
            "confidence": 0.8,
        }

    # If rule clearly detects generate, trust rule
    if rule_intent == "generate":
        return {
            "intent": "generate",
            "polygon_id": None,
            "target_description": None,
            "confidence": 0.9,
        }

    return intent_result

def recommend_urban_types_from_goal(user_goal: str):
    text = user_goal.lower()

    if any(k in text for k in ["heat", "hot", "cool", "temperature", "热岛", "降温", "凉快"]):
        return {
            "goal": "Urban heat mitigation",
            "recommended_types": ["Dense trees", "Low Plants", "Open low-rise"],
            "explanation": "To reduce urban heat, the system recommends increasing vegetation and reducing impervious or dense built-up surfaces."
        }

    if any(k in text for k in ["green", "vegetation", "tree", "park", "绿化", "树", "公园"]):
        return {
            "goal": "Urban greening",
            "recommended_types": ["Dense trees", "Low Plants"],
            "explanation": "The user goal is related to greening, so vegetation-based urban types are recommended."
        }

    if any(k in text for k in ["water", "pond", "lake", "水", "池塘", "湖"]):
        return {
            "goal": "Blue infrastructure",
            "recommended_types": ["Water"],
            "explanation": "The user goal suggests adding water-related surface types."
        }

    if any(k in text for k in ["parking", "road", "asphalt", "停车", "道路"]):
        return {
            "goal": "Paved surface transformation",
            "recommended_types": ["Bare rock or paved"],
            "explanation": "The user goal suggests a paved or transport-related surface."
        }

    return {
        "goal": "Unclear",
        "recommended_types": [],
        "explanation": "The system cannot confidently identify the transformation goal yet."
    }


def should_use_rag_for_chat(user_text: str):
    text = user_text.lower()

    rag_keywords = [
        "document", "pdf", "paper", "report", "uploaded file",
        "knowledge base", "according to", "in the file",
        "search", "retrieve", "citation",
        "文档", "文件", "论文", "报告", "知识库", "检索",
    ]

    return any(k in text for k in rag_keywords)


def answer_general_chat(user_text: str):
    use_rag = should_use_rag_for_chat(user_text)

    answer = ""

    if use_rag:
        try:
            result = st.session_state.rag_engine.query(user_text)
            if result and result.get("success"):
                answer = result.get("answer", "")
        except Exception:
            answer = ""

    if not answer:
        answer = run_llm_once(f"""
        {LAYER_AGENT_SYSTEM_PROMPT}

        User question:
        {user_text}

        Answer naturally as the assistant.
        """)

    if not answer:
        answer = (
            "I can help with GIS, GeoAI, urban simulation, "
            "Layer Alterator workflows, and spatial planning questions."
        )

    return answer


def get_predictor_summary_for_type(matched_type: str, values: dict):
    key_vars_by_type = {
        "Dense trees": ["F_TV", "TCH", "F_G", "IMD", "F_AC"],
        "Scattered trees": ["F_TV", "TCH", "F_G", "IMD", "F_AC"],
        "Low Plants": ["F_G", "F_TV", "TCH", "IMD", "F_AC"],
        "Water": ["F_W", "IMD", "F_AC", "F_G", "F_TV"],
        "Bare rock or paved": ["F_AC", "IMD", "F_G", "F_TV", "F_W"],
        "Bare Soil or sand": ["F_BS", "IMD", "F_G", "F_TV", "F_AC"],
        "Open midrise": ["BH", "BSF", "IMD", "SVF", "F_AC"],
        "Open low-rise": ["BH", "BSF", "IMD", "SVF", "F_G"],
        "Compact midrise": ["BH", "BSF", "IMD", "SVF", "F_AC"],
        "Compact low-rise": ["BH", "BSF", "IMD", "SVF", "F_AC"],
        "Large low-rise": ["BH", "BSF", "IMD", "F_AC", "SVF"],
    }

    explanations = {
        "Dense trees": "This type is mainly characterized by high tall vegetation and tree canopy.",
        "Scattered trees": "This type represents mixed vegetation with moderate tree coverage.",
        "Low Plants": "This type is mainly characterized by grass or low vegetation.",
        "Water": "This type is mainly characterized by high water fraction.",
        "Bare rock or paved": "This type is mainly characterized by impervious or paved surface.",
        "Bare Soil or sand": "This type is mainly characterized by bare soil or sand fraction.",
        "Open midrise": "This type is mainly characterized by medium building height with relatively open morphology.",
        "Open low-rise": "This type is mainly characterized by low buildings and relatively open space.",
        "Compact midrise": "This type is mainly characterized by compact built-up morphology and medium building height.",
        "Compact low-rise": "This type is mainly characterized by compact low-rise built-up morphology.",
        "Large low-rise": "This type is mainly characterized by large low-rise built-up surfaces.",
    }

    key_vars = key_vars_by_type.get(
        matched_type,
        ["TCH", "IMD", "BH", "BSF", "SVF", "F_AC", "F_G", "F_TV"]
    )

    key_summary = {
        k: values[k]
        for k in key_vars
        if k in values
    }

    full_summary = {
        k: values[k]
        for k in [
            "TCH", "IMD", "BH", "BSF", "SVF",
            "F_AC", "F_S", "F_M", "F_BS", "F_G", "F_TV", "F_W"
        ]
        if k in values
    }

    return key_summary, full_summary, explanations.get(
        matched_type,
        "This type is matched from the LCZ reference predictor table."
    )



def build_scenario_candidates(global_goal: str):
    text = global_goal.lower()

    if any(k in text for k in ["heat", "hot", "cool", "temperature", "热岛", "降温", "凉快"]):
        return [
            {
                "name": "Scenario A - Greening priority",
                "description": "Maximize vegetation coverage to support urban cooling.",
                "types": ["Dense trees", "Low Plants"],
            },
            {
                "name": "Scenario B - Open morphology cooling",
                "description": "Combine vegetation with more open urban morphology.",
                "types": ["Dense trees", "Open low-rise"],
            },
            {
                "name": "Scenario C - Blue-green infrastructure",
                "description": "Combine vegetation and water surfaces for cooling-oriented transformation.",
                "types": ["Dense trees", "Water"],
            },
        ]

    if any(k in text for k in ["green", "vegetation", "tree", "park", "绿化", "树", "公园"]):
        return [
            {
                "name": "Scenario A - Tree-based greening",
                "description": "Prioritize tall vegetation and tree canopy.",
                "types": ["Dense trees", "Scattered trees"],
            },
            {
                "name": "Scenario B - Grassland greening",
                "description": "Prioritize low vegetation and open green space.",
                "types": ["Low Plants", "Dense trees"],
            },
            {
                "name": "Scenario C - Park-like transformation",
                "description": "Create mixed green urban open space.",
                "types": ["Low Plants", "Scattered trees"],
            },
        ]

    if any(k in text for k in ["water", "pond", "lake", "水", "池塘", "湖"]):
        return [
            {
                "name": "Scenario A - Water surface",
                "description": "Prioritize water-covered transformation.",
                "types": ["Water"],
            },
            {
                "name": "Scenario B - Blue-green space",
                "description": "Combine water with vegetation.",
                "types": ["Water", "Dense trees"],
            },
        ]

    return [
        {
            "name": "Scenario A - Greening",
            "description": "A general vegetation-oriented transformation.",
            "types": ["Dense trees", "Low Plants"],
        },
        {
            "name": "Scenario B - Open urban form",
            "description": "A general open morphology transformation.",
            "types": ["Open low-rise", "Open midrise"],
        },
        {
            "name": "Scenario C - Paved surface",
            "description": "A general paved or impervious surface transformation.",
            "types": ["Bare rock or paved"],
        },
    ]




def build_before_after_table(polygon_descriptions, reference_df):
    rows = []

    for pid, desc in polygon_descriptions.items():
        if not str(desc).strip():
            continue

        matched_type = match_urban_type(desc)
        values = get_predictor_values(reference_df, matched_type)
        key_summary, full_summary, explanation = get_predictor_summary_for_type(
            matched_type,
            values
        )

        rows.append({
            "polygon_id": pid,
            "user_description": desc,
            "matched_urban_type": matched_type,
            "key_predictors": ", ".join([f"{k}={v:.3f}" for k, v in key_summary.items()]),
            "explanation": explanation,
        })

    return pd.DataFrame(rows)


def write_simulation_explanation_md(
    output_dir,
    global_goal,
    selected_scenario,
    polygon_descriptions,
    reference_df,
    result,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explanation_path = output_dir / "simulation_explanation.md"

    lines = []
    lines.append("# Simulation Explanation Report\n")
    lines.append("## 1. User Simulation Goal\n")
    lines.append(f"{global_goal}\n")

    lines.append("## 2. Selected Scenario\n")
    if selected_scenario:
        lines.append(f"**Name:** {selected_scenario.get('name', 'N/A')}\n")
        lines.append(f"**Description:** {selected_scenario.get('description', 'N/A')}\n")
        lines.append("**Recommended urban types:**\n")
        for t in selected_scenario.get("types", []):
            lines.append(f"- {t}\n")
    else:
        lines.append("No explicit scenario was selected.\n")

    lines.append("\n## 3. Polygon-level Decisions\n")

    for pid, desc in polygon_descriptions.items():
        matched_type = match_urban_type(desc)
        values = get_predictor_values(reference_df, matched_type)
        key_summary, full_summary, explanation = get_predictor_summary_for_type(
            matched_type,
            values
        )

        lines.append(f"\n### Polygon {pid}\n")
        lines.append(f"- User description: {desc}\n")
        lines.append(f"- Matched urban type: {matched_type}\n")
        lines.append(f"- Explanation: {explanation}\n")

        lines.append("\nKey predictor values:\n")
        for k, v in key_summary.items():
            lines.append(f"- {k}: {v:.3f}\n")

        lines.append("\nFull predictor values:\n")
        for k, v in full_summary.items():
            lines.append(f"- {k}: {v:.3f}\n")

    lines.append("\n## 4. Generated Layer Alterator Inputs\n")
    lines.append(f"- Updated vector: `{result['updated_vector_path']}`\n")
    lines.append(f"- Rules JSON: `{result['rules_path']}`\n")
    lines.append(f"- Simulation config: `{result['config_path']}`\n")

    lines.append("\n## 5. Methodological Note\n")
    lines.append(
        "The system does not freely generate numerical urban parameters. "
        "Urban typology descriptions are matched to reference LCZ-based predictor values, "
        "which are then used to populate the Layer Alterator input vector and rule files.\n"
    )

    explanation_path.write_text("".join(lines), encoding="utf-8")

    return str(explanation_path)

def run_llm_once(prompt: str) -> str:
    """
    Use the existing local LLM streaming interface as one-shot text generation.
    No need to modify rag_engine.py.
    """
    if "rag_engine" not in st.session_state:
        return ""

    if st.session_state.rag_engine is None:
        return ""

    llm = getattr(st.session_state.rag_engine, "llm", None)

    if llm is None:
        return ""

    output = ""

    try:
        for chunk in llm.generate_stream(prompt):
            if isinstance(chunk, dict):
                if chunk.get("type") == "answer":
                    output += chunk.get("content", "")
            else:
                output += str(chunk)

        return output.strip()

    except Exception as e:
        print("LLM intent router failed:", e)
        return ""

def classify_user_intent(user_text: str):
    text = user_text.lower().strip()

    if any(k in text for k in ["generate", "run", "create files", "生成", "运行", "创建文件"]):
        return "generate"

    if any(k in text for k in ["why", "explain", "reason", "为什么", "解释"]):
        return "ask_explanation"

    if any(k in text for k in ["all polygons", "all areas", "全部", "所有"]):
        return "apply_all"

    if re.search(r"polygon\s*\d+", text) or re.search(r"多边形\s*\d+", text):
        return "set_polygon"

    if any(k in text for k in ["heat", "cool", "green", "tree", "water", "park", "热岛", "降温", "绿化", "水体", "公园"]):
        return "set_goal"

    return "unknown"

def should_use_rag_for_chat(user_text: str):
    text = user_text.lower()

    rag_keywords = [
        "document",
        "pdf",
        "paper",
        "report",
        "uploaded file",
        "knowledge base",
        "according to",
        "in the file",
        "search",
        "retrieve",
        "citation",
        "文档",
        "文件",
        "论文",
        "报告",
        "知识库",
        "检索",
    ]

    return any(k in text for k in rag_keywords)

def extract_polygon_id_and_description(user_text: str):
    text = user_text.strip()

    match = re.search(r"polygon\s*(\d+)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"多边形\s*(\d+)", text)

    if not match:
        return None, None

    polygon_id = int(match.group(1))

    desc = re.sub(r"polygon\s*\d+", "", text, flags=re.IGNORECASE)
    desc = re.sub(r"多边形\s*\d+", "", desc)
    for phrase in ["should become", "change to", "改成", "变成", "设置为"]:
        desc = desc.replace(phrase, "")

    desc = desc.strip(" :，。")
    return polygon_id, desc


def extract_apply_all_description(user_text: str):
    text = user_text.strip()

    for phrase in [
        "make all polygons",
        "change all polygons to",
        "all polygons",
        "all areas",
        "全部改成",
        "所有改成",
        "全部变成",
        "所有变成",
    ]:
        text = text.replace(phrase, "")

    return text.strip(" :，。")


def build_polygon_status_text(polygon_descriptions):
    lines = ["Current polygon settings:"]
    for pid, desc in polygon_descriptions.items():
        value = desc if str(desc).strip() else "not defined yet"
        lines.append(f"- Polygon {pid}: {value}")
    return "\n".join(lines)


def explain_current_recommendation(global_goal, polygon_descriptions, reference_df):
    lines = []

    if global_goal:
        rec = recommend_urban_types_from_goal(global_goal)
        lines.append(f"Overall goal: **{global_goal}**")
        lines.append(f"Detected intention: **{rec['goal']}**")
        lines.append(rec["explanation"])
        lines.append("")

    for pid, desc in polygon_descriptions.items():
        if not str(desc).strip():
            continue

        matched_type = match_urban_type(desc)
        values = get_predictor_values(reference_df, matched_type)
        key_summary, full_summary, explanation = get_predictor_summary_for_type(matched_type, values)

        lines.append(f"Polygon {pid}: **{matched_type}**")
        lines.append(explanation)
        lines.append("Key predictors:")
        for k, v in key_summary.items():
            lines.append(f"- {k}: {v:.3f}")
        lines.append("")

    if not lines:
        return "There is no confirmed goal or polygon setting to explain yet."

    return "\n".join(lines)

# Page configuration
st.set_page_config(
    page_title="AI-assisted Urban Spatial Simulation System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_streamlit_style = """
<style>

/* 隐藏右上角 Deploy 和菜单 */
[data-testid="stToolbar"] {
    display: none;
}

/* 隐藏顶部 header */
header {
    visibility: hidden;
}

/* 隐藏 footer */
footer {
    visibility: hidden;
}

/* 隐藏 MainMenu */
#MainMenu {
    visibility: hidden;
}

/* 页面顶部留白缩小 */
.block-container {
    padding-top: 1.5rem;
}

</style>
"""

st.markdown(
    hide_streamlit_style,
    unsafe_allow_html=True
)

# Add custom CSS to improve UI stability
st.markdown("""
<style>
    /* Disable minimal height animation for dataframe container */
    .stDataFrame {
        min-height: 100px;
    }
    /* Simplify scrollbar appearance */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    /* Stabilize sidebar layout */
    [data-testid="stSidebar"] {
        min-width: 300px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'think_mode' not in st.session_state:
    st.session_state.think_mode = False
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None

# Conversation history directory
CONVERSATIONS_DIR = settings.PROJECT_ROOT / "data" / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

def get_conversation_list():
    """Get all conversation history list"""
    conversations = []
    if CONVERSATIONS_DIR.exists():
        for f in sorted(CONVERSATIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                import json
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    conversations.append({
                        "id": f.stem,
                        "title": data.get("title", "Untitled"),
                        "created_at": data.get("created_at", ""),
                        "message_count": len(data.get("messages", []))
                    })
            except:
                pass
    return conversations

def save_conversation(conv_id: str, title: str, messages: list):
    """Save conversation"""
    import json
    from datetime import datetime

    data = {
        "id": conv_id,
        "title": title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": messages
    }

    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_conversation(conv_id: str):
    """Load conversation"""
    import json
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", []), data.get("title", "Untitled")
    return [], "Untitled"

def delete_conversation(conv_id: str):
    """Delete conversation"""
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    if filepath.exists():
        filepath.unlink()

def generate_conversation_id():
    """Generate conversation ID"""
    import uuid
    return str(uuid.uuid4())[:8]

def get_conversation_title(messages):
    """Generate conversation title from first message"""
    if messages and len(messages) > 0:
        first_q = messages[0][0] if isinstance(messages[0], tuple) else messages[0].get("question", "")
        return first_q[:20] + "..." if len(first_q) > 20 else first_q
    return "New Chat"


@st.cache_data(show_spinner=False)
def _get_uploaded_files_info(pdf_dir: str, vector_dir: str, raster_dir: str):
    files_info = []

    pdf_path = Path(pdf_dir)
    if pdf_path.exists():
        for p in sorted(pdf_path.glob("*.pdf"), key=lambda x: x.name.lower()):
            files_info.append({
                "Filename": p.name,
                "Type": "PDF",
                "Size(MB)": f"{p.stat().st_size / 1024 / 1024:.2f}",
                "Uploaded": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })

    vector_path = Path(vector_dir)
    if vector_path.exists():
        for ext in [".shp", ".geojson"]:
            for p in sorted(vector_path.glob(f"*{ext}"), key=lambda x: x.name.lower()):
                files_info.append({
                    "Filename": p.name,
                    "Type": "Vector",
                    "Size(MB)": f"{p.stat().st_size / 1024 / 1024:.2f}",
                    "Uploaded": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })

    raster_path = Path(raster_dir)
    if raster_path.exists():
        for ext in [".tif", ".tiff", ".jp2"]:
            for p in sorted(raster_path.glob(f"*{ext}"), key=lambda x: x.name.lower()):
                files_info.append({
                    "Filename": p.name,
                    "Type": "Raster",
                    "Size(MB)": f"{p.stat().st_size / 1024 / 1024:.2f}",
                    "Uploaded": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })

    return files_info

@st.cache_resource
def load_rag_engine():
    """Load RAG engine with caching"""
    try:
        engine = RAGEngine()
        return engine
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def count_uploaded_files():
    """Count uploaded files"""
    pdf_count = 0
    vector_count = 0
    raster_count = 0

    # Count PDF files
    pdf_dir = settings.PDF_DATA_DIR
    if pdf_dir.exists():
        pdf_count = len(list(pdf_dir.glob("*.pdf")))

    # Count vector files
    vector_dir = settings.VECTOR_DATA_DIR
    if vector_dir.exists():
        for ext in ['.shp', '.geojson', '.gpkg', '.kml']:
            vector_count += len(list(vector_dir.glob(f"*{ext}")))

    # Count raster files
    raster_dir = settings.RASTER_DATA_DIR
    if raster_dir.exists():
        for ext in ['.tif', '.tiff', '.jp2', '.img', '.nc']:
            raster_count += len(list(raster_dir.glob(f"*{ext}")))

    return pdf_count, vector_count, raster_count

def main():
    """Main application function"""



    # ---------- Page state ----------
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Urban Simulation Assistant"

    pages = [
        "Urban Simulation Assistant",
        "Simulation Workspace",
        "Knowledge Explorer",
        "System Configuration",
    ]

    # ---------- Floating top navigation ----------
    top_space1, top_space2 = st.columns([5, 4])

    with top_space2:

        nav1, nav2, nav3, nav4 = st.columns(4)

        nav_map = {
            "Urban Simulation Assistant": "Assistant",
            "Simulation Workspace": "Workspace",
            "Knowledge Explorer": "Knowledge",
            "System Configuration": "System",
        }

        for col, p in zip(
                [nav1, nav2, nav3, nav4],
                pages
        ):

            with col:

                if st.button(
                        nav_map[p],
                        key=f"top_nav_{p}",
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == p else "secondary",
                ):
                    st.session_state.current_page = p
                    st.rerun()

    # ---------- Minimal sidebar ----------

    # ---------- Top navigation ----------

    # ---------- Hero header ----------
    st.markdown("""
    <div style="
    padding: 0.9rem 1.2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef6ff 0%, #f7fbf7 100%);
    border: 1px solid #e5e7eb;
    margin-top: -0.5rem;
    ">

    <h1 style="
    margin-bottom:0.3rem;
    font-size:1.7rem;
    color:#111827;
    ">
    AI-assisted Urban Spatial Simulation System
    </h1>

    <p style="
    margin:0;
    color:#4b5563;
    font-size:0.95rem;
    ">
    Conversational GeoAI Interface for Urban Transformation Simulation
    </p>

    </div>
    """, unsafe_allow_html=True)


    page = st.session_state.current_page
    st.divider()

    # ---------- Initialize RAG engine ----------
    if st.session_state.rag_engine is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Starting System")
            st.markdown("Please wait, loading RAG engine and local models...")
            with st.spinner("Initializing..."):
                engine = load_rag_engine()
                if engine:
                    st.session_state.rag_engine = engine
                    st.toast("System started successfully!", icon="🚀")
                    st.rerun()
                else:
                    st.error("System startup failed, please check configuration.")
        return

    # ---------- Render page ----------
    if page == "Urban Simulation Assistant":
        show_layer_alterator_agent_page()
    elif page == "Simulation Workspace":
        show_file_management_page()
    elif page == "Knowledge Explorer":
        show_search_page()
    elif page == "System Configuration":
        show_settings_page()

def show_chat_page():
    """Q&A Page"""
    st.header("💬 Q&A")

    if "is_answering" not in st.session_state:
        st.session_state.is_answering = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "active_question" not in st.session_state:
        st.session_state.active_question = None
    if "queued_question" not in st.session_state:
        st.session_state.queued_question = None
    if "clear_question_input" not in st.session_state:
        st.session_state.clear_question_input = False
    if "question_input_field" not in st.session_state:
        st.session_state.question_input_field = ""

    if st.session_state.clear_question_input:
        st.session_state.question_input_field = ""
        st.session_state.clear_question_input = False

    if st.session_state.queued_question and not st.session_state.is_answering:
        st.session_state.question_input_field = st.session_state.queued_question
        st.session_state.queued_question = None

    def _clear_chat():
        st.session_state.chat_history = []
        st.session_state.current_conversation_id = None
        st.session_state.selected_file = None
        st.session_state.last_visualized_file = None
        st.session_state.selected_file_idx = 0
        st.session_state.pending_question = None
        st.session_state.active_question = None
        st.session_state.queued_question = None
        st.session_state.clear_question_input = True

    # Think mode toggle
    status_col, toggle_col = st.columns([7, 3], vertical_alignment="center")

    with status_col:
        if st.session_state.think_mode:
            st.markdown("🧠 Thinking mode enabled · Deep reasoning")
        else:
            st.markdown("⚡ Standard mode · Fast response")

    with toggle_col:
        left_pad, switch_col, label_col = st.columns([2, 1, 3], vertical_alignment="center")
        with switch_col:
            new_think_mode = st.toggle(
                "Deep Think",
                value=st.session_state.think_mode,
                key="deep_think_toggle",
                help="Enable thinking model for deeper analysis, but slower response",
                label_visibility="collapsed",
                disabled=st.session_state.is_answering,
            )
        with label_col:
            st.markdown("🧠 Deep Think")

        if new_think_mode != st.session_state.think_mode:
            st.session_state.think_mode = new_think_mode
            if st.session_state.rag_engine:
                with st.spinner(
                    f"{'🧠 Loading thinking model...' if new_think_mode else '⚡ Loading standard model...'}"
                ):
                    success = st.session_state.rag_engine.switch_model(new_think_mode)
                    if success:
                        st.success(f"Switched to {'thinking mode' if new_think_mode else 'standard mode'}")
                    else:
                        st.error("Model switch failed")
                st.rerun()

    # File selector for GIS queries
    st.markdown("---")

    # Load uploaded file list
    uploaded_files = []

    # Collect vector files
    if settings.VECTOR_DATA_DIR.exists():
        for ext in ['.shp', '.geojson', '.gpkg']:
            for f in settings.VECTOR_DATA_DIR.glob(f"*{ext}"):
                uploaded_files.append({"name": f.name, "path": str(f), "type": "Vector"})

    # Collect raster files
    if settings.RASTER_DATA_DIR.exists():
        for ext in ['.tif', '.tiff', '.jp2']:
            for f in settings.RASTER_DATA_DIR.glob(f"*{ext}"):
                uploaded_files.append({"name": f.name, "path": str(f), "type": "Raster"})

    # Collect PDF files
    if settings.PDF_DATA_DIR.exists():
        for f in settings.PDF_DATA_DIR.glob("*.pdf"):
            uploaded_files.append({"name": f.name, "path": str(f), "type": "PDF"})

    # File selection
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if "selected_file_idx" not in st.session_state:
        st.session_state.selected_file_idx = 0

    st.markdown("##### 🎯 Select file to query")
    file_options = ["No specific file (search all)"] + [f"📄 {f['name']} ({f['type']})" for f in uploaded_files]
    selected_idx = st.selectbox(
        "Select file to query",
        range(len(file_options)),
        format_func=lambda x: file_options[x],
        help="Select a file to query, AI will answer based on its content",
        label_visibility="collapsed",
        disabled=st.session_state.is_answering,
        key="selected_file_idx",
    )

    if selected_idx > 0:
        st.session_state.selected_file = uploaded_files[selected_idx - 1]
        st.info(f"Selected: **{st.session_state.selected_file['name']}**")
    else:
        st.session_state.selected_file = None

    def _prefill_question(q: str):
        if st.session_state.is_answering:
            st.session_state.queued_question = q
            return
        st.session_state.question_input_field = q

    # Q&A examples (click to fill input without sending)
    with st.expander("💡 Quick Question Examples", expanded=False):
        st.caption("📄 **Document Questions:**")
        doc_questions = [
            "What is the definition of Spatial Data Science?",
            "What is vector data in Python using Shapely?",
            "What are the main spatial indexing algorithms?"
        ]

        for i, q in enumerate(doc_questions):
            st.button(
                f"💬 {q}",
                key=f"doc_q_{i}",
                on_click=_prefill_question,
                args=(q,),
                disabled=st.session_state.is_answering,
            )

        st.caption("🗺️ **GIS File Questions:**")
        gis_questions = [
            "What is the coordinate system of this file?",
            "What type of file is this?",
            "What is the spatial extent of this data?"
        ]

        for i, q in enumerate(gis_questions):
            st.button(
                f"💬 {q}",
                key=f"gis_q_{i}",
                on_click=_prefill_question,
                args=(q,),
                disabled=st.session_state.is_answering,
            )

    # Chat history container
    chat_container = st.container()

    # Render chat history
    with chat_container:
        for i, item in enumerate(st.session_state.chat_history):
            # Support both old (question, answer) and new (question, answer, sources) format
            if len(item) == 3:
                question, answer, sources = item
            else:
                question, answer = item
                sources = []
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    with st.expander("📚 Related Documents", expanded=False):
                        for j, doc in enumerate(sources):
                            st.write(f"**Document {j+1}:**")
                            st.write(doc.get('content', ''))
                            st.divider()

    # Question input area
    st.markdown("---")
    st.markdown("##### 💬 Enter Your Question")

    with st.form("question_form", clear_on_submit=False):
        col_input, col_btn = st.columns([6, 1], vertical_alignment="center")
        with col_input:
            question_input = st.text_input(
                "Question input",
                placeholder="Enter your question here, then click Send...",
                label_visibility="collapsed",
                key="question_input_field",
                disabled=st.session_state.is_answering,
            )
        with col_btn:
            send_clicked = st.form_submit_button(
                "🚀 Send",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_answering,
            )

    if send_clicked and (not st.session_state.is_answering) and question_input.strip():
        st.session_state.pending_question = question_input.strip()
        st.session_state.active_question = None
        st.session_state.is_answering = True
        st.session_state.clear_question_input = True
        st.rerun()

    if st.session_state.is_answering and st.session_state.active_question is None:
        if st.session_state.pending_question:
            st.session_state.active_question = st.session_state.pending_question
            st.session_state.pending_question = None
        else:
            st.session_state.is_answering = False

    question = st.session_state.active_question if st.session_state.is_answering else None

    if question:
        # Append user question to history
        with chat_container:
            with st.chat_message("user"):
                st.write(question)

        # Stream-answer the question
        with chat_container:
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                answer_placeholder = st.empty()
                source_expander = None

                thinking_content = ""
                answer_content = ""
                source_docs = []
                is_thinking = False

                st.session_state.is_answering = True

                try:
                    # Build file filter if a file is selected
                    file_filter = None
                    if st.session_state.selected_file:
                        file_filter = st.session_state.selected_file.get("path")

                    for chunk in st.session_state.rag_engine.query_stream(question, file_filter=file_filter):
                        chunk_type = chunk.get("type", "")
                        content = chunk.get("content", "")

                        if chunk_type == "source":
                            source_docs = content
                        elif chunk_type == "thinking_start":
                            is_thinking = True
                            thinking_placeholder.info("🧠 Thinking...")
                        elif chunk_type == "thinking":
                            thinking_content += content
                            thinking_placeholder.markdown(f"🧠 **Thinking...**\n\n{thinking_content}")
                        elif chunk_type == "thinking_end":
                            is_thinking = False
                            # Collapse thinking content
                            with thinking_placeholder.container():
                                with st.expander("🧠 View Thinking Process", expanded=False):
                                    st.markdown(thinking_content)
                        elif chunk_type == "answer":
                            answer_content += content
                            answer_placeholder.markdown(answer_content)
                        elif chunk_type == "error":
                            st.error(f"Error: {content}")

                    # Show related documents during streaming
                    if source_docs:
                        with st.expander("📚 Related Documents", expanded=False):
                            for i, doc in enumerate(source_docs):
                                st.write(f"**Document {i+1}:**")
                                st.write(doc.get('content', ''))
                                st.divider()

                    # Add final answer to chat history (now including source_docs)
                    if answer_content:
                        full_answer = answer_content
                        if thinking_content:
                            full_answer = f"[Thinking process collapsed]\n\n{answer_content}"
                        # Store as (question, answer, sources) tuple
                        st.session_state.chat_history.append((question, full_answer, source_docs))

                        # Auto-save conversation
                        if st.session_state.current_conversation_id is None:
                            st.session_state.current_conversation_id = generate_conversation_id()

                        title = get_conversation_title(st.session_state.chat_history)
                        save_conversation(
                            st.session_state.current_conversation_id,
                            title,
                            st.session_state.chat_history
                        )

                except Exception as e:
                    st.error(f"Error processing question: {e}")
                finally:
                    st.session_state.is_answering = False
                    st.session_state.active_question = None
                    st.rerun()

    # Clear chat history button
    st.button(
        "🗑️ Clear Chat History",
        disabled=st.session_state.is_answering,
        on_click=_clear_chat,
    )

    # GIS file visualization - auto open when a file is selected
    if (
        (not st.session_state.is_answering)
        and st.session_state.selected_file
        and st.session_state.selected_file.get("type") in ["Vector", "Raster"]
    ):
        # Detect newly selected file
        current_file = st.session_state.selected_file.get("path")
        last_visualized = st.session_state.get("last_visualized_file")

        if current_file != last_visualized:
            st.session_state.last_visualized_file = current_file
            show_gis_visualization_dialog()

        # Also keep manual button for re-opening visualization
        if st.button("🗺️ View Visualization Again", type="secondary", disabled=st.session_state.is_answering):
            show_gis_visualization_dialog()


@st.dialog("🗺️ GIS File Visualization", width="large")
def show_gis_visualization_dialog():
    """GIS file visualization dialog"""
    if not st.session_state.selected_file:
        st.warning("Please select a GIS file first")
        return

    file_path = st.session_state.selected_file["path"]
    file_type = st.session_state.selected_file["type"]
    file_name = st.session_state.selected_file["name"]

    st.subheader(f"📄 {file_name}")

    try:
        if file_type == "Vector":
            import geopandas as gpd
            import folium
            import re

            with st.spinner("🗺️ Loading vector layer..."):
                gdf = gpd.read_file(file_path)

                if gdf.crs and gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")

                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                center_lat = (bounds[1] + bounds[3]) / 2
                center_lon = (bounds[0] + bounds[2]) / 2

                m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

                gdf_copy = gdf.copy()
                for col in gdf_copy.columns:
                    if col != 'geometry':
                        if gdf_copy[col].dtype == 'datetime64[ns]' or 'datetime' in str(gdf_copy[col].dtype):
                            gdf_copy[col] = gdf_copy[col].astype(str)

                geojson_data = gdf_copy.to_json()
                folium.GeoJson(
                    geojson_data,
                    name="data"
                ).add_to(m)

                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

            from streamlit.components.v1 import html
            map_html = m._repr_html_()
            map_var = None
            tile_var = None
            m_map = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map", map_html)
            if m_map:
                map_var = m_map.group(1)
            m_tile = re.search(r"var\s+(tile_layer_[A-Za-z0-9_]+)\s*=\s*L\.tileLayer", map_html)
            if m_tile:
                tile_var = m_tile.group(1)

            if map_var:
                overlay_html = f"""
<style>
.gis-rag-map-overlay {{
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.88);
  z-index: 9999;
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}
.gis-rag-map-overlay .box {{
  padding: 10px 14px;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 10px;
  background: white;
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}}
</style>
<script>
(function() {{
  const map = window.{map_var};
  if (!map || !map.getContainer) return;
  const container = map.getContainer();
  container.style.position = 'relative';
  const overlay = document.createElement('div');
  overlay.className = 'gis-rag-map-overlay';
  overlay.innerHTML = '<div class="box">🗺️ Loading map layers...</div>';
  container.appendChild(overlay);
  function hide() {{
    if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
  }}
  const tile = {(f"window.{tile_var}" if tile_var else "null")};
  if (tile && tile.on) {{
    tile.on('load', hide);
    tile.on('tileerror', hide);
  }}
  setTimeout(hide, 10000);
}})();
</script>
"""
                map_html = map_html + overlay_html

            html(map_html, height=450)

            # Show attribute table preview
            st.caption(f"📊 Attribute Table Preview ({len(gdf)} features)")
            display_df = gdf.drop(columns=['geometry']).head(10)
            for col in display_df.columns:
                if display_df[col].dtype == 'datetime64[ns]':
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df, width="stretch")

        elif file_type == "Raster":
            import rasterio
            import numpy as np
            import matplotlib.pyplot as plt

            with st.spinner("🛰️ Loading raster preview..."):
                src = rasterio.open(file_path)
            with src:
                # Display metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Size:** {src.width} × {src.height} pixels")
                    st.write(f"**Bands:** {src.count}")
                with col2:
                    st.write(f"**CRS:** {src.crs}")
                    st.write(f"**Data Type:** {src.dtypes[0]}")

                if src.bounds:
                    st.write(f"**Spatial Extent:** ({src.bounds.left:.4f}, {src.bounds.bottom:.4f}) to ({src.bounds.right:.4f}, {src.bounds.top:.4f})")

                # Read and display image preview
                st.caption("🖼️ Image Preview")

                # Downsample for performance
                scale_factor = max(1, max(src.width, src.height) // 500)
                out_shape = (src.count, src.height // scale_factor, src.width // scale_factor)

                if src.count >= 3:
                    # RGB display
                    data = src.read([1, 2, 3], out_shape=(3, out_shape[1], out_shape[2]))
                    data = np.moveaxis(data, 0, -1)
                    if data.max() > 1:
                        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                    st.image(data, caption="RGB Preview", width="stretch")
                else:
                    # Single band grayscale display
                    data = src.read(1, out_shape=(out_shape[1], out_shape[2]))
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(data, cmap='viridis')
                    plt.colorbar(im, ax=ax, label='Value')
                    ax.set_title('Band 1')
                    st.pyplot(fig)
                    plt.close()

    except ImportError as e:
        st.warning(f"Additional dependencies required: {e}")
        st.code("pip install folium geopandas rasterio matplotlib", language="bash")
    except Exception as e:
        st.error(f"Visualization failed: {e}")

def show_vector_map(gdf, matched_type_column=None):
    """
    Display vector polygons with polygon labels and optional simulation result colors.
    Click interaction is disabled to avoid Leaflet focus rectangle.
    Hover tooltip and hover highlight are preserved.
    """
    try:
        import folium
        from streamlit.components.v1 import html

        gdf_map = gdf.copy()

        if "polygon_id" not in gdf_map.columns:
            gdf_map["polygon_id"] = range(1, len(gdf_map) + 1)

        if gdf_map.crs and str(gdf_map.crs) != "EPSG:4326":
            gdf_map = gdf_map.to_crs("EPSG:4326")

        bounds = gdf_map.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="OpenStreetMap"
        )

        color_map = {
            "Dense trees": "#2ca25f",
            "Scattered trees": "#99d8c9",
            "Low Plants": "#a1d99b",
            "Water": "#3182bd",
            "Bare rock or paved": "#969696",
            "Bare Soil or sand": "#d9b38c",
            "Compact midrise": "#de2d26",
            "Compact low-rise": "#fc9272",
            "Open midrise": "#fd8d3c",
            "Open low-rise": "#fdae6b",
            "Large low-rise": "#756bb1",
        }

        def style_function(feature):
            props = feature.get("properties", {})
            matched_type = props.get(matched_type_column) if matched_type_column else None
            fill_color = color_map.get(matched_type, "#3388ff")

            return {
                "fillColor": fill_color,
                "color": "#222222",
                "weight": 2,
                "fillOpacity": 0.45,
            }

        tooltip_fields = [
            col for col in gdf_map.columns
            if col != "geometry"
        ][:8]

        folium.GeoJson(
            gdf_map.to_json(),
            name="Polygons",
            style_function=style_function,
            highlight_function=lambda feature: {
                "color": "#ffcc00",
                "weight": 5,
                "fillOpacity": 0.65,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_fields,
                localize=True,
            ),
            popup=None,
        ).add_to(m)

        # Add polygon labels
        for _, row in gdf_map.iterrows():
            point = row.geometry.representative_point()
            polygon_id = row["polygon_id"]

            folium.Marker(
                location=[point.y, point.x],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size: 13px;
                        font-weight: bold;
                        color: white;
                        background-color: #1f2937;
                        border-radius: 12px;
                        padding: 3px 7px;
                        border: 1px solid white;
                    ">
                        Polygon {polygon_id}
                    </div>
                    """
                ),
            ).add_to(m)

        m.fit_bounds([
            [bounds[1], bounds[0]],
            [bounds[3], bounds[2]]
        ])

        map_html = m._repr_html_()

        map_html += """
        <style>
        *:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        .leaflet-container *:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        .leaflet-interactive:focus,
        path.leaflet-interactive:focus,
        svg path:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        </style>

        <script>
        setTimeout(function() {
            document.querySelectorAll('.leaflet-interactive').forEach(function(el) {
                el.removeAttribute('tabindex');

                el.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (document.activeElement) {
                        document.activeElement.blur();
                    }
                    return false;
                }, true);

                el.addEventListener('mousedown', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (document.activeElement) {
                        document.activeElement.blur();
                    }
                    return false;
                }, true);
            });
        }, 300);
        </script>
        """

        html(map_html, height=450)

    except Exception as e:
        st.warning(f"Map preview failed: {e}")

def show_layer_alterator_agent_page():
    """
    V3.2 State-machine conversational Layer Alterator Agent.
    """



    reference_table_path = settings.PROJECT_ROOT / "data" / "reference" / "ref_predictor_values_mean.csv"

    if not reference_table_path.exists():
        st.error(f"Reference table not found: {reference_table_path}")
        st.stop()

    reference_df = load_reference_table(str(reference_table_path))

    # ---------- Session state ----------
    defaults = {
        "la_stage": "waiting_for_vector",
        "la_chat_history": [],
        "la_vector_path": None,
        "la_vector_name": None,
        "la_gdf": None,
        "la_polygon_ids": [],
        "la_id_column": None,
        "la_global_goal": "",
        "la_polygon_descriptions": {},
        "la_current_polygon_index": 0,
        "la_generation_result": None,
        "la_scenario_candidates": [],
        "la_selected_scenario": None,
        "la_last_recommendation": None,
        "la_pending_user_message": None,
        "la_processing": False,
        "la_agent_state": AgentState(),
        "la_layer_alterator_result": None,
        "la_ucp_folder": "",
        "la_fractions_folder": "",
        "la_pending_batch": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    def add_message(role, content=None, msg_type="text"):
        st.session_state.la_chat_history.append({
            "role": role,
            "content": content,
            "type": msg_type,
        })

    def reset_agent():
        for key in defaults.keys():
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()





    def process_user_message(user_text: str):
        user_text = user_text.strip()
        if not user_text:
            return

        agent_state = st.session_state.la_agent_state
        safe_intent = route_intent(user_text)

        if st.session_state.la_pending_batch:
            batch = st.session_state.la_pending_batch
            if safe_intent.intent == Intent.CONFIRM:
                for polygon_id in st.session_state.la_polygon_ids:
                    st.session_state.la_polygon_descriptions[polygon_id] = batch["urban_type"]
                    agent_state.confirmed_decisions[polygon_id] = batch["urban_type"]
                agent_state.stage = AgentStage.READY_TO_GENERATE
                st.session_state.la_pending_batch = None
                st.session_state.la_before_after_df = build_before_after_table(
                    st.session_state.la_polygon_descriptions,
                    reference_df,
                )
                add_message("assistant", f"Confirmed: all polygons → **{batch['urban_type']}**.")
                return
            if safe_intent.intent == Intent.REVISE:
                st.session_state.la_pending_batch = None
                add_message("assistant", "The batch proposal was discarded.")
                return
            add_message("assistant", "A batch proposal is waiting. Type `Confirm` or `Revise`.")
            return

        if agent_state.stage == AgentStage.WAITING_FOR_CLARIFICATION:
            proposal = agent_state.pending_proposal
            selected_type = next(
                (item for item in proposal.candidate_types if item.lower() in user_text.lower()),
                None,
            )
            if not selected_type:
                add_message("assistant", proposal.clarification_question)
                return
            values = get_predictor_values(reference_df, selected_type)
            clarify_proposal(agent_state, selected_type, values)
            add_message(
                "assistant",
                (
                    f"Proposed type for Polygon {proposal.polygon_id}: **{selected_type}**\n\n"
                    "The numerical values come from the LCZ reference table. "
                    "Type `Confirm` to accept or `Revise` to choose again."
                ),
            )
            return

        if agent_state.stage == AgentStage.WAITING_FOR_CONFIRMATION:
            proposal = agent_state.pending_proposal
            if safe_intent.intent == Intent.CONFIRM:
                confirm_proposal(agent_state)
                st.session_state.la_polygon_descriptions[proposal.polygon_id] = proposal.recommended_type
                st.session_state.la_before_after_df = build_before_after_table(
                    st.session_state.la_polygon_descriptions,
                    reference_df,
                )
                add_message(
                    "assistant",
                    (
                        f"Confirmed: Polygon {proposal.polygon_id} → "
                        f"**{proposal.recommended_type}**.\n\n"
                        f"{build_polygon_status_text(st.session_state.la_polygon_descriptions)}"
                    ),
                )
                return
            if safe_intent.intent == Intent.REVISE:
                revise_proposal(agent_state)
                add_message("assistant", f"Proposal for Polygon {proposal.polygon_id} was discarded. Please describe it again.")
                return
            add_message("assistant", "A proposal is waiting for your decision. Type `Confirm` or `Revise`.")
            return

        intent_result = classify_user_intent_with_llm(user_text)
        intent_result = normalize_intent_result(user_text, intent_result)
        intent = intent_result["intent"]

        # 1. Set global goal
        if intent == "set_goal":
            st.session_state.la_global_goal = user_text
            register_goal(agent_state, user_text)

            recommendation = recommend_urban_types_from_goal(user_text)
            scenarios = build_scenario_candidates(user_text)

            st.session_state.la_last_recommendation = recommendation
            st.session_state.la_scenario_candidates = scenarios

            reply = (
                f"I understand your overall simulation goal as: **{recommendation['goal']}**.\n\n"
                f"{recommendation['explanation']}\n\n"
            )

            if recommendation["recommended_types"]:
                reply += "Recommended urban types:\n"
                for t in recommendation["recommended_types"]:
                    reply += f"- {t}\n"

            reply += "\nCandidate scenarios:\n"
            for i, scenario in enumerate(scenarios, start=1):
                reply += f"\n**{i}. {scenario['name']}**\n"
                reply += f"{scenario['description']}\n"
                reply += "Urban types: " + ", ".join(scenario["types"]) + "\n"

            reply += (
                "\nYou can continue freely. For example:\n"
                "- `Make all polygons Dense trees`\n"
                f"- `Polygon {st.session_state.la_polygon_ids[0]} should become Water`\n"
                "- `Why did you recommend this?`\n"
                "- `Generate`"
            )

            add_message("assistant", reply)
            st.session_state.la_stage = "free_chat"
            return

        # 2. Apply one type to all polygons
        if intent == "apply_all":
            desc = extract_apply_all_description(user_text)

            if not desc:
                add_message(
                    "assistant",
                    "I understand you want to apply one transformation to all polygons, but I could not detect the target urban type."
                )
                return

            try:
                matched_type = match_urban_type(desc)
                get_predictor_values(reference_df, matched_type)
            except Exception as e:
                add_message(
                    "assistant",
                    (
                        f"I could not match **{desc}** to a supported urban type, "
                        f"so no polygon was changed.\n\nDetails: {e}"
                    ),
                )
                return

            st.session_state.la_pending_batch = {
                "description": desc,
                "urban_type": matched_type,
            }
            add_message(
                "assistant",
                (
                    f"Proposed type for all polygons: **{matched_type}**.\n\n"
                    "Type `Confirm` to accept or `Revise` to cancel."
                )
            )
            return

        # 3. Set or modify one polygon
        if intent == "set_polygon":
            # Prefer the user's original wording. The LLM router may paraphrase
            # "a park" and accidentally remove the keyword needed for
            # clarification.
            parsed_pid, parsed_desc = extract_polygon_id_and_description(user_text)
            pid = parsed_pid if parsed_pid is not None else intent_result.get("polygon_id")
            desc = parsed_desc or intent_result.get("target_description")

            if pid is not None:
                try:
                    pid = int(pid)
                except Exception:
                    pid = None

            if pid is None or pid not in st.session_state.la_polygon_ids:
                add_message(
                    "assistant",
                    (
                        "I could not identify a valid polygon ID. "
                        f"Please write something like: `Polygon {st.session_state.la_polygon_ids[0]} "
                        "should become Water`."
                    ),
                )
                return

            if not desc:
                add_message(
                    "assistant",
                    f"I detected Polygon {pid}, but I could not detect the target urban type."
                )
                return

            try:
                proposal = propose_typology(pid, desc)
                set_proposal(agent_state, proposal)
                if proposal.needs_clarification:
                    add_message("assistant", proposal.clarification_question)
                    return

                values = get_predictor_values(reference_df, proposal.recommended_type)
                proposal.predictor_values = values
                key_summary, _, explanation = get_predictor_summary_for_type(
                    proposal.recommended_type,
                    values,
                )
                reply = (
                    f"Proposed type for Polygon {pid}: **{proposal.recommended_type}**\n\n"
                    f"Confidence: **{proposal.confidence:.2f}**\n\n"
                    f"{explanation}\n\nKey predictor values:\n"
                )
                for key, value in key_summary.items():
                    reply += f"- {key}: {value:.3f}\n"
                reply += "\nType `Confirm` to accept or `Revise` to choose again."
                add_message("assistant", reply)

            except Exception as e:
                add_message(
                    "assistant",
                    (
                        f"I could not match **{desc}** to a supported urban type, "
                        f"so Polygon {pid} was not changed.\n\nDetails: {e}"
                    ),
                )

            st.session_state.la_stage = "free_chat"
            return

        # 4. Ask explanation
        if intent == "ask_explanation":
            explanation = explain_current_recommendation(
                st.session_state.la_global_goal,
                st.session_state.la_polygon_descriptions,
                reference_df,
            )
            add_message("assistant", explanation)
            st.session_state.la_stage = "free_chat"
            return

        # 5. Generate files
        if intent == "generate":
            if not st.session_state.la_vector_path:
                add_message(
                    "assistant",
                    "I can generate the Layer Alterator files only after a polygon vector is uploaded. Please upload a GeoJSON, GPKG, or Shapefile first."
                )
                return

            missing = [
                pid for pid, desc in st.session_state.la_polygon_descriptions.items()
                if not str(desc).strip()
            ]

            if missing:
                add_message(
                    "assistant",
                    (
                        f"I cannot generate yet because these polygons are missing target urban types: {missing}.\n\n"
                        "You can say for example:\n"
                        "- `Make all polygons Dense trees`\n"
                        "- `Polygon 1 should become Dense trees`\n"
                        "- `Polygon 2 should become Water`"
                    )
                )
                return

            if agent_state.stage != AgentStage.READY_TO_GENERATE:
                add_message(
                    "assistant",
                    "All polygon decisions must be explicitly confirmed before generation.",
                )
                return

            try:
                result = generate_layer_alterator_inputs(
                    vector_path=st.session_state.la_vector_path,
                    reference_table_path=str(reference_table_path),
                    polygon_descriptions=st.session_state.la_polygon_descriptions,
                    id_column=st.session_state.la_id_column,
                    output_dir=str(settings.PROJECT_ROOT / "data" / "layer_alterator_outputs"),
                )

                explanation_path = write_simulation_explanation_md(
                    output_dir=str(settings.PROJECT_ROOT / "data" / "layer_alterator_outputs"),
                    global_goal=st.session_state.la_global_goal,
                    selected_scenario=st.session_state.la_selected_scenario,
                    polygon_descriptions=st.session_state.la_polygon_descriptions,
                    reference_df=reference_df,
                    result=result,
                )

                result["explanation_path"] = explanation_path

                st.session_state.la_generation_result = result
                st.session_state.la_stage = "generated"
                agent_state.stage = AgentStage.INPUTS_GENERATED
                agent_state.outputs = {
                    key: value for key, value in result.items()
                    if key.endswith("_path") and isinstance(value, str)
                }

                add_message("assistant", msg_type="generation_result")

            except Exception as e:
                add_message("assistant", f"Generation failed: {e}")

            return

        if intent == "chat":
            answer = answer_general_chat(user_text)
            add_message("assistant", answer)
            return

        if intent == "upload_help":
            add_message(
                "assistant",
                (
                    "Please upload a polygon vector file from the left panel. "
                    "Supported formats are GeoJSON, GPKG, and SHP. "
                    "For Shapefile, all companion files such as .shx, .dbf, and .prj may be required."
                )
            )
            return

        # unknown fallback
        answer = answer_general_chat(user_text)
        add_message("assistant", answer)

        st.session_state.la_stage = "free_chat"
        return

    # ---------- Layout ----------
    context_col, chat_col, preview_col = st.columns([1.05, 2.2, 1.35], gap="medium")

    # ---------- Left panel ----------
    with context_col:
        st.subheader("Simulation Context")
        st.caption("Upload data and monitor the current simulation workflow.")

        # =========================================================
        # Input Vector
        # =========================================================
        with st.container(border=True):
            st.markdown("##### Input Vector")

            uploaded_vector = st.file_uploader(
                "Upload polygon vector",
                type=["geojson", "gpkg", "shp"],
                help="Recommended: GeoJSON or GPKG.",
                key="layer_agent_vector_upload_v32",
            )

        if (
                uploaded_vector is not None
                and st.session_state.la_vector_name != uploaded_vector.name
        ):
            vector_save_dir = settings.VECTOR_DATA_DIR
            vector_save_dir.mkdir(parents=True, exist_ok=True)

            vector_path = vector_save_dir / uploaded_vector.name

            with open(vector_path, "wb") as f:
                f.write(uploaded_vector.getbuffer())

            try:
                gdf = load_vector(str(vector_path))

                if gdf.empty:
                    st.error("The uploaded vector contains no features.")
                    return

                id_column = resolve_id_column(gdf)
                polygon_ids = get_polygon_ids(gdf, id_column)

                st.session_state.la_vector_path = str(vector_path)
                st.session_state.la_vector_name = uploaded_vector.name
                st.session_state.la_gdf = gdf
                st.session_state.la_polygon_ids = polygon_ids
                st.session_state.la_id_column = id_column
                st.session_state.la_polygon_descriptions = {
                    pid: "" for pid in polygon_ids
                }

                st.session_state.la_global_goal = ""
                st.session_state.la_current_polygon_index = 0
                st.session_state.la_generation_result = None
                st.session_state.la_stage = "waiting_for_goal"
                st.session_state.la_chat_history = []
                register_vector(
                    st.session_state.la_agent_state,
                    str(vector_path),
                    id_column,
                    polygon_ids,
                )

                add_message(
                    "assistant",
                    "Vector uploaded successfully. "
                    f"I detected **{len(polygon_ids)} polygon(s)** "
                    f"using **{id_column}** as the zone ID."
                )

                add_message("assistant", msg_type="vector_preview")

                add_message(
                    "assistant",
                    "Please describe the overall simulation goal. "
                    "For example: "
                    "*I want to reduce urban heat island effect by adding more vegetation.*"
                )

                st.rerun()

            except Exception as e:
                st.error(f"Failed to read vector: {e}")
                return

        # =========================================================
        # Workflow Progress
        # =========================================================
        with st.container(border=True):

            st.markdown("##### Workflow Progress")

            has_vector = (
                    st.session_state.la_vector_path is not None
            )

            has_goal = bool(
                st.session_state.la_global_goal
            )

            total = len(
                st.session_state.la_polygon_ids
            )

            completed = len([
                pid
                for pid, desc in st.session_state.la_polygon_descriptions.items()
                if str(desc).strip()
            ])

            has_all_polygons = (
                    total > 0
                    and completed == total
            )

            has_outputs = (
                    st.session_state.la_generation_result
                    is not None
            )

            progress_steps = [
                has_vector,
                has_goal,
                has_all_polygons,
                has_outputs,
            ]

            progress_value = (
                    sum(progress_steps)
                    / len(progress_steps)
            )

            st.progress(progress_value)

            st.caption(
                f"{int(progress_value * 100)}% completed"
            )

            st.write(
                "Input ready"
                if has_vector
                else "Input pending"
            )

            st.write(
                "Goal defined"
                if has_goal
                else "Goal pending"
            )

            st.write(
                f"Polygon decisions completed ({completed}/{total})"
                if has_all_polygons
                else f"Polygon decisions pending ({completed}/{total})"
            )

            st.write(
                "Outputs generated"
                if has_outputs
                else "Outputs pending"
            )

        # =========================================================
        # Current Context
        # =========================================================
        with st.container(border=True):

            st.markdown("##### Current Context")

            if has_vector:
                st.write(
                    f"**Vector:** {st.session_state.la_vector_name}"
                )

                st.write(
                    f"**Polygons:** "
                    f"{len(st.session_state.la_polygon_ids)}"
                )

            else:
                st.info(
                    "No vector uploaded yet."
                )

            if has_goal:
                st.write(
                    f"**Goal:** "
                    f"{st.session_state.la_global_goal}"
                )

            if total > 0:
                st.write(
                    f"**Polygon decisions:** "
                    f"{completed} of {total} completed"
                )

        # =========================================================
        # Reset Assistant
        # =========================================================
        with st.container(border=True):

            st.markdown("##### Assistant Control")

            if st.button(
                    "Reset Assistant",
                    type="secondary",
                    use_container_width=True
            ):
                reset_agent()

        # =========================================================
        # Generated Outputs
        # =========================================================
        with st.container(border=True):

            st.markdown("##### Generated Outputs")

            result = st.session_state.la_generation_result

            if result:

                with open(
                        result["updated_vector_path"],
                        "rb"
                ) as f:

                    st.download_button(
                        "Download updated_vector.geojson",
                        data=f,
                        file_name="updated_vector.geojson",
                        mime="application/geo+json",
                        use_container_width=True,
                    )

                with open(
                        result["rules_path"],
                        "rb"
                ) as f:

                    st.download_button(
                        "Download rules.json",
                        data=f,
                        file_name="rules.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with open(
                        result["config_path"],
                        "rb"
                ) as f:

                    st.download_button(
                        "Download simulation_config.json",
                        data=f,
                        file_name="simulation_config.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                if result.get("explanation_path"):
                    with open(
                            result["explanation_path"],
                            "rb"
                    ) as f:
                        st.download_button(
                            "Download simulation_explanation.md",
                            data=f,
                            file_name="simulation_explanation.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

            else:
                st.caption(
                    "Generated files will appear here."
                )

        with st.container(border=True):
            st.markdown("##### Run C1 Layer Alterator")
            st.caption("Available after the vector and rules files have been generated.")
            st.session_state.la_ucp_folder = st.text_input(
                "UCP raster folder",
                value=st.session_state.la_ucp_folder,
                placeholder="Folder containing TCH.tif, IMD.tif, BH.tif, BSF.tif, SVF.tif",
            )
            st.session_state.la_fractions_folder = st.text_input(
                "Fraction raster folder",
                value=st.session_state.la_fractions_folder,
                placeholder="Folder containing the seven F_*.tif layers",
            )
            if st.button("Run C1 Layer Alterator", use_container_width=True, disabled=not bool(result)):
                try:
                    st.session_state.la_agent_state.stage = AgentStage.RUNNING_LAYER_ALTERATOR
                    layer_result = run_c1_layer_alterator(
                        vector_mask_path=result["updated_vector_path"],
                        rules_path=result["rules_path"],
                        ucp_folder=st.session_state.la_ucp_folder,
                        fractions_folder=st.session_state.la_fractions_folder,
                        output_folder=str(settings.PROJECT_ROOT / "data" / "layer_alterator_rasters"),
                    )
                    st.session_state.la_layer_alterator_result = layer_result
                    st.session_state.la_agent_state.stage = AgentStage.COMPLETED
                    st.success(f"Generated {len(layer_result.raster_outputs)} modified raster layers.")
                except Exception as exc:
                    st.session_state.la_agent_state.stage = AgentStage.ERROR
                    st.session_state.la_agent_state.last_error = str(exc)
                    st.error(f"Layer Alterator failed: {exc}")

            if st.session_state.la_layer_alterator_result:
                st.write(f"Output folder: `{st.session_state.la_layer_alterator_result.output_dir}`")
    # ---------- Chat panel ----------
    with chat_col:
        st.subheader("Conversational Planning")
        st.caption("Describe goals, modify polygon targets, ask explanations, or generate outputs.")

        # fixed-height chat box
        chat_box = st.container(height=650, border=True)

        with chat_box:
            if not st.session_state.la_chat_history:
                with st.chat_message("assistant"):
                    st.write(
                        "Hello. I will help you prepare Layer Alterator inputs through a guided dialogue."
                    )
                    st.write("Please upload a polygon vector file from the left panel to start.")

            for msg in st.session_state.la_chat_history:
                role = msg.get("role", "assistant")
                msg_type = msg.get("type", "text")
                content = msg.get("content", "")

                with st.chat_message(role):
                    if msg_type == "text":
                        st.markdown(content)

                    elif msg_type == "vector_preview":
                        st.info("Vector preview is available in the Map / Result Preview panel on the right.")

                    elif msg_type == "generation_result":
                        result = st.session_state.la_generation_result

                        if result:
                            st.success("Layer Alterator inputs generated successfully.")

                            result_df = pd.DataFrame(result["attribute_table"])
                            st.dataframe(result_df, width="stretch")

                            try:
                                import geopandas as gpd

                                updated_gdf = gpd.read_file(result["updated_vector_path"])
                                show_vector_map(
                                    updated_gdf,
                                    matched_type_column="matched_urban_type",
                                )

                            except Exception as e:
                                st.warning(f"Simulation result preview failed: {e}")

                            st.code(result["updated_vector_path"], language="text")
                            st.code(result["rules_path"], language="text")
                            st.code(result["config_path"], language="text")

            if st.session_state.get("la_processing"):
                with st.chat_message("assistant"):
                    st.info("Thinking...")

            pending_msg = st.session_state.get("la_pending_user_message")

            if pending_msg:
                st.session_state.la_pending_user_message = None
                st.session_state.la_processing = False

                process_user_message(pending_msg)

                st.rerun()




        # ---------- Chat input ----------
        disabled_input = False

        prompt_map = {
            "waiting_for_vector": "Ask a question, describe a goal, or upload a vector to start...",
            "waiting_for_goal": "Describe your overall simulation goal...",
            "free_chat": "Type freely: set goal, modify polygons, ask why, or generate...",
            "generated": "Files generated. You can still ask questions or reset.",
        }

        user_text = st.chat_input(
            prompt_map.get(st.session_state.la_stage, "Type your message..."),
            disabled=disabled_input,
            key="la_v32_chat_input",
        )

        if user_text:
            st.session_state.la_chat_history.append({
                "role": "user",
                "content": user_text,
                "type": "text",
            })
            st.session_state.la_pending_user_message = user_text
            st.session_state.la_processing = True
            st.rerun()

    with preview_col:
        st.subheader("Spatial Preview")
        st.caption("Review the uploaded vector, generated outputs, and decision table.")

        with st.container(border=True):
            st.markdown("##### Map Preview")

            gdf = st.session_state.la_gdf
            result = st.session_state.la_generation_result

            if result:
                st.caption("Generated simulation input preview")

                try:
                    import geopandas as gpd

                    updated_gdf = gpd.read_file(result["updated_vector_path"])

                    show_vector_map(
                        updated_gdf,
                        matched_type_column="matched_urban_type",
                    )

                except Exception as e:
                    st.warning(f"Result preview failed: {e}")

            elif gdf is not None:
                st.caption("Uploaded vector preview")
                show_vector_map(gdf)

            else:
                st.info("Upload a polygon vector to see the map preview here.")
                st.markdown(
                    """
                    <div style="
                        min-height: 260px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        color:#9ca3af;
                        border:1px dashed #d1d5db;
                        border-radius:12px;
                        margin-top:0.6rem;
                    ">
                        Spatial preview will appear after vector upload
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            st.markdown("##### Generated Outputs")

            result = st.session_state.la_generation_result

            if result:
                st.success("updated_vector.geojson")
                st.success("rules.json")
                st.success("simulation_config.json")

                if result.get("explanation_path"):
                    st.success("simulation_explanation.md")
            else:
                st.markdown(
                    """
                    <div style="
                        min-height: 90px;
                        display:flex;
                        align-items:center;
                        color:#9ca3af;
                    ">
                        No generated outputs yet.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            st.markdown("##### Data Preview")

            gdf = st.session_state.la_gdf

            if "la_before_after_df" in st.session_state:
                st.caption("Decision table")
                st.dataframe(st.session_state.la_before_after_df, width="stretch")

            elif gdf is not None:
                st.caption("Attribute table")
                preview_df = gdf.drop(columns=["geometry"], errors="ignore").copy()
                st.dataframe(preview_df.head(20), width="stretch")

            else:
                st.markdown(
                    """
                    <div style="
                        min-height: 90px;
                        display:flex;
                        align-items:center;
                        color:#9ca3af;
                    ">
                        No data preview available.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
def show_file_management_page():
    """Simulation Workspace page"""
    st.header("📂 Simulation Workspace")
    st.markdown(
        "**Manage the input data used by the Urban Simulation Assistant.**"
    )

    st.divider()

    # ---------- Upload cards ----------
    vector_card, knowledge_card, raster_card = st.columns(3, vertical_alignment="top")

    # ---------- Vector Upload ----------
    with vector_card:
        with st.container(border=True):
            st.subheader("🗺️ Vector Inputs")
            st.caption("Polygon layers for urban transformation areas.")

            vector_files = st.file_uploader(
                "Upload vector data",
                type=["shp", "shx", "dbf", "prj", "cpg", "geojson", "gpkg"],
                accept_multiple_files=True,
                help="For Shapefile, upload .shp, .shx, .dbf, .prj together.",
                key=f"workspace_vector_upload_{st.session_state.uploader_key}",
            )

            if vector_files and st.button("Upload Vector Data", type="primary", use_container_width=True):
                try:
                    save_dir = settings.VECTOR_DATA_DIR
                    save_dir.mkdir(parents=True, exist_ok=True)

                    saved_count = 0

                    for vf in vector_files:
                        file_path = save_dir / vf.name
                        with open(file_path, "wb") as f:
                            f.write(vf.getbuffer())
                        saved_count += 1

                    st.success(f"Uploaded {saved_count} vector file(s).")
                    _get_uploaded_files_info.clear()
                    st.session_state.uploader_key += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"Vector upload failed: {e}")

    # ---------- Knowledge Upload ----------
    with knowledge_card:
        with st.container(border=True):
            st.subheader("📄 Knowledge Base")
            st.caption("PDF documents used for RAG-based explanation and reference.")

            pdf_file = st.file_uploader(
                "Upload PDF document",
                type=["pdf"],
                help="Upload papers, reports, manuals, or thesis reference documents.",
                key=f"workspace_pdf_upload_{st.session_state.uploader_key}",
            )

            if pdf_file and st.button("Upload Knowledge Document", type="primary", use_container_width=True):
                try:
                    pdf_path = settings.PDF_DATA_DIR / pdf_file.name
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(pdf_path, "wb") as f:
                        f.write(pdf_file.getbuffer())

                    success = st.session_state.rag_engine.add_pdf_document(pdf_path)

                    if success:
                        st.success(f"Added to knowledge base: {pdf_file.name}")
                        _get_uploaded_files_info.clear()
                        st.session_state.uploader_key += 1
                        st.rerun()
                    else:
                        st.error("PDF processing failed.")

                except Exception as e:
                    st.error(f"Knowledge upload failed: {e}")

    # ---------- Raster Upload ----------
    with raster_card:
        with st.container(border=True):
            st.subheader("🛰️ Raster Layers")
            st.caption("Raster predictors or contextual layers for urban simulation.")

            raster_files = st.file_uploader(
                "Upload raster data",
                type=["tif", "tiff", "jp2", "img", "nc"],
                accept_multiple_files=True,
                help="Upload raster layers such as F_AC, F_G, F_TV, or other predictors.",
                key=f"workspace_raster_upload_{st.session_state.uploader_key}",
            )

            if raster_files and st.button("Upload Raster Data", type="primary", use_container_width=True):
                try:
                    save_dir = settings.RASTER_DATA_DIR
                    save_dir.mkdir(parents=True, exist_ok=True)

                    saved_count = 0

                    for rf in raster_files:
                        file_path = save_dir / rf.name
                        with open(file_path, "wb") as f:
                            f.write(rf.getbuffer())
                        saved_count += 1

                    st.success(f"Uploaded {saved_count} raster file(s).")
                    _get_uploaded_files_info.clear()
                    st.session_state.uploader_key += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"Raster upload failed: {e}")

    st.divider()

    # ---------- Workspace inventory ----------
    st.subheader("📋 Workspace Inventory")

    if st.button("🔄 Refresh Workspace", key="refresh_workspace_files"):
        _get_uploaded_files_info.clear()

    files_info = _get_uploaded_files_info(
        str(settings.PDF_DATA_DIR),
        str(settings.VECTOR_DATA_DIR),
        str(settings.RASTER_DATA_DIR),
    )

    if files_info:
        df = pd.DataFrame(files_info)
        st.dataframe(df, width="stretch", hide_index=True)

        total_files = len(files_info)
        pdf_files = len([f for f in files_info if f["Type"] == "PDF"])
        vector_files_count = len([f for f in files_info if f["Type"] == "Vector"])
        raster_files_count = len([f for f in files_info if f["Type"] == "Raster"])

        st.markdown("### Workspace Summary")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total files", total_files)
        with m2:
            st.metric("Knowledge docs", pdf_files)
        with m3:
            st.metric("Vector inputs", vector_files_count)
        with m4:
            st.metric("Raster layers", raster_files_count)

    else:
        st.info("No workspace files uploaded yet.")

def show_search_page():
    """Document search page"""
    st.header("🔎 Knowledge Explorer")

    # Search input
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("🔍 Search Query", placeholder="Enter keywords to search documents...")

    with col2:
        doc_type = st.selectbox("Document Type", ["All", "PDF", "GIS Metadata"])

    if search_query and st.button("Search", type="primary"):
        with st.spinner("🔍 Searching..."):
            try:
                filter_type = None
                if doc_type == "PDF":
                    filter_type = "pdf"
                elif doc_type == "GIS Metadata":
                    filter_type = "gis_metadata"

                # Execute search
                results = st.session_state.rag_engine.search_documents(
                    query=search_query,
                    doc_type=filter_type
                )

                if results:
                    st.success(f"Found {len(results)} related document chunks")

                    # Display search results
                    for i, result in enumerate(results):
                        with st.expander(f"📄 Document Chunk {i+1} (Similarity: {result.get('similarity_score', 0):.3f})"):
                            st.write("**Content:**")
                            st.write(result.get('content', ''))

                            st.write("**Metadata:**")
                            metadata = result.get('metadata', {})

                            # Format metadata for display
                            col1, col2 = st.columns(2)
                            with col1:
                                if metadata.get('source'):
                                    st.write(f"**Source:** {Path(metadata['source']).name}")
                                if metadata.get('doc_type'):
                                    st.write(f"**Type:** {metadata['doc_type']}")

                            with col2:
                                if metadata.get('file_size'):
                                    st.write(f"**Size:** {metadata['file_size'] / 1024 / 1024:.2f} MB")
                                if metadata.get('chunk_id') is not None:
                                    st.write(f"**Chunk ID:** {metadata['chunk_id']}")
                else:
                    st.info("No related documents found")

            except Exception as e:
                st.error(f"Search failed: {e}")

def show_settings_page():
    """System settings page"""
    st.header("⚙️ System Configuration")

    st.subheader("📊 System Information")

    try:
        system_info = st.session_state.rag_engine.get_system_info()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Vector Store Info:**")
            vector_info = system_info.get('vector_store', {})
            st.json(vector_info)

        with col2:
            st.write("**Model Info:**")
            llm_info = system_info.get('llm_model', {})
            if llm_info:
                st.json(llm_info)
            else:
                st.warning("LLM model not configured")

        st.write("**Processor Status:**")
        processors = system_info.get('processors', {})
        for processor, status in processors.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {processor}: {'Available' if status else 'Not Available'}")

        # Detailed file statistics
        st.write("**File Statistics:**")
        pdf_count, vector_count, raster_count = count_uploaded_files()

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("📄 PDF Documents", pdf_count)
        with metrics_col2:
            st.metric("🗺️ Vector Data", vector_count)
        with metrics_col3:
            st.metric("🛰️ Raster Data", raster_count)

        # Vector database statistics
        if vector_info:
            st.write("**Vector Database Statistics:**")
            doc_chunks = vector_info.get('document_count', 0)
            st.write(f"- Total document chunks: {doc_chunks}")
            st.write(f"- Average chunks per PDF: {doc_chunks / max(pdf_count, 1):.1f}")

    except Exception as e:
        st.error(f"Failed to get system info: {e}")

    st.divider()

    # Database management
    st.subheader("🗄️ Database Management")

    st.warning("⚠️ Dangerous operation: Clearing database will delete all processed document data")

    if st.button("🗑️ Clear Database", type="secondary"):
        if st.button("⚠️ Confirm Clear", type="primary"):
            with st.spinner("Clearing database..."):
                try:
                    success = st.session_state.rag_engine.clear_database()
                    if success:
                        st.success("✅ Database cleared")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear database")
                except Exception as e:
                    st.error(f"Error clearing database: {e}")

if __name__ == "__main__":
    main()
