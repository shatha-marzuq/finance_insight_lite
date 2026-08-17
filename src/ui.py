"""
Finance Insight Lite - Streamlit UI (v3)
"Ledger & Brass" theme, bilingual (AR/EN), welcome->dashboard flow.
"""

import shutil
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import sys
import plotly.graph_objects as go
import json
import re
import pandas as pd
import chat_db

current_dir = str(Path(__file__).resolve().parent)
if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.insert(0, current_dir)

from finance_insight_lite.modules.processor import clear_cache, load_documents_fastest, pdf_to_documents
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.rag_agent import FinancialRAGAgent

from i18n import t, bind as i18n_bind, get_lang, set_lang
from styles import get_css

project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

st.set_page_config(page_title="Finance Insight Lite", page_icon="./images/logo.png", layout="wide", initial_sidebar_state="collapsed")

_defaults = {
    "agent": None, "chat_history": None, "vector_db": None, "chunks": None,
    "pending_question": None, "processed_files": set(), "num_files": 0, "num_docs": 0,
    "proc_time": 0.0, "default_chart_type": "bar", "show_data_table": True, "lang": "en",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

chat_db.init_db()
session_id = chat_db.get_or_create_session_id(st)
if st.session_state.chat_history is None:
    st.session_state.chat_history = chat_db.load_history(session_id)

i18n_bind(st)
lang = get_lang(st)
st.html(get_css(lang))

def _toggle_lang():
    new = "ar" if st.session_state.lang == "en" else "en"
    set_lang(st, new)

_LEGACY_ANSWER_NOTE_PATTERNS = (
    re.compile(
        r"\n*\s*---\s*\n\s*\*\*Note:\*\*\s*.*?appropriate caution\.?",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\n*\s*---\s*\n\s*\*\*ملاحظة:\*\*\s*.*?الحذر المناسب\.?",
        re.DOTALL,
    ),
)

def strip_legacy_answer_note(answer):
    cleaned = answer or ""
    for pattern in _LEGACY_ANSWER_NOTE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()

def build_llm_chat_history(chat_history, max_turns: int = 6):
    recent = chat_history[-max_turns:] if max_turns else chat_history
    return [
        {"question": turn.get("question", ""), "answer": strip_legacy_answer_note(turn.get("answer", ""))}
        for turn in recent
    ]

def process_uploaded_files(uploaded_files):
    if os.path.exists("./data/uploaded/"): shutil.rmtree("./data/uploaded")
    os.makedirs("./data/uploaded")
    paths = []
    for f in uploaded_files:
        p = f"./data/uploaded/{f.name}"
        with open(p, "wb") as fh: fh.write(f.getbuffer())
        paths.append(p)
    start = time.time()
    all_docs = []
    progress = st.progress(0)
    for idx, fp in enumerate(paths):
        progress.progress((idx + 1) / len(paths))
        result = load_documents_fastest(fp, use_cache=True, max_workers=1)
        all_docs.extend(result["documents"])
    progress.empty()
    with st.spinner(t("building_db")):
        st.session_state.vector_db, st.session_state.chunks = build_vector_db(all_docs, db_path="./database", source_paths=paths)
    with st.spinner(t("initializing_agent")):
        st.session_state.agent = FinancialRAGAgent(st.session_state.vector_db, chunks=st.session_state.chunks)
    st.session_state.num_files = len(paths)
    st.session_state.num_docs = len(all_docs)
    st.session_state.proc_time = time.time() - start
    st.session_state.processed_files = {f.name for f in uploaded_files}

def reset_to_welcome():
    st.session_state.agent = None; st.session_state.vector_db = None; st.session_state.chunks = None
    st.session_state.processed_files = set(); st.session_state.num_files = 0; st.session_state.num_docs = 0
    st.session_state.proc_time = 0.0; st.session_state.pending_question = None; st.session_state.chat_history = []
    chat_db.clear_session(session_id)

def clear_chat_only():
    st.session_state.chat_history = []; chat_db.clear_session(session_id)

def render_chat_turn(chat, idx):
    answer = strip_legacy_answer_note(chat.get("answer", ""))
    _ar = len(re.findall(r"[\u0600-\u06FF]", answer))
    _lt = len(re.findall(r"[A-Za-z]", answer))
    if _ar and _ar >= _lt:
        # إجابة عربية: اعرضها من اليمين لليسار مع الحفاظ على تنسيق الماركداون
        st.markdown(
            f'<div dir="rtl" style="text-align: right;">\n\n{answer}\n\n</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(answer)
    chart_data = chat.get("chart")
    if chart_data and chart_data.get("success"):
        chart_json = chart_data.get("chart")
        if chart_json:
            try:
                fig = go.Figure(json.loads(chart_json))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,29,50,0.6)", font_color="#E2E8F0", font_family="IBM Plex Mono, monospace", title_font_color="#60A5FA", legend_font_color="#94A3B8", margin=dict(l=20, r=20, t=20, b=20), title_text="")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
            except Exception as e: st.error(t("chart_error", error=str(e)))
        if st.session_state.get("show_data_table", True) and chart_data.get("data_preview"):
            with st.expander(t("view_data_table")): st.dataframe(pd.DataFrame(chart_data.get("data_preview")), use_container_width=True)
    elif chart_data and chart_data.get("error"):
        st.info(t("chart_unavailable", error=chart_data["error"]))

    source_pages = chat.get("source_pages", [])
    pages_val = ', '.join(source_pages) if source_pages else "N/A"
    confidence_val = chat.get('confidence') or 'N/A'
    docs_val = chat.get('relevant_docs_count') or 0

    confidence_class = {
        "high": "meta-badge--high", "medium": "meta-badge--medium", "low": "meta-badge--low",
    }.get(str(confidence_val).lower(), "meta-badge--default")

    chart_badge = f'<div class="meta-badge">{t("chart_included")}</div>' if chart_data else ""

    st.html(f"""
    <div class="meta-row">
        <div class="meta-badge">
            <span class="meta-label">{t('pages_label')}</span>
            <span class="meta-value">{pages_val}</span>
        </div>
        <div class="meta-badge {confidence_class}">
            <span class="meta-label">{t('confidence_label')}</span>
            <span class="meta-value">{confidence_val}</span>
        </div>
        <div class="meta-badge">
            <span class="meta-label">{t('docs_label')}</span>
            <span class="meta-value">{docs_val}</span>
        </div>
        {chart_badge}
    </div>
    <style>
    .meta-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }}
    .meta-badge {{
        display: flex;
        align-items: center;
        gap: 6px;
        background: var(--ink-800, rgba(15,31,56,0.6));
        border: 1px solid var(--card-border, rgba(30,48,80,0.8));
        border-radius: 6px;
        padding: 4px 12px;
        font-family: var(--font-mono, monospace);
        font-size: 12px;
        line-height: 1.4;
    }}
    .meta-label {{
        color: var(--text-muted, #64748B);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 10.5px;
    }}
    .meta-value {{
        color: var(--text-primary, #E2E8F0);
        font-weight: 600;
    }}
    .meta-badge--high .meta-value {{ color: var(--success, #22C55E); }}
    .meta-badge--medium .meta-value {{ color: var(--warning, #F59E0B); }}
    .meta-badge--low .meta-value {{ color: var(--error, #EF4444); }}
    </style>
    """)

    if chat.get("verification"):
        with st.expander(t("view_verification")):
            _notes = chat["verification"].get("notes", t("no_verification_notes"))
            if get_lang(st) == "ar":
                _dir, _align = "rtl", "right"
            else:
                _dir, _align = "ltr", "left"
            st.markdown(
                f'<div style="direction: {_dir}; text-align: {_align};">{_notes}</div>',
                unsafe_allow_html=True,
            )

def handle_question(question: str):
    if st.session_state.agent is None: return st.warning(t("upload_first_warning"))
    with st.spinner(t("thinking")):
        result = st.session_state.agent.process_query(question, chat_history=build_llm_chat_history(st.session_state.chat_history))
    answer = strip_legacy_answer_note(result.get("answer"))
    entry = {
        "question": question, "answer": answer, "chart": result.get("chart"),
        "source_pages": result.get("source_pages"), "confidence": result.get("confidence"),
        "relevant_docs_count": result.get("relevant_docs_count"), "verification": result.get("verification"),
        "retrieved": answer,
        "relevant": result.get("source_texts"),
        "source_texts": result.get("source_texts"),
    }
    st.session_state.chat_history.append(entry)
    chat_db.save_entry(session_id, entry)
    st.rerun()

_ICON_SEARCH_URI = "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2360A5FA%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Ccircle%20cx%3D%2211%22%20cy%3D%2211%22%20r%3D%227.5%22/%3E%3Cpath%20d%3D%22M20.5%2020.5%2016.4%2016.4%22/%3E%3Cpath%20d%3D%22M8%2011h6%22/%3E%3Cpath%20d%3D%22M11%208v6%22/%3E%3C/svg%3E"
_ICON_CHART_URI = "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2360A5FA%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M3.5%203.5v17h17%22/%3E%3Crect%20x%3D%227%22%20y%3D%2213%22%20width%3D%222.6%22%20height%3D%225.2%22%20rx%3D%220.4%22%20fill%3D%22%2360A5FA%22%20stroke%3D%22none%22/%3E%3Crect%20x%3D%2211.7%22%20y%3D%229%22%20width%3D%222.6%22%20height%3D%229.2%22%20rx%3D%220.4%22%20fill%3D%22%2360A5FA%22%20stroke%3D%22none%22/%3E%3Crect%20x%3D%2216.4%22%20y%3D%225.5%22%20width%3D%222.6%22%20height%3D%2212.7%22%20rx%3D%220.4%22%20fill%3D%22%2360A5FA%22%20stroke%3D%22none%22/%3E%3C/svg%3E"
_ICON_CHAT_URI = "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2360A5FA%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M4%205.5h16a1%201%200%200%201%201%201V16a1%201%200%200%201-1%201H9l-4.6%203.4a.6.6%200%200%201-.96-.48V17H4a1%201%200%200%201-1-1V6.5a1%201%200%200%201%201-1Z%22/%3E%3Cpath%20d%3D%22M8%2010h8%22/%3E%3Cpath%20d%3D%22M8%2013.2h5%22/%3E%3C/svg%3E"

ICON_SEARCH = f'<img src="{_ICON_SEARCH_URI}" alt="" />'
ICON_CHART = f'<img src="{_ICON_CHART_URI}" alt="" />'
ICON_CHAT = f'<img src="{_ICON_CHAT_URI}" alt="" />'

def show_welcome():
    import base64
    logo_path = Path("./images/logo.png")
    logo_tag = f'<img class="welcome-logo" src="data:image/png;base64,{base64.b64encode(logo_path.read_bytes()).decode()}" />' if logo_path.exists() else ""

    with st.container(key="welcome_wrap"):
        st.html(f"""
        <div class="welcome-container">
            <span class="logo-halo-wrap">{logo_tag}</span>
            <div class="welcome-title">Finance Insight Lite</div>
            <div class="welcome-subtitle">{t("welcome_title")}</div>
            <div class="welcome-desc">{t("welcome_desc")}</div>
        </div>
        """)

        spacer_l, mid, spacer_r = st.columns([1, 2, 1])
        with mid:
            cta_align = "right" if lang == "ar" else "left"
            st.html(f'<div class="upload-cta-label" style="text-align: {cta_align};">{t("upload_cta")}</div>')
            uploaded = st.file_uploader(t("upload_label"), type=["pdf", "xlsx", "xls", "csv", "png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")

        if uploaded:
            new_names = {f.name for f in uploaded}
            if new_names != st.session_state.processed_files:
                with mid:
                    with st.spinner(t("processing")):
                        try:
                            process_uploaded_files(uploaded)
                            st.success(t("process_success", files=st.session_state.num_files, docs=st.session_state.num_files, time=st.session_state.proc_time))
                        except Exception as e:
                            st.error(t("process_error", error=str(e)))
                            st.stop()
                time.sleep(0.8)
                st.rerun()

        st.html(f"""
        <div class="features-grid">
            <div class="feature-card"><div class="feature-icon">{ICON_SEARCH}</div><div class="feature-title">{t("feature_1_title")}</div><div class="feature-desc">{t("feature_1_desc")}</div></div>
            <div class="feature-card"><div class="feature-icon">{ICON_CHART}</div><div class="feature-title">{t("feature_2_title")}</div><div class="feature-desc">{t("feature_2_desc")}</div></div>
            <div class="feature-card"><div class="feature-icon">{ICON_CHAT}</div><div class="feature-title">{t("feature_3_title")}</div><div class="feature-desc">{t("feature_3_desc")}</div></div>
        </div>
        """)

def show_menu_bar():
    import base64
    logo_path = Path("./images/logo.png")
    logo_tag = f'<img class="menu-logo" src="data:image/png;base64,{base64.b64encode(logo_path.read_bytes()).decode()}" />' if logo_path.exists() else ""

    st.html('<span class="menu-bar-anchor"></span>')
    col_brand, col_actions = st.columns([3, 2])
    with col_brand:
        st.html(f'<div class="menu-brand">{logo_tag}<span class="menu-wordmark">Finance Insight Lite</span></div>')
    with col_actions:
        with st.container(key="menu_actions"):
            in_dashboard = st.session_state.agent is not None
            if in_dashboard:
                if st.button(t("clear_chat_btn"), help=t("clear_chat_help"), key="menu_clear_chat"):
                    clear_chat_only(); st.rerun()
                if st.button(t("back_btn"), help=t("back_btn_help"), key="menu_back"):
                    reset_to_welcome(); st.rerun()
            st.button(t("lang_btn"), key="menu_lang_toggle", on_click=_toggle_lang)

def show_footer():
    import base64
    logo_path = Path("./images/logo.png")
    logo_tag = f'<img class="footer-logo" src="data:image/png;base64,{base64.b64encode(logo_path.read_bytes()).decode()}" />' if logo_path.exists() else ""

    st.html(f"""
    <div class="app-footer">
        <div class="footer-main">
            <div class="footer-identity">
                {logo_tag}
                <div class="footer-copy">
                    <div class="footer-title">Finance Insight Lite</div>
                    <div class="footer-summary">{t("footer_summary")}</div>
                </div>
            </div>
        </div>
        <div class="footer-legal">{t("footer")}</div>
    </div>
    """)

def show_dashboard_header():
    st.html("""
    <div class="dash-header-block">
        <div class="dash-title">Finance <span class="accent">Insight Lite</span></div>
    </div>
    """)
    st.divider()

def show_dashboard():
    for i, chat in enumerate(st.session_state.chat_history):
        st.html('<span class="msg-user-anchor"></span>')
        with st.chat_message("user", avatar="./images/user_icon.png"): st.write(chat.get("question"))
        st.html('<span class="msg-bot-anchor"></span>')
        with st.chat_message("assistant", avatar="./images/chatbots_icon.png"): render_chat_turn(chat, i)

    if len(st.session_state.chat_history) == 0:
        SVG_SPARKLE = "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2360A5FA%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolygon%20points%3D%2212%202%2015.09%208.26%2022%209.27%2017%2014.14%2018.18%2021.02%2012%2017.77%205.82%2021.02%207%2014.14%202%209.27%208.91%208.26%2012%202%22/%3E%3C/svg%3E"
        st.html(f'<div class="ask-header"><img src="{SVG_SPARKLE}" width="20" height="20" /><p class="ask-title">{t("ask_questions_title")}</p></div><p class="ask-subtitle">{t("ask_questions_subtitle")}</p>')

        samples = [t("sample_q_1"), t("sample_q_2"), t("sample_q_3"), t("sample_q_4"), t("sample_q_5"), t("sample_q_6")]
        def _set_q(q): st.session_state.pending_question = q
        cols = st.columns(3)
        for i, q in enumerate(samples):
            with cols[i % 3]:
                st.button(f">  {q}", key=f"sq_{i}", on_click=_set_q, args=(q,))

    user_q = st.chat_input(t("chat_placeholder"))
    st.html(f'<div class="chat-input-legal">{t("footer")}</div>')
    if st.session_state.pending_question:
        q = st.session_state.pending_question; st.session_state.pending_question = None; handle_question(q)
    if user_q: handle_question(user_q)

show_menu_bar()

if st.session_state.agent is not None:
    show_dashboard()
else:
    show_welcome()
    show_footer()
