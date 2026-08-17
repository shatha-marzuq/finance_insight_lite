"""
Bilingual support for Finance Insight Lite.
Usage:
    from i18n import t, get_lang
    lang = get_lang(st)          # "en" or "ar"
    label = t("welcome_title")   # returns the right translation
"""

TRANSLATIONS = {
    # ── Welcome Screen ──────────────────────────────────────────
    "welcome_title": {
        "en": "Your Intelligent Financial Partner",
        "ar": "شريكك المالي الذكي",
    },
    "welcome_subtitle": {
        "en": "Your AI-Powered Financial Analyst",
        "ar": "محللك المالي المدعوم بالذكاء الاصطناعي",
    },
    "welcome_desc": {
        "en": (
            "Upload your reports (PDF, Excel, or Images) and let our AI engine "
            "extract insights, build visualizations, and answer your questions "
            "in seconds."
        ),
        "ar": (
            "ارفع تقاريرك (PDF أو Excel أو صور) ودع محرك الذكاء الاصطناعي "
            "يستخرج لك المعلومات، ويبني الرسوم البيانية، ويجاوب على أسئلتك "
            "خلال ثوانٍ."
        ),
    },
    "upload_cta": {
        "en": "Drop your files below to get started",
        "ar": "ارفع ملفاتك هنا للبدء",
    },
    "feature_1_title": {
        "en": "Smart Analysis",
        "ar": "تحليل ذكي",
    },
    "feature_1_desc": {
        "en": "AI-driven extraction of key financial metrics and KPIs",
        "ar": "استخراج ذكي للمؤشرات والبيانات المالية الرئيسية",
    },
    "feature_2_title": {
        "en": "Interactive Charts",
        "ar": "رسوم تفاعلية",
    },
    "feature_2_desc": {
        "en": "Auto-generated visualizations from your document data",
        "ar": "رسوم بيانية تلقائية من بيانات مستنداتك",
    },
    "feature_3_title": {
        "en": "AI Chat",
        "ar": "محادثة ذكية",
    },
    "feature_3_desc": {
        "en": "Ask anything about your documents and get sourced answers",
        "ar": "اسأل أي سؤال عن مستنداتك واحصل على إجابات موثّقة",
    },

    # ── Upload & Processing ─────────────────────────────────────
    "upload_label": {
        "en": "Upload PDF, Excel, or Image",
        "ar": "ارفع ملف PDF أو Excel أو صورة",
    },
    "upload_help": {
        "en": "Drag and drop or click to browse. Supports PDF, XLSX, XLS, PNG, JPG.",
        "ar": "اسحب وأفلت أو اضغط للاستعراض. يدعم PDF و XLSX و XLS و PNG و JPG.",
    },
    "browse_files_btn": {
        "en": "Browse files",
        "ar": "استعراض الملفات",
    },
    "processing": {
        "en": "Processing your documents...",
        "ar": "جاري معالجة مستنداتك...",
    },
    "processing_file": {
        "en": "Processing file {idx} of {total}: {name}",
        "ar": "جاري معالجة الملف {idx} من {total}: {name}",
    },
    "building_db": {
        "en": "Building knowledge base...",
        "ar": "جاري بناء قاعدة المعرفة...",
    },
    "initializing_agent": {
        "en": "Initializing AI agent...",
        "ar": "جاري تهيئة المحلل الذكي...",
    },
    "process_success": {
        "en": "Processed {docs} document(s)",
        "ar": "تمت معالجة {docs} مستند",
    },
    "process_error": {
        "en": "Error: {error}",
        "ar": "خطأ: {error}",
    },

    # ── Dashboard ───────────────────────────────────────────────
    "dashboard_title": {
        "en": "Document Analysis",
        "ar": "تحليل المستندات",
    },
    "files_processed": {
        "en": "Files Processed",
        "ar": "ملفات معالجة",
    },
    "total_documents": {
        "en": "Document Chunks",
        "ar": "أجزاء المستندات",
    },
    "ai_status": {
        "en": "AI Status",
        "ar": "حالة الذكاء",
    },
    "ready": {
        "en": "Ready",
        "ar": "جاهز",
    },
    "processing_time": {
        "en": "Processing Time",
        "ar": "وقت المعالجة",
    },
    "session_info": {
        "en": "Session Info",
        "ar": "معلومات الجلسة",
    },
    "tab_chat": {
        "en": "Chat",
        "ar": "المحادثة",
    },
    "tab_settings": {
        "en": "Settings",
        "ar": "الإعدادات",
    },

    # ── Header actions ────────────────────────────────────────
    "back_btn": {
        "en": "New Analysis",
        "ar": "تحليل جديد",
    },
    "back_btn_help": {
        "en": "Return to the upload screen and start a new analysis",
        "ar": "الرجوع لشاشة الرفع وبدء تحليل جديد",
    },
    "clear_chat_help": {
        "en": "Delete all chat messages in this session",
        "ar": "حذف جميع رسائل المحادثة في هذه الجلسة",
    },

    # ── Ask Questions section ────────────────────────────────────
    "ask_questions_title": {
        "en": "Ask Questions",
        "ar": "اطرح الأسئلة",
    },
    "ask_questions_subtitle": {
        "en": "Get AI-powered insights from your financial documents",
        "ar": "احصل على رؤى مدعومة بالذكاء الاصطناعي من مستنداتك المالية",
    },

    # ── Chat ────────────────────────────────────────────────────
    "chat_placeholder": {
        "en": "Type your question here...",
        "ar": "اكتب سؤالك هنا...",
    },
    "thinking": {
        "en": "Thinking...",
        "ar": "جاري التحليل...",
    },
    "upload_first_warning": {
        "en": "Please upload and process a document first.",
        "ar": "يرجى رفع ومعالجة مستند أولاً.",
    },
    "sample_q_1": {
        "en": "What is the net income?",
        "ar": "ما هو صافي الدخل؟",
    },
    "sample_q_2": {
        "en": "What is the free cash flow?",
        "ar": "ما هو التدفق النقدي الحر؟",
    },
    "sample_q_3": {
        "en": "What is the gearing ratio?",
        "ar": "ما هي نسبة المديونية؟",
    },
    "sample_q_4": {
        "en": "Summarize the financial highlights",
        "ar": "لخّص أبرز النقاط المالية",
    },
    "sample_q_5": {
        "en": "Draw a bar chart of revenue breakdown",
        "ar": "ارسم رسم بياني لتوزيع الإيرادات",
    },
    "sample_q_6": {
        "en": "Visualize the expense categories",
        "ar": "اعرض تصنيفات المصروفات بيانياً",
    },
    "confidence_label": {
        "en": "Confidence",
        "ar": "الثقة",
    },
    "pages_label": {
        "en": "Pages",
        "ar": "الصفحات",
    },
    "docs_label": {
        "en": "Sources",
        "ar": "المصادر",
    },
    "chart_included": {
        "en": "Chart included",
        "ar": "يتضمن رسم بياني",
    },
    "view_data_table": {
        "en": "View Data Table",
        "ar": "عرض جدول البيانات",
    },
    "view_verification": {
        "en": "View Verification",
        "ar": "عرض التحقق",
    },
    "no_verification_notes": {
        "en": "No verification notes available.",
        "ar": "لا توجد ملاحظات تحقق متاحة.",
    },
    "chart_error": {
        "en": "Error rendering chart: {error}",
        "ar": "خطأ في عرض الرسم البياني: {error}",
    },
    "chart_unavailable": {
        "en": "Chart unavailable: {error}",
        "ar": "الرسم البياني غير متوفر: {error}",
    },

    # ── Settings ────────────────────────────────────────────────
    "settings_title": {
        "en": "Settings",
        "ar": "الإعدادات",
    },
    "rag_config": {
        "en": "RAG Configuration",
        "ar": "إعدادات RAG",
    },
    "self_rag_toggle": {
        "en": "Enable Self-RAG",
        "ar": "تفعيل Self-RAG",
    },
    "self_rag_help": {
        "en": "Higher accuracy but slower",
        "ar": "دقة أعلى لكن أبطأ",
    },
    "relevance_threshold": {
        "en": "Relevance Threshold",
        "ar": "حد الصلة",
    },
    "relevance_help": {
        "en": "Higher = stricter filtering",
        "ar": "أعلى = تصفية أدق",
    },
    "num_docs": {
        "en": "Number of Documents",
        "ar": "عدد المستندات",
    },
    "num_docs_help": {
        "en": "More docs = better coverage",
        "ar": "مستندات أكثر = تغطية أفضل",
    },
    "default_chart": {
        "en": "Default Chart Type",
        "ar": "نوع الرسم الافتراضي",
    },
    "chart_type_bar": {"en": "Bar", "ar": "أعمدة"},
    "chart_type_line": {"en": "Line", "ar": "خطي"},
    "chart_type_pie": {"en": "Pie", "ar": "دائري"},
    "chart_type_scatter": {"en": "Scatter", "ar": "مبعثر"},
    "chart_type_area": {"en": "Area", "ar": "مساحي"},
    "show_data_table": {
        "en": "Show Data Table",
        "ar": "عرض جدول البيانات",
    },
    "clear_cache_btn": {
        "en": "Clear Cache",
        "ar": "مسح الكاش",
    },
    "cache_cleared": {
        "en": "Cache cleared!",
        "ar": "تم مسح الكاش!",
    },
    "cached_size": {
        "en": "{size:.1f} MB cached",
        "ar": "{size:.1f} ميجا مخزنة",
    },
    "clear_chat_btn": {
        "en": "Clear Chat History",
        "ar": "مسح سجل المحادثة",
    },
    "upload_new": {
        "en": "Upload New Documents",
        "ar": "رفع مستندات جديدة",
    },
    "footer": {
        "en": "Finance Insight Lite 2026. All rights reserved.",
        "ar": "Finance Insight Lite 2026. جميع الحقوق محفوظة.",
    },
    "footer_summary": {
        "en": "AI-assisted document analysis for financial reports, tables, charts, and sourced Q&A.",
        "ar": "تحليل مستندات مالي مدعوم بالذكاء الاصطناعي للتقارير والجداول والرسوم والإجابات الموثّقة.",
    },
    "lang_btn": {
        "en": "العربية",
        "ar": "English",
    },
}


# ── Helpers ─────────────────────────────────────────────────────
_DEFAULT_LANG = "en"
_SUPPORTED_LANGS = {"en", "ar"}


def get_lang(st_module) -> str:
    """Return the current language from Streamlit session state."""
    if "lang" not in st_module.session_state:
        st_module.session_state.lang = _DEFAULT_LANG
    return st_module.session_state.lang


def set_lang(st_module, lang: str):
    if lang not in _SUPPORTED_LANGS:
        lang = _DEFAULT_LANG
    st_module.session_state.lang = lang


def _resolve_lang() -> str:
    """Resolve the active Streamlit session language without module-level state."""
    try:
        import streamlit as st

        lang = st.session_state.get("lang", _DEFAULT_LANG)
    except Exception:
        lang = _DEFAULT_LANG

    return lang if lang in _SUPPORTED_LANGS else _DEFAULT_LANG


def t(key: str, **kwargs) -> str:
    """Translate *key* into the current language. Extra kwargs are .format()-ed."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    lang = _resolve_lang()
    text = entry.get(lang, entry.get(_DEFAULT_LANG, key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


def bind(st_module):
    """Ensure the current Streamlit session has a language initialized."""
    get_lang(st_module)
