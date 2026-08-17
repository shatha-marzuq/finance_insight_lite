"""
All custom CSS for Finance Insight Lite.
Theme: "Ledger & Brass" — a financial-analyst register (ink, hairlines,
brass accent, tabular mono for figures) in place of a generic dark-glow UI.
Injected via st.html() to avoid the st.markdown style-rendering bug.
"""

def get_css(lang: str = "en") -> str:
    direction = "rtl" if lang == "ar" else "ltr"
    text_align = "right" if lang == "ar" else "left"
    footer_justify = "flex-end" if lang == "ar" else "flex-start"
    footer_identity_direction = "row-reverse" if lang == "ar" else "row"

    heading_font = "'IBM Plex Sans Arabic', sans-serif" if lang == "ar" else "'Fraunces', serif"
    body_font = "'IBM Plex Sans Arabic', sans-serif" if lang == "ar" else "'IBM Plex Sans', sans-serif"

    lang_btn_side = "right" if lang == "en" else "left"
    browse_text = "إضافة ملفات" if lang == "ar" else "Add Files"

    # جهة كل رسالة: انجليزي = يوزر يسار / بوت يمين || عربي = يوزر يمين / بوت يسار
    if lang == "ar":
        user_side, bot_side = "right", "left"
        user_flex, bot_flex = "row-reverse", "row"
    else:
        user_side, bot_side = "left", "right"
        user_flex, bot_flex = "row", "row-reverse"

    def _margins(side):
        return ("auto", "0") if side == "right" else ("0", "auto")

    def _radius(side):
        # حواف مسطّحة أكثر (سجل/ledger) بدل شكل الفقاعة
        return "12px 12px 3px 12px" if side == "right" else "12px 12px 12px 3px"

    user_margin_l, user_margin_r = _margins(user_side)
    bot_margin_l, bot_margin_r = _margins(bot_side)
    user_radius = _radius(user_side)
    bot_radius = _radius(bot_side)

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,900&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
    --ink-950: #060F1E;
    --ink-900: #0A1628;
    --ink-850: #0F1D32;
    --ink-800: #0F1F38;
    --ink-750: #132039;
    --line-700: #1E3050;
    --line-600: #243656;

    --brass:       #3B82F6;
    --brass-light: #60A5FA;
    --brass-dim:   rgba(59, 130, 246, 0.14);
    --brass-ring:  rgba(59, 130, 246, 0.45);

    --text-primary:   #E2E8F0;
    --text-secondary: #A6B2C4;
    --text-muted:     #808C9E;

    --success: #22C55E;
    --warning: #F59E0B;
    --error:   #EF4444;

    --card-bg: var(--ink-850);
    --card-border: var(--line-700);
    --radius: 10px;
    --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);

    --font-heading: {heading_font};
    --font-body: {body_font};
    --font-mono: 'IBM Plex Mono', monospace;

    --chat-user-bg: var(--ink-800);
    --chat-bot-bg:  var(--ink-850);
}}

.stApp {{
    background: var(--ink-950);
    direction: {direction};
    text-align: {text_align};
}}
.main .block-container {{
    direction: {direction};
    text-align: {text_align};
}}
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
    font-family: var(--font-body);
    color: var(--text-primary);
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

[data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    width: 0 !important;
    height: 0 !important;
    line-height: 0 !important;
    overflow: hidden !important;
    display: inline-block !important;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--ink-900); }}
::-webkit-scrollbar-thumb {{ background: var(--line-600); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--brass); }}

/* ═══════════ أزرار عامة — لوحة تحكم مسطّحة بدل الأزرار الكبسولية ═══════════ */
div.stButton > button {{
    background: var(--ink-850);
    color: var(--text-primary);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.55rem 1.6rem;
    font-family: var(--font-heading);
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: var(--transition);
    width: 100%;
}}
div.stButton > button:hover {{
    background: var(--brass);
    border-color: var(--brass);
    color: var(--ink-950);
    box-shadow: 0 4px 16px rgba(201, 162, 39, 0.22);
    transform: translateY(-1px);
}}

div[data-testid="stHorizontalBlock"] div.stButton > button {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    font-family: var(--font-mono);
    font-size: 0.86rem;
    font-weight: 500;
    padding: 0.75rem 1.1rem;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.6rem;
    text-align: {text_align};
    color: var(--text-secondary);
}}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {{
    background: var(--ink-750);
    border-color: var(--brass);
    color: var(--brass-light);
    box-shadow: none;
    transform: translateY(-2px);
}}

/* ═══════════ شعار — توهج ثابت هادئ، بدون نبض مستمر ═══════════ */
.logo-halo-wrap {{
    position: relative;
    display: inline-block;
    z-index: 1;
    animation: logoSettle 0.9s cubic-bezier(0.2, 0.8, 0.2, 1);
}}
.logo-halo-wrap::before {{
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 130%;
    height: 130%;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    background: radial-gradient(circle,
        rgba(96, 165, 250, 0.10) 0%,
        rgba(124, 140, 248, 0.055) 44%,
        rgba(167, 139, 250, 0.025) 64%,
        transparent 78%);
    filter: blur(32px);
    z-index: -1;
    pointer-events: none;
}}
@keyframes logoSettle {{
    0%   {{ opacity: 0; transform: scale(0.92) translateY(6px); }}
    100% {{ opacity: 1; transform: scale(1) translateY(0); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .logo-halo-wrap {{ animation: none; }}
}}

div[data-testid="stFileUploader"] label {{
    display: none !important;
}}
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"] {{
    display: flex !important;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    text-align: center;
    padding: 1.6rem 1.2rem;
    outline: 1.5px dashed var(--line-600);
    outline-offset: -2px;
    border-radius: var(--radius);
    background: rgba(255, 255, 255, 0.015);
    transition: var(--transition);
    width: 100%;
}}
[data-testid="stFileUploaderDropzone"] button span {{
    display: none !important;
}}
[data-testid="stFileUploaderDropzone"] button::after {{
    content: "{browse_text}";
    display: block;
    font-family: var(--font-body);
    letter-spacing: 0;
}}
section[data-testid="stFileUploaderDropzone"]::before,
div[data-testid="stFileUploaderDropzone"]::before {{
    content: "";
    display: block;
    width: 36px;
    height: 36px;
    margin: 0 auto 0.4rem;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2360A5FA%22%20stroke-width%3D%221.6%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M7%2018a4.5%204.5%200%2001-1.2-8.84A5.5%205.5%200%200116.5%207.5%204%204%200%200117%2015.9%22/%3E%3Cpath%20d%3D%22M12%2011v8%22/%3E%3Cpath%20d%3D%22M9%2014l3-3%203%203%22/%3E%3C/svg%3E");
}}

div[data-testid="stElementContainer"]:has(> .upload-cta-label) {{
    margin-bottom: -0.25rem !important;
}}
.upload-cta-label {{
    font-family: var(--font-heading);
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.005em;
    color: var(--text-secondary);
    transition: var(--transition);
}}
div[data-testid="stElementContainer"]:has(> .upload-cta-label):hover ~ div[data-testid="stElementContainer"] section[data-testid="stFileUploaderDropzone"],
div[data-testid="stElementContainer"]:has(> .upload-cta-label) + div[data-testid="stElementContainer"]:hover section[data-testid="stFileUploaderDropzone"],
div[data-testid="stElementContainer"]:has(> .upload-cta-label):hover ~ div[data-testid="stElementContainer"] div[data-testid="stFileUploaderDropzone"],
div[data-testid="stElementContainer"]:has(> .upload-cta-label) + div[data-testid="stElementContainer"]:hover div[data-testid="stFileUploaderDropzone"] {{
    outline-color: var(--brass) !important;
    background: var(--brass-dim);
}}
div[data-testid="stElementContainer"]:has(> .upload-cta-label):hover {{
    color: var(--brass-light);
}}

div[data-testid="stFileUploaderFile"] {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin-top: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.85rem;
}}
div[data-testid="stFileUploaderFile"] button {{
    background: rgba(181, 86, 74, 0.14) !important;
    border: 1px solid rgba(181, 86, 74, 0.5) !important;
    border-radius: 50% !important;
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    position: relative;
    transition: var(--transition);
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    overflow: hidden;
}}
div[data-testid="stFileUploaderFile"] button:hover {{
    background: rgba(181, 86, 74, 0.3) !important;
}}
div[data-testid="stFileUploaderFile"] button [data-testid="stIconMaterial"] {{
    display: none !important;
}}
div[data-testid="stFileUploaderFile"] button span,
div[data-testid="stFileUploaderFile"] button p,
div[data-testid="stFileUploaderFile"] button div {{
    display: none !important;
}}
div[data-testid="stFileUploaderFile"] button::after {{
    content: "×" !important;
    color: var(--error) !important;
    font-size: 1.05rem !important;
    font-weight: 800;
    line-height: 1;
}}
div[data-testid="stFileUploaderFile"] ~ button {{
    display: none !important;
}}

/* ═══ عند وجود ملف مرفوع: يبقى فقط زر الحذف × ═══
   نُخفي كل الأزرار داخل عنصر الرفع (بما فيها زر "Add Files/إضافة ملفات"
   أينما كان في الـ DOM)، ثم نُعيد إظهار زر حذف الملف × فقط بأولوية أعلى.
   (إضافة ملف آخر تبقى ممكنة بالسحب والإفلات على المنطقة) */
div[data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"]) button {{
    display: none !important;
}}
div[data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"]) [data-testid="stFileUploaderFile"] button {{
    display: inline-flex !important;
}}

/* ═══════════════════════════════════════════════════════════
   محادثة الشات — بطاقات دفتر أستاذ (ledger) بدل فقاعات ملونة
   شريط رفيع بلون البراس/الحبر يميّز الدور، بدون توهج نابض.
   ═══════════════════════════════════════════════════════════ */
.msg-user-anchor, .msg-bot-anchor {{ display: none; }}

div[data-testid="stElementContainer"]:has(> .msg-user-anchor),
div[data-testid="stElementContainer"]:has(> .msg-bot-anchor) {{
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible;
    border: none !important;
}}

div[data-testid="stChatMessage"] {{
    display: flex !important;
    direction: ltr !important;
    align-items: flex-start;
    gap: 0.7rem;
    width: fit-content;
    max-width: 74%;
    padding: 0.85rem 1rem;
    margin-bottom: 0.85rem;
    position: relative;
    animation: turnIn 0.35s ease-out;
}}
@keyframes turnIn {{
    0%   {{ opacity: 0; transform: translateY(6px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
@media (prefers-reduced-motion: reduce) {{
    div[data-testid="stChatMessage"] {{ animation: none; }}
}}
@media (max-width: 720px) {{
    div[data-testid="stChatMessage"] {{
        max-width: 94%;
        padding: 0.78rem 0.82rem;
    }}
}}
div[data-testid="stChatMessage"] img {{
    width: 34px;
    height: 34px;
    flex-shrink: 0;
    border-radius: 50%;
    filter: grayscale(0.18) brightness(0.98);
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {{
    text-align: {text_align} !important;
    direction: {direction} !important;
    width: 100%;
    min-width: 0;
    overflow-wrap: anywhere;
}}

div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] p {{
    margin: 0 0 0.65rem 0;
    color: var(--text-primary);
    font-size: 0.98rem;
    line-height: 1.74;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] p:last-child {{
    margin-bottom: 0;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] strong {{
    color: var(--text-primary);
    font-weight: 700;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] ul,
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] ol {{
    margin: 0.35rem 0 0.72rem;
    padding-inline-start: 1.25rem;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] li {{
    margin: 0.22rem 0;
    line-height: 1.68;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] hr {{
    margin: 1rem 0 !important;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] code {{
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.14);
    border-radius: 5px;
    padding: 0.08rem 0.32rem;
    color: var(--brass-light);
    font-family: var(--font-mono);
    font-size: 0.88em;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] table {{
    width: 100%;
    margin: 0.8rem 0;
    border-collapse: collapse;
    overflow: hidden;
    border-radius: 8px;
    font-size: 0.9rem;
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] th,
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] td {{
    border: 1px solid var(--card-border);
    padding: 0.55rem 0.68rem;
    text-align: {text_align};
}}
div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] th {{
    background: rgba(96, 165, 250, 0.1);
    color: var(--text-primary);
    font-weight: 700;
}}

/* رسالة اليوزر */
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {{
    flex-direction: {user_flex} !important;
    margin-left: {user_margin_l} !important;
    margin-right: {user_margin_r} !important;
    border-radius: {user_radius} !important;
    background: var(--chat-user-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-{"left" if user_side == "right" else "right"}: 3px solid var(--brass) !important;
}}

/* رسالة البوت */
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {{
    flex-direction: {bot_flex} !important;
    margin-left: {bot_margin_l} !important;
    margin-right: {bot_margin_r} !important;
    border-radius: {bot_radius} !important;
    background:
        linear-gradient(180deg, rgba(96, 165, 250, 0.045), rgba(96, 165, 250, 0)),
        var(--chat-bot-bg) !important;
    border: 1px solid rgba(96, 165, 250, 0.18) !important;
    border-{"left" if bot_side == "right" else "right"}: 3px solid var(--line-600) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
}}

div[data-testid="stChatInput"] {{
    border: 1px solid rgba(96, 165, 250, 0.18) !important;
    border-radius: 8px !important;
    background:
        linear-gradient(180deg, rgba(96, 165, 250, 0.05), rgba(96, 165, 250, 0)),
        var(--ink-850) !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
}}
div[data-testid="stChatInput"]:focus-within {{
    border-color: var(--brass) !important;
    box-shadow: 0 0 0 1px var(--brass-ring), 0 16px 38px rgba(0, 0, 0, 0.28) !important;
}}
div[data-testid="stChatInput"] textarea {{
    color: var(--text-primary) !important;
    direction: {direction};
    font-family: var(--font-body) !important;
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
    min-height: 2.75rem !important;
    caret-color: var(--brass-light);
}}
div[data-testid="stChatInput"] textarea::placeholder {{
    color: var(--text-muted) !important;
    opacity: 1;
}}
div[data-testid="stChatInput"] button {{
    border-radius: 6px !important;
    transition: background 0.18s ease, transform 0.18s ease;
}}
div[data-testid="stChatInput"] button:hover {{
    background: rgba(96, 165, 250, 0.14) !important;
    transform: translateY(-1px);
}}
.chat-input-legal {{
    position: fixed;
    inset-inline: 0;
    bottom: 0.35rem;
    z-index: 999998;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    line-height: 1.2;
    text-align: center;
    direction: {direction};
    pointer-events: none;
}}

hr {{
    border: none !important;
    border-top: 1px solid var(--card-border) !important;
    height: 0 !important;
    margin: 0.9rem 0 !important;
}}

.stApp {{ min-height: 100vh; }}
.main .block-container {{
    padding-top: 0.5rem !important;
}}

.st-key-welcome_wrap {{
    min-height: 76vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.welcome-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 1rem;
    direction: {direction};
}}
.welcome-logo {{ width: 180px; height: 180px; border-radius: 50%; margin-bottom: 0.3rem; position: relative; z-index: 1; filter: drop-shadow(0 5px 14px rgba(96, 165, 250, 0.08)); }}

.welcome-title {{
    font-family: var(--font-heading);
    font-size: 3.2rem;
    font-weight: 700;
    font-style: normal;
    letter-spacing: -0.015em;
    background: linear-gradient(
        100deg,
        #FFFFFF 0%,
        #C7D2FE 16%,
        #6D8BF0 34%,
        #2E4C9E 50%,
        #7C5CE0 66%,
        #C4B5FD 84%,
        #FFFFFF 100%
    );
    background-size: 220% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    animation: titleFlow 7s linear infinite;
    margin-bottom: 0.3rem;
}}
@keyframes titleFlow {{
    0%   {{ background-position: 0% 50%; }}
    100% {{ background-position: 220% 50%; }}
}}
@media (prefers-reduced-motion: reduce) {{
    .welcome-title {{ animation: none; }}
}}
.welcome-subtitle {{
    font-family: var(--font-heading);
    font-size: 1.15rem;
    font-weight: 500;
    font-style: italic;
    letter-spacing: 0.01em;
    color: var(--brass-light) !important;
    margin-bottom: 1.4rem;
}}
.welcome-desc {{
    font-size: 1rem;
    color: #79828F !important;
    max-width: 540px;
    line-height: 1.65;
    margin-bottom: 1.6rem;
}}

/* ═══════════ شريط القائمة العلوي — ثابت (sticky)، يحتوي الشعار + أزرار التحكم ═══════════ */
.menu-bar-anchor {{ display: none; }}
div[data-testid="stElementContainer"]:has(.menu-bar-anchor) {{
    height: 0 !important; margin: 0 !important; padding: 0 !important; overflow: visible; border: none !important;
}}
div[data-testid="stElementContainer"]:has(.menu-bar-anchor) + div[data-testid="stLayoutWrapper"],
div[data-testid="stElementContainer"]:has(.menu-bar-anchor) + div[data-testid="stHorizontalBlock"] {{
    position: sticky;
    top: 0;
    z-index: 999997;
    direction: ltr;
    text-align: left;
    background: rgba(6, 15, 30, 0.9);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--card-border);
    min-height: 48px;
    padding: 0.35rem 0.2rem;
    margin: 0 0 1.1rem 0;
    align-items: center;
}}
div[data-testid="stElementContainer"]:has(.menu-bar-anchor) + div[data-testid="stLayoutWrapper"] div[data-testid="stHorizontalBlock"],
div[data-testid="stElementContainer"]:has(.menu-bar-anchor) + div[data-testid="stHorizontalBlock"] {{
    direction: ltr;
    align-items: center;
}}
div[data-testid="stElementContainer"]:has(.menu-bar-anchor) + div[data-testid="stLayoutWrapper"] div[data-testid="stHorizontalBlock"] > div,
div[data-testid="stElementContainer"]:has(.menu-bar-anchor) + div[data-testid="stHorizontalBlock"] > div {{
    display: flex;
    align-items: center;
}}

.menu-brand {{
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.6rem;
    direction: ltr;
}}
.menu-logo {{
    display: block;
    width: 70px;
    height: 70px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
}}
.menu-wordmark {{
    font-family: var(--font-heading);
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--text-primary);
}}
@media (max-width: 640px) {{
    .menu-wordmark {{ display: none; }}
}}

.st-key-menu_actions {{
    display: flex;
    flex-direction: row !important;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 0.5rem;
    width: 100%;
    direction: ltr;
}}
.st-key-menu_actions > div[data-testid="stElementContainer"] {{
    width: auto !important;
    flex: 0 0 auto !important;
}}
.st-key-menu_actions div.stButton {{ width: auto; }}
.st-key-menu_actions div.stButton > button {{
    width: auto;
    min-width: max-content;
    background: transparent;
    border: 1px solid var(--card-border);
    border-radius: 999px;
    padding: 0.4rem 1.1rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1.1;
    white-space: nowrap;
    justify-content: center;
    direction: {direction};
    text-align: center;
    color: var(--text-secondary);
}}
.st-key-menu_actions div.stButton > button:hover {{
    background: transparent;
    border-color: var(--brass);
    color: var(--brass-light);
    box-shadow: none;
    transform: none;
}}

/* ═══════════ تذييل الصفحة ═══════════ */
.app-footer {{
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    width: 100%;
    direction: {direction};
    text-align: {text_align};
    margin-top: 1.6rem;
    padding: 1.15rem 0 0.4rem;
    border-top: 1px solid var(--card-border);
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 1rem;
    letter-spacing: 0.02em;
}}
.footer-main {{
    display: flex;
    justify-content: {footer_justify};
    width: 100%;
    direction: ltr;
    text-align: {text_align};
}}
.footer-identity {{
    display: flex;
    flex-direction: {footer_identity_direction};
    align-items: center;
    gap: 0.85rem;
    max-width: 620px;
    direction: ltr;
}}
.footer-logo {{
    display: block;
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
}}
.footer-copy {{
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
    max-width: 560px;
    direction: {direction};
    text-align: {text_align};
}}
.footer-title {{
    font-family: var(--font-heading);
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
}}
.footer-summary {{
    color: var(--text-secondary);
    line-height: 1.45;
}}
.footer-legal {{
    width: 100%;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(30, 48, 80, 0.55);
    color: var(--text-muted);
    text-align: center;
}}
@media (max-width: 640px) {{
    .app-footer {{
        text-align: {text_align};
    }}
    .footer-main {{
        justify-content: {footer_justify};
        text-align: {text_align};
    }}
}}

.dash-header-block {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding-top: 0.2rem;
    margin-bottom: 0.6rem;
}}
.dash-title {{
    font-family: var(--font-heading);
    font-size: 2.3rem;
    font-weight: 600;
    font-style: {"normal" if lang == "ar" else "italic"};
    margin: 0;
    line-height: 1.1;
    color: var(--text-primary);
}}
.dash-title .accent {{
    color: var(--brass);
}}



.features-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    max-width: 700px;
    margin: 0.4rem auto 0;
    direction: ltr;
}}
.feature-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-top: 2px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1rem 0.8rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 0.3rem;
    transition: var(--transition);
    cursor: default;
    direction: {direction};
}}
.feature-card:hover {{
    border-top-color: var(--brass);
    transform: translateY(-3px);
}}
.feature-icon {{ margin-bottom: 0.15rem; }}
.feature-icon img {{ width: 22px; height: 22px; }}
.feature-title {{
    font-family: var(--font-heading);
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--text-primary);
}}
.feature-desc {{
    font-size: 0.76rem;
    color: var(--text-muted);
    line-height: 1.4;
}}
@media (max-width: 640px) {{
    .features-grid {{ grid-template-columns: 1fr; }}
}}

.ask-header {{ display: flex; align-items: center; gap: 0.55rem; margin: 0.4rem 0 0.15rem; direction: {direction}; }}
.ask-title {{ font-family: var(--font-heading); font-weight: 600; font-size: 1.3rem; margin: 0; color: var(--text-primary); }}
.ask-subtitle {{ color: var(--text-muted); font-size: 0.9rem; margin: 0 0 1.3rem; direction: {direction}; }}
</style>
"""
