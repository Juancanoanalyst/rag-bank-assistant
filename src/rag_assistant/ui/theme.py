"""Look and feel of the UI: a header band and a few CSS refinements.

The palette is inspired by BBVA's public brand colours. This is an unofficial
prototype, so it carries no logo and says so in the header.
"""

NAVY = "#072146"
CORE_BLUE = "#004481"
AQUA = "#2DCCCD"

CSS = f"""
<style>
  .block-container {{ padding-top: 1.5rem; max-width: 900px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  .app-header {{
    background: {NAVY};
    color: #FFFFFF;
    padding: 1.1rem 1.4rem;
    border-radius: 4px;
    border-bottom: 4px solid {AQUA};
    margin-bottom: 1rem;
  }}
  .app-header h1 {{ color: #FFFFFF; font-size: 1.45rem; font-weight: 600; margin: 0; padding: 0; }}
  .app-header p {{ color: #D4EDFC; font-size: 0.9rem; margin: 0.25rem 0 0 0; }}
  .app-header .tag {{
    display: inline-block; margin-left: 0.6rem; padding: 0.1rem 0.55rem;
    border: 1px solid {AQUA}; border-radius: 999px;
    font-size: 0.7rem; font-weight: 400; color: {AQUA}; vertical-align: middle;
  }}

  section[data-testid="stSidebar"] {{ border-right: 1px solid #E1E6EC; }}
  section[data-testid="stSidebar"] h2 {{ color: {NAVY}; font-size: 1.05rem; }}

  button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 600; }}
  h3 {{ color: {NAVY}; font-size: 1.1rem; margin-top: 1.2rem; }}

  div[data-testid="stMetric"] {{
    background: #F4F6F9;
    border-left: 4px solid {CORE_BLUE};
    border-radius: 4px;
    padding: 0.8rem 1rem;
  }}
  div[data-testid="stMetricValue"] {{ color: {NAVY}; font-size: 1.6rem; }}
  div[data-testid="stMetricLabel"] p {{ color: #5A6B7B; font-size: 0.8rem; }}

  div[data-testid="stChatMessage"] {{ border-radius: 4px; padding: 0.9rem 1rem; }}
  div[data-testid="stChatMessageAvatarUser"] {{ background-color: {CORE_BLUE}; }}
  div[data-testid="stChatMessageAvatarAssistant"] {{ background-color: {NAVY}; }}
  div[data-testid="stChatMessageAvatarUser"] svg,
  div[data-testid="stChatMessageAvatarAssistant"] svg {{ color: #FFFFFF; }}
  .stButton button {{ border-radius: 4px; border-color: {CORE_BLUE}; color: {CORE_BLUE}; }}
</style>
"""

HEADER = """
<div class="app-header">
  <h1>Asistente de contenido web <span class="tag">Prototipo no oficial</span></h1>
  <p>Consulta la información publicada en bbva.com.co sin buscarla a mano.</p>
</div>
"""
