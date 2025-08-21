# Reaction Plot Simulator — Streamlit single-file app
# ---------------------------------------------------
# Features:
# - Mermaid input + live preview (client-side mermaid.js)
# - Regex-based Mermaid flowchart parser (nodes/edges)
# - Emoji/label mapping UI
# - Traversal from start node(s) to build a scene sequence
# - Animated GIF/MP4 export (Pillow + imageio)
# - Storyboard PDF export (reportlab)
#
# Note: Color emoji fonts vary by OS. If your server font lacks emoji,
# frames will still render text; emojis may appear as tofu boxes.

from __future__ import annotations
import re
import io
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import imageio
import networkx as nx

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ----------------------------
# Utilities & Defaults
# ----------------------------

st.set_page_config(page_title="Reaction Plot Simulator", page_icon="🧪", layout="wide")

EMOJI_HINT = "🧪🔬➡️🛡️💥👨🏼‍🍳👴🏻🧑‍💼👔❓💻📖"
DEFAULT_FONT = ImageFont.load_default()  # Simple fallback

DINNER_PARTY_MERMAID = """flowchart LR
A[Stable State: Dave & Chloe hosting] -->|Trigger| B[Uncle Brian arrives]
B -->|Escalates to| C[Brexit Comment]
C -->|Buffer| D[Chloe diffuses]
D --> E[Resolution: Dessert]
"""

JOB_INTERVIEW_MERMAID = """flowchart TD
S[Introductions] -->|Trigger| Q[Tough Technical Question]
Q -->|Escalation| P[Panic]
P -->|Buffer Story| B["Well, I learned..." story]
B --> R[Resolution: Handshakes]
"""

JOB_PRESET_MAP = {
    "S": {"name": "Introductions", "emoji": "🤝"},
    "Q": {"name": "Technical Question", "emoji": "❓💻"},
    "P": {"name": "Panic", "emoji": "😰"},
    "B": {"name": "Buffer Story", "emoji": "🛡️📖"},
    "R": {"name": "Resolution", "emoji": "✅"},
}
DINNER_PRESET_MAP = {
    "A": {"name": "Stable State", "emoji": "👨🏼‍🍳👩🏼"},
    "B": {"name": "Uncle Brian Arrives", "emoji": "👴🏻"},
    "C": {"name": "Brexit Comment", "emoji": "🗣️💥"},
    "D": {"name": "Diffuse", "emoji": "🛡️"},
    "E": {"name": "Dessert", "emoji": "🍰"},
}

# Narration pool (overlay/subtitles). Users can edit this in UI.
DEFAULT_NARRATION = [
    "Now, the catalyst enters. Always with an opinion, never with a bottle.",
    "And... we have ignition. The exotherm. Point of no return.",
]

# ----------------------------
# Data models
# ----------------------------

@dataclass
class Node:
    id: str
    label: str
    emoji: str = ""
    role: str = ""
    chemistry: str = ""

@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    emoji: str = ""

@dataclass
class Plot:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

# ----------------------------
# Mermaid Parser (simple flowchart subset)
# Supports lines like:
#   A[Label] --> B[Label]
#   X(Decision) -->|Escalates to| Y{Crisis}
#   shapes: [], (), {}, [[]]
# ----------------------------

NODE_PATTERN = re.compile(r"""
    (?P<id>[A-Za-z0-9_]+)           # Node ID
    \s*
    (?P<bracket>[\[\(\{]{1,2})      # opening bracket(s)
    (?P<label>.+?)
    (?P=bracket).?                  # Balancing handled loosely (visual only)
""", re.VERBOSE)

EDGE_PATTERN = re.compile(r"""
    (?P<src>[A-Za-z0-9_]+)        # src id
    \s*-\->\s*                    # arrow -->
    (?:\|\s*(?P<lbl>[^|]+?)\s*\|\s*)?   # optional |label|
    (?P<dst>[A-Za-z0-9_]+)        # dst id
""", re.VERBOSE)

NODE_DEF_PATTERN = re.compile(r"""
    ^\s*(?P<id>[A-Za-z0-9_]+)
    \s*(?P<bracket>[\[\(\{]{1,2})
    (?P<label>.+?)
    (?P=bracket).*
""", re.VERBOSE)

def parse_mermaid(mermaid: str) -> Plot:
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    lines = [ln.strip() for ln in mermaid.splitlines()]
    for ln in lines:
        if not ln or ln.startswith("%") or ln.startswith("%%"):
            continue
        # Node definitions (e.g., A[Label])
        nd = NODE_DEF_PATTERN.match(ln)
        if nd:
            nid = nd.group("id")
            label = nd.group("label").strip()
            if nid not in nodes:
                nodes[nid] = Node(id=nid, label=_strip_bracket(label))
        # Edges (e.g., A -->|Escalates| B)
        ed = EDGE_PATTERN.search(ln)
        if ed:
            src = ed.group("src")
            dst = ed.group("dst")
            lbl = (ed.group("lbl") or "").strip()
            edges.append(Edge(src=src, dst=dst, label=lbl))
            # Ensure nodes exist if only referenced in edges
            if src not in nodes:
                nodes[src] = Node(id=src, label=src)
            if dst not in nodes:
                nodes[dst] = Node(id=dst, label=dst)
    return Plot(nodes=nodes, edges=edges)

def _strip_bracket(label: str) -> str:
    return label.replace("]", "").replace("[", "").replace("(", "").replace(")", "").replace("{","").replace("}","")

# ----------------------------
# Traversal (deterministic)
# - Find nodes with no inbound edges as starts
# - Follow edges in insertion order; avoid cycles with visited set
# ----------------------------

def traverse(plot: Plot) -> List[Tuple[Optional[Edge], Node]]:
    g = nx.DiGraph()
    for node in plot.nodes.values():
        g.add_node(node.id)
    for e in plot.edges:
        g.add_edge(e.src, e.dst, label=e.label)
    # Start nodes = in_degree == 0; if none, fallback to first defined
    starts = [n for n in g.nodes if g.in_degree(n) == 0] or [next(iter(g.nodes))]
    sequence: List[Tuple[Optional[Edge], Node]] = []
    visited: Set[str] = set()

    def walk(nid: str):
        if nid in visited:
            return
        visited.add(nid)
        node = plot.nodes[nid]
        sequence.append((None, node))
        # Traverse outgoing edges
        for _, dst, data in g.out_edges(nid, data=True):
            e = _find_edge(plot.edges, nid, dst)
            if e:
                sequence.append((e, plot.nodes[dst]))
                walk(dst)

    for s in starts:
        walk(s)
    # Collapse consecutive (edge,node) so pattern is: Node0, (Edge,Node1), (Edge,Node2)...
    # Already constructed that way above.
    return sequence

def _find_edge(edges: List[Edge], src: str, dst: str) -> Optional[Edge]:
    for e in edges:
        if e.src == src and e.dst == dst:
            return e
    return None

# ----------------------------
# Frame / Animation builder
# ----------------------------

def make_frame_base(w=960, h=540, bg="#111111") -> Image.Image:
    return Image.new("RGB", (w, h), color=bg)

def draw_text_centered(img: Image.Image, text: str, y: int, font: ImageFont.ImageFont, fill=(240,240,240)):
    d = ImageDraw.Draw(img)
    W, H = img.size
    w, h = d.textsize(text, font=font)
    d.text(((W - w) // 2, y), text, font=font, fill=fill)

def compose_scene_frame(title: str, subtitle: str, emoji: str, w=960, h=540) -> Image.Image:
    img = make_frame_base(w, h)
    big = DEFAULT_FONT
    title_font = DEFAULT_FONT
    # Try to load a nicer font if available
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 36)
        big = ImageFont.truetype("DejaVuSans.ttf", 68)
    except Exception:
        pass
    draw_text_centered(img, emoji or "", 120, big)
    draw_text_centered(img, title, 240, title_font)
    if subtitle:
        draw_text_centered(img, subtitle, 300, title_font)
    return img

def compose_transition_frame(edge_lbl: str, edge_emoji: str, w=960, h=540) -> Image.Image:
    label = edge_lbl or "Transition"
    emoji = edge_emoji or "➡️"
    return compose_scene_frame(label, "→", emoji, w, h)

def shake_frames(base: Image.Image, n=6, px=6) -> List[Image.Image]:
    frames = []
    for i in range(n):
        dx = ((i % 2) * 2 - 1) * px
        dy = -px if i % 3 == 0 else px
        shaken = base.copy()
        canvas = Image.new("RGB", base.size, "#111111")
        canvas.paste(shaken, (dx, dy))
        frames.append(canvas)
    return frames

def build_animation_frames(sequence: List[Tuple[Optional[Edge], Node]]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for idx, item in enumerate(sequence):
        edge, node = item
        if edge is None:
            # First node in a branch
            base = compose_scene_frame(node.label, node.role or node.chemistry, node.emoji)
            frames.extend([base] * 6)
        else:
            # Transition
            tframe = compose_transition_frame(edge.label, edge.emoji)
            frames.extend([tframe] * 4)
            # Arrival node
            base = compose_scene_frame(node.label, node.role or node.chemistry, node.emoji)
            # Add a little shake on likely "Trigger/Conflict" words
            if any(k in (node.label.lower() + " " + (edge.label or "").lower()) for k in ["trigger", "conflict", "crisis", "panic", "exotherm"]):
                frames.extend(shake_frames(base, n=6))
            frames.extend([base] * 6)
    return frames

def save_gif(frames: List[Image.Image], fps: int = 6) -> bytes:
    buf = io.BytesIO()
    # duration is per-frame ms
    imageio.mimsave(buf, [f.convert("P", palette=Image.ADAPTIVE) for f in frames], format='GIF', duration=1.0/fps)
    return buf.getvalue()

def save_mp4(frames: List[Image.Image], fps: int = 24) -> Optional[bytes]:
    try:
        arr = [imageio.v3.imread(imageio.core.asarray(f)) if not isinstance(f, (bytes, bytearray)) else f for f in frames]
        # imageio can write from array data directly
        tmp = io.BytesIO()
        # imageio-ffmpeg requires a filename; fallback to temp file
        import tempfile, numpy as np
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            imageio.mimsave(tf.name, [np.array(f) for f in frames], fps=fps, codec="libx264", quality=7)
            tf.flush()
            tf.seek(0)
            data = open(tf.name, "rb").read()
        return data
    except Exception as e:
        st.warning(f"MP4 export unavailable: {e}")
        return None

def save_storyboard_pdf(sequence: List[Tuple[Optional[Edge], Node]]) -> Optional[bytes]:
    if not REPORTLAB_OK:
        st.warning("reportlab is not installed; PDF export disabled.")
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    margin = 2 * cm
    y = H - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Reaction Plot Storyboard")
    y -= 1.2 * cm
    c.setFont("Helvetica", 12)
    step = 1
    for edge, node in sequence:
        if y < margin + 4*cm:
            c.showPage()
            y = H - margin
        title = f"{step}. {node.label}"
        sub = f"Emoji: {node.emoji or '-'}   Role: {node.role or '-'}   Chemistry: {node.chemistry or '-'}"
        c.drawString(margin, y, title)
        y -= 0.7*cm
        if edge is not None:
            c.drawString(margin, y, f"Transition: {edge.label or '-'} {edge.emoji or ''}")
            y -= 0.7*cm
        c.line(margin, y, W - margin, y)
        y -= 0.6*cm
        step += 1
    c.save()
    return buf.getvalue()

# ----------------------------
# UI Components
# ----------------------------

st.markdown(
    "<h1>🧪 Reaction Plot Simulator</h1><p>Map stories to a universal reaction structure, visualize, and export animations.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("📁 Input Panel")
    story_type = st.selectbox(
        "Story Type",
        ["Dinner Party", "Job Interview", "Custom"],
        index=0
    )
    if story_type == "Dinner Party":
        mermaid = st.text_area("Mermaid Flowchart", value=DINNER_PARTY_MERMAID, height=180)
        preset = DINNER_PRESET_MAP
    elif story_type == "Job Interview":
        mermaid = st.text_area("Mermaid Flowchart", value=JOB_INTERVIEW_MERMAID, height=180)
        preset = JOB_PRESET_MAP
    else:
        mermaid = st.text_area("Mermaid Flowchart", value="flowchart LR\nA[Start] --> B[Conflict]\nB --> C[Resolution]\n", height=180)
        preset = {}

    narration_pool = st.text_area("Narration lines (optional, one per line)", value="\n".join(DEFAULT_NARRATION), height=110)
    fps = st.slider("GIF FPS", min_value=4, max_value=15, value=6)
    st.caption(f"Emoji ideas: {EMOJI_HINT}")

# Parse Mermaid
plot = parse_mermaid(mermaid)

# Node & Edge mapping forms
st.subheader("🧬 Character & Scene Mapping")
cols = st.columns([2, 1, 1, 1])
cols[0].markdown("**Node (ID → Label)**")
cols[1].markdown("**Emoji**")
cols[2].markdown("**Role**")
cols[3].markdown("**Chemistry**")

for nid, node in plot.nodes.items():
    p_emoji = preset.get(nid, {}).get("emoji", "")
    p_name = preset.get(nid, {}).get("name", node.label)
    c0, c1, c2, c3 = st.columns([2,1,1,1])
    new_label = c0.text_input(f"{nid}", value=p_name or node.label, key=f"node_label_{nid}")
    new_emoji = c1.text_input(" ", value=p_emoji, key=f"node_emoji_{nid}")
    new_role = c2.text_input(" ", value="", key=f"node_role_{nid}")
    new_chem = c3.text_input(" ", value="", key=f"node_chem_{nid}")
    node.label = new_label
    node.emoji = new_emoji
    node.role = new_role
    node.chemistry = new_chem

st.subheader("🧪 Transition Mapping")
ec1, ec2, ec3 = st.columns([3, 2, 1])
ec1.markdown("**Edge (src → dst) label**")
ec2.markdown("**Edge Emoji**")
ec3.write("")

for i, e in enumerate(plot.edges):
    label_val = ec1.text_input(f"{e.src} → {e.dst}", value=e.label, key=f"edge_label_{i}")
    emoji_val = ec2.text_input(" ", value=e.emoji or "➡️", key=f"edge_emoji_{i}")
    e.label = label_val
    e.emoji = emoji_val

# ----------------------------
# Live Preview (Mermaid)
# ----------------------------

st.subheader("🎬 Preview Canvas")
preview_col, actions_col = st.columns([3, 1])

with preview_col:
    # Render Mermaid client-side via CDN
    mermaid_container = f"""
    <div class="mermaid">
    {mermaid}
    </div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: true, theme: "dark" }});
      // trigger re-render on load
      setTimeout(() => mermaid.run(), 50);
    </script>
    """
    st.components.v1.html(mermaid_container, height=420, scrolling=True)

with actions_col:
    st.markdown("**Generate & Export**")
    gen = st.button("Generate Plot")
    export_gif = st.button("Export GIF")
    export_mp4 = st.button("Export MP4")
    export_pdf = st.button("Export Storyboard (PDF)")

# Build sequence and frames on demand
if "sequence_cache" not in st.session_state:
    st.session_state.sequence_cache = None
if "frames_cache" not in st.session_state:
    st.session_state.frames_cache = None

if gen:
    seq = traverse(plot)
    st.session_state.sequence_cache = seq
    frames = build_animation_frames(seq)
    st.session_state.frames_cache = frames
    st.success(f"Generated {len(frames)} frames from {len(plot.nodes)} nodes and {len(plot.edges)} edges.")

# Show last render as a static preview
if st.session_state.frames_cache:
    st.image(st.session_state.frames_cache[0], caption="First frame preview", use_column_width=True)

# Exports
if export_gif:
    if not st.session_state.frames_cache:
        st.warning("Generate the plot first.")
    else:
        data = save_gif(st.session_state.frames_cache, fps=fps)
        st.download_button("Download GIF", data, file_name="reaction_plot.gif", mime="image/gif")

if export_mp4:
    if not st.session_state.frames_cache:
        st.warning("Generate the plot first.")
    else:
        data = save_mp4(st.session_state.frames_cache, fps=24)
        if data:
            st.download_button("Download MP4", data, file_name="reaction_plot.mp4", mime="video/mp4")

if export_pdf:
    if not st.session_state.sequence_cache:
        st.warning("Generate the plot first.")
    else:
        data = save_storyboard_pdf(st.session_state.sequence_cache)
        if data:
            st.download_button("Download Storyboard.pdf", data, file_name="reaction_storyboard.pdf", mime="application/pdf")

st.caption("Tip: emojis render best when the OS font supports them. Consider installing a color emoji font on your server for richer frames.")
