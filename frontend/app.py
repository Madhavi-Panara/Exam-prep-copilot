"""Streamlit UI for Exam Prep Copilot.  Run:  streamlit run frontend/app.py"""
import os
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
DOC_TYPES = {"Notes": "notes", "Syllabus": "syllabus", "Past paper": "past_paper"}
CUSTOM_TOPIC = "✏️ Type my own topic…"

st.set_page_config(page_title="Exam Prep Copilot", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------------- helpers

def api(method, path, *, quiet=False, timeout=180, **kwargs):
    """Call the backend. Returns parsed JSON, or None after showing an error."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        if not quiet:
            st.error("Can't reach the backend. Start it with: `uvicorn backend.app.main:app --reload`")
        return None
    except requests.exceptions.Timeout:
        if not quiet:
            st.error("The request timed out. Try again.")
        return None
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if not quiet:
            st.error(detail if isinstance(detail, str) else str(detail))
        return None
    return r.json()


def init_state():
    defaults = {"chat": [], "quiz": None, "quiz_result": None, "topics": None, "plan": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def page_label(page):
    return f", p.{page}" if page else ""


def md_table(headers, rows):
    """Markdown table. Used instead of st.dataframe so the app doesn't need pandas
    (Windows Application Control sometimes blocks pandas' compiled files)."""

    def esc(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def bar_list(items, max_value, fmt):
    """Horizontal bars built from st.progress (no pandas needed). items = [(label, value)]."""
    for label, value in items:
        c1, c2 = st.columns([2, 3])
        c1.write(label)
        c2.progress(max(0.0, min(float(value) / max_value, 1.0)), text=fmt(value))


def show_sources(sources):
    if not sources:
        return
    cited = [s for s in sources if s["cited"]]
    with st.expander(f"Sources ({len(cited)} cited, {len(sources)} retrieved)"):
        for s in sources:
            tag = "✅ cited" if s["cited"] else "retrieved"
            st.markdown(f"**[{s['id']}]** {s['source']}{page_label(s['page'])} · _{tag}_")
            st.caption(s["snippet"])


# ----------------------------------------------------------------------------- sidebar

def sidebar():
    st.sidebar.title("📚 Exam Prep Copilot")
    health = api("GET", "/health", quiet=True, timeout=15)
    if health is None:
        st.sidebar.error("Backend offline")
        st.title("📚 Exam Prep Copilot")
        st.error("Can't reach the backend. Start it in another terminal:")
        st.code("uvicorn backend.app.main:app --reload", language="bash")
        st.stop()
    if not health["llm_configured"]:
        st.sidebar.warning(
            f"No API key set for provider '{health['llm_provider']}'. Add it to `.env` and restart the backend."
        )
    st.sidebar.caption(
        f"AI: {health['llm_provider']} · {health['model']}  \n"
        f"Search: {health['vector_backend']} · {health['chunks_indexed']} chunks indexed"
    )

    st.sidebar.subheader("Add study material")
    label = st.sidebar.selectbox("What is it?", list(DOC_TYPES), help="Syllabus and past papers power the topic map.")
    files = st.sidebar.file_uploader("PDF, TXT or Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.sidebar.button("Add to my library", disabled=not files, use_container_width=True):
        for f in files:
            with st.spinner(f"Reading {f.name}…"):
                res = api(
                    "POST", "/upload", files={"file": (f.name, f.getvalue())}, data={"doc_type": DOC_TYPES[label]}
                )
            if res:
                st.sidebar.success(f"{res['source']}: {res['chunks']} chunks")
        st.session_state.topics = None

    docs = (api("GET", "/documents", quiet=True) or {"documents": []})["documents"]
    st.sidebar.subheader("Your library")
    if not docs:
        st.sidebar.caption("Nothing yet. Upload a syllabus, notes or past papers.")
    for d in docs:
        c1, c2 = st.sidebar.columns([5, 1])
        c1.caption(f"**{d['source']}**  \n{d['doc_type'].replace('_', ' ')} · {d['chunks']} chunks")
        if c2.button("🗑", key=f"del_{d['source']}", help="Remove"):
            api("DELETE", f"/documents/{quote(d['source'], safe='')}")
            st.session_state.topics = None
            st.rerun()


# ----------------------------------------------------------------------------- tabs

def tab_ask():
    st.caption("Ask anything. Answers come from your own notes, with citations.")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources"))

    prompt = st.chat_input("Ask a question about your notes…")
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching your notes…"):
                res = api("POST", "/ask", json={"question": prompt, "history": history})
            if res:
                st.markdown(res["answer"])
                show_sources(res["sources"])
                st.session_state.chat.append({"role": "user", "content": prompt})
                st.session_state.chat.append({"role": "assistant", "content": res["answer"], "sources": res["sources"]})
    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()


def load_topics():
    if st.session_state.topics is None:
        st.session_state.topics = api("GET", "/topics", quiet=True) or {"topics": []}
    return st.session_state.topics


def render_quiz_results(result):
    st.subheader(f"Score: {result['score']:g} / {result['out_of']}  ({result['percent']}%)")
    st.progress(min(result["percent"], 100) / 100)
    for res in result["results"]:
        score = res["score"]
        icon = "❔" if score is None else "✅" if score == 1 else "🟡" if score == 0.5 else "❌"
        with st.expander(f"{icon} {res['question']}"):
            st.markdown(f"**Your answer:** {res['your_answer'] or '—'}")
            st.markdown(f"**Correct answer:** {res['correct_answer']}")
            if res["feedback"]:
                st.info(res["feedback"])
            if res["explanation"]:
                st.caption(f"Why: {res['explanation']}")
            if res.get("source"):
                st.caption(f"Source: {res['source']}")
    if st.button("Take another quiz"):
        st.session_state.quiz = None
        st.session_state.quiz_result = None
        st.rerun()


def tab_quiz():
    if st.session_state.quiz_result:
        render_quiz_results(st.session_state.quiz_result)
        return

    quiz = st.session_state.quiz
    if quiz is None:
        names = [t["name"] for t in load_topics().get("topics", [])]
        c1, c2, c3 = st.columns([3, 1, 1])
        choice = c1.selectbox("Topic", names + [CUSTOM_TOPIC])
        topic = c1.text_input("Your topic") if choice == CUSTOM_TOPIC else choice
        n = c2.slider("Questions", 3, 10, 5)
        difficulty = c3.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        if st.button("Generate quiz", type="primary", disabled=not topic):
            with st.spinner("Writing questions from your notes…"):
                res = api("POST", "/quiz", json={"topic": topic, "n": n, "difficulty": difficulty})
            if res:
                st.session_state.quiz = res
                st.rerun()
        if not names:
            st.caption("Tip: upload a syllabus and use the Topics tab to get a topic list to pick from.")
        return

    st.subheader(f"Quiz: {quiz['topic']}  ·  {quiz['difficulty']}")
    with st.form(f"quiz_{quiz['quiz_id']}"):
        answers = {}
        for i, q in enumerate(quiz["questions"], 1):
            st.markdown(f"**Q{i}.** {q['question']}")
            key = f"{quiz['quiz_id']}_{q['id']}"
            if q["type"] == "mcq":
                choice = st.radio("Choose one", q["options"], index=None, key=key, label_visibility="collapsed")
                answers[q["id"]] = str(q["options"].index(choice)) if choice is not None else ""
            else:
                answers[q["id"]] = st.text_area("Your answer", key=key, height=100, label_visibility="collapsed")
        submitted = st.form_submit_button("Submit answers", type="primary")
    if submitted:
        with st.spinner("Marking…"):
            res = api("POST", f"/quiz/{quiz['quiz_id']}/submit", json={"answers": answers})
        if res:
            st.session_state.quiz_result = res
            st.rerun()
    if st.button("Discard quiz"):
        st.session_state.quiz = None
        st.rerun()


def tab_topics():
    st.caption("Builds a topic map from your syllabus, ranked by how much past papers weight each topic.")
    if st.button("Analyze my syllabus", type="primary"):
        with st.spinner("Reading your syllabus and past papers…"):
            res = api("POST", "/topics/analyze")
        if res:
            st.session_state.topics = res
    data = load_topics()
    topics = data.get("topics", [])
    if not topics:
        st.info("No topic map yet. Upload a syllabus (and past papers if you have them), then click Analyze.")
        return
    if not data.get("has_past_papers"):
        st.caption("No past papers uploaded, so importance is based on the syllabus only.")
    bar_list([(t["name"], t["importance"]) for t in topics], 5, lambda v: f"{v}/5")
    st.markdown(
        md_table(
            ["Topic", "Importance (1-5)", "Past-paper questions", "Subtopics"],
            [[t["name"], t["importance"], t["past_paper_questions"], ", ".join(t["subtopics"])] for t in topics],
        )
    )


def tab_progress():
    data = api("GET", "/progress")
    if not data:
        return
    if not data["questions_answered"]:
        st.info("Take a quiz and your results will show up here.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Quizzes taken", data["quizzes_taken"])
    c2.metric("Questions answered", data["questions_answered"])
    c3.metric("Overall accuracy", f"{round(data['overall_accuracy'] * 100)}%")

    if data["weak_topics"]:
        st.warning("Weak topics to revise: " + ", ".join(data["weak_topics"]))
    else:
        st.success("No weak topics right now. Nice work.")

    st.subheader("Accuracy by topic")
    bar_list([(t["topic"], t["accuracy"]) for t in data["topics"]], 1.0, lambda v: f"{round(v * 100)}%")
    st.markdown(
        md_table(
            ["Topic", "Accuracy", "Questions", "Status"],
            [
                [t["topic"], f"{round(t['accuracy'] * 100)}%", t["attempts"], "⚠️ weak" if t["weak"] else "✅ ok"]
                for t in data["topics"]
            ],
        )
    )


def plan_to_markdown(plan):
    lines = ["# Study plan", "", plan["summary"], ""]
    for d in plan["days"]:
        lines += [f"## Day {d['day']}: {d['title']} ({d['hours']} h)", ""]
        if d["focus_topics"]:
            lines.append("Focus: " + ", ".join(d["focus_topics"]))
        lines += [f"- {t}" for t in d["tasks"]] + [""]
    return "\n".join(lines)


def tab_plan():
    st.caption("A day-by-day revision plan that weights important topics and your weak spots.")
    c1, c2 = st.columns(2)
    days = c1.slider("Days until exam", 1, 30, 7)
    hours = c2.slider("Study hours per day", 0.5, 8.0, 2.0, step=0.5)
    if st.button("Generate plan", type="primary"):
        with st.spinner("Planning…"):
            res = api("POST", "/plan", json={"days": days, "hours_per_day": hours})
        if res:
            st.session_state.plan = res
    plan = st.session_state.plan
    if not plan:
        return
    st.write(plan["summary"])
    if plan["weak_topics"]:
        st.caption("Prioritising your weak topics: " + ", ".join(plan["weak_topics"]))
    for d in plan["days"]:
        with st.expander(f"Day {d['day']}: {d['title']}  ·  {d['hours']} h", expanded=d["day"] == 1):
            if d["focus_topics"]:
                st.markdown("**Focus:** " + ", ".join(d["focus_topics"]))
            for t in d["tasks"]:
                st.markdown(f"- {t}")
    st.download_button("Download plan (Markdown)", plan_to_markdown(plan), file_name="study_plan.md")


# ----------------------------------------------------------------------------- main

init_state()
sidebar()
st.title("📚 Exam Prep Copilot")
ask, quiz_tab, topics, progress, plan = st.tabs(
    ["💬 Ask your notes", "📝 Quiz me", "🗺️ Topics", "📈 Progress", "📅 Study plan"]
)
with ask:
    tab_ask()
with quiz_tab:
    tab_quiz()
with topics:
    tab_topics()
with progress:
    tab_progress()
with plan:
    tab_plan()