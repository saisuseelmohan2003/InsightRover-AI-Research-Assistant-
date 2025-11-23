# streamlit code

import os
import requests, sqlite3, time, re
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from openai import OpenAI

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================================
# PAGE CONFIG (Beautiful UI)
# ================================
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
/* Center title */
h1 {text-align: center;}

/* Beautiful card styling */
.result-card {
    padding: 1.1rem;
    border-radius: 10px;
    background-color: #ffffff10;
    border: 1px solid #444;
    margin-bottom: 15px;
}

/* Summary box */
.summary-box {
    padding: 1.2rem;
    border-radius: 10px;
    background-color: #111;
    border-left: 4px solid #6a5acd;
}

/* Improve input box look */
textarea, input {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# ENV KEYS
# ================================
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# ================================
# GUARDRAILS
# ================================
PROFANITY = {
    "kill","hack","murder","harass",
    "assault","attack","abuse","threaten","stalk",
    "shoot","stab","kidnap","torture","poison",
    "destroy","bomb","explode","harm","injure","sabotage","strangle","burn"
}
_PROFANITY_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in PROFANITY) + r")\b", flags=re.IGNORECASE)

def sanitize_input(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    return text

def simple_moderation(text: str):
    if not text:
        return True, "OK"
    m = _PROFANITY_PATTERN.search(text.lower())
    if m:
        return False, f"Contains banned word: {m.group(1)}"
    return True, "OK"

# ================================
# MEMORY DB
# ================================
DB = "research_memory.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

def store(role, content):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO memory (role, content) VALUES (?,?)", (role, content))
    conn.commit()
    conn.close()

def fetch_recent(n=10):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, role, content, created_at FROM memory ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    return rows

# ================================
# RETRIEVER
# ================================
class SimpleRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []
        self.vecs = None

    def add(self, docs):
        texts = [t for _, t in docs]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if self.vecs is None:
            self.vecs = embeddings
        else:
            self.vecs = np.vstack([self.vecs, embeddings])
        self.docs.extend(docs)

    def retrieve(self, query, k=3):
        if self.vecs is None or len(self.docs) == 0:
            return []
        qv = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(qv, self.vecs)[0]
        idx = np.argsort(-sims)[:k]
        return [(self.docs[i][0], self.docs[i][1], float(sims[i])) for i in idx]


# ================================
# SERPAPI SEARCH TOOL
# ================================
def serpapi_search(query):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=8, verify=False)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    if "organic_results" not in data:
        return []

    results = []
    for item in data.get("organic_results", [])[:5]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", "")
        })
    return results


# ================================
# URL FETCH TOOL
# ================================
def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return r.text[:3000]
    except Exception as e:
        return f"Unable to fetch content: {e}"

# ================================
# OPENAI SUMMARY TOOL
# ================================
def openai_summarize(prompt):
    prompt = sanitize_input(prompt)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        return response.choices[0].message.content
    except:
        return "No summary returned."

# ================================
# AGENTS
# ================================
class ResearchAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def research(self, query):
        query = sanitize_input(query)
        ok, msg = simple_moderation(query)
        if not ok:
            raise ValueError(msg)

        search_results = serpapi_search(query)
        findings = []

        for r in search_results:
            page_text = fetch_url_text(r["url"])
            store("research_raw", r["title"] + "\n" + page_text[:2000])
            findings.append({
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"],
                "text": page_text
            })

        docs = [(f"doc{i}", f["text"]) for i, f in enumerate(findings)]
        if docs:
            self.retriever.add(docs)

        return findings

class SummaryAgent:
    def summarize(self, findings, question):
        prompt = f"Summarize the following research for the query: {question}\n\n"
        for f in findings:
            prompt += f"- {f['title']}\n  Snippet: {f['snippet']}\n  Content: {f['text'][:400]}\n\n"

        prompt += """
Provide a clear executive summary:
1. Key insights
2. Trends
3. Recommended next steps
"""

        summary = openai_summarize(prompt)
        store("summary", summary)
        return summary

# ================================
# WORKFLOW
# ================================
def run_workflow(query):
    init_db()
    retriever = SimpleRetriever()
    researcher = ResearchAgent(retriever)
    summarizer = SummaryAgent()

    findings = researcher.research(query)
    rag_hits = retriever.retrieve(query, k=3)

    for doc_id, text, score in rag_hits:
        findings.append({"title": f"RAG: {doc_id}", "snippet": text[:200], "text": text})

    summary = summarizer.summarize(findings, query)
    return findings, summary


# =======================================================
# ====================== UI BODY ========================
# =======================================================
st.title("🔍 AI Research Assistant")
st.subheader("Enter any research question and get web findings + executive summary")

query = st.text_input("Enter your research query:")

if st.button("Run Research", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("Running research..."):
        try:
            findings, summary = run_workflow(query)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()

    st.success("Research completed!")

    # ===========================
    # Summary
    # ===========================
    st.markdown("## 📝 Executive Summary")
    st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

    # ===========================
    # Findings
    # ===========================
    st.markdown("## 🌍 Web Findings")

    for f in findings:
        with st.expander(f"📌 {f['title']}"):
            st.markdown(f"**URL:** {f['url']}")
            st.markdown(f"**Snippet:** {f['snippet']}")
            st.markdown("---")
            st.text(f["text"])

    # ===========================
    # Memory
    # ===========================
    st.markdown("## 🧠 Recent Memory")
    rows = fetch_recent(5)

    for r in rows:
        st.markdown(
            f"""
            <div class="result-card">
                <strong>{r[1]}</strong><br>
                <small>{r[3]}</small><br><br>
                {r[2]}
            </div>
            """,
            unsafe_allow_html=True,
        )
