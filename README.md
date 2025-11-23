# 🔍 InsightRover  
### **AI-Powered Research Assistant with Web Search, Retrieval & Summarization**

InsightRover is an **AI-driven automated research system** built using **Python**, **OpenAI**, **SERPAPI**, **SentenceTransformers**, and **SQLite memory** — designed to fetch fresh information from the web, retrieve context using embeddings (RAG), and generate high-quality executive summaries.

This project showcases skills in:

✅ Data Science  
✅ NLP  
✅ Embeddings  
✅ Retrieval-Augmented Generation (RAG)
✅ Information Extraction  
✅ Agentic Workflows  
✅ LLM Integrations  
✅ Reasoning & Automation  

---

# 🚀 Features

### 🔎 **1. Web Research Agent (SERPAPI + URL Reader)**
The system fetches:
- Top Google results
- Webpage content
- Key snippets  
- Stores the raw research in memory

---

### 🧠 **2. RAG Retrieval (Embedding Search)**  
Uses **SentenceTransformer (MiniLM-L6-v2)** to embed and store documents.  
Retrieves the **top-k most relevant text chunks** using cosine similarity.

---

### 📝 **3. AI Summarization Agent (OpenAI GPT-4o-mini)**  
Generates a clean, structured summary that includes:
- Key insights  
- Trends  
- Recommendations  

---

### 💾 **4. Persistent Memory (SQLite)**  
Stores:
- Research snapshots  
- Summaries  
- Historical search data  

Allows you to track previous results and improves retrieval.

---

# 🎯 Use Cases

### 📚 **Academic / Market Research**
Ask: *“Impact of AI on supply chain industry 2025?”*

The tool fetches latest insights + summary.

---

### 📰 **News & Trend Analysis**
Ask: *“Latest updates on electric vehicle battery technology.”*

Gets breaking updates instantly.

---

### 💼 **Business & Competitor Insights**
Ask: *“Competitors of Swiggy in India and their business models.”*

You get multi-source consolidated insights.

---

### 👩‍💻 **Developer Tooling**
Useful for:
- Code understanding  
- Framework comparisons  
- Architecture summaries  

---

### 🤖 **AI Agent Demonstration**
This project is a great example of:
- Multi-step agent workflow  
- Tool calling  
- Retrieval pipelines  
- Guardrails & Moderation  

---

# 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| LLM API | OpenAI GPT-4o-mini |
| Web Search | SERPAPI |
| Web Content Extraction | Requests + Custom Parser |
| Embeddings | SentenceTransformer MiniLM-L6-v2 |
| Similarity Search | Cosine Similarity |
| Database | SQLite |
| Backend | Python |
| Memory | Local DB Persistence |
| Other | Requests, Regex Moderation |

---
