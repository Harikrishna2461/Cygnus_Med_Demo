# Quick Start Guide - Dual Mode Medical Assistant

## 🚀 Start the App

```bash
cd backend
python app.py
```

Then open: `http://localhost:7860`

---

## 🏠 Landing Page (NEW!)

**Minimalistic Dark Theme**
```
╔════════════════════════════════════════╗
║                                        ║
║      Medical Assistant                 ║
║      Select a mode to begin            ║
║                                        ║
║  ┌──────────────┐  ┌──────────────┐   ║
║  │ 🩺 Clinical  │  │ 📚 Knowledge │   ║
║  │   Support    │  │    Chat      │   ║
║  │              │  │              │   ║
║  │  • Shunt     │  │  • Medical   │   ║
║  │    class.    │  │    research  │   ║
║  │  • Ligation  │  │  • General   │   ║
║  │  • Clinical  │  │    knowledge │   ║
║  │              │  │              │   ║
║  │ [Enter →]    │  │ [Enter →]    │   ║
║  └──────────────┘  └──────────────┘   ║
║                                        ║
╚════════════════════════════════════════╝

Colors: Dark Navy (#0f172a) + White text
```

---

## 📋 Mode 1: Clinical Support (`/clinical`)

**UI:** `frontend/index.html` (Unchanged)
**Database:** Sessions with `mode='clinical'`

### Features
- ✓ Describe venous condition in natural language
- ✓ Get shunt type classification (Type 1, 2, 3, etc.)
- ✓ Ligation planning and strategy
- ✓ Clinical reasoning and hemodynamics
- ✓ Complications & contraindications
- ✓ Submit feedback and ratings
- ✓ Full chat history

### Example Flow
```
User: "There is reflux at the SFJ flowing into the GSV..."
  ↓
[Clinical Pipeline]
  ↓
Response: "Type 1 Shunt (90% confidence) - Ligation at SFJ..."
```

---

## 💬 Mode 2: General Medical Chat (`/general`)

**UI:** `frontend/general.html` (NEW!)
**Database:** Sessions with `mode='general'`

### Features
- ✓ Ask ANY medical/surgical question
- ✓ Intelligent search (BM25 + Vector + Reranking)
- ✓ Domain guardrails (refuses off-topic questions)
- ✓ Evidence-based answers from 2,653 medical texts
- ✓ Full chat history (separate from clinical)
- ✓ Clean, minimalistic interface

### Example Flow
```
User: "What is Doppler ultrasound used for?"
  ↓
[Domain Check] ✓ Medical question
  ↓
[RAG Pipeline]
  - Vector search: top 50 results
  - BM25 search: top 50 results
  - Reranking: top 5 results
  ↓
[LLM] Uses top 5 + query → Answer
  ↓
Response: "Doppler ultrasound is used to assess blood flow..."
```

---

## 🛡️ Domain Guardrails (General Mode Only)

### Protected Questions (Accepted)
```javascript
✓ "What is Doppler ultrasound used for?"
✓ "Explain venous insufficiency"
✓ "What causes deep vein thrombosis?"
✓ "How is ligation performed?"
✓ "Describe the anatomy of veins"
```

### Rejected Questions
```javascript
✗ "Tell me a joke" → Politely refused
✗ "What's the weather?" → Politely refused
✗ "Write Python code" → Politely refused
✗ "Recommend a movie" → Politely refused
```

### Response Example
```
User: "Tell me a joke"
  ↓
[Guardrail Check] ✗ Not medical
  ↓
Response: "I can only assist with medical and surgical topics. 
Please ask me questions related to medical knowledge, procedures, 
anatomy, pathology, or clinical guidelines. 
How can I help with your medical question?"
```

---

## 📊 Database Structure

### Sessions Table
```sql
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  title TEXT,
  mode TEXT CHECK(mode IN ('clinical', 'general')),  -- NEW!
  created_at TEXT,
  updated_at TEXT
)
```

### Session Separation
```
Database: chat_history.db

sessions (1 table, 2 modes):
├── Clinical Sessions (mode='clinical')
│   ├── "Type 1 — Describe CHIVA Shunt..."
│   ├── "Type 3 — SFJ is incompetent..."
│   └── ...
└── General Sessions (mode='general')
    ├── "What is venous insufficiency?"
    ├── "How does ultrasound work?"
    └── ...
```

---

## 🔄 Data Flow

### Clinical Mode Flow
```
User Input
    ↓
[Natural Language Interpreter]
    ↓
[CHIVA Rules Engine]
    ↓
[LLM Classification]
    ↓
[Clinical Output Card] with:
  - Shunt type
  - Confidence
  - Ligation points
  - Clinical reasoning
    ↓
[Feedback Section]
```

### General Mode Flow
```
User Input
    ↓
[Domain Guardrail Check]
    ├→ ✗ Off-topic? → Polite Refusal
    └→ ✓ Medical? → Continue
        ↓
    [Vector Embedding]
        ↓
    [Hybrid Search]
    ├→ Vector Search (cosine) → top 50
    ├→ BM25 Search (keyword) → top 50
    └→ Merge & Deduplicate → candidates
        ↓
    [Cross-Encoder Reranking]
        ↓
    [LLM with Context]
        ↓
    [Text Response]
```

---

## 📁 File Structure

```
Cygnus_Med_Demo/Task_1_App/
├── backend/
│   ├── app.py                          [MODIFIED] Dual-mode routes
│   ├── chat_db.py                      [MODIFIED] Mode support
│   ├── general_chat_engine.py          [NEW] RAG + guardrails
│   ├── ingest_final_collection_fast.py [NEW] Collection builder
│   ├── rag_engine.py                   [unchanged] Clinical RAG
│   ├── config.py                       [unchanged]
│   ├── qdrant_storage/
│   │   ├── collection/
│   │   │   ├── ligation_knowledgebase_db_v2/     [Clinical]
│   │   │   └── final_structured_rag/             [General] 2,653 records
│   │   └── ...
│   └── chat_history.db                 [NEW] Mode column added
├── frontend/
│   ├── index.html                      [MODIFIED] Clinical UI
│   └── general.html                    [NEW] General chat UI
└── ...
```

---

## ✅ What You Get

| Aspect | Clinical | General |
|--------|----------|---------|
| **URL** | /clinical | /general |
| **UI** | Clinical-specific | Minimalistic |
| **Database** | Separate sessions | Separate sessions |
| **Input** | Describe venous condition | Ask any question |
| **Output** | Structured analysis | Text answer |
| **Knowledge** | Shunt classification | 2,653 medical texts |
| **Search** | Clinical rules | RAG (Vector + BM25 + Rerank) |
| **Guardrails** | N/A | Domain-specific ✓ |
| **Feedback** | Yes | No |
| **Chat History** | Separate | Separate |

---

## 🧪 Test It Out

### Test 1: Clinical Mode
1. Go to `http://localhost:7860`
2. Click "Clinical Support"
3. Type: "There is reflux at the SFJ flowing into the GSV down to mid-thigh"
4. See classification with ligation strategy

### Test 2: General Mode
1. Go to `http://localhost:7860`
2. Click "Knowledge Chat"
3. Type: "What is Doppler ultrasound used for?"
4. Get evidence-based answer

### Test 3: Guardrails
1. In General mode, type: "Tell me a joke"
2. See polite refusal message
3. Try again with: "What is venous insufficiency?"
4. Get medical answer

### Test 4: Separation
1. Create a chat in Clinical mode with title "Shunt Type 1"
2. Create a chat in General mode with title "Ultrasound Question"
3. Go to Clinical mode → see only "Shunt Type 1"
4. Go to General mode → see only "Ultrasound Question"

---

## 🎨 Theme & Colors

### Dark Navy + White (Matches Clinical UI)
```css
--slate-900: #0f172a   /* Dark navy background */
--white: #ffffff       /* Text */
--blue-600: #2563eb    /* Buttons */
--slate-200: #e2e8f0   /* Borders */
```

### Landing Page
- **Background:** Dark navy (#0f172a)
- **Cards:** Slightly lighter slate (#1e293b)
- **Text:** White with light slate for secondary
- **Borders:** Subtle (#334155)

---

## 🚨 Important Notes

1. **Database Migration:** First run will initialize the new `mode` column
2. **Session Filtering:** Old sessions (without mode) will still work as clinical
3. **Independent Histories:** Switching modes shows different chat histories
4. **RAG Collection:** Must have `final_structured_rag` with 2,653 records
5. **Guardrails:** Only active in general mode, not in clinical

---

## 📞 Troubleshooting

**Q: Both modes showing the same chats?**
A: Restart the app. DB needs fresh init.

**Q: Landing page not loading?**
A: Check `http://localhost:7860/` exactly. No trailing slash issues.

**Q: General mode not using RAG?**
A: Verify collection exists: `backend/qdrant_storage/collection/final_structured_rag/`

**Q: Guardrails not working?**
A: Make sure you're in `/general` route, not `/clinical`

---

## 🎉 You're All Set!

Everything is configured and ready. Both modes have:
- ✓ Separate databases
- ✓ Separate UIs
- ✓ Separate chat histories
- ✓ Minimalistic dark theme landing page
- ✓ Domain guardrails on general chat
- ✓ Full RAG functionality on general chat

Start the app and explore! 🚀
