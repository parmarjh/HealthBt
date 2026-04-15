# PharmaRAG 🏥

A production-ready **Pharmaceutical Retrieval-Augmented Generation** system implementing the full architecture from the diagram — built in pure Python with zero mandatory external dependencies for local testing.

---

## Architecture

```
Actor → UI → Cache
              ↓
        Query Expander
         ├── Entity Extractor (NER)
         └── Ontology Expander
                    ↑
            Ontology Layer
             ├── Domain Ontology
             ├── NER (rule-based)
             └── Semantic Synonyms
              ↓
   Query Router & Retriever
    ├── Knowledge Source Classifier
    └── Context Retriever & Reranker
              ↓
       Retriever Validator
              ↓
      Response Generator (LLM)
              ↓
       Response Validator
              ↓
           └── Answer

🆕 **NEW: 3D Vision Layer**
     └── 3D Medical Recon Engine
          ├── Voxel Reconstruction
          └── Anatomical SDF Logic
```

---

## Quick Start

```bash
# 1. Clone / copy this folder
cd pharma_rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set LLM API key for real answers
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# 4. Run demo
python main.py

# 5. Ask a custom question
python main.py "What medication is used for high blood pressure?"

# 6. Run 3D Medical Vision (NEW)
streamlit run app_3d.py
```

---

## File Structure

| File | Layer | Description |
|------|-------|-------------|
| `ontology_layer.py` | Ontology Layer | Domain ontology, NER, semantic synonyms |
| `query_expander.py` | Query Expander | Entity extractor + ontology expander |
| `cache.py` | Cache | TTL-based in-memory query cache |
| `query_router.py` | Query Router & Retriever | Classifier, retriever, reranker, validator |
| `response_generator.py` | Response Generator | LLM synthesis + response validator |
| `main.py` | Orchestrator | Full pipeline wiring + CLI |
| `app_3d.py` | 3D Vision Layer | Streamlit-based 3D reconstructor |

---

## 🎓 3D Image Base Development (Zero to End)

This project now includes a **working algorithm for 3D image-based reconstruction**. We follow a knowledge-graded approach:

### 1. Zero Knowledge (Coordinate Prep)
We initialize a 3D coordinate tensor using `np.ogrid`. This creates the spatial vacuum where our medical models exist.

### 2. Low Knowledge (Basic SDF)
We implement **Signed Distance Fields (SDF)**. For example, a heart is not just a bunch of dots, but a mathematical intersection of ellipsoids:
`(x² + 2.25y² + z² - 1)³ - x²z³ - 0.1125y²z³ = 0`

### 3. Mid Knowledge (Biological Simulation)
Tissue isn't mathematically perfect. We apply **Random Normal Noise** and **Gaussian Smoothing** via `scipy.ndimage` to simulate cellular density and organic transparency.

### 4. High Knowledge (Isosurface Extraction)
We use the **Marching Cubes algorithm** logic (abstracted via Plotly `Isosurface`) to convert high-dimensional scalar fields into a viewable 3D mesh for surgeons and pharmacists.

---

## 🔄 Working with Latest Updates

To ensure you are using the latest 3D features:
1. **Update Dependencies**: `pip install streamlit plotly scipy`
2. **Launch Portal**: Run `streamlit run app_3d.py` to open the interactive 3D dashboard.
3. **Interactive Tuning**: Use the sidebar to increase "Voxel Resolution" to transition from Low to High knowledge models in real-time.

---

## Upgrading to Production

| Component | Current (Dev) | Production Upgrade |
|-----------|--------------|-------------------|
| NER | Rule-based dict matching | `spaCy` + `en_core_sci_md` model |
| Vector Store | Python list + term overlap | `ChromaDB` / `FAISS` + embeddings |
| Reranker | Score-based sort | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Cache | In-memory dict | Redis with TTL |
| LLM | Anthropic/OpenAI stub | Claude claude-opus-4-5 / GPT-4o |
| Database | Hardcoded lists | PostgreSQL + pgvector |

---

## LLM Provider Config

```python
# Anthropic (default)
pipeline = PharmaRAGPipeline(llm_provider="anthropic")

# OpenAI
pipeline = PharmaRAGPipeline(llm_provider="openai")

# Stub (no API key needed — for testing)
pipeline = PharmaRAGPipeline(llm_provider="stub")
```

---

## Example Output

```
════════════════════════════════════════════════════════════
  QUERY: What can I take for a headache?
════════════════════════════════════════════════════════════
  [Cache] MISS — proceeding through pipeline

▶ Step 1: Query Expander
  Entities   : drugs=[], conditions=['headache']
  Drug classes: ['analgesic']
  Variants    : 5 generated

▶ Step 2: Query Router & Retriever
  [Router] Sources: ['pharmacy_necessity'] | Triggered by: pharmacy drug class
  [Retriever] Retrieved 4 docs before validation
  [Validator] 3 docs passed validation (threshold=0.1).
  Top doc scores: [0.5, 0.33, 0.17]

▶ Step 3: Response Generator
  Sources used: ['ph002', 'ph001', 'ph003']

▶ Step 4: Response Validator
  ✅ PASSED
  Warning: Safety disclaimer auto-appended.

────────────────────────────────────────────────────────────
  FINAL ANSWER:
────────────────────────────────────────────────────────────
Paracetamol 500mg is the first-line option for headaches.
Ibuprofen 400mg is also effective but should be taken with food.

⚠️ Always consult a licensed pharmacist or physician before taking any medication.
════════════════════════════════════════════════════════════
```

---

## Contributing 🤝

We welcome contributions to the PharmaRAG project! Whether you're fixing bugs, adding new features, or improving documentation, here's how you can help:

1. **Fork the Repository**: Create your own copy of the project.
2. **Create a Branch**: Use a descriptive name for your branch (e.g., `feature/improved-ner`).
3. **Commit Your Changes**: Keep your commits small and descriptive.
4. **Open a Pull Request**: Describe your changes and why they are needed.

Please ensure your code follows the existing style and includes comments where necessary. For major changes, please open an issue first to discuss what you would like to change.

