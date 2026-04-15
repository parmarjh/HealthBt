"""
Query Router & Retriever Layer
# Components:
#   - KnowledgeSourceClassifier  -> decides which DB(s) to query
#   - ContextRetrieverReranker   -> fetches docs and reranks by relevance
#   - RetrieverValidator         -> post-retrieval quality gate
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from query_expander import ExpandedQuery


# --- Simulated Document Databases -------------------------------------------
# In production replace these with FAISS / ChromaDB / Pinecone vector stores
# or SQL/NoSQL backends backed by real pharmacy data.

PHARMACY_NECESSITY_DB = [
    {"id": "ph001", "text": "Ibuprofen 400mg is used for mild to moderate pain and fever. Take with food.", "tags": ["pain", "fever", "nsaid", "ibuprofen"]},
    {"id": "ph002", "text": "Paracetamol 500mg is the first-line treatment for headache and fever in adults.", "tags": ["headache", "fever", "analgesic", "paracetamol"]},
    {"id": "ph003", "text": "Cetirizine 10mg is an antihistamine used for allergic rhinitis and urticaria.", "tags": ["allergy", "antihistamine", "cetirizine"]},
    {"id": "ph004", "text": "Omeprazole 20mg reduces stomach acid; indicated for GERD and peptic ulcers.", "tags": ["acid", "gerd", "ulcer", "omeprazole"]},
    {"id": "ph005", "text": "Amoxicillin 500mg is a broad-spectrum antibiotic for bacterial infections.", "tags": ["infection", "antibiotic", "amoxicillin"]},
    {"id": "ph006", "text": "Salbutamol inhaler (100mcg/dose) provides rapid bronchodilation during asthma attacks.", "tags": ["asthma", "bronchodilator", "salbutamol"]},
]

DRUG_NECESSITY_DB = [
    {"id": "dn001", "text": "Metformin 500mg is the first-line drug for Type 2 Diabetes. Improves insulin sensitivity.", "tags": ["diabetes", "antidiabetic", "metformin"]},
    {"id": "dn002", "text": "Lisinopril 10mg is an ACE inhibitor for hypertension and heart failure.", "tags": ["hypertension", "antihypertensive", "lisinopril"]},
    {"id": "dn003", "text": "Atorvastatin 20mg lowers LDL cholesterol; reduces cardiovascular risk.", "tags": ["cholesterol", "statin", "atorvastatin"]},
    {"id": "dn004", "text": "Sertraline 50mg is an SSRI indicated for depression, OCD, and anxiety disorders.", "tags": ["depression", "anxiety", "antidepressant", "sertraline"]},
    {"id": "dn005", "text": "Warfarin is an anticoagulant used for deep vein thrombosis and atrial fibrillation.", "tags": ["anticoagulant", "dvt", "warfarin"]},
    {"id": "dn006", "text": "Amlodipine 5mg is a calcium channel blocker for hypertension and stable angina.", "tags": ["hypertension", "angina", "amlodipine"]},
]

MLM_LLM_KNOWLEDGE = [
    {"id": "llm001", "text": "Drug-drug interactions: NSAIDs combined with anticoagulants increase bleeding risk.", "tags": ["interaction", "nsaid", "anticoagulant"]},
    {"id": "llm002", "text": "Contraindication: ACE inhibitors are contraindicated in pregnancy (Category D).", "tags": ["contraindication", "ace inhibitor", "pregnancy"]},
    {"id": "llm003", "text": "Paracetamol overdose causes acute liver failure; max dose 4g/day in adults.", "tags": ["overdose", "paracetamol", "liver"]},
]


# --- Knowledge Source Classifier ---------------------------------------------

@dataclass
class ClassificationResult:
    sources: list[str]              # which DBs to query
    reasoning: str
    confidence: float = 1.0


class KnowledgeSourceClassifier:
    """
    Decides which knowledge sources to route the query to
    based on drug classes and entities found in the expanded query.
    """

    PHARMACY_SIGNALS = {"analgesic", "antihistamine", "bronchodilator", "antibiotic", "nsaid"}
    DRUG_SIGNALS     = {"antihypertensive", "antidiabetic", "statin", "antidepressant", "anticoagulant"}
    LLM_SIGNALS      = {"interaction", "contraindication", "overdose", "pregnancy", "side effect"}

    def classify(self, expanded: ExpandedQuery) -> ClassificationResult:
        sources = []
        signals_found = []

        dc_set = set(dc.lower() for dc in expanded.drug_classes)
        all_text = (expanded.original + " " + " ".join(expanded.expanded_terms)).lower()

        if dc_set & self.PHARMACY_SIGNALS:
            sources.append("pharmacy_necessity")
            signals_found.append("pharmacy drug class")

        if dc_set & self.DRUG_SIGNALS:
            sources.append("drug_necessity")
            signals_found.append("chronic disease drug class")

        for sig in self.LLM_SIGNALS:
            if sig in all_text:
                sources.append("mlm_llm")
                signals_found.append(f"LLM signal: {sig}")
                break

        # Default: query both primary DBs if classifier is uncertain
        if not sources:
            sources = ["pharmacy_necessity", "drug_necessity"]
            signals_found.append("default fallback")

        return ClassificationResult(
            sources=list(set(sources)),
            reasoning=f"Triggered by: {', '.join(signals_found)}",
        )


# --- Context Retriever & Reranker ---------------------------------------------

@dataclass
class RetrievedDoc:
    id: str
    text: str
    source: str
    score: float
    tags: list[str]


class ContextRetrieverReranker:
    """
    Retrieves documents from classified knowledge sources,
    then reranks using a simple TF-IDF-style term overlap score.
    """

    SOURCE_MAP = {
        "pharmacy_necessity": PHARMACY_NECESSITY_DB,
        "drug_necessity":     DRUG_NECESSITY_DB,
        "mlm_llm":            MLM_LLM_KNOWLEDGE,
    }

    def _score(self, doc: dict, query_terms: list[str]) -> float:
        doc_text = (doc["text"] + " " + " ".join(doc.get("tags", []))).lower()
        matches = sum(1 for t in query_terms if t.lower() in doc_text)
        return matches / max(len(query_terms), 1)

    def retrieve(
        self,
        expanded: ExpandedQuery,
        sources: list[str],
        top_k: int = 4
    ) -> list[RetrievedDoc]:

        query_terms = (
            expanded.entities.drugs
            + expanded.entities.conditions
            + expanded.drug_classes
            + expanded.ontology_terms
            + [expanded.original]
        )

        all_docs: list[RetrievedDoc] = []

        for source in sources:
            db = self.SOURCE_MAP.get(source, [])
            for doc in db:
                score = self._score(doc, query_terms)
                all_docs.append(RetrievedDoc(
                    id=doc["id"],
                    text=doc["text"],
                    source=source,
                    score=score,
                    tags=doc.get("tags", []),
                ))

        # Rerank: sort descending by score, return top_k
        reranked = sorted(all_docs, key=lambda d: d.score, reverse=True)
        return reranked[:top_k]


# --- Retriever Validator ------------------------------------------------------

@dataclass
class RetrieverValidationResult:
    passed: bool
    docs: list[RetrievedDoc]
    rejected: list[RetrievedDoc]
    reason: str


class RetrieverValidator:
    """
    Post-retrieval quality gate.
    Rejects documents below a score threshold or off-topic.
    """

    def __init__(self, min_score: float = 0.1):
        self.min_score = min_score

    def validate(
        self,
        docs: list[RetrievedDoc],
        expanded: ExpandedQuery
    ) -> RetrieverValidationResult:

        passed_docs   = [d for d in docs if d.score >= self.min_score]
        rejected_docs = [d for d in docs if d.score <  self.min_score]

        if not passed_docs:
            return RetrieverValidationResult(
                passed=False,
                docs=[],
                rejected=docs,
                reason="All retrieved documents scored below threshold. Consider broadening query.",
            )

        return RetrieverValidationResult(
            passed=True,
            docs=passed_docs,
            rejected=rejected_docs,
            reason=f"{len(passed_docs)} docs passed validation (threshold={self.min_score}).",
        )


# --- Unified Query Router & Retriever -------------------------------------------

class QueryRouterRetriever:
    """
    Orchestrates: KnowledgeSourceClassifier -> ContextRetrieverReranker -> RetrieverValidator
    """

    def __init__(self):
        self.classifier = KnowledgeSourceClassifier()
        self.retriever  = ContextRetrieverReranker()
        self.validator  = RetrieverValidator()

    def run(self, expanded: ExpandedQuery) -> RetrieverValidationResult:
        classification = self.classifier.classify(expanded)
        print(f"  [Router] Sources: {classification.sources} | {classification.reasoning}")

        raw_docs = self.retriever.retrieve(expanded, classification.sources)
        print(f"  [Retriever] Retrieved {len(raw_docs)} docs before validation")

        result = self.validator.validate(raw_docs, expanded)
        print(f"  [Validator] {result.reason}")
        return result
