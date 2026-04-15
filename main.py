"""
PharmaRAG — Main Pipeline Orchestrator
Wires together all layers from the architecture diagram:

Actor → UI → [Cache] → Query Expander → [Ontology Layer]
           → Query Router & Retriever → [Retriever Validator]
           → Response Generator → Response Validator → UI
"""

import sys
import os
from dotenv import load_dotenv
from cache import QueryCache
from query_expander import QueryExpander
from query_router import QueryRouterRetriever, RetrievedDoc
from response_generator import ResponseGenerator, ResponseValidator
from realtime_data_fetcher import MedicalDataFetcher

# Load environment variables from .env
load_dotenv()


class PharmaRAGPipeline:

    def __init__(self, llm_provider: str = "google"):
        self.cache              = QueryCache(ttl_seconds=3600)
        self.query_expander     = QueryExpander()
        self.router_retriever   = QueryRouterRetriever()
        self.response_generator = ResponseGenerator(provider=llm_provider)
        self.response_validator = ResponseValidator()
        self.realtime_fetcher  = MedicalDataFetcher()

    def run(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"  QUERY: {user_query}")
        print(f"{'='*60}")

        # ── Step 0: Cache check ──────────────────────────────────
        cached = self.cache.get(user_query)
        if cached:
            print("  [Cache] HIT — returning cached response")
            return cached

        print("  [Cache] MISS — proceeding through pipeline\n")

        # -- Step 1: Query Expansion ------------------------------
        print("> Step 1: Query Expander")
        expanded = self.query_expander.expand(user_query)
        print(f"  Entities   : drugs={expanded.entities.drugs}, conditions={expanded.entities.conditions}")
        print(f"  Drug classes: {expanded.drug_classes}")
        print(f"  Variants    : {len(expanded.search_variants)} generated\n")

        # -- Step 2: Query Routing & Retrieval ---------------------
        print("> Step 2: Query Router & Retriever")
        retrieval_result = self.router_retriever.run(expanded)

        if not retrieval_result.passed:
            answer = (
                f"I couldn't find relevant information for: '{user_query}'. "
                "Please rephrase or consult a pharmacist directly."
            )
            self.cache.set(user_query, answer)
            return answer

        docs = retrieval_result.docs
        
        # -- Step 2.5: Real-time Data Augmentation -----------------
        if expanded.entities.drugs:
            print("> Step 2.5: Fetching Real-time Industry Data")
            for drug in expanded.entities.drugs:
                insights = self.realtime_fetcher.fetch_drug_events(drug)
                insights += self.realtime_fetcher.fetch_label_warnings(drug)
                for ins in insights:
                    print(f"  [Realtime] Found: {ins.headline}")
                    # Convert insight to RetrievedDoc format to inject into LLM context
                    docs.append(RetrievedDoc(
                        id=f"rt_{ins.headline[:5]}",
                        text=f"{ins.headline}: {ins.details}",
                        source=ins.source,
                        score=1.0, # Give it high priority
                        tags=["realtime", drug.lower()]
                    ))

        print(f"  Top doc scores: {[round(d.score,2) for d in docs]}\n")

        # -- Step 3: Response Generation ---------------------------
        print("> Step 3: Response Generator")
        raw_response = self.response_generator.generate(expanded, docs)
        print(f"  Sources used: {raw_response.sources_used}\n")

        # -- Step 4: Response Validation ---------------------------
        print("> Step 4: Response Validator")
        validated = self.response_validator.validate(raw_response, user_query)

        status = "[PASSED]" if validated.passed else "[FLAGGED]"
        print(f"  {status}")
        for w in validated.warnings:
            print(f"  Warning: {w}")

        # -- Cache & Return -----------------------------------------
        self.cache.set(user_query, validated.final_answer)

        print(f"\n{'-'*60}")
        print("  FINAL ANSWER:")
        print(f"{'-'*60}")
        print(validated.final_answer)
        print(f"{'='*60}\n")

        return validated.final_answer


# ─── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use 'google' as the MedTech upgraded provider
    pipeline = PharmaRAGPipeline(llm_provider="google")

    demo_queries = [
        "What can I take for a headache?",
        "I have a fever and mild pain, what's recommended?",
        "What medication is used for type 2 diabetes?",
        "Can I take ibuprofen for my allergy?",
        "What's the treatment for high blood pressure?",
    ]

    queries = sys.argv[1:] if len(sys.argv) > 1 else demo_queries

    for q in queries:
        pipeline.run(q)

    print("\nCache stats:", pipeline.cache.stats())
