"""
Query Expander Layer
# Components: Entity Extractor -> Ontology Expander
Transforms a raw user query into a rich, domain-enriched query bundle.
"""

from dataclasses import dataclass, field
from ontology_layer import OntologyLayer, ExtractedEntities


@dataclass
class ExpandedQuery:
    original: str
    entities: ExtractedEntities
    expanded_terms: list[str]
    drug_classes: list[str]
    ontology_terms: list[str]
    search_variants: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Original    : {self.original}\n"
            f"Drugs found : {self.entities.drugs}\n"
            f"Conditions  : {self.entities.conditions}\n"
            f"Drug classes: {self.drug_classes}\n"
            f"Synonyms    : {self.expanded_terms}\n"
            f"Ontology    : {self.ontology_terms}\n"
            f"Variants    : {self.search_variants}"
        )


class EntityExtractor:
    """
    Step 1 of Query Expander.
    Pulls raw named entities (drugs, conditions, dosages) from the query text.
    """
    def __init__(self, ontology_layer: OntologyLayer):
        self.ontology_layer = ontology_layer

    def extract(self, query: str) -> dict:
        return self.ontology_layer.process(query)


class OntologyExpander:
    """
    Step 2 of Query Expander.
    Takes extracted entities and builds multiple search variants
    by combining synonyms, drug classes, and ontology terms.
    """

    def expand(self, query: str, ontology_output: dict) -> list[str]:
        entities      = ontology_output["entities"]
        synonyms      = ontology_output["synonyms"]
        drug_classes  = ontology_output["drug_classes"]
        ontology_terms = ontology_output["ontology_terms"]

        variants = [query]  # always include original

        # Variant: replace condition with each synonym
        for cond in entities.conditions:
            for syn in synonyms:
                if syn.lower() != cond.lower():
                    variant = query.lower().replace(cond, syn)
                    if variant not in variants:
                        variants.append(variant)

        # Variant: query + drug class
        for dc in drug_classes:
            v = f"{query} {dc}"
            if v not in variants:
                variants.append(v)

        # Variant: drug class + ontology expansion
        for ot in ontology_terms[:3]:   # cap at 3 to avoid explosion
            v = f"{query} {ot}"
            if v not in variants:
                variants.append(v)

        return variants[:8]  # return top-8 variants max


class QueryExpander:
    """
    Orchestrates Entity Extractor -> Ontology Expander pipeline.
    Entry point for the Query Expander layer in the architecture.
    """

    def __init__(self):
        self.ontology_layer    = OntologyLayer()
        self.entity_extractor  = EntityExtractor(self.ontology_layer)
        self.ontology_expander = OntologyExpander()

    def expand(self, query: str) -> ExpandedQuery:
        # Step 1: Extract entities via NER + ontology processing
        ontology_output = self.entity_extractor.extract(query)

        # Step 2: Build search variants
        search_variants = self.ontology_expander.expand(query, ontology_output)

        return ExpandedQuery(
            original      = query,
            entities      = ontology_output["entities"],
            expanded_terms = ontology_output["synonyms"],
            drug_classes  = ontology_output["drug_classes"],
            ontology_terms = ontology_output["ontology_terms"],
            search_variants = search_variants,
        )
