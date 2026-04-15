"""
Ontology Layer
Provides: Domain Ontology, Named Entity Recognition (NER), Semantic Synonyms
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# --- Domain Ontology --------------------------------------------------------

DOMAIN_ONTOLOGY: dict[str, list[str]] = {
    "analgesic":       ["pain reliever", "painkiller", "antipyretic"],
    "antibiotic":      ["antimicrobial", "antibacterial", "anti-infective"],
    "antihypertensive":["blood pressure medication", "BP drug", "vasodilator"],
    "antidiabetic":    ["hypoglycemic agent", "insulin sensitizer", "glucose lowering drug"],
    "antihistamine":   ["allergy medication", "H1 blocker", "anti-allergic"],
    "nsaid":           ["ibuprofen", "naproxen", "aspirin", "COX inhibitor"],
    "opioid":          ["morphine", "codeine", "tramadol", "narcotic analgesic"],
    "statin":          ["cholesterol lowering drug", "HMG-CoA reductase inhibitor", "lipid lowering agent"],
    "bronchodilator":  ["inhaler", "beta-agonist", "asthma medication", "SABA"],
    "antidepressant":  ["SSRI", "SNRI", "tricyclic", "mood stabilizer"],
}

DRUG_CONDITIONS: dict[str, str] = {
    "headache":        "analgesic",
    "fever":           "antipyretic",
    "infection":       "antibiotic",
    "high blood pressure": "antihypertensive",
    "diabetes":        "antidiabetic",
    "allergy":         "antihistamine",
    "inflammation":    "nsaid",
    "pain":            "analgesic",
    "cholesterol":     "statin",
    "asthma":          "bronchodilator",
    "depression":      "antidepressant",
    "anxiety":         "antidepressant",
}


class DomainOntology:
    """Maps medical conditions to drug classes and expands terminology."""

    def get_drug_class(self, condition: str) -> Optional[str]:
        condition_lower = condition.lower()
        for key, drug_class in DRUG_CONDITIONS.items():
            if key in condition_lower:
                return drug_class
        return None

    def expand_drug_class(self, drug_class: str) -> list[str]:
        return DOMAIN_ONTOLOGY.get(drug_class.lower(), [])

    def get_related_terms(self, term: str) -> list[str]:
        """Traverse ontology to find all related terms for a given input."""
        drug_class = self.get_drug_class(term) or term.lower()
        synonyms = self.expand_drug_class(drug_class)
        return list({drug_class, *synonyms})


# --- Named Entity Recognizer -------------------------------------------------

KNOWN_DRUGS = {
    "ibuprofen", "paracetamol", "acetaminophen", "amoxicillin", "metformin",
    "lisinopril", "atorvastatin", "omeprazole", "cetirizine", "salbutamol",
    "aspirin", "codeine", "tramadol", "morphine", "insulin", "warfarin",
    "diazepam", "sertraline", "fluoxetine", "amlodipine", "simvastatin",
}

KNOWN_CONDITIONS = {
    "headache", "fever", "pain", "infection", "diabetes", "hypertension",
    "allergy", "asthma", "depression", "anxiety", "cholesterol", "inflammation",
    "cold", "flu", "nausea", "vomiting", "diarrhea", "constipation",
    "high blood pressure", "gerd", "peptic ulcers", "heart failure",
}

DOSAGE_PATTERN = re.compile(r'\b\d+\s*(?:mg|mcg|ml|g|iu|units?)\b', re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(
    r'\b(?:once|twice|thrice|\d+\s*times?)\s*(?:a\s*)?(?:day|daily|week|weekly|month|monthly)\b',
    re.IGNORECASE
)


@dataclass
class ExtractedEntities:
    drugs: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    dosages: list[str] = field(default_factory=list)
    frequencies: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.drugs, self.conditions, self.dosages, self.frequencies])


class NERExtractor:
    """Rule-based Named Entity Recognizer for pharmaceutical queries."""

    def extract(self, text: str) -> ExtractedEntities:
        text_lower = text.lower()
        tokens = set(re.findall(r'\b\w+\b', text_lower))

        # Single token matches
        found_drugs      = tokens & KNOWN_DRUGS
        found_conditions = tokens & KNOWN_CONDITIONS

        # Multi-word phrase matches
        for condition in KNOWN_CONDITIONS:
            if " " in condition and condition in text_lower:
                found_conditions.add(condition)

        return ExtractedEntities(
            drugs=sorted(list(found_drugs)),
            conditions=sorted(list(found_conditions)),
            dosages=DOSAGE_PATTERN.findall(text),
            frequencies=FREQUENCY_PATTERN.findall(text),
        )


# --- Semantic Synonyms --------------------------------------------------------

SEMANTIC_SYNONYMS: dict[str, list[str]] = {
    "headache":    ["migraine", "cephalgia", "head pain", "cranial pain"],
    "fever":       ["pyrexia", "high temperature", "hyperthermia", "febrile"],
    "pain":        ["ache", "discomfort", "soreness", "tenderness"],
    "allergy":     ["hypersensitivity", "allergic reaction", "atopy"],
    "infection":   ["bacterial infection", "viral infection", "sepsis", "pathogen"],
    "diabetes":    ["hyperglycemia", "type 2 diabetes", "T2DM", "insulin resistance"],
    "hypertension":["high blood pressure", "elevated BP", "arterial hypertension"],
    "ibuprofen":   ["advil", "motrin", "brufen", "nurofen"],
    "paracetamol": ["acetaminophen", "tylenol", "panadol"],
    "amoxicillin": ["amoxil", "trimox", "penicillin-like antibiotic"],
}


class SemanticSynonymExpander:
    """Expands query terms with semantically equivalent medical synonyms."""

    def expand(self, term: str) -> list[str]:
        return SEMANTIC_SYNONYMS.get(term.lower(), [term])

    def expand_all(self, terms: list[str]) -> list[str]:
        expanded = []
        seen = set()
        for term in terms:
            for syn in self.expand(term):
                if syn not in seen:
                    seen.add(syn)
                    expanded.append(syn)
        return expanded


# --- Unified Ontology Layer ---------------------------------------------------

class OntologyLayer:
    """Aggregates Domain Ontology, NER, and Semantic Synonyms."""

    def __init__(self):
        self.ontology  = DomainOntology()
        self.ner       = NERExtractor()
        self.synonyms  = SemanticSynonymExpander()

    def process(self, query: str) -> dict:
        entities      = self.ner.extract(query)
        all_terms     = entities.drugs + entities.conditions
        expanded_syns = self.synonyms.expand_all(all_terms)
        drug_classes  = [
            self.ontology.get_drug_class(c)
            for c in entities.conditions
            if self.ontology.get_drug_class(c)
        ]
        ontology_terms = []
        for dc in drug_classes:
            ontology_terms.extend(self.ontology.expand_drug_class(dc))

        return {
            "entities":      entities,
            "synonyms":      expanded_syns,
            "drug_classes":  list(set(drug_classes)),
            "ontology_terms": list(set(ontology_terms)),
        }
