"""
Real-time Medical Data Fetcher 🌍
Fetches live pharmacological data from OpenFDA API to augment the RAG pipeline.
"""

import requests
import json
from dataclasses import dataclass

@dataclass
class RealtimeInsight:
    source: str
    headline: str
    details: str
    timestamp: str

class MedicalDataFetcher:
    """
    Connects to OpenFDA and other medical APIs for real-time safety data.
    """
    
    BASE_URL_DRUG = "https://api.fda.gov/drug/event.json"
    
    def fetch_drug_events(self, drug_name: str, limit: int = 3) -> list[RealtimeInsight]:
        """
        Fetches the latest reported adverse events for a given drug.
        """
        try:
            params = {
                'search': f'patient.drug.medicinalproduct:"{drug_name}"',
                'limit': limit
            }
            response = requests.get(self.BASE_URL_DRUG, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            insights = []
            
            for result in data.get('results', []):
                # Extract primary side effect from patient reaction list
                reactions = [r.get('reactionmeddrapt', 'Unknown') for r in result.get('patient', {}).get('reaction', [])]
                reaction_summary = ", ".join(reactions[:3])
                
                insights.append(RealtimeInsight(
                    source="OpenFDA Real-time",
                    headline=f"Recent Adverse Event Report: {drug_name}",
                    details=f"Patient reported reactions: {reaction_summary}. Serious: {result.get('serious', 'N/A')}",
                    timestamp=result.get('receiptdate', 'N/A')
                ))
            return insights
            
        except Exception as e:
            print(f"[Real-time Fetch Error] {e}")
            return []

    def fetch_label_warnings(self, drug_name: str) -> list[RealtimeInsight]:
        """
        Fetches latest FDA label warnings for a drug.
        """
        try:
            url = "https://api.fda.gov/drug/label.json"
            params = {
                'search': f'openfda.brand_name:"{drug_name}"',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            insights = []
            
            for result in data.get('results', []):
                warnings = result.get('warnings', ['No recent warnings found.'])[0][:200] + "..."
                insights.append(RealtimeInsight(
                    source="FDA Label",
                    headline=f"Official Warning for {drug_name}",
                    details=warnings,
                    timestamp="Current"
                ))
            return insights
        except Exception as e:
            print(f"[Label Fetch Error] {e}")
            return []
