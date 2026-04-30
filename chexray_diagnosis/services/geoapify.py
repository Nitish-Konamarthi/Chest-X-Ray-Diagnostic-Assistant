"""
Geoapify-based nearby doctor / specialist finder.

Maps detected pathologies to specialist types and searches the Geoapify
Places API to find real clinics near the user's location.

API keys are resolved through ``chexray_diagnosis.config`` — never embedded.
"""

import requests
from typing import Dict, List, Optional, Tuple

from chexray_diagnosis.config import get_geoapify_api_key


class GeoapifyDoctorFinder:
    """Find specialized doctors using Geoapify Places API."""

    API_BASE_URL = "https://api.geoapify.com/v2/places"
    MAX_RADIUS_METERS = 100_000  # 100 km

    # pathology → (display_label, geoapify_category)
    PATHOLOGY_SPECIALIST_MAP = {
        "Pneumonia":          ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Pneumothorax":       ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Atelectasis":        ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Infiltration":       ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Consolidation":      ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Emphysema":          ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Fibrosis":           ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Pleural_Thickening": ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Effusion":           ("pulmonologist",   "healthcare.clinic_or_praxis.pulmonology"),
        "Cardiomegaly":       ("cardiologist",    "healthcare.clinic_or_praxis.cardiology"),
        "Edema":              ("cardiologist",    "healthcare.clinic_or_praxis.cardiology"),
        "Mass":               ("oncologist",      "healthcare.clinic_or_praxis"),
        "Nodule":             ("oncologist",      "healthcare.clinic_or_praxis"),
        "Hernia":             ("general surgeon", "healthcare.hospital"),
    }

    GENERAL_CATEGORY = (
        "healthcare.clinic_or_praxis.general,"
        "healthcare.clinic_or_praxis,"
        "healthcare.hospital"
    )

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = get_geoapify_api_key(api_key)
        if not self.api_key:
            print("⚠️  GEOAPIFY_API_KEY not set — doctor finder disabled")
        else:
            print("✅ Geoapify API key loaded")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_specialist(
        self, pathologies: List[Dict]
    ) -> Tuple[str, str, str]:
        if not pathologies:
            return "general practitioner", "General Health Check", self.GENERAL_CATEGORY

        significant = sorted(
            [p for p in pathologies if p.get("probability", 0) > 0.4],
            key=lambda x: x.get("probability", 0),
            reverse=True,
        )
        if not significant:
            return "general practitioner", "Preventive Care", self.GENERAL_CATEGORY

        primary = significant[0].get("name", "")
        mapping = self.PATHOLOGY_SPECIALIST_MAP.get(primary)
        if mapping:
            label, category = mapping
        else:
            label, category = "general practitioner", self.GENERAL_CATEGORY

        return label, primary, category

    def _search(
        self,
        lat: float,
        lon: float,
        category: Optional[str] = None,
        limit: int = 4,
    ) -> List[Dict]:
        if not self.api_key:
            return []

        try:
            params = {
                "categories": category or self.GENERAL_CATEGORY,
                "filter": f"circle:{lon},{lat},{self.MAX_RADIUS_METERS}",
                "bias": f"proximity:{lon},{lat}",
                "limit": limit * 2,
                "apiKey": self.api_key,
            }
            resp = requests.get(self.API_BASE_URL, params=params, timeout=10)
            if resp.status_code != 200:
                return []

            doctors = []
            for feat in resp.json().get("features", [])[:limit]:
                props = feat.get("properties", {})
                contact = props.get("contact", {})
                phone = (
                    contact.get("phone", contact.get("telephone", "Not available"))
                    if isinstance(contact, dict)
                    else "Not available"
                )
                doctors.append({
                    "name": props.get("name", "Medical Facility"),
                    "address": props.get("formatted", "Address not available"),
                    "distance_km": round(props.get("distance", 0) / 1000, 1),
                    "phone": phone,
                    "website": props.get("datasource", {}).get(
                        "url", props.get("website", "Not available"),
                    ),
                    "rating": props.get("rating"),
                })
            return doctors

        except Exception:
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_doctors_for_pathology(
        self,
        latitude: float,
        longitude: float,
        pathologies: List[Dict],
        include_general_practitioner: bool = True,
    ) -> Dict:
        """Find specialists + GPs near a location based on pathology results."""
        specialist_type, primary, category = self._determine_specialist(pathologies)

        specialists = self._search(latitude, longitude, category, limit=4)

        # Broader fallback
        if not specialists:
            specialists = self._search(
                latitude, longitude,
                "healthcare.clinic_or_praxis,healthcare.hospital",
                limit=4,
            )

        gps: List[Dict] = []
        if include_general_practitioner:
            gps = self._search(latitude, longitude, self.GENERAL_CATEGORY, limit=2)
            specialist_names = {s["name"] for s in specialists}
            gps = [g for g in gps if g["name"] not in specialist_names]

        return {
            "specialist_type": specialist_type,
            "primary_pathology": primary,
            "specialists": specialists,
            "general_practitioners": gps,
            "search_radius_km": self.MAX_RADIUS_METERS / 1000,
            "user_location": {"latitude": latitude, "longitude": longitude},
        }

    @staticmethod
    def get_fallback_recommendations() -> Dict:
        """Return generic advice when the API is unavailable."""
        return {
            "specialist_type": "healthcare provider",
            "primary_pathology": "General Health",
            "specialists": [],
            "general_practitioners": [],
            "search_radius_km": 100,
            "fallback_message": (
                "Unable to fetch nearby doctors. "
                "Please consult your primary care physician or nearest hospital."
            ),
            "generic_advice": [
                "Contact your primary care physician for follow-up",
                "Visit the nearest hospital if symptoms worsen",
                "Use online directories to find specialists",
                "Check with your insurance for in-network specialists",
            ],
        }
