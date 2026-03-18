"""
Geoapify Service for Doctor Location Finder
============================================

NEW FILE - Add this to your backend/src/ or backend/ directory

Uses Geoapify Places API to find specialized doctors near user location.

Features:
- Finds 3-4 specialists based on detected pathology
- Searches within 100km radius  
- Priority-based ranking
- Includes general practitioners as fallback
- Returns doctor details with contact info and ratings

API Documentation: https://apidocs.geoapify.com/docs/places/
"""

import os
import requests
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_dotenv_path = find_dotenv(usecwd=False)
if _dotenv_path:
    load_dotenv(_dotenv_path)

class GeoapifyDoctorFinder:
    """Find specialized doctors near user location using Geoapify API"""

    API_BASE_URL = "https://api.geoapify.com/v2/places"
    MAX_RADIUS_METERS = 100000  # 100 km
    
    # Pathology to specialist mapping — values are (display_label, geoapify_category)
    PATHOLOGY_SPECIALIST_MAP = {
        'Pneumonia':          ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Pneumothorax':       ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Atelectasis':        ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Infiltration':       ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Consolidation':      ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Emphysema':          ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Fibrosis':           ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Pleural_Thickening': ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Effusion':           ('pulmonologist',     'healthcare.clinic_or_praxis.pulmonology'),
        'Cardiomegaly':       ('cardiologist',      'healthcare.clinic_or_praxis.cardiology'),
        'Edema':              ('cardiologist',      'healthcare.clinic_or_praxis.cardiology'),
        'Mass':               ('oncologist',        'healthcare.clinic_or_praxis'),
        'Nodule':             ('oncologist',        'healthcare.clinic_or_praxis'),
        'Hernia':             ('general surgeon',   'healthcare.hospital'),
    }

    # GP / general fallback category
    GENERAL_CATEGORY = 'healthcare.clinic_or_praxis.general,healthcare.clinic_or_praxis,healthcare.hospital'

    def __init__(self):
        self.api_key = os.getenv('GEOAPIFY_API_KEY')
        
        if not self.api_key:
            print("⚠️  GEOAPIFY_API_KEY not found in environment variables")
            print("    Get your free API key from: https://myprojects.geoapify.com/")
        else:
            print("✅ Geoapify API key loaded successfully")

    def _determine_specialist_type(self, pathologies: List[Dict]) -> Tuple[str, str, str]:
        """
        Determine the primary specialist needed based on detected pathologies.
        Returns:
            Tuple of (specialist_label, primary_pathology, geoapify_category)
        """
        if not pathologies:
            return 'general practitioner', 'General Health Check', self.GENERAL_CATEGORY

        # Filter high-probability findings (>40%)
        significant = [p for p in pathologies if p.get('probability', 0) > 0.4]

        if not significant:
            return 'general practitioner', 'Preventive Care', self.GENERAL_CATEGORY

        # Sort by probability
        significant.sort(key=lambda x: x.get('probability', 0), reverse=True)

        primary_name = significant[0].get('name', '')
        mapping = self.PATHOLOGY_SPECIALIST_MAP.get(primary_name)

        if mapping:
            label, category = mapping
        else:
            label, category = 'general practitioner', self.GENERAL_CATEGORY

        return label, primary_name, category

    def _search_doctors(
        self,
        latitude: float,
        longitude: float,
        specialist_type: str,
        geoapify_category: Optional[str] = None,
        limit: int = 4
    ) -> List[Dict]:
        """
        Search for doctors using Geoapify Places API with correct category names.
        """
        if not self.api_key:
            return []

        try:
            # Use provided category, or fall back to generic healthcare
            category = geoapify_category or self.GENERAL_CATEGORY

            params = {
                'categories': category,
                'filter': f'circle:{longitude},{latitude},{self.MAX_RADIUS_METERS}',
                'bias': f'proximity:{longitude},{latitude}',
                'limit': limit * 2,
                'apiKey': self.api_key,
            }
            
            response = requests.get(
                f"{self.API_BASE_URL}",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"⚠️  Geoapify API error: {response.status_code}")
                print(f"    Response: {response.text}")
                return []
            
            data = response.json()
            features = data.get('features', [])
            
            # Parse and format results
            doctors = []
            for feature in features[:limit]:
                props = feature.get('properties', {})
                
                # Extract contact info (Geoapify nests it differently)
                contact = props.get('contact', {})
                if isinstance(contact, dict):
                    phone = contact.get('phone', contact.get('telephone', 'Not available'))
                else:
                    phone = 'Not available'
                
                doctor_info = {
                    'name': props.get('name', 'Medical Facility'),
                    'address': props.get('formatted', props.get('address_line1', 'Address not available')),
                    'distance_km': round(props.get('distance', 0) / 1000, 1),
                    'phone': phone,
                    'website': props.get('datasource', {}).get('url', props.get('website', 'Not available')),
                    'rating': props.get('rating', None),
                }
                
                doctors.append(doctor_info)
            
            return doctors
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error calling Geoapify API: {e}")
            return []
        except Exception as e:
            print(f"⚠️  Error searching for doctors: {e}")
            return []

    def find_doctors_for_pathology(
        self, 
        latitude: float, 
        longitude: float,
        pathologies: List[Dict],
        include_general_practitioner: bool = True
    ) -> Dict:
        """
        Find specialized doctors and general practitioners near user location.
        
        Args:
            latitude: User's latitude
            longitude: User's longitude
            pathologies: List of detected pathologies from ML model
            include_general_practitioner: Whether to include GP in results
            
        Returns:
            Dictionary with specialist recommendations and doctor listings
        """
        # Determine specialist type needed (returns 3-tuple)
        specialist_type, primary_pathology, specialist_category = self._determine_specialist_type(pathologies)

        # ── Level 1: Search for specialists using the exact specialty category ──
        specialists = self._search_doctors(
            latitude,
            longitude,
            specialist_type,
            geoapify_category=specialist_category,
            limit=4
        )

        # ── Level 2 fallback: if exact category has no data, try broader healthcare ──
        # (Common in many regions where clinics are tagged only as 'clinic_or_praxis'
        #  or 'hospital', not by specific specialty.)
        if not specialists:
            print(f"ℹ️  No '{specialist_category}' results — falling back to broader search")
            specialists = self._search_doctors(
                latitude,
                longitude,
                specialist_type,
                geoapify_category='healthcare.clinic_or_praxis,healthcare.hospital',
                limit=4
            )

        # ── GP / general search (always runs, for secondary recommendations) ──
        general_practitioners = []
        if include_general_practitioner:
            general_practitioners = self._search_doctors(
                latitude,
                longitude,
                'general practitioner',
                geoapify_category=self.GENERAL_CATEGORY,
                limit=2
            )

        # De-duplicate: remove GPs that are already in the specialist list
        specialist_names = {s['name'] for s in specialists}
        general_practitioners = [g for g in general_practitioners if g['name'] not in specialist_names]

        return {
            'specialist_type': specialist_type,
            'primary_pathology': primary_pathology,
            'specialists': specialists,
            'general_practitioners': general_practitioners,
            'search_radius_km': self.MAX_RADIUS_METERS / 1000,
            'user_location': {
                'latitude': latitude,
                'longitude': longitude
            }
        }

    def get_fallback_recommendations(self) -> Dict:
        """
        Return fallback recommendations when API is unavailable.
        
        Returns:
            Dictionary with generic health advice
        """
        return {
            'specialist_type': 'healthcare provider',
            'primary_pathology': 'General Health',
            'specialists': [],
            'general_practitioners': [],
            'search_radius_km': 100,
            'fallback_message': (
                "Unable to fetch nearby doctors at this time. "
                "Please consult your primary care physician or visit your nearest hospital."
            ),
            'generic_advice': [
                "Contact your primary care physician for follow-up",
                "Visit the nearest hospital emergency department if symptoms worsen",
                "Use online healthcare directories to find specialists in your area",
                "Check with your health insurance provider for in-network specialists"
            ]
        }


# Global instance
geoapify_finder = GeoapifyDoctorFinder()
