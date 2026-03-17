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
    
    # Pathology to specialist mapping
    PATHOLOGY_SPECIALIST_MAP = {
        'Pneumonia': 'pulmonologist',
        'Pneumothorax': 'pulmonologist',
        'Atelectasis': 'pulmonologist',
        'Infiltration': 'pulmonologist',
        'Consolidation': 'pulmonologist',
        'Emphysema': 'pulmonologist',
        'Fibrosis': 'pulmonologist',
        'Pleural_Thickening': 'pulmonologist',
        'Effusion': 'pulmonologist',
        'Cardiomegaly': 'cardiologist',
        'Edema': 'cardiologist',
        'Mass': 'oncologist',
        'Nodule': 'oncologist',
        'Hernia': 'general surgeon',
    }

    def __init__(self):
        self.api_key = os.getenv('GEOAPIFY_API_KEY')
        
        if not self.api_key:
            print("⚠️  GEOAPIFY_API_KEY not found in environment variables")
            print("    Get your free API key from: https://myprojects.geoapify.com/")
        else:
            print("✅ Geoapify API key loaded successfully")

    def _determine_specialist_type(self, pathologies: List[Dict]) -> Tuple[str, str]:
        """
        Determine the primary specialist needed based on detected pathologies.
        
        Args:
            pathologies: List of detected pathologies with probabilities
            
        Returns:
            Tuple of (specialist_type, primary_pathology)
        """
        if not pathologies:
            return 'general practitioner', 'General Health Check'
        
        # Filter high-probability findings (>40%)
        significant = [p for p in pathologies if p.get('probability', 0) > 0.4]
        
        if not significant:
            return 'general practitioner', 'Preventive Care'
        
        # Sort by probability
        significant.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        # Get primary pathology
        primary = significant[0]
        primary_name = primary.get('name', '')
        
        # Map to specialist
        specialist = self.PATHOLOGY_SPECIALIST_MAP.get(
            primary_name, 
            'general practitioner'
        )
        
        return specialist, primary_name

    def _search_doctors(
        self, 
        latitude: float, 
        longitude: float, 
        specialist_type: str,
        limit: int = 4
    ) -> List[Dict]:
        """
        Search for doctors using Geoapify Places API.
        
        Args:
            latitude: User's latitude
            longitude: User's longitude
            specialist_type: Type of specialist (e.g., 'pulmonologist')
            limit: Maximum number of results
            
        Returns:
            List of doctor/clinic information dictionaries
        """
        if not self.api_key:
            return []
        
        try:
            # Geoapify categories for healthcare
            # https://apidocs.geoapify.com/docs/places/categories/
            params = {
                'categories': 'healthcare.doctor,healthcare.clinic,healthcare.hospital',
                'filter': f'circle:{longitude},{latitude},{self.MAX_RADIUS_METERS}',
                'bias': f'proximity:{longitude},{latitude}',
                'limit': limit * 2,  # Request more to filter better results
                'apiKey': self.api_key,
            }
            
            # Add text filter for specialist type if not general practitioner
            if specialist_type != 'general practitioner':
                params['text'] = specialist_type
            
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
                
                doctor_info = {
                    'name': props.get('name', 'Medical Facility'),
                    'address': props.get('formatted', 'Address not available'),
                    'distance_km': round(props.get('distance', 0) / 1000, 1),
                    'latitude': props.get('lat'),
                    'longitude': props.get('lon'),
                    'phone': props.get('contact', {}).get('phone', 'Not available'),
                    'website': props.get('website', 'Not available'),
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
        # Determine specialist type needed
        specialist_type, primary_pathology = self._determine_specialist_type(pathologies)
        
        # Search for specialists
        specialists = self._search_doctors(
            latitude, 
            longitude, 
            specialist_type,
            limit=4
        )
        
        # Search for general practitioners
        general_practitioners = []
        if include_general_practitioner:
            general_practitioners = self._search_doctors(
                latitude,
                longitude,
                'general practitioner',
                limit=2
            )
        
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
