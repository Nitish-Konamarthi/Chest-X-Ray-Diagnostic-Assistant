import React, { useState, useEffect, useCallback } from 'react';

/**
 * NearbyDoctors
 * -------------
 * Calls the real /find-doctors backend API using the user's geolocation + pathologies.
 * Falls back to curated mock data if location is denied or the API is unavailable.
 * Shows a "routine checkup" General Physician section for NORMAL X-ray results.
 */

const API_BASE = 'http://localhost:8000';

// ── Specialty mapping (for display labels / emojis) ───────────────────────────
const PATHOLOGY_SPECIALTY_MAP = {
  Atelectasis: { specialty: 'Pulmonologist', keyword: 'pulmonologist', emoji: '🫁' },
  Cardiomegaly: { specialty: 'Cardiologist', keyword: 'cardiologist', emoji: '❤️' },
  Effusion: { specialty: 'Pulmonologist', keyword: 'pulmonologist', emoji: '🫁' },
  Infiltration: { specialty: 'Pulmonologist', keyword: 'chest physician', emoji: '🫁' },
  Mass: { specialty: 'Oncologist', keyword: 'oncologist', emoji: '🔬' },
  Nodule: { specialty: 'Pulmonologist', keyword: 'pulmonologist', emoji: '🫁' },
  Pneumonia: { specialty: 'Pulmonologist', keyword: 'chest physician', emoji: '🫁' },
  Pneumothorax: { specialty: 'Thoracic Surgeon', keyword: 'thoracic surgeon', emoji: '🏥' },
  Consolidation: { specialty: 'Pulmonologist', keyword: 'pulmonologist', emoji: '🫁' },
  Edema: { specialty: 'Cardiologist', keyword: 'cardiologist', emoji: '❤️' },
  Emphysema: { specialty: 'Pulmonologist', keyword: 'pulmonologist', emoji: '🫁' },
  Fibrosis: { specialty: 'Rheumatologist', keyword: 'rheumatologist', emoji: '💊' },
  Pleural_Thickening: { specialty: 'Pulmonologist', keyword: 'chest physician', emoji: '🫁' },
  Hernia: { specialty: 'General Surgeon', keyword: 'general surgeon', emoji: '🏥' },
};

// ── Mock fallback doctors (used when API is unavailable / location denied) ────
const MOCK_DOCTORS = {
  Pulmonologist: [
    { name: 'Dr. Arjun Mehta', hospital: 'Apollo Hospitals', address: 'Jubilee Hills, Hyderabad', phone: '+91-40-23607777', distance: '1.2 km', rating: 4.8 },
    { name: 'Dr. Priya Sharma', hospital: 'AIIMS', address: 'Ansari Nagar, New Delhi', phone: '+91-11-26588500', distance: '3.5 km', rating: 4.9 },
    { name: 'Dr. Vikram Rao', hospital: 'Fortis Healthcare', address: 'Cunningham Road, Bengaluru', phone: '+91-80-66214444', distance: '2.1 km', rating: 4.7 },
  ],
  Cardiologist: [
    { name: 'Dr. Sunita Patel', hospital: 'Kokilaben Dhirubhai Ambani Hospital', address: 'Andheri West, Mumbai', phone: '+91-22-42696969', distance: '0.9 km', rating: 4.9 },
    { name: 'Dr. Rajesh Kumar', hospital: 'Medanta', address: 'Sector 38, Gurugram', phone: '+91-124-4141414', distance: '4.2 km', rating: 4.8 },
    { name: 'Dr. Ananya Das', hospital: 'Narayana Health', address: 'Bommasandra, Bengaluru', phone: '+91-80-71222222', distance: '5.0 km', rating: 4.7 },
  ],
  Oncologist: [
    { name: 'Dr. Pooja Nair', hospital: 'Tata Memorial Hospital', address: 'Dr. Ernest Borges Rd, Mumbai', phone: '+91-22-24177000', distance: '2.3 km', rating: 4.9 },
    { name: 'Dr. Sameer Gupta', hospital: 'Rajiv Gandhi Cancer Institute', address: 'Rohini, New Delhi', phone: '+91-11-47022222', distance: '3.8 km', rating: 4.8 },
  ],
  'Thoracic Surgeon': [
    { name: 'Dr. Anil Saxena', hospital: 'Max Super Speciality Hospital', address: 'Saket, New Delhi', phone: '+91-11-26519050', distance: '2.8 km', rating: 4.7 },
    { name: 'Dr. Kavitha Reddy', hospital: 'Yashoda Hospitals', address: 'Malakpet, Hyderabad', phone: '+91-40-45677777', distance: '3.3 km', rating: 4.6 },
  ],
  'General Surgeon': [
    { name: 'Dr. Deepak Joshi', hospital: 'Lilavati Hospital', address: 'Bandra West, Mumbai', phone: '+91-22-26751000', distance: '1.5 km', rating: 4.7 },
    { name: 'Dr. Nalini Singh', hospital: 'Sir Ganga Ram Hospital', address: 'Old Rajinder Nagar, Delhi', phone: '+91-11-25750000', distance: '2.7 km', rating: 4.8 },
  ],
  Rheumatologist: [
    { name: 'Dr. Sanjay Bhatia', hospital: 'PD Hinduja Hospital', address: 'Mahim, Mumbai', phone: '+91-22-24452222', distance: '1.8 km', rating: 4.7 },
    { name: 'Dr. Geeta Verma', hospital: 'AIIMS Rishikesh', address: 'Veerbhadra Rd, Rishikesh', phone: '+91-135-2462946', distance: '4.5 km', rating: 4.6 },
  ],
  'General Physician': [
    { name: 'Dr. Anita Verma', hospital: 'City Clinic', address: 'MG Road, Pune', phone: '+91-20-26130000', distance: '0.7 km', rating: 4.6 },
    { name: 'Dr. Rohan Kapoor', hospital: 'Health First Clinic', address: 'Sector 15, Noida', phone: '+91-120-4567890', distance: '1.1 km', rating: 4.5 },
  ],
};

// ── Build a Google Maps search URL ────────────────────────────────────────────
const buildMapsUrl = (keyword, lat, lng) => {
  if (lat && lng) {
    return `https://www.google.com/maps/search/${encodeURIComponent(keyword)}/@${lat},${lng},13z`;
  }
  return `https://www.google.com/maps/search/${encodeURIComponent(keyword)}`;
};

// ── Star rating ───────────────────────────────────────────────────────────────
const StarRating = ({ rating }) => {
  if (!rating) return null;
  const full = Math.floor(rating);
  const half = rating - full >= 0.5;
  return (
    <div className="doctor-stars">
      {'★'.repeat(full)}
      {half ? '½' : ''}
      {'☆'.repeat(5 - full - (half ? 1 : 0))}
      <span className="doctor-rating-num">{rating}</span>
    </div>
  );
};

// ── Single doctor flashcard ───────────────────────────────────────────────────
const DoctorCard = ({ doctor, specialty, emoji, keyword, lat, lng }) => (
  <div className="doctor-card">
    <div className="doctor-card-header">
      <span className="doctor-emoji">{emoji}</span>
      <span className="doctor-specialty-badge">{specialty}</span>
    </div>
    <div className="doctor-name">{doctor.name}</div>
    {doctor.hospital && doctor.hospital !== doctor.name && (
      <div className="doctor-hospital">{doctor.hospital}</div>
    )}
    <div className="doctor-address">📍 {doctor.address}</div>
    {doctor.rating && <StarRating rating={doctor.rating} />}
    {doctor.distance && <div className="doctor-distance">🚗 {doctor.distance} away</div>}
    {doctor.phone && doctor.phone !== 'Not available' && (
      <a className="doctor-phone" href={`tel:${doctor.phone.replace(/[\s\-()]/g, '')}`}>
        📞 {doctor.phone}
      </a>
    )}
    <div className="doctor-card-footer">
      <button
        className="btn-book"
        onClick={() => window.open(
          buildMapsUrl(keyword + ' near me', lat, lng),
          '_blank', 'noopener'
        )}
      >
        🗺️ Find Nearby on Maps
      </button>
    </div>
  </div>
);

// ── Main component ────────────────────────────────────────────────────────────
const NearbyDoctors = ({ pathologies, isNormal }) => {
  const [locationGranted, setLocationGranted] = useState(false);
  const [locationDenied, setLocationDenied] = useState(false);
  const [userCity, setUserCity] = useState(null);
  const [userCoords, setUserCoords] = useState({ lat: null, lng: null });
  const [specialists, setSpecialists] = useState([]);
  const [gpDoctors, setGpDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [apiUsed, setApiUsed] = useState(false);
  const [targetSpecialty, setTargetSpecialty] = useState(null);
  const [targetKeyword, setTargetKeyword] = useState('pulmonologist');
  const [targetEmoji, setTargetEmoji] = useState('🏥');

  // Determine primary specialty from top pathology
  const getTargetSpecialty = useCallback(() => {
    if (isNormal || !pathologies || pathologies.length === 0) {
      return { specialty: 'General Physician', keyword: 'general physician', emoji: '🩺' };
    }
    const sorted = [...pathologies].sort((a, b) => b.probability - a.probability);
    for (const p of sorted) {
      const mapped = PATHOLOGY_SPECIALTY_MAP[p.pathology];
      if (mapped && p.probability > 0.2) return mapped;
    }
    return PATHOLOGY_SPECIALTY_MAP[sorted[0]?.pathology] ||
      { specialty: 'Pulmonologist', keyword: 'pulmonologist', emoji: '🫁' };
  }, [pathologies, isNormal]);

  // Call backend /find-doctors API
  const fetchDoctorsFromAPI = useCallback(async (lat, lng) => {
    if (!pathologies && !isNormal) return false;

    const pathologiesPayload = isNormal
      ? []
      : pathologies.map(p => ({ name: p.pathology, probability: p.probability }));

    try {
      const response = await fetch(`${API_BASE}/find-doctors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: lat,
          longitude: lng,
          pathologies: pathologiesPayload,
        }),
      });

      if (!response.ok) return false;

      const data = await response.json();

      if (data.success && (data.specialists?.length || data.general_practitioners?.length)) {
        setSpecialists(data.specialists || []);
        setGpDoctors(data.general_practitioners || []);
        setApiUsed(true);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, [pathologies, isNormal]);

  // Load mock fallback data
  const loadMockData = useCallback((specialtyInfo) => {
    const pool = MOCK_DOCTORS[specialtyInfo.specialty] || MOCK_DOCTORS['Pulmonologist'];
    const gpPool = MOCK_DOCTORS['General Physician'];
    setSpecialists(pool);
    setGpDoctors(isNormal ? [] : gpPool);
    setApiUsed(false);
  }, [isNormal]);

  useEffect(() => {
    const specialtyInfo = getTargetSpecialty();
    setTargetSpecialty(specialtyInfo.specialty);
    setTargetKeyword(specialtyInfo.keyword);
    setTargetEmoji(specialtyInfo.emoji);

    if (!navigator.geolocation) {
      setLocationDenied(true);
      loadMockData(specialtyInfo);
      setLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        setLocationGranted(true);
        setUserCoords({ lat, lng });

        // Reverse geocode for city name
        fetch(
          `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`,
          { headers: { 'Accept-Language': 'en' } }
        )
          .then(r => r.json())
          .then(data => {
            const city =
              data.address?.city ||
              data.address?.town ||
              data.address?.suburb ||
              data.address?.village ||
              'your area';
            setUserCity(city);
          })
          .catch(() => setUserCity('your area'));

        // Try real API first
        const apiSuccess = await fetchDoctorsFromAPI(lat, lng);
        if (!apiSuccess) {
          loadMockData(specialtyInfo);
        }
        setLoading(false);
      },
      () => {
        // Location denied — use mock data
        setLocationDenied(true);
        loadMockData(specialtyInfo);
        setLoading(false);
      },
      { timeout: 6000, maximumAge: 60000 }
    );
  }, [getTargetSpecialty, fetchDoctorsFromAPI, loadMockData]);

  if (!targetSpecialty) return null;

  const isSectionForNormal = isNormal;

  return (
    <div className="nearby-section">
      {/* Header */}
      <div className="nearby-header">
        <div>
          <h3 className="nearby-title">
            {isSectionForNormal ? '🩺 Recommended for Routine Checkup' : '📍 Recommended Specialists Near You'}
          </h3>
          <p className="nearby-subtitle">
            {isSectionForNormal
              ? 'Your X-ray appears normal. We still recommend a periodic health checkup with a '
              : 'Based on the detected conditions, we recommend consulting a '}
            <strong>{targetSpecialty}</strong>
            {locationGranted && userCity ? ` near ${userCity}` : ''}.
            {apiUsed && <span className="api-live-badge"> ✅ Live results near you</span>}
          </p>
        </div>

        {locationDenied && (
          <div className="location-denied-note">
            📌 Location not shared — showing representative listings.
          </div>
        )}
      </div>

      {/* Loading state */}
      {loading ? (
        <div className="nearby-status">
          <span className="spinner" /> Finding nearby {targetSpecialty}s…
        </div>
      ) : (
        <>
          {/* Specialist cards */}
          {specialists.length > 0 && (
            <div className="doctor-cards-scroll">
              {specialists.map((doc, i) => (
                <DoctorCard
                  key={i}
                  doctor={doc}
                  specialty={targetSpecialty}
                  emoji={targetEmoji}
                  keyword={targetKeyword}
                  lat={userCoords.lat}
                  lng={userCoords.lng}
                />
              ))}
            </div>
          )}

          {/* General Physician fallback section (only for abnormal cases) */}
          {!isSectionForNormal && gpDoctors.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <div className="gp-section-label">
                🩺 Also consider a General Physician for initial consultation:
              </div>
              <div className="doctor-cards-scroll">
                {gpDoctors.map((doc, i) => (
                  <DoctorCard
                    key={i}
                    doctor={doc}
                    specialty="General Physician"
                    emoji="🩺"
                    keyword="general physician"
                    lat={userCoords.lat}
                    lng={userCoords.lng}
                  />
                ))}
              </div>
            </div>
          )}

          <p className="nearby-disclaimer">
            {apiUsed
              ? '* Live results from Geoapify. Verify availability and credentials before booking.'
              : '* Representative listings shown. Enable location access for results near you.'}
            {locationDenied && ' Allow location access for personalized results.'}
          </p>
        </>
      )}
    </div>
  );
};

export default NearbyDoctors;
