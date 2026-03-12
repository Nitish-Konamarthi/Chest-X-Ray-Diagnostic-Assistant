import React, { useState, useEffect, useCallback } from 'react';

/**
 * NearbyDoctors
 * -------------
 * Shows specialist doctor flashcards based on detected pathologies.
 * Auto-requests geolocation on mount and loads doctors immediately —
 * no button click needed. Falls back gracefully if location is denied.
 * "Find Nearby on Maps" uses the user's real coordinates when available.
 */

// ── Specialty mapping ─────────────────────────────────────────────────────────
const PATHOLOGY_SPECIALTY_MAP = {
  Atelectasis:        { specialty: 'Pulmonologist',    keyword: 'pulmonologist',     emoji: '🫁' },
  Cardiomegaly:       { specialty: 'Cardiologist',     keyword: 'cardiologist',      emoji: '❤️' },
  Effusion:           { specialty: 'Pulmonologist',    keyword: 'pulmonologist',     emoji: '🫁' },
  Infiltration:       { specialty: 'Pulmonologist',    keyword: 'chest physician',   emoji: '🫁' },
  Mass:               { specialty: 'Oncologist',       keyword: 'oncologist',        emoji: '🔬' },
  Nodule:             { specialty: 'Pulmonologist',    keyword: 'pulmonologist',     emoji: '🫁' },
  Pneumonia:          { specialty: 'Pulmonologist',    keyword: 'chest physician',   emoji: '🫁' },
  Pneumothorax:       { specialty: 'Thoracic Surgeon', keyword: 'thoracic surgeon',  emoji: '🏥' },
  Consolidation:      { specialty: 'Pulmonologist',    keyword: 'pulmonologist',     emoji: '🫁' },
  Edema:              { specialty: 'Cardiologist',     keyword: 'cardiologist',      emoji: '❤️' },
  Emphysema:          { specialty: 'Pulmonologist',    keyword: 'pulmonologist',     emoji: '🫁' },
  Fibrosis:           { specialty: 'Rheumatologist',   keyword: 'rheumatologist',    emoji: '💊' },
  Pleural_Thickening: { specialty: 'Pulmonologist',    keyword: 'chest physician',   emoji: '🫁' },
  Hernia:             { specialty: 'General Surgeon',  keyword: 'general surgeon',   emoji: '🏥' },
};

// ── Mock fallback doctors ─────────────────────────────────────────────────────
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
    { name: 'Dr. Meena Krishnan', hospital: 'Cancer Institute (WIA)', address: 'Sardar Patel Rd, Chennai', phone: '+91-44-22350241', distance: '4.1 km', rating: 4.7 },
  ],
  'Thoracic Surgeon': [
    { name: 'Dr. Anil Saxena', hospital: 'Max Super Speciality Hospital', address: 'Saket, New Delhi', phone: '+91-11-26519050', distance: '2.8 km', rating: 4.7 },
    { name: 'Dr. Kavitha Reddy', hospital: 'Yashoda Hospitals', address: 'Malakpet, Hyderabad', phone: '+91-40-45677777', distance: '3.3 km', rating: 4.6 },
    { name: 'Dr. Suresh Babu', hospital: 'Global Hospitals', address: 'Lakdi Ka Pul, Hyderabad', phone: '+91-40-30244444', distance: '1.9 km', rating: 4.8 },
  ],
  'General Surgeon': [
    { name: 'Dr. Deepak Joshi', hospital: 'Lilavati Hospital', address: 'Bandra West, Mumbai', phone: '+91-22-26751000', distance: '1.5 km', rating: 4.7 },
    { name: 'Dr. Nalini Singh', hospital: 'Sir Ganga Ram Hospital', address: 'Old Rajinder Nagar, Delhi', phone: '+91-11-25750000', distance: '2.7 km', rating: 4.8 },
    { name: 'Dr. Rajan Pillai', hospital: 'Christian Medical College', address: 'Vellore, Tamil Nadu', phone: '+91-416-2281000', distance: '3.2 km', rating: 4.9 },
  ],
  Rheumatologist: [
    { name: 'Dr. Sanjay Bhatia', hospital: 'PD Hinduja Hospital', address: 'Mahim, Mumbai', phone: '+91-22-24452222', distance: '1.8 km', rating: 4.7 },
    { name: 'Dr. Geeta Verma', hospital: 'AIIMS Rishikesh', address: 'Veerbhadra Rd, Rishikesh', phone: '+91-135-2462946', distance: '4.5 km', rating: 4.6 },
    { name: 'Dr. Harish Nanda', hospital: 'BLK-Max Super Speciality', address: 'Pusa Road, New Delhi', phone: '+91-11-30403040', distance: '3.0 km', rating: 4.8 },
  ],
  'General Physician': [
    { name: 'Dr. Anita Verma', hospital: 'City Clinic', address: 'MG Road, Pune', phone: '+91-20-26130000', distance: '0.7 km', rating: 4.6 },
    { name: 'Dr. Rohan Kapoor', hospital: 'Health First Clinic', address: 'Sector 15, Noida', phone: '+91-120-4567890', distance: '1.1 km', rating: 4.5 },
  ],
};

// ── Build a Google Maps search URL ────────────────────────────────────────────
// If we have the user's coordinates, open Maps centered on them for that specialty.
const buildMapsUrl = (keyword, lat, lng) => {
  if (lat && lng) {
    return `https://www.google.com/maps/search/${encodeURIComponent(keyword)}/@${lat},${lng},13z`;
  }
  return `https://www.google.com/maps/search/${encodeURIComponent(keyword)}`;
};

// ── Star rating ───────────────────────────────────────────────────────────────
const StarRating = ({ rating }) => {
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
    <div className="doctor-hospital">{doctor.hospital}</div>
    <div className="doctor-address">📍 {doctor.address}</div>
    <StarRating rating={doctor.rating} />
    <div className="doctor-distance">🚗 {doctor.distance} away</div>
    <a className="doctor-phone" href={`tel:${doctor.phone.replace(/\s/g, '')}`}>
      📞 {doctor.phone}
    </a>
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
const NearbyDoctors = ({ pathologies }) => {
  const [locationGranted, setLocationGranted] = useState(false);
  const [locationDenied, setLocationDenied]   = useState(false);
  const [userCity, setUserCity]               = useState(null);
  const [userCoords, setUserCoords]           = useState({ lat: null, lng: null });
  const [doctors, setDoctors]                 = useState([]);
  const [gpDoctors, setGpDoctors]             = useState([]);
  const [loading, setLoading]                 = useState(true);
  const [targetSpecialty, setTargetSpecialty] = useState(null);
  const [targetKeyword, setTargetKeyword]     = useState('pulmonologist');
  const [targetEmoji, setTargetEmoji]         = useState('🏥');

  // Determine primary specialty from top pathology (above 0.2 probability)
  const getTargetSpecialty = useCallback(() => {
    if (!pathologies || pathologies.length === 0) return null;
    const sorted = [...pathologies].sort((a, b) => b.probability - a.probability);
    for (const p of sorted) {
      const mapped = PATHOLOGY_SPECIALTY_MAP[p.pathology];
      if (mapped && p.probability > 0.2) return mapped;
    }
    const mapped = PATHOLOGY_SPECIALTY_MAP[sorted[0]?.pathology];
    return mapped || { specialty: 'Pulmonologist', keyword: 'pulmonologist', emoji: '🫁' };
  }, [pathologies]);

  // Auto-load doctors immediately + try geolocation
  useEffect(() => {
    const mapped = getTargetSpecialty();
    if (!mapped) return;

    setTargetSpecialty(mapped.specialty);
    setTargetKeyword(mapped.keyword);
    setTargetEmoji(mapped.emoji);

    const pool   = MOCK_DOCTORS[mapped.specialty] || MOCK_DOCTORS['Pulmonologist'];
    const gpPool = MOCK_DOCTORS['General Physician'];

    // Try geolocation (non-blocking — always show doctors regardless)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = pos.coords.latitude;
          const lng = pos.coords.longitude;
          setLocationGranted(true);
          setUserCoords({ lat, lng });

          // Free Nominatim reverse geocode — no API key needed
          fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`,
            { headers: { 'Accept-Language': 'en' } }
          )
            .then(r => r.json())
            .then(data => {
              const city =
                data.address?.city    ||
                data.address?.town    ||
                data.address?.suburb  ||
                data.address?.village ||
                'your area';
              setUserCity(city);
            })
            .catch(() => setUserCity('your area'));
        },
        () => {
          // Denied — still show cards
          setLocationDenied(true);
        },
        { timeout: 5000, maximumAge: 60000 }
      );
    } else {
      setLocationDenied(true);
    }

    // Always show cards after a brief simulated loading delay
    const timer = setTimeout(() => {
      setDoctors(pool);
      setGpDoctors(gpPool);
      setLoading(false);
    }, 700);

    return () => clearTimeout(timer);
  }, [getTargetSpecialty]);

  if (!targetSpecialty) return null;

  return (
    <div className="nearby-section">
      {/* Header */}
      <div className="nearby-header">
        <div>
          <h3 className="nearby-title">📍 Recommended Specialists Near You</h3>
          <p className="nearby-subtitle">
            Based on the detected conditions, we recommend consulting a{' '}
            <strong>{targetSpecialty}</strong>
            {locationGranted && userCity ? ` near ${userCity}` : ''}.
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
          <div className="doctor-cards-scroll">
            {doctors.map((doc, i) => (
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

          {/* General Physician fallback section */}
          <div style={{ marginTop: '20px' }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '10px' }}>
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

          <p className="nearby-disclaimer">
            * Listings are representative. Verify availability and credentials before booking.
            {locationDenied && ' Enable location access for results near you.'}
          </p>
        </>
      )}
    </div>
  );
};

export default NearbyDoctors;
