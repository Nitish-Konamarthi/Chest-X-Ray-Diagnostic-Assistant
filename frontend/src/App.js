import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import ImagePanel from './components/ImagePanel';
import BinaryPipelineStatus from './components/BinaryPipelineStatus';
import PathologyResults from './components/PathologyResults';
import AIExplanationCard from './components/AIExplanationCard';
import NearbyDoctors from './components/NearbyDoctors';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const form = new FormData();
    form.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setResults(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />

      <main className="app-main">
        <UploadSection
          onFileSelect={handleFileSelect}
          onAnalyze={handleAnalyze}
          selectedFile={selectedFile}
          loading={loading}
        />

        {error && (
          <div className="error-banner">
            ❌ {error}
          </div>
        )}

        <ImagePanel
          imagePreview={imagePreview}
          heatmapB64={results?.heatmap_b64}
        />

        {results && (
          <>
            {/* Validation pipeline */}
            <BinaryPipelineStatus binaryPipeline={results.binary_pipeline} />

            {results.valid_for_analysis ? (
              <>
                {/* Pathology findings + bar chart + model footer */}
                <PathologyResults
                  pathologies={results.pathologies}
                  assessmentLevel={results.assessment_level}
                  clinicalSummary={results.clinical_summary}
                />

                {/* Gemini AI Explanation */}
                <AIExplanationCard
                  explanation={results.ai_explanation}
                  apiUsed={results.api_used}
                />

                {/* Nearby specialist doctors */}
                <NearbyDoctors pathologies={results.pathologies} />
              </>
            ) : (
              <div className="invalid-banner">
                {results.clinical_summary || 'Image validation failed. Please upload a valid chest X-ray.'}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}