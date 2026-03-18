import React from 'react';

const Header = () => (
  <header className="app-header">
    <div className="header-brand">
      <span className="header-icon">🩺</span>
      <div>
        <h1 className="header-title"> Chest X-ray Diagnostic Assistant</h1>
        <p className="header-subtitle">Chest X-ray Pathology Detection &amp; Explainability</p>
      </div>
    </div>
    <div className="header-features">
      <span className="feature-badge">🔍 Clinical Confidence Levels</span>
      <span className="feature-badge">🔥 Grad-CAM++ Heatmaps</span>
      <span className="feature-badge">📊 14-Pathology DenseNet-121</span>
      <span className="feature-badge">🤖 Gemini 3.1 Flash Lite AI</span>
      <span className="feature-badge">📍 Nearby Specialists</span>
    </div>
  </header>
);

export default Header;
