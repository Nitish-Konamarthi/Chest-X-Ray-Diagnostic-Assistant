import React from 'react';

const ImagePanel = ({ imagePreview, heatmapB64 }) => {
  if (!imagePreview && !heatmapB64) return null;

  return (
    <div className="image-panel">
      {imagePreview && (
        <div className="image-card">
          <div className="image-card-title">📸 Uploaded X-ray</div>
          <img src={imagePreview} alt="Uploaded chest X-ray" />
        </div>
      )}
      {heatmapB64 && (
        <div className="image-card">
          <div className="image-card-title">🔥 Grad-CAM++ Attention Map</div>
          <img src={heatmapB64} alt="Grad-CAM attention heatmap" />
          <div className="image-card-note">Highlighted regions indicate areas the model focused on</div>
        </div>
      )}
    </div>
  );
};

export default ImagePanel;
