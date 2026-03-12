import React, { useRef, useState } from 'react';

const UploadSection = ({ onFileSelect, onAnalyze, selectedFile, loading }) => {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFileSelect(file);
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  };

  return (
    <section className="upload-section">
      <h2 className="section-header">📤 Upload Chest X-ray</h2>
      <div
        className={`drop-zone ${dragging ? 'drop-zone--active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/tiff"
          onChange={handleChange}
          style={{ display: 'none' }}
          id="file-input"
        />
        <div className="drop-zone-icon">
          {selectedFile ? '📄' : '☁️'}
        </div>
        <div className="drop-zone-text">
          {selectedFile
            ? <><strong>{selectedFile.name}</strong><br /><span className="muted">Click to change</span></>
            : <><strong>Drag &amp; drop</strong> or <strong>click to browse</strong><br /><span className="muted">PNG, JPG, TIFF — max 200 MB</span></>
          }
        </div>
      </div>

      <button
        id="analyze-btn"
        className="btn-analyze"
        onClick={onAnalyze}
        disabled={!selectedFile || loading}
      >
        {loading
          ? <><span className="spinner" />🔬 Analyzing with AI...</>
          : '🚀 Run Pathology Analysis'}
      </button>
    </section>
  );
};

export default UploadSection;
