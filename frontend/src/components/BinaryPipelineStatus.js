import React from 'react';

const BinaryPipelineStatus = ({ binaryPipeline }) => {
  if (!binaryPipeline || binaryPipeline.length === 0) return null;

  return (
    <div className="pipeline-section">
      <h3 className="pipeline-title">🔒 Image Validation Pipeline</h3>
      <div className="pipeline-cards">
        {binaryPipeline.map((item, i) => (
          <div key={i} className={`pipeline-card ${item.is_valid ? 'pipeline-card--pass' : 'pipeline-card--fail'}`}>
            <div className="pipeline-card-icon">{item.is_valid ? '✅' : '❌'}</div>
            <div className="pipeline-card-body">
              <div className="pipeline-card-model">{item.model}</div>
              <div className="pipeline-card-msg">{item.message}</div>
              <div className="pipeline-card-conf">{(item.confidence * 100).toFixed(1)}% confidence</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BinaryPipelineStatus;
