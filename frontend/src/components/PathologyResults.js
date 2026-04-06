import React from 'react';
import DiseaseTag from './DiseaseTag';

const getClinicalIcon = (level) => {
  switch (level) {
    case 'HIGH': return '🔴';
    case 'MEDIUM': return '🟡';
    case 'LOW': return '🟢';
    default: return '⚪';
  }
};

const FindingRow = ({ pathology }) => {
  const { probability: prob, confidence_level, pathology: name } = pathology;
  const isHigh = confidence_level === 'HIGH';
  const isMedium = confidence_level === 'MEDIUM';

  const badge = isHigh
    ? <span className="badge badge--positive">✅ POSITIVE</span>
    : isMedium
      ? <span className="badge badge--consider">⚠️ CONSIDER</span>
      : <span className="badge badge--monitor">ℹ️ MONITOR</span>;

  const confClass = isHigh ? 'conf-high' : isMedium ? 'conf-medium' : 'conf-low';

  return (
    <div className="finding-row">
      <div className="finding-name"><DiseaseTag name={name} />{badge}</div>
      <div className="finding-prob-col">
        <div className="finding-prob-label">Probability</div>
        <div className="finding-prob">{(prob * 100).toFixed(1)}%</div>
      </div>
      <div className={`finding-conf ${confClass}`}>
        {getClinicalIcon(confidence_level)} {confidence_level}
      </div>
    </div>
  );
};

const PathologyResults = ({ pathologies, assessmentLevel, clinicalSummary }) => {
  if (!pathologies || pathologies.length === 0) return null;

  const high = pathologies.filter(p => p.confidence_level === 'HIGH');
  const medium = pathologies.filter(p => p.confidence_level === 'MEDIUM');
  const top5 = [...pathologies].sort((a, b) => b.probability - a.probability).slice(0, 5);

  return (
    <div className="pathology-section">
      {/* Clinical summary banner */}
      <div className={`clinical-banner clinical-banner--${assessmentLevel.toLowerCase()}`}>
        {assessmentLevel === 'NORMAL' ? '✅' : '⚠️'} {clinicalSummary}
      </div>

      {/* Detailed findings */}
      {high.length > 0 && (
        <div className="findings-group">
          <h3>🔴 High Confidence Findings</h3>
          <div className="findings-subtitle">Probability ≥ 65% — Highly likely present</div>
          {high.map(p => <FindingRow key={p.pathology} pathology={p} />)}
        </div>
      )}

      {medium.length > 0 && (
        <div className="findings-group">
          <h3>🟡 Medium Confidence Findings</h3>
          <div className="findings-subtitle">Probability 40–65% — Possibly present</div>
          {medium.map(p => <FindingRow key={p.pathology} pathology={p} />)}
        </div>
      )}

      {/* Top 5 bar chart */}
      <div className="top-predictions">
        <h3>📊 Top Predictions</h3>
        {top5.map((p, i) => (
          <div className="pred-bar-row" key={i}>
            <div className="pred-bar-label">#{i + 1} <DiseaseTag name={p.pathology} /></div>
            <div className="pred-bar-track">
              <div className="pred-bar-fill" style={{ width: `${p.probability * 100}%` }} />
            </div>
            <div className="pred-bar-value">{(p.probability * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      {/* Model footer */}
      <div className="model-footer">
        <div className="model-metric"><span className="metric-label">Model</span><span className="metric-value">m-30012020-104001.pth.tar</span></div>
        <div className="model-metric"><span className="metric-label">Architecture</span><span className="metric-value">DenseNet-121</span></div>
        <div className="model-metric"><span className="metric-label">Classes</span><span className="metric-value">14 Pathologies</span></div>
      </div>
    </div>
  );
};

export default PathologyResults;
