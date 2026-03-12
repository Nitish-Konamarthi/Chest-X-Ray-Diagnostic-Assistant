import React, { useState } from 'react';

/**
 * AIExplanationCard
 * -----------------
 * Renders the Gemini 2.0 Flash Markdown response as a beautifully structured card.
 * The backend returns Markdown with sections like:
 *   ## 🩺 Overall Assessment
 *   ## 🔍 Key Findings
 *   ## 📋 Recommendations
 *   ## ⚠️ Disclaimer
 *
 * We parse and display each section individually for a clean, readable layout.
 */

// Simple inline Markdown renderer (no external deps needed)
const renderMarkdown = (text) => {
  if (!text) return null;

  return text.split('\n').map((line, i) => {
    // Bold text **...**
    const boldParsed = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    if (line.trim().startsWith('- ')) {
      return (
        <li key={i} dangerouslySetInnerHTML={{ __html: boldParsed.trim().slice(2) }} />
      );
    }
    if (line.trim() === '') return null;

    return <p key={i} dangerouslySetInnerHTML={{ __html: boldParsed }} />;
  }).filter(Boolean);
};

// Wrap consecutive <li> elements inside a <ul>
const wrapListItems = (elements) => {
  const result = [];
  let listBuffer = [];

  elements.forEach((el, i) => {
    if (el && el.type === 'li') {
      listBuffer.push(el);
    } else {
      if (listBuffer.length > 0) {
        result.push(<ul key={`ul-${i}`}>{listBuffer}</ul>);
        listBuffer = [];
      }
      if (el) result.push(el);
    }
  });

  if (listBuffer.length > 0) result.push(<ul key="ul-last">{listBuffer}</ul>);
  return result;
};

// Parse the full Markdown into named sections
const parseSections = (markdown) => {
  const sections = {};
  const sectionPattern = /^## (.+)$/gm;
  const parts = markdown.split(/^## .+$/m);
  const headers = [];
  let match;

  while ((match = sectionPattern.exec(markdown)) !== null) {
    headers.push(match[1].trim());
  }

  headers.forEach((header, i) => {
    sections[header] = (parts[i + 1] || '').trim();
  });

  return sections;
};

const SectionBlock = ({ title, content, defaultOpen = true, colorClass }) => {
  const [open, setOpen] = useState(defaultOpen);
  const items = renderMarkdown(content);
  const wrapped = wrapListItems(items);

  return (
    <div className={`ai-section ${colorClass}`}>
      <button
        className="ai-section-header"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span className="ai-section-title">{title}</span>
        <span className="ai-section-chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="ai-section-body">{wrapped}</div>
      )}
    </div>
  );
};

const SECTION_COLORS = {
  'Overall Assessment': 'ai-section--assessment',
  'Key Findings':       'ai-section--findings',
  'Recommendations':    'ai-section--recs',
  'Disclaimer':         'ai-section--disclaimer',
};

// Fallback: strip ## headings and render everything as plain paragraphs
const PlainExplanation = ({ text }) => {
  const lines = text.split('\n').filter(l => l.trim() && !l.startsWith('##'));
  return (
    <div className="ai-plain-body">
      {lines.map((line, i) => (
        <p key={i} dangerouslySetInnerHTML={{
          __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        }} />
      ))}
    </div>
  );
};

const AIExplanationCard = ({ explanation, apiUsed }) => {
  if (!explanation) return null;

  const sections = parseSections(explanation);
  const hasSections = Object.keys(sections).length > 0;

  return (
    <div className="ai-card">
      <div className="ai-card-header">
        <h3 className="ai-card-title">
          🤖 AI Clinical Explanation
        </h3>
        <span className="gemini-badge">
          <span className="gemini-badge-dot" />
          {apiUsed || 'AI Generated'}
        </span>
      </div>

      <div className="ai-card-body">
        {hasSections ? (
          Object.entries(sections).map(([title, content]) => {
            // Match section title loosely (strip emoji prefix for key lookup)
            const colorKey = Object.keys(SECTION_COLORS).find(k =>
              title.toLowerCase().includes(k.toLowerCase().split(' ')[0])
            );
            return (
              <SectionBlock
                key={title}
                title={title}
                content={content}
                colorClass={SECTION_COLORS[colorKey] || ''}
                defaultOpen={title.toLowerCase().includes('disclaimer') ? false : true}
              />
            );
          })
        ) : (
          <PlainExplanation text={explanation} />
        )}
      </div>

      <div className="ai-card-footer">
        <span className="ai-footer-note">
          ⚕️ This explanation is AI-generated. Always consult a licensed physician.
        </span>
      </div>
    </div>
  );
};

export default AIExplanationCard;
