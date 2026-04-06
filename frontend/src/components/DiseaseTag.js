import React, { useState, useRef, useCallback } from 'react';

// ── Map CheXNet class names → correct Wikipedia article titles ──────────────
const WIKI_MAP = {
  'Atelectasis': 'Atelectasis',
  'Cardiomegaly': 'Cardiomegaly',
  'Effusion': 'Pleural effusion',
  'Infiltration': 'Pulmonary infiltrate',
  'Mass': 'Lung tumor',
  'Nodule': 'Lung nodule',
  'Pneumonia': 'Pneumonia',
  'Pneumothorax': 'Pneumothorax',
  'Consolidation': 'Pulmonary consolidation',
  'Edema': 'Pulmonary edema',
  'Emphysema': 'Emphysema',
  'Fibrosis': 'Pulmonary fibrosis',
  'Pleural_Thickening': 'Pleural thickening',
  'Hernia': 'Hiatal hernia',
};

// Module-level cache — persists for the entire session
const wikiCache = {};

const DiseaseTag = ({ name }) => {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const [data, setData] = useState(null);   // { extract, url, thumb }
  const [loading, setLoading] = useState(false);
  const tagRef = useRef(null);
  const hideTimer = useRef(null);

  const fetchWiki = useCallback(async () => {
    if (wikiCache[name]) {
      setData(wikiCache[name]);
      return;
    }
    setLoading(true);
    const wikiTitle = WIKI_MAP[name] || name.replace(/_/g, ' ');
    const query = encodeURIComponent(wikiTitle.replace(/ /g, '_'));

    try {
      const res = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${query}`);
      if (!res.ok) throw new Error('Not found');
      const json = await res.json();

      const result = {
        extract: json.extract
          ? json.extract.split('. ').slice(0, 2).join('. ') + '.'
          : 'No summary available.',
        url: json.content_urls?.desktop?.page
          || `https://en.wikipedia.org/w/index.php?search=${query}`,
        thumb: json.thumbnail?.source || null,
      };
      wikiCache[name] = result;
      setData(result);
    } catch {
      const fallback = {
        extract: `No Wikipedia summary found for "${wikiTitle}".`,
        url: `https://en.wikipedia.org/w/index.php?search=${encodeURIComponent(wikiTitle)}`,
        thumb: null,
      };
      wikiCache[name] = fallback;
      setData(fallback);
    } finally {
      setLoading(false);
    }
  }, [name]);

  const handleMouseEnter = useCallback(() => {
    clearTimeout(hideTimer.current);

    // Use fixed positioning so we escape any parent overflow:hidden
    if (tagRef.current) {
      const rect = tagRef.current.getBoundingClientRect();
      const POPOVER_W = 280;
      const viewportW = window.innerWidth;

      let left = rect.left;
      // Flip left if it would overflow right edge
      if (left + POPOVER_W > viewportW - 16) {
        left = Math.max(8, rect.right - POPOVER_W);
      }

      setPos({ top: rect.bottom + 8, left });
    }

    setVisible(true);
    if (!wikiCache[name]) {
      fetchWiki();
    } else {
      setData(wikiCache[name]);
    }
  }, [name, fetchWiki]);

  const handleMouseLeave = useCallback(() => {
    hideTimer.current = setTimeout(() => setVisible(false), 200);
  }, []);

  const cancelHide = useCallback(() => {
    clearTimeout(hideTimer.current);
  }, []);

  const isMissing = data && !data.thumb &&
    data.extract.startsWith('No Wikipedia summary');

  return (
    <>
      <span
        ref={tagRef}
        className="disease-tag"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {name.replace(/_/g, ' ')}
      </span>

      {visible && (
        <div
          className="disease-popover"
          style={{ top: pos.top, left: pos.left }}
          onMouseEnter={cancelHide}
          onMouseLeave={handleMouseLeave}
        >
          {/* Header */}
          <div className="disease-popover-title">
            {name.replace(/_/g, ' ')}
          </div>

          {/* Thumbnail */}
          {data?.thumb && (
            <img
              src={data.thumb}
              alt={name}
              className="disease-popover-thumb"
            />
          )}

          {/* Loading */}
          {loading && (
            <div className="disease-popover-loading">
              <span className="wiki-spinner" /> Fetching Wikipedia…
            </div>
          )}

          {/* Extract + link */}
          {!loading && data && (
            <>
              <p className="disease-popover-extract">{data.extract}</p>
              <a
                href={data.url}
                target="_blank"
                rel="noopener noreferrer"
                className={`disease-popover-link${isMissing ? ' disease-popover-link--search' : ''}`}
              >
                {isMissing ? '🔍 Search on Wikipedia →' : '📖 Learn more on Wikipedia →'}
              </a>
            </>
          )}
        </div>
      )}
    </>
  );
};

export default DiseaseTag;
