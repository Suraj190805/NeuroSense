import { useState, useEffect, useCallback, useRef, useMemo, createContext, useContext, Fragment } from 'react';
import { Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom';
import './App.css';

const API_BASE = 'http://localhost:8000';

const STAGE_CONFIG = {
  pre_manifest: { label: 'Pre-manifest', color: 'var(--stage-pre)' },
  early: { label: 'Early HD', color: 'var(--stage-early)' },
  advanced: { label: 'Advanced HD', color: 'var(--stage-advanced)' },
};

const FEATURE_LABELS = {
  cag_repeat: 'CAG Repeat',
  uhdrs_motor: 'UHDRS Motor',
  uhdrs_cognitive: 'Cognitive',
  tfc: 'TFC Score',
  age: 'Age',
};

/* ═══════════════════════════════════════════
   FLOATING PARTICLES BACKGROUND
   ═══════════════════════════════════════════ */

function Particles() {
  const particles = Array.from({ length: 18 }, (_, i) => {
    const size = 4 + Math.random() * 14;
    const left = Math.random() * 100;
    const duration = 15 + Math.random() * 25;
    const delay = Math.random() * 20;
    const colors = ['rgba(224,122,95,0.10)', 'rgba(212,168,83,0.08)', 'rgba(45,53,97,0.05)', 'rgba(224,122,95,0.06)'];
    const color = colors[i % colors.length];
    return (
      <div
        key={i}
        className="particle"
        style={{
          width: size,
          height: size,
          left: `${left}%`,
          background: color,
          animationDuration: `${duration}s`,
          animationDelay: `${delay}s`,
        }}
      />
    );
  });
  return <div className="particles">{particles}</div>;
}

/* ═══════════════════════════════════════════
   DARK MODE HOOK
   ═══════════════════════════════════════════ */

function useTheme() {
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('neurosense-theme');
    if (saved) return saved === 'dark';
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches || false;
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('neurosense-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  return { isDark, toggle: () => setIsDark((d) => !d) };
}

/* ═══════════════════════════════════════════
   THEME CONTEXT
   ═══════════════════════════════════════════ */

const ThemeContext = createContext({ isDark: false, toggle: () => {} });

/* ═══════════════════════════════════════════
   ACTIVE SECTION CONTEXT
   ═══════════════════════════════════════════ */

const ActiveSectionContext = createContext('hero');

/* ═══════════════════════════════════════════
   SCROLL REVEAL HOOK
   ═══════════════════════════════════════════ */

function useScrollReveal() {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    );

    document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  });
}

/* ═══════════════════════════════════════════
   ACTIVE SECTION TRACKER HOOK
   ═══════════════════════════════════════════ */

function useActiveSection(sectionIds) {
  const [activeSection, setActiveSection] = useState(sectionIds[0] || '');

  useEffect(() => {
    const observers = [];
    const visibleSections = new Map();

    sectionIds.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            visibleSections.set(id, entry.intersectionRatio);
          } else {
            visibleSections.delete(id);
          }

          // Pick the section with highest visibility
          if (visibleSections.size > 0) {
            let best = '';
            let bestRatio = 0;
            visibleSections.forEach((ratio, key) => {
              if (ratio >= bestRatio) {
                bestRatio = ratio;
                best = key;
              }
            });
            setActiveSection(best);
          }
        },
        {
          threshold: [0, 0.1, 0.3, 0.5, 0.7],
          rootMargin: '-80px 0px -40% 0px',
        }
      );

      observer.observe(el);
      observers.push(observer);
    });

    return () => observers.forEach((o) => o.disconnect());
  }, [sectionIds]);

  return activeSection;
}

/* ═══════════════════════════════════════════
   NAVBAR
   ═══════════════════════════════════════════ */

function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === '/';
  const isMemoryTest = location.pathname === '/memory-test';
  const activeSection = useContext(ActiveSectionContext);
  const { isDark, toggle: toggleTheme } = useContext(ThemeContext);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 30);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    document.body.style.overflow = mobileOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [mobileOpen]);

  const scrollTo = (id) => {
    setMobileOpen(false);
    if (!isHome) {
      navigate('/');
      setTimeout(() => {
        document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
      }, 150);
    } else {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const isActive = (sectionId) => {
    if (!isHome) return false;
    return activeSection === sectionId;
  };

  const navLinks = [
    { id: 'about', label: 'About HD', section: true },
    { id: 'how-it-works', label: 'How It Works', section: true },
    { id: 'analysis', label: 'Analysis', section: true },
  ];

  return (
    <>
      <nav className={`navbar ${scrolled ? 'scrolled' : ''}`} role="navigation" aria-label="Main navigation">
        <div className="navbar-inner">
          <div
            className="navbar-brand"
            onClick={() => { navigate('/'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
            role="link"
            tabIndex={0}
            aria-label="NeuroSense home"
            onKeyDown={(e) => { if (e.key === 'Enter') { navigate('/'); window.scrollTo({ top: 0, behavior: 'smooth' }); } }}
          >
            <div className="navbar-logo">🧬</div>
            <span className="navbar-name">NeuroSense</span>
          </div>

          {/* Desktop links */}
          <ul className="navbar-links">
            {navLinks.map((link) => (
              <li key={link.id}>
                <a
                  href={`#${link.id}`}
                  className={isActive(link.id) ? 'active' : ''}
                  onClick={(e) => { e.preventDefault(); scrollTo(link.id); }}
                >
                  {link.label}
                </a>
              </li>
            ))}
            <li>
              <Link
                to="/memory-test"
                className={`navbar-link-page ${isMemoryTest ? 'active' : ''}`}
              >
                Memory Test
              </Link>
            </li>
            <li>
              <Link
                to="/history"
                className={`navbar-link-page ${location.pathname === '/history' ? 'active' : ''}`}
              >
                History
              </Link>
            </li>
            <li>
              <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle dark mode">
                {isDark ? '☀️' : '🌙'}
              </button>
            </li>
            <li>
              <a
                href="#analysis"
                className="navbar-cta"
                onClick={(e) => { e.preventDefault(); scrollTo('analysis'); }}
              >
                Run Analysis
              </a>
            </li>
          </ul>

          {/* Mobile hamburger */}
          <button
            className={`navbar-hamburger ${mobileOpen ? 'open' : ''}`}
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={mobileOpen}
          >
            <span className="hamburger-line" />
            <span className="hamburger-line" />
            <span className="hamburger-line" />
          </button>
        </div>
      </nav>

      {/* Mobile overlay */}
      <div
        className={`mobile-overlay ${mobileOpen ? 'open' : ''}`}
        onClick={() => setMobileOpen(false)}
        aria-hidden="true"
      />

      {/* Mobile drawer */}
      <aside className={`mobile-drawer ${mobileOpen ? 'open' : ''}`} role="dialog" aria-label="Mobile navigation">
        <div className="mobile-drawer-header">
          <div className="navbar-brand" onClick={() => { navigate('/'); window.scrollTo({ top: 0, behavior: 'smooth' }); setMobileOpen(false); }}>
            <div className="navbar-logo">🧬</div>
            <span className="navbar-name">NeuroSense</span>
          </div>
        </div>
        <ul className="mobile-nav-links">
          {navLinks.map((link) => (
            <li key={link.id}>
              <a
                href={`#${link.id}`}
                className={isActive(link.id) ? 'active' : ''}
                onClick={(e) => { e.preventDefault(); scrollTo(link.id); }}
              >
                {link.label}
              </a>
            </li>
          ))}
          <li>
            <Link
              to="/memory-test"
              className={isMemoryTest ? 'active' : ''}
              onClick={() => setMobileOpen(false)}
            >
              Memory Test
            </Link>
          </li>
        </ul>
        <div className="mobile-drawer-cta">
          <a
            href="#analysis"
            className="btn btn-primary"
            onClick={(e) => { e.preventDefault(); scrollTo('analysis'); }}
          >
            🔬 Run Analysis
          </a>
        </div>
      </aside>
    </>
  );
}

/* ═══════════════════════════════════════════
   HERO SECTION
   ═══════════════════════════════════════════ */

function HeroSection() {
  const features = [
    'MRI-Based Analysis',
    'Clinical Biomarkers',
    'Explainable AI',
    'Progression Forecasting',
  ];

  const trustBadges = [
    { icon: '⚡', label: 'AI-Powered' },
    { icon: '📚', label: 'Research Based' },
    { icon: '🏥', label: 'Clinical Workflow' },
  ];

  return (
    <section className="hero" id="hero">
      <div className="hero-inner">
        <div className="hero-content">
          <div className="hero-badge">
            🔬 AI-Powered Neuroscience Research
          </div>
          <h1 className="hero-title">
            Early Detection of{' '}
            <span className="highlight">Huntington's Disease</span>{' '}
            Through AI
          </h1>
          <p className="hero-description">
            NeuroSense combines 3D brain MRI analysis with clinical biomarkers
            using deep learning to enable early HD detection, stage classification,
            and 12/24-month progression forecasting — giving clinicians actionable
            insights before symptoms fully manifest.
          </p>

          {/* Feature checklist */}
          <div className="hero-features">
            {features.map((feature, i) => (
              <div className="hero-feature" key={i} style={{ animationDelay: `${0.4 + i * 0.08}s` }}>
                <span className="hero-feature-check">✓</span>
                <span>{feature}</span>
              </div>
            ))}
          </div>

          <div className="hero-actions">
            <a href="#analysis" className="btn btn-primary hero-btn" onClick={(e) => { e.preventDefault(); document.getElementById('analysis')?.scrollIntoView({ behavior: 'smooth' }); }}>
              <span className="btn-icon">🔬</span>
              Start Analysis
              <span className="btn-arrow">→</span>
            </a>
            <a href="#about" className="btn btn-secondary hero-btn" onClick={(e) => { e.preventDefault(); document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' }); }}>
              Learn About HD
            </a>
          </div>

          {/* Trust badges */}
          <div className="hero-trust">
            {trustBadges.map((badge, i) => (
              <div className="hero-trust-badge" key={i} style={{ animationDelay: `${0.7 + i * 0.1}s` }}>
                <span className="hero-trust-icon">{badge.icon}</span>
                <span className="hero-trust-label">{badge.label}</span>
              </div>
            ))}
          </div>

          <div className="hero-stats">
            <div className="hero-stat">
              <div className="hero-stat-value">≥87%</div>
              <div className="hero-stat-label">Target AUC-ROC</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">3-Stage</div>
              <div className="hero-stat-label">HD Classification</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">24-Mo</div>
              <div className="hero-stat-label">Progression Forecast</div>
            </div>
          </div>
        </div>
        <div className="hero-image">
          <div className="hero-ring" />
          <div className="hero-ring hero-ring-2" />
          <div className="hero-glow" />
          <img src="/hero-brain.png" alt="3D visualization of a human brain with neural pathways highlighted" />
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   ABOUT HD SECTION
   ═══════════════════════════════════════════ */

function AboutSection() {
  return (
    <section className="section about-section" id="about">
      <div className="section-inner">
        <div className="about-grid">
          <div className="about-image reveal">
            <img src="/dna-helix.png" alt="DNA double helix representing the HTT gene mutation in Huntington's Disease" />
            <div className="about-image-badge">
              <div className="about-image-badge-icon">🧬</div>
              <div className="about-image-badge-text">
                <strong>CAG ≥ 36</strong>
                HD Gene Mutation
              </div>
            </div>
          </div>
          <div>
            <div className="section-label reveal">Understanding the Disease</div>
            <h2 className="section-title reveal">
              What is Huntington's Disease?
            </h2>
            <p className="section-subtitle reveal" style={{ maxWidth: '520px' }}>
              Huntington's Disease (HD) is a progressive neurodegenerative disorder caused
              by an expanded CAG trinucleotide repeat in the <em>HTT</em> gene. It leads to
              the gradual breakdown of nerve cells in the brain, affecting movement, cognition,
              and behaviour.
            </p>
            <div className="about-facts stagger">
              <div className="fact-card reveal">
                <div className="fact-icon">🧠</div>
                <div className="fact-content">
                  <h4>Affects the Basal Ganglia</h4>
                  <p>HD primarily damages the caudate nucleus and putamen, brain regions critical for motor control and cognitive processing.</p>
                </div>
              </div>
              <div className="fact-card reveal">
                <div className="fact-icon">🔗</div>
                <div className="fact-content">
                  <h4>Autosomal Dominant Inheritance</h4>
                  <p>A single copy of the mutated HTT gene (CAG ≥ 36 repeats) is sufficient to cause the disease. Each child of a carrier has a 50% chance of inheriting it.</p>
                </div>
              </div>
              <div className="fact-card reveal">
                <div className="fact-icon">📊</div>
                <div className="fact-content">
                  <h4>30,000+ Affected in the U.S.</h4>
                  <p>Approximately 30,000 Americans have symptomatic HD, with another 200,000+ at risk of inheriting the gene. Onset typically occurs between ages 30–50.</p>
                </div>
              </div>
              <div className="fact-card reveal">
                <div className="fact-icon">⏱️</div>
                <div className="fact-content">
                  <h4>Early Detection is Critical</h4>
                  <p>Brain changes begin 10–15 years before motor symptoms appear. AI-powered early detection can enable proactive clinical interventions during the pre-manifest stage.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   HOW IT WORKS
   ═══════════════════════════════════════════ */

function HowItWorksSection() {
  const steps = [
    {
      num: '01',
      icon: '🧠',
      title: 'Upload MRI + Clinical Data',
      desc: 'Provide a structural T1-weighted MRI scan (.nii.gz) along with clinical measurements: CAG repeat count, UHDRS motor/cognitive scores, TFC, and patient age.',
    },
    {
      num: '02',
      icon: '⚡',
      title: 'Multi-Modal AI Analysis',
      desc: 'Our 3D ResNet-50 processes the brain MRI while a Bi-LSTM encodes longitudinal clinical data. Cross-modal attention fusion combines both streams for robust prediction.',
    },
    {
      num: '03',
      icon: '📋',
      title: 'Explainable Results',
      desc: 'Receive HD stage classification, 12/24-month progression forecasts, GradCAM++ brain heatmaps showing important regions, and SHAP feature attribution scores.',
    },
  ];

  return (
    <section className="section" id="how-it-works">
      <div className="section-inner">
        <div className="section-label reveal">The Pipeline</div>
        <h2 className="section-title reveal">How NeuroSense Works</h2>
        <p className="section-subtitle reveal">
          A three-step clinical workflow from data input to explainable AI predictions.
        </p>
        <div className="steps-grid stagger">
          {steps.map((step) => (
            <div className="step-card reveal" key={step.num}>
              <div className="step-number">{step.num}</div>
              <div className="step-icon">{step.icon}</div>
              <h3 className="step-title">{step.title}</h3>
              <p className="step-desc">{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   CLINICAL FORM (Enhanced Validation & UX)
   ═══════════════════════════════════════════ */

const FIELD_CONFIG = {
  cag_repeat: { label: 'CAG Repeat Count', min: 36, max: 120, step: '1', placeholder: 'e.g. 44', section: 'genetic', hint: 'Must be between 36–120' },
  uhdrs_motor: { label: 'UHDRS Motor Score', min: 0, max: 124, step: '0.1', placeholder: 'e.g. 18', section: 'clinical', hint: 'Must be between 0–124' },
  uhdrs_cognitive: { label: 'UHDRS Cognitive Score', min: 0, max: 999, step: '0.1', placeholder: 'e.g. 142', section: 'clinical', hint: 'Must be ≥ 0' },
  tfc_score: { label: 'Total Functional Capacity', min: 0, max: 13, step: '0.1', placeholder: '13 = fully functional', section: 'clinical', hint: 'Must be between 0–13' },
  age: { label: 'Patient Age', min: 18, max: 90, step: '1', placeholder: 'e.g. 42', section: 'clinical', hint: 'Must be between 18–90' },
};

const MAX_FILE_SIZE_MB = 500;

function ClinicalForm({ onSubmit, isLoading }) {
  const [form, setForm] = useState({
    cag_repeat: '', uhdrs_motor: '', uhdrs_cognitive: '', tfc_score: '13', age: '',
  });
  const [mriFile, setMriFile] = useState(null);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [dragOver, setDragOver] = useState(false);
  const [fileError, setFileError] = useState(null);

  const validateField = (key, value) => {
    const cfg = FIELD_CONFIG[key];
    if (!cfg) return null;
    if (value === '' || value === undefined) return 'This field is required';
    const num = parseFloat(value);
    if (isNaN(num)) return 'Please enter a valid number';
    if (num < cfg.min) return `Minimum value is ${cfg.min}`;
    if (num > cfg.max) return `Maximum value is ${cfg.max}`;
    return null;
  };

  const isFieldValid = (key) => {
    return form[key] !== '' && !validateField(key, form[key]);
  };

  const set = (k, v) => {
    setForm((p) => ({ ...p, [k]: v }));
    setTouched((p) => ({ ...p, [k]: true }));
    const err = validateField(k, v);
    setErrors((p) => ({ ...p, [k]: err }));
  };

  const handleBlur = (k) => {
    setTouched((p) => ({ ...p, [k]: true }));
    const err = validateField(k, form[k]);
    setErrors((p) => ({ ...p, [k]: err }));
  };

  const allRequiredValid = ['cag_repeat', 'uhdrs_motor', 'uhdrs_cognitive', 'tfc_score', 'age'].every(
    (k) => isFieldValid(k)
  );

  const validCount = ['cag_repeat', 'uhdrs_motor', 'uhdrs_cognitive', 'tfc_score', 'age'].filter(
    (k) => isFieldValid(k)
  ).length;

  const handleFile = (file) => {
    setFileError(null);
    if (!file) { setMriFile(null); return; }
    const validExts = ['.nii', '.nii.gz', '.gz'];
    const name = file.name.toLowerCase();
    if (!validExts.some((ext) => name.endsWith(ext))) {
      setFileError('Invalid format. Please upload a NIfTI file (.nii, .nii.gz)');
      return;
    }
    if (file.size > MAX_FILE_SIZE_MB * 1e6) {
      setFileError(`File too large. Maximum size is ${MAX_FILE_SIZE_MB} MB`);
      return;
    }
    setMriFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  };

  const handleSubmit = (ev) => {
    ev.preventDefault();
    // Touch all fields to show errors
    const allTouched = {};
    const allErrors = {};
    Object.keys(FIELD_CONFIG).forEach((k) => {
      allTouched[k] = true;
      allErrors[k] = validateField(k, form[k]);
    });
    setTouched(allTouched);
    setErrors(allErrors);
    if (!allRequiredValid) return;
    onSubmit(form, mriFile);
  };

  const renderField = (key) => {
    const cfg = FIELD_CONFIG[key];
    const hasError = touched[key] && errors[key];
    const valid = isFieldValid(key);
    const rangeText = key === 'uhdrs_cognitive' ? '≥ 0' : `${cfg.min}–${cfg.max}`;
    return (
      <div className={`form-group ${hasError ? 'has-error' : ''} ${valid ? 'is-valid' : ''}`} key={key}>
        <label className="form-label">
          {cfg.label}
          <span className="form-range">{rangeText}</span>
        </label>
        <div className="form-input-wrap">
          <input
            className={`form-input ${hasError ? 'error' : ''} ${valid ? 'valid' : ''}`}
            type="number"
            step={cfg.step}
            min={cfg.min}
            max={cfg.max}
            placeholder={cfg.placeholder}
            value={form[key]}
            onChange={(e) => set(key, e.target.value)}
            onBlur={() => handleBlur(key)}
          />
          {valid && <span className="form-field-check">✓</span>}
        </div>
        {hasError && <p className="form-error-msg">{errors[key]}</p>}
      </div>
    );
  };

  return (
    <div className="form-card">
      <h3 className="form-card-title">📋 Patient Assessment</h3>
      <form onSubmit={handleSubmit} noValidate>
        {/* MRI Upload */}
        <div className="form-section-label">Neuroimaging</div>
        <div
          className={`file-drop ${mriFile ? 'has-file' : ''} ${dragOver ? 'drag-over' : ''} ${fileError ? 'has-error' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".nii,.nii.gz,.gz"
            onChange={(e) => handleFile(e.target.files?.[0] || null)}
          />
          {mriFile ? (
            <>
              <div className="file-drop-icon">✅</div>
              <p className="file-drop-text">
                <strong>{mriFile.name}</strong>
              </p>
              <p className="file-drop-meta">
                {(mriFile.size / 1e6).toFixed(1)} MB • NIfTI format
              </p>
              <button
                type="button"
                className="file-remove-btn"
                onClick={(e) => { e.stopPropagation(); setMriFile(null); setFileError(null); }}
              >
                ✕ Remove
              </button>
            </>
          ) : (
            <>
              <div className="file-drop-icon">{dragOver ? '📂' : '🧠'}</div>
              <p className="file-drop-text">
                {dragOver ? 'Drop file here' : <>Drop MRI or <strong>browse</strong></>}
              </p>
              <p className="file-drop-hint">NIfTI format (.nii, .nii.gz) • Max {MAX_FILE_SIZE_MB} MB • Optional</p>
            </>
          )}
        </div>
        {fileError && <p className="form-error-msg file-error">{fileError}</p>}

        {/* Genetic */}
        <div className="form-section-label">Genetic Biomarker</div>
        {renderField('cag_repeat')}

        {/* Clinical */}
        <div className="form-section-label">Clinical Assessment</div>
        {renderField('uhdrs_motor')}
        {renderField('uhdrs_cognitive')}
        {renderField('tfc_score')}
        {renderField('age')}

        {/* Validation summary */}
        <div className="form-validation-bar">
          <div className="form-validation-dots">
            {['cag_repeat', 'uhdrs_motor', 'uhdrs_cognitive', 'tfc_score', 'age'].map((k) => (
              <span key={k} className={`form-val-dot ${isFieldValid(k) ? 'valid' : touched[k] && errors[k] ? 'error' : ''}`} />
            ))}
          </div>
          <span className="form-validation-text">{validCount}/5 fields completed</span>
        </div>

        <button type="submit" className="btn-analyze" disabled={isLoading || !allRequiredValid}>
          {isLoading && <span className="spinner" />}
          {isLoading ? 'Analysing…' : allRequiredValid ? '🔬 Run HD Analysis' : `Complete all fields (${validCount}/5)`}
        </button>
      </form>
    </div>
  );
}

/* ═══════════════════════════════════════════
   RESULTS PANEL (3-state: idle / loading / results)
   ═══════════════════════════════════════════ */

function ResultsPanel({ result, error, isLoading, lastForm }) {
  const [loadingStep, setLoadingStep] = useState(0);

  const loadingSteps = [
    { icon: '📤', label: 'Receiving patient data', detail: 'Validating clinical inputs…' },
    { icon: '🧠', label: 'Processing MRI scan', detail: 'Running 3D ResNet-50 inference…' },
    { icon: '⚡', label: 'Generating predictions', detail: 'Cross-modal attention fusion…' },
    { icon: '📊', label: 'Computing explanations', detail: 'SHAP values & GradCAM++ maps…' },
  ];

  useEffect(() => {
    if (!isLoading) { setLoadingStep(0); return; }
    setLoadingStep(0);
    const interval = setInterval(() => {
      setLoadingStep((prev) => (prev < loadingSteps.length - 1 ? prev + 1 : prev));
    }, 1800);
    return () => clearInterval(interval);
  }, [isLoading]);

  /* ─── Error state ─── */
  if (error) {
    return (
      <div className="dash-panel">
        <div className="error-box">⚠️ {error}</div>
      </div>
    );
  }

  /* ─── Loading state ─── */
  if (isLoading) {
    const progress = ((loadingStep + 1) / loadingSteps.length) * 100;
    return (
      <div className="dash-panel dash-loading">
        <div className="dash-loading-header">
          <div className="dash-loading-pulse" />
          <div>
            <h3 className="dash-loading-title">Analyzing Patient Data</h3>
            <p className="dash-loading-sub">NeuroSense AI pipeline is processing…</p>
          </div>
        </div>

        <div className="dash-progress-wrap">
          <div className="dash-progress-bar">
            <div className="dash-progress-fill" style={{ width: `${progress}%` }} />
            <div className="dash-progress-glow" style={{ left: `${progress}%` }} />
          </div>
          <span className="dash-progress-pct">{Math.round(progress)}%</span>
        </div>

        <div className="dash-loading-steps">
          {loadingSteps.map((step, i) => (
            <div className={`dash-load-step ${i < loadingStep ? 'done' : i === loadingStep ? 'active' : ''}`} key={i}>
              <div className="dash-load-step-icon">
                {i < loadingStep ? <span className="dash-check">✓</span> : <span>{step.icon}</span>}
              </div>
              <div className="dash-load-step-text">
                <span className="dash-load-step-label">{step.label}</span>
                <span className="dash-load-step-detail">{step.detail}</span>
              </div>
              {i === loadingStep && <div className="dash-load-spinner" />}
            </div>
          ))}
        </div>

        <div className="dash-loading-footer">
          <span className="dash-loading-dot" />
          <span>This typically takes 5–15 seconds depending on input complexity</span>
        </div>
      </div>
    );
  }

  /* ─── Idle / placeholder state ─── */
  if (!result) {
    const workflowSteps = [
      { icon: '📤', label: 'Upload', desc: 'MRI scan + clinical biomarkers' },
      { icon: '⚙️', label: 'Process', desc: '3D ResNet + Bi-LSTM encoding' },
      { icon: '🎯', label: 'Predict', desc: 'Stage classification & forecast' },
      { icon: '📊', label: 'Explain', desc: 'GradCAM++ & SHAP attribution' },
    ];

    return (
      <div className="dash-panel dash-idle">
        {/* MRI Preview placeholder */}
        <div className="dash-preview-card">
          <div className="dash-preview-visual">
            <div className="dash-mri-placeholder">
              <div className="dash-mri-grid">
                <div className="dash-mri-slice dash-mri-axial">
                  <div className="dash-mri-brain" />
                  <span className="dash-mri-label">Axial</span>
                </div>
                <div className="dash-mri-slice dash-mri-sagittal">
                  <div className="dash-mri-brain" />
                  <span className="dash-mri-label">Sagittal</span>
                </div>
                <div className="dash-mri-slice dash-mri-coronal">
                  <div className="dash-mri-brain" />
                  <span className="dash-mri-label">Coronal</span>
                </div>
              </div>
              <div className="dash-mri-overlay">
                <span className="dash-mri-overlay-icon">🧠</span>
                <span className="dash-mri-overlay-text">Sample MRI Preview</span>
              </div>
            </div>
          </div>
          <div className="dash-preview-info">
            <h3 className="dash-preview-title">Ready for Analysis</h3>
            <p className="dash-preview-desc">
              Enter patient data on the left panel to run NeuroSense's multi-modal AI pipeline.
              Results will appear here with full explainability outputs.
            </p>
          </div>
        </div>

        {/* Workflow pipeline */}
        <div className="dash-workflow">
          <div className="dash-workflow-label">
            <span className="dash-workflow-dot" />
            Analysis Pipeline
          </div>
          <div className="dash-workflow-steps">
            {workflowSteps.map((step, i) => (
              <div className="dash-wf-step" key={i}>
                <div className="dash-wf-icon">{step.icon}</div>
                <div className="dash-wf-label">{step.label}</div>
                <div className="dash-wf-desc">{step.desc}</div>
                {i < workflowSteps.length - 1 && <div className="dash-wf-arrow">→</div>}
              </div>
            ))}
          </div>
        </div>

        {/* Output preview cards */}
        <div className="dash-output-grid">
          <div className="dash-output-card">
            <div className="dash-output-icon">🎯</div>
            <div className="dash-output-label">Stage Classification</div>
            <div className="dash-output-preview">
              <div className="dash-output-bar-group">
                <div className="dash-output-bar" style={{ width: '65%', background: 'var(--stage-pre)', opacity: 0.3 }} />
                <div className="dash-output-bar" style={{ width: '25%', background: 'var(--stage-early)', opacity: 0.3 }} />
                <div className="dash-output-bar" style={{ width: '10%', background: 'var(--stage-advanced)', opacity: 0.3 }} />
              </div>
            </div>
          </div>
          <div className="dash-output-card">
            <div className="dash-output-icon">📈</div>
            <div className="dash-output-label">Progression Forecast</div>
            <div className="dash-output-preview">
              <div className="dash-output-sparkline">
                <svg viewBox="0 0 100 40" className="dash-sparkline-svg">
                  <path d="M0,35 Q25,30 40,22 T70,15 T100,5" fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="4 3" opacity="0.3" />
                </svg>
              </div>
            </div>
          </div>
          <div className="dash-output-card">
            <div className="dash-output-icon">🔥</div>
            <div className="dash-output-label">GradCAM++ Heatmap</div>
            <div className="dash-output-preview">
              <div className="dash-output-heatmap">
                <div className="dash-heatmap-dot" style={{ top: '30%', left: '40%', width: '40px', height: '40px' }} />
                <div className="dash-heatmap-dot" style={{ top: '45%', left: '55%', width: '28px', height: '28px' }} />
                <div className="dash-heatmap-dot" style={{ top: '25%', left: '58%', width: '22px', height: '22px' }} />
              </div>
            </div>
          </div>
        </div>

        {/* Capabilities footer */}
        <div className="dash-capabilities">
          <div className="dash-cap-item">
            <span className="dash-cap-check">✓</span>
            <span>3-stage HD classification</span>
          </div>
          <div className="dash-cap-item">
            <span className="dash-cap-check">✓</span>
            <span>12 & 24-month forecasts</span>
          </div>
          <div className="dash-cap-item">
            <span className="dash-cap-check">✓</span>
            <span>Spatial brain heatmaps</span>
          </div>
          <div className="dash-cap-item">
            <span className="dash-cap-check">✓</span>
            <span>SHAP feature attribution</span>
          </div>
        </div>
      </div>
    );
  }

  /* ─── Results state ─── */
  const sc = STAGE_CONFIG[result.stage] || STAGE_CONFIG.early;

  return (
    <div className="dash-panel dash-results">
      {/* Results header */}
      <div className="dash-results-header">
        <div className="dash-results-header-dot" />
        <span className="dash-results-header-label">Analysis Complete</span>
        <span className="dash-results-header-time">
          {result.processing_time_s ? `${result.processing_time_s.toFixed(2)}s` : '—'}
        </span>
      </div>

      {/* Stage Classification */}
      <div className="result-card">
        <div className="result-header">
          <span>🎯</span>
          <span className="result-title">HD Stage Classification</span>
        </div>
        <div className="stage-display">
          <div className={`stage-badge ${result.stage}`}>{sc.label}</div>
          <div className="confidence-block">
            <p className="confidence-label">Confidence</p>
            <p className="confidence-number" style={{ color: sc.color }}>{(result.confidence * 100).toFixed(1)}%</p>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: `${result.confidence * 100}%`, background: sc.color }} />
            </div>
          </div>
        </div>
        {result.stage_probabilities && (
          <div className="prob-list" style={{ marginTop: 20 }}>
            {[
              { name: 'Pre-manifest', val: result.stage_probabilities.pre_manifest, color: 'var(--stage-pre)' },
              { name: 'Early HD', val: result.stage_probabilities.early, color: 'var(--stage-early)' },
              { name: 'Advanced HD', val: result.stage_probabilities.advanced, color: 'var(--stage-advanced)' },
            ].map((p) => (
              <div className="prob-row" key={p.name}>
                <span className="prob-name">{p.name}</span>
                <div className="prob-track">
                  <div className="prob-fill" style={{ width: `${p.val * 100}%`, background: p.color }} />
                </div>
                <span className="prob-val">{(p.val * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Progression Forecast */}
      <div className="result-card">
        <div className="result-header">
          <span>📈</span>
          <span className="result-title">Progression Forecast</span>
        </div>
        <div className="prog-grid">
          <div className="prog-item">
            <p className="prog-label">12-Month Δ</p>
            <p className={`prog-value ${result.progression_12mo >= 8 ? 'risk-high' : result.progression_12mo >= 3 ? 'risk-medium' : 'risk-low'}`}>
              {result.progression_12mo >= 0 ? '+' : ''}{result.progression_12mo?.toFixed(1)}
            </p>
            <p className="prog-unit">UHDRS Motor</p>
          </div>
          <div className="prog-item">
            <p className="prog-label">24-Month Δ</p>
            <p className={`prog-value ${result.progression_24mo >= 8 ? 'risk-high' : result.progression_24mo >= 3 ? 'risk-medium' : 'risk-low'}`}>
              {result.progression_24mo >= 0 ? '+' : ''}{result.progression_24mo?.toFixed(1)}
            </p>
            <p className="prog-unit">UHDRS Motor</p>
          </div>
          <div className="prog-item">
            <p className="prog-label">Risk Level</p>
            <p className={`prog-value risk-${result.risk_category}`}>{result.risk_category?.toUpperCase()}</p>
            <p className="prog-unit">Category</p>
          </div>
        </div>
      </div>

      {/* SHAP */}
      {result.shap_features?.length > 0 && (
        <div className="result-card">
          <div className="result-header">
            <span>📊</span>
            <span className="result-title">Feature Attribution (SHAP)</span>
          </div>
          <div className="shap-list">
            {result.shap_features.map((f) => {
              const maxI = Math.max(...result.shap_features.map((x) => Math.abs(x.impact)), 0.01);
              const w = Math.min((Math.abs(f.impact) / maxI) * 50, 50);
              const pos = f.impact >= 0;
              return (
                <div className="shap-row" key={f.name}>
                  <span className="shap-name">{FEATURE_LABELS[f.name] || f.name}</span>
                  <div className="shap-bar-wrap">
                    <div className="shap-midline" />
                    <div className={`shap-bar ${pos ? 'pos' : 'neg'}`} style={{ width: `${w}%` }} />
                  </div>
                  <span className={`shap-val ${pos ? 'pos' : 'neg'}`}>
                    {f.impact >= 0 ? '+' : ''}{f.impact.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* GradCAM */}
      <div className="result-card">
        <div className="result-header">
          <span>🔥</span>
          <span className="result-title">GradCAM++ Activation Map</span>
        </div>
        {result.gradcam_url ? (
          <div className="gradcam-wrap">
            <img src={`${API_BASE}${result.gradcam_url}`} alt="GradCAM++ heatmap overlay on axial MRI slices" />
          </div>
        ) : (
          <div className="gradcam-empty">
            <div className="gradcam-empty-icon">🧠</div>
            <p>Upload an MRI scan to generate spatial activation heatmaps</p>
          </div>
        )}
      </div>

      {/* Processing info */}
      <div className="proc-info">
        <span>Request: <span>{result.request_id || '—'}</span></span>
        <span>Processed in <span>{result.processing_time_s?.toFixed(2)}s</span></span>
      </div>

      {/* Download Report */}
      {lastForm && (
        <button className="btn-download-report" onClick={() => downloadReport(lastForm, result)}>
          📄 Download Report
        </button>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════
   ANALYSIS SECTION
   ═══════════════════════════════════════════ */

function AnalysisSection({ onSubmit, isLoading, result, error, lastForm }) {
  return (
    <section className="section analysis-section" id="analysis">
      <div className="section-inner">
        <div className="section-label reveal">Clinical Tool</div>
        <h2 className="section-title reveal">HD Analysis Dashboard</h2>
        <p className="section-subtitle reveal">
          Enter patient data below to generate AI-powered staging, progression forecasts, and explainability outputs.
        </p>
        <div className="analysis-grid">
          <ClinicalForm onSubmit={onSubmit} isLoading={isLoading} />
          <ResultsPanel result={result} error={error} isLoading={isLoading} lastForm={lastForm} />
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   MEMORY TEST SECTION
   ═══════════════════════════════════════════ */

// ── Word pools for Word Recall test ──
const WORD_POOLS = [
  ['MOUNTAIN', 'GARDEN', 'SHADOW', 'CRYSTAL', 'THUNDER', 'VELVET', 'HARBOR', 'CANDLE', 'FEATHER', 'ORCHID', 'LANTERN', 'MARBLE'],
  ['SUNSET', 'PIANO', 'DESERT', 'RIBBON', 'BEACON', 'FOREST', 'SILVER', 'BREEZE', 'TEMPLE', 'IVORY', 'CORAL', 'WALNUT'],
  ['GLACIER', 'PUZZLE', 'DRAGON', 'MEADOW', 'COPPER', 'VOYAGE', 'BASKET', 'SAFFRON', 'WILLOW', 'QUARTZ', 'NECTAR', 'FALCON'],
];

// ── Shape/color sets for Visual Change Detection ──
const VCD_SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star', 'hexagon'];
const VCD_COLORS = [
  '#e07a5f', '#3d9970', '#e09f3e', '#6366f1', '#ec4899', '#06b6d4',
  '#f97316', '#8b5cf6', '#14b8a6', '#ef4444', '#84cc16', '#d4a853',
];

// ── Sequence Memory tile colors ──
const SEQ_TILE_COLORS = [
  '#e07a5f', '#3d9970', '#6366f1', '#e09f3e', '#ec4899',
  '#06b6d4', '#f97316', '#8b5cf6', '#14b8a6',
];

function generateShapeGrid(seed, gridSize = 16) {
  const rng = mulberry32(seed);
  const grid = [];
  for (let i = 0; i < gridSize; i++) {
    grid.push({
      id: i,
      shape: VCD_SHAPES[Math.floor(rng() * VCD_SHAPES.length)],
      color: VCD_COLORS[Math.floor(rng() * VCD_COLORS.length)],
      rotation: Math.floor(rng() * 4) * 90,
      size: 0.7 + rng() * 0.5,
    });
  }
  return grid;
}

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ── Sub-components for each memory test ──

function WordRecallTest({ onComplete }) {
  const [phase, setPhase] = useState('intro'); // intro, study, distraction, recall, done
  const [words, setWords] = useState([]);
  const [timer, setTimer] = useState(0);
  const [recalledWords, setRecalledWords] = useState([]);
  const [currentInput, setCurrentInput] = useState('');
  const [studyStartTime, setStudyStartTime] = useState(null);
  const [distractionAnswer, setDistractionAnswer] = useState('');
  const [distractionProblems, setDistractionProblems] = useState([]);
  const [currentProblem, setCurrentProblem] = useState(0);
  const [distractionCorrect, setDistractionCorrect] = useState(0);
  const inputRef = useRef(null);
  const timerRef = useRef(null);

  const STUDY_TIME = 20;
  const DISTRACTION_TIME = 30;
  const RECALL_TIME = 45;
  const NUM_WORDS = 8;

  useEffect(() => {
    // Pick random word pool and select NUM_WORDS
    const poolIndex = Math.floor(Math.random() * WORD_POOLS.length);
    const pool = [...WORD_POOLS[poolIndex]];
    const selected = [];
    for (let i = 0; i < NUM_WORDS; i++) {
      const idx = Math.floor(Math.random() * pool.length);
      selected.push(pool.splice(idx, 1)[0]);
    }
    setWords(selected);

    // Generate distraction math problems
    const problems = [];
    for (let i = 0; i < 10; i++) {
      const a = Math.floor(Math.random() * 50) + 10;
      const b = Math.floor(Math.random() * 30) + 5;
      const ops = ['+', '-'];
      const op = ops[Math.floor(Math.random() * ops.length)];
      const answer = op === '+' ? a + b : a - b;
      problems.push({ question: `${a} ${op} ${b} = ?`, answer });
    }
    setDistractionProblems(problems);
  }, []);

  useEffect(() => {
    if (phase === 'study') {
      setStudyStartTime(Date.now());
      setTimer(STUDY_TIME);
      timerRef.current = setInterval(() => {
        setTimer((t) => {
          if (t <= 1) {
            clearInterval(timerRef.current);
            setPhase('distraction');
            return 0;
          }
          return t - 1;
        });
      }, 1000);
    } else if (phase === 'distraction') {
      setTimer(DISTRACTION_TIME);
      timerRef.current = setInterval(() => {
        setTimer((t) => {
          if (t <= 1) {
            clearInterval(timerRef.current);
            setPhase('recall');
            return 0;
          }
          return t - 1;
        });
      }, 1000);
    } else if (phase === 'recall') {
      setTimer(RECALL_TIME);
      setTimeout(() => inputRef.current?.focus(), 100);
      timerRef.current = setInterval(() => {
        setTimer((t) => {
          if (t <= 1) {
            clearInterval(timerRef.current);
            setPhase('done');
            return 0;
          }
          return t - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timerRef.current);
  }, [phase]);

  useEffect(() => {
    if (phase === 'done') {
      const correct = recalledWords.filter((w) =>
        words.includes(w.toUpperCase())
      ).length;
      onComplete({
        test: 'word_recall',
        score: correct,
        total: NUM_WORDS,
        percentage: Math.round((correct / NUM_WORDS) * 100),
        wordsShown: words,
        wordsRecalled: recalledWords,
      });
    }
  }, [phase]);

  const handleRecallSubmit = (e) => {
    e.preventDefault();
    const word = currentInput.trim().toUpperCase();
    if (word && !recalledWords.map((w) => w.toUpperCase()).includes(word)) {
      setRecalledWords((prev) => [...prev, word]);
    }
    setCurrentInput('');
    inputRef.current?.focus();
  };

  const handleDistractionSubmit = (e) => {
    e.preventDefault();
    if (distractionProblems[currentProblem]) {
      if (parseInt(distractionAnswer) === distractionProblems[currentProblem].answer) {
        setDistractionCorrect((c) => c + 1);
      }
      setDistractionAnswer('');
      if (currentProblem < distractionProblems.length - 1) {
        setCurrentProblem((p) => p + 1);
      }
    }
  };

  const finishRecall = () => {
    clearInterval(timerRef.current);
    setPhase('done');
  };

  if (phase === 'intro') {
    return (
      <div className="mt-test-card">
        <div className="mt-test-icon">📝</div>
        <h4 className="mt-test-name">Word Recall Test</h4>
        <p className="mt-test-desc">
          You will see <strong>{NUM_WORDS} words</strong> for {STUDY_TIME} seconds. Memorize as many as you can.
          After a {DISTRACTION_TIME}-second distraction task (math problems), type back all the words you remember.
        </p>
        <button className="mt-btn-start" onClick={() => setPhase('study')}>
          🧠 Begin Test
        </button>
      </div>
    );
  }

  if (phase === 'study') {
    return (
      <div className="mt-test-card mt-active">
        <div className="mt-phase-header">
          <span className="mt-phase-badge study">📖 Study Phase</span>
          <span className="mt-timer">{timer}s</span>
        </div>
        <div className="mt-timer-bar">
          <div className="mt-timer-fill" style={{ width: `${(timer / STUDY_TIME) * 100}%` }} />
        </div>
        <p className="mt-instruction">Memorize these words:</p>
        <div className="mt-word-grid">
          {words.map((word, i) => (
            <div key={i} className="mt-word-chip" style={{ animationDelay: `${i * 0.08}s` }}>
              {word}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (phase === 'distraction') {
    const problem = distractionProblems[currentProblem];
    return (
      <div className="mt-test-card mt-active">
        <div className="mt-phase-header">
          <span className="mt-phase-badge distraction">🔢 Distraction Phase</span>
          <span className="mt-timer">{timer}s</span>
        </div>
        <div className="mt-timer-bar">
          <div className="mt-timer-fill distraction" style={{ width: `${(timer / DISTRACTION_TIME) * 100}%` }} />
        </div>
        <p className="mt-instruction">Solve these math problems (to clear short-term memory):</p>
        {problem && (
          <form className="mt-distraction-form" onSubmit={handleDistractionSubmit}>
            <div className="mt-math-problem">{problem.question}</div>
            <input
              className="mt-input"
              type="number"
              value={distractionAnswer}
              onChange={(e) => setDistractionAnswer(e.target.value)}
              placeholder="Your answer"
              autoFocus
            />
            <button type="submit" className="mt-btn-submit">Submit</button>
          </form>
        )}
        <div className="mt-distraction-progress">
          Solved: {distractionCorrect}/{currentProblem + (distractionAnswer ? 0 : 0)} correct
        </div>
      </div>
    );
  }

  if (phase === 'recall') {
    return (
      <div className="mt-test-card mt-active">
        <div className="mt-phase-header">
          <span className="mt-phase-badge recall">💭 Recall Phase</span>
          <span className="mt-timer">{timer}s</span>
        </div>
        <div className="mt-timer-bar">
          <div className="mt-timer-fill recall" style={{ width: `${(timer / RECALL_TIME) * 100}%` }} />
        </div>
        <p className="mt-instruction">Type the words you remember (one at a time):</p>
        <form className="mt-recall-form" onSubmit={handleRecallSubmit}>
          <input
            ref={inputRef}
            className="mt-input"
            type="text"
            value={currentInput}
            onChange={(e) => setCurrentInput(e.target.value)}
            placeholder="Type a word and press Enter"
            autoComplete="off"
          />
          <button type="submit" className="mt-btn-submit">Add</button>
        </form>
        <div className="mt-recalled-words">
          {recalledWords.map((word, i) => (
            <span key={i} className={`mt-recalled-chip ${words.includes(word.toUpperCase()) ? 'correct' : 'incorrect'}`}>
              {word}
            </span>
          ))}
        </div>
        <div className="mt-recall-count">
          {recalledWords.length} word{recalledWords.length !== 1 ? 's' : ''} recalled
        </div>
        <button className="mt-btn-finish" onClick={finishRecall}>
          ✅ Done Recalling
        </button>
      </div>
    );
  }

  return null;
}

function SequenceMemoryTest({ onComplete }) {
  const [phase, setPhase] = useState('intro'); // intro, watch, input, result, done
  const [level, setLevel] = useState(1);
  const [sequence, setSequence] = useState([]);
  const [playerSequence, setPlayerSequence] = useState([]);
  const [activeTile, setActiveTile] = useState(null);
  const [showingSequence, setShowingSequence] = useState(false);
  const [isCorrect, setIsCorrect] = useState(null);
  const [maxLevel, setMaxLevel] = useState(0);
  const [lives, setLives] = useState(3);
  const GRID_SIZE = 9;

  const generateSequence = useCallback((len) => {
    const seq = [];
    for (let i = 0; i < len; i++) {
      seq.push(Math.floor(Math.random() * GRID_SIZE));
    }
    return seq;
  }, []);

  const playSequence = useCallback((seq) => {
    setShowingSequence(true);
    setActiveTile(null);
    let i = 0;
    const interval = setInterval(() => {
      if (i < seq.length) {
        setActiveTile(seq[i]);
        setTimeout(() => setActiveTile(null), 500);
        i++;
      } else {
        clearInterval(interval);
        setShowingSequence(false);
        setPhase('input');
      }
    }, 800);
  }, []);

  const startLevel = useCallback(() => {
    const seq = generateSequence(level);
    setSequence(seq);
    setPlayerSequence([]);
    setIsCorrect(null);
    setPhase('watch');
    setTimeout(() => playSequence(seq), 600);
  }, [level, generateSequence, playSequence]);

  const handleTileClick = (index) => {
    if (showingSequence || phase !== 'input') return;

    const newPlayerSeq = [...playerSequence, index];
    setPlayerSequence(newPlayerSeq);
    setActiveTile(index);
    setTimeout(() => setActiveTile(null), 200);

    const currentPos = newPlayerSeq.length - 1;
    if (newPlayerSeq[currentPos] !== sequence[currentPos]) {
      // Wrong
      const newLives = lives - 1;
      setLives(newLives);
      setIsCorrect(false);
      setPhase('result');
      if (newLives <= 0) {
        setTimeout(() => {
          setPhase('done');
          onComplete({
            test: 'sequence_memory',
            score: maxLevel,
            total: maxLevel,
            percentage: Math.min(Math.round((maxLevel / 12) * 100), 100),
            maxSequenceLength: maxLevel,
          });
        }, 1500);
      }
    } else if (newPlayerSeq.length === sequence.length) {
      // Completed sequence correctly
      const newMax = Math.max(maxLevel, level);
      setMaxLevel(newMax);
      setIsCorrect(true);
      setPhase('result');
      setTimeout(() => {
        setLevel((l) => l + 1);
      }, 1200);
    }
  };

  useEffect(() => {
    if (phase === 'result' && isCorrect && lives > 0) {
      const timeout = setTimeout(() => {
        startLevel();
      }, 300);
      return () => clearTimeout(timeout);
    }
  }, [level]);

  const stopTest = () => {
    setPhase('done');
    onComplete({
      test: 'sequence_memory',
      score: maxLevel,
      total: maxLevel,
      percentage: Math.min(Math.round((maxLevel / 12) * 100), 100),
      maxSequenceLength: maxLevel,
    });
  };

  if (phase === 'intro') {
    return (
      <div className="mt-test-card">
        <div className="mt-test-icon">🔲</div>
        <h4 className="mt-test-name">Sequence Memory</h4>
        <p className="mt-test-desc">
          Watch the tiles light up in a sequence, then repeat the pattern by clicking them in the same order.
          The sequence grows longer each round. You have <strong>3 lives</strong>.
        </p>
        <button className="mt-btn-start" onClick={startLevel}>
          🎯 Begin Test
        </button>
      </div>
    );
  }

  if (phase === 'done') return null;

  return (
    <div className="mt-test-card mt-active">
      <div className="mt-phase-header">
        <span className="mt-phase-badge study">
          {phase === 'watch' ? '👁️ Watch' : phase === 'input' ? '👆 Repeat' : isCorrect ? '✅ Correct!' : '❌ Wrong'}
        </span>
        <div className="mt-seq-info">
          <span className="mt-level">Level {level}</span>
          <span className="mt-lives">{'❤️'.repeat(lives)}{'🖤'.repeat(3 - lives)}</span>
        </div>
      </div>

      {phase === 'result' && (
        <div className={`mt-seq-result ${isCorrect ? 'correct' : 'incorrect'}`}>
          {isCorrect ? `Level ${level - 1} complete!` : 'Wrong sequence!'}
        </div>
      )}

      <div className="mt-tile-grid">
        {Array.from({ length: GRID_SIZE }, (_, i) => (
          <button
            key={i}
            className={`mt-tile ${activeTile === i ? 'active' : ''} ${phase === 'input' ? 'clickable' : ''}`}
            onClick={() => handleTileClick(i)}
            disabled={phase !== 'input'}
            style={{ '--tile-color': SEQ_TILE_COLORS[i] }}
          />
        ))}
      </div>

      <div className="mt-seq-progress">
        {phase === 'input' && (
          <div className="mt-seq-dots">
            {sequence.map((_, i) => (
              <span key={i} className={`mt-seq-dot ${i < playerSequence.length ? 'filled' : ''}`} />
            ))}
          </div>
        )}
      </div>

      <button className="mt-btn-stop" onClick={stopTest}>
        🛑 End Test
      </button>
    </div>
  );
}

function VisualChangeDetectionTest({ onComplete }) {
  const [phase, setPhase] = useState('intro'); // intro, study, blank, detect, done
  const [originalGrid, setOriginalGrid] = useState([]);
  const [changedGrid, setChangedGrid] = useState([]);
  const [changedIndices, setChangedIndices] = useState([]);
  const [selectedIndices, setSelectedIndices] = useState([]);
  const [timer, setTimer] = useState(0);
  const [round, setRound] = useState(0);
  const [roundResults, setRoundResults] = useState([]);
  const timerRef = useRef(null);

  const STUDY_TIME = 8;
  const BLANK_TIME = 3;
  const DETECT_TIME = 15;
  const TOTAL_ROUNDS = 3;
  const GRID_SIZE = 16;
  const CHANGES_PER_ROUND = [2, 3, 4];

  const startRound = useCallback((roundNum) => {
    const seed = Date.now() + roundNum * 1000;
    const grid = generateShapeGrid(seed, GRID_SIZE);
    setOriginalGrid(grid);
    setSelectedIndices([]);

    // Create changed version
    const numChanges = CHANGES_PER_ROUND[roundNum] || 2;
    const indices = [];
    const availableIndices = [...Array(GRID_SIZE).keys()];
    for (let i = 0; i < numChanges; i++) {
      const idx = Math.floor(Math.random() * availableIndices.length);
      indices.push(availableIndices.splice(idx, 1)[0]);
    }
    setChangedIndices(indices);

    const modified = grid.map((item, i) => {
      if (indices.includes(i)) {
        const changeType = Math.floor(Math.random() * 3);
        const newItem = { ...item };
        if (changeType === 0) {
          // Change color
          let newColor;
          do {
            newColor = VCD_COLORS[Math.floor(Math.random() * VCD_COLORS.length)];
          } while (newColor === item.color);
          newItem.color = newColor;
        } else if (changeType === 1) {
          // Change shape
          let newShape;
          do {
            newShape = VCD_SHAPES[Math.floor(Math.random() * VCD_SHAPES.length)];
          } while (newShape === item.shape);
          newItem.shape = newShape;
        } else {
          // Remove (make invisible)
          newItem.shape = 'removed';
        }
        return newItem;
      }
      return { ...item };
    });
    setChangedGrid(modified);

    setPhase('study');
    setTimer(STUDY_TIME);
  }, []);

  useEffect(() => {
    clearInterval(timerRef.current);

    if (phase === 'study') {
      timerRef.current = setInterval(() => {
        setTimer((t) => {
          if (t <= 1) {
            clearInterval(timerRef.current);
            setPhase('blank');
            return 0;
          }
          return t - 1;
        });
      }, 1000);
    } else if (phase === 'blank') {
      setTimer(BLANK_TIME);
      timerRef.current = setInterval(() => {
        setTimer((t) => {
          if (t <= 1) {
            clearInterval(timerRef.current);
            setPhase('detect');
            return 0;
          }
          return t - 1;
        });
      }, 1000);
    } else if (phase === 'detect') {
      setTimer(DETECT_TIME);
      timerRef.current = setInterval(() => {
        setTimer((t) => {
          if (t <= 1) {
            clearInterval(timerRef.current);
            finishRound();
            return 0;
          }
          return t - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timerRef.current);
  }, [phase]);

  const handleShapeClick = (index) => {
    if (phase !== 'detect') return;
    setSelectedIndices((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  const finishRound = () => {
    clearInterval(timerRef.current);
    const correctSelections = selectedIndices.filter((i) => changedIndices.includes(i)).length;
    const incorrectSelections = selectedIndices.filter((i) => !changedIndices.includes(i)).length;
    const missed = changedIndices.filter((i) => !selectedIndices.includes(i)).length;

    const result = {
      round: round + 1,
      totalChanges: changedIndices.length,
      correctlyIdentified: correctSelections,
      incorrectSelections,
      missed,
      score: Math.max(0, correctSelections - incorrectSelections),
    };

    const newResults = [...roundResults, result];
    setRoundResults(newResults);

    if (round + 1 < TOTAL_ROUNDS) {
      setRound((r) => r + 1);
      setTimeout(() => startRound(round + 1), 1500);
    } else {
      setPhase('done');
      const totalScore = newResults.reduce((s, r) => s + r.correctlyIdentified, 0);
      const totalPossible = newResults.reduce((s, r) => s + r.totalChanges, 0);
      onComplete({
        test: 'visual_change',
        score: totalScore,
        total: totalPossible,
        percentage: Math.round((totalScore / totalPossible) * 100),
        rounds: newResults,
      });
    }

    // Brief show-answer phase
    if (round + 1 < TOTAL_ROUNDS) {
      setPhase('showing_answers');
    }
  };

  const renderShape = (item, isChanged, isSelected) => {
    const baseClass = `mt-shape ${item.shape}`;
    const stateClass = isSelected ? 'selected' : '';
    const removedClass = item.shape === 'removed' ? 'removed' : '';
    return (
      <div
        key={item.id}
        className={`mt-shape-cell ${stateClass} ${removedClass} ${phase === 'detect' ? 'clickable' : ''}`}
        onClick={() => handleShapeClick(item.id)}
        style={{ '--shape-color': item.color }}
      >
        <div className={baseClass} style={{ transform: `rotate(${item.rotation}deg) scale(${item.size})` }}>
          {renderShapeSVG(item.shape, item.color)}
        </div>
      </div>
    );
  };

  if (phase === 'intro') {
    return (
      <div className="mt-test-card">
        <div className="mt-test-icon">🔍</div>
        <h4 className="mt-test-name">Visual Change Detection</h4>
        <p className="mt-test-desc">
          Study a grid of shapes and colors for {STUDY_TIME} seconds. After a brief blank screen,
          the grid reappears with <strong>some changes</strong>. Click the shapes that changed.
          {TOTAL_ROUNDS} rounds with increasing difficulty.
        </p>
        <button className="mt-btn-start" onClick={() => startRound(0)}>
          👁️ Begin Test
        </button>
      </div>
    );
  }

  if (phase === 'done') return null;

  const displayGrid = phase === 'study' ? originalGrid : phase === 'blank' ? [] : changedGrid;

  return (
    <div className="mt-test-card mt-active">
      <div className="mt-phase-header">
        <span className={`mt-phase-badge ${phase === 'study' ? 'study' : phase === 'blank' ? 'distraction' : 'recall'}`}>
          {phase === 'study' ? '📖 Memorize' : phase === 'blank' ? '⏳ Wait...' : phase === 'showing_answers' ? '📊 Results' : '🔍 Find Changes'}
        </span>
        <div className="mt-seq-info">
          <span className="mt-level">Round {round + 1}/{TOTAL_ROUNDS}</span>
          <span className="mt-timer">{timer}s</span>
        </div>
      </div>
      <div className="mt-timer-bar">
        <div
          className={`mt-timer-fill ${phase === 'study' ? '' : phase === 'detect' ? 'recall' : 'distraction'}`}
          style={{
            width: `${(timer / (phase === 'study' ? STUDY_TIME : phase === 'blank' ? BLANK_TIME : DETECT_TIME)) * 100}%`,
          }}
        />
      </div>

      {phase === 'blank' ? (
        <div className="mt-blank-screen">
          <div className="mt-blank-icon">🧠</div>
          <p>Hold the image in your mind...</p>
        </div>
      ) : (
        <>
          {phase === 'detect' && (
            <p className="mt-instruction">Click the shapes that are different from before:</p>
          )}
          <div className="mt-shape-grid">
            {displayGrid.map((item) =>
              renderShape(
                item,
                changedIndices.includes(item.id),
                selectedIndices.includes(item.id)
              )
            )}
          </div>
          {phase === 'detect' && (
            <div className="mt-detect-footer">
              <span className="mt-selected-count">
                {selectedIndices.length} selected · {CHANGES_PER_ROUND[round]} changes to find
              </span>
              <button className="mt-btn-submit" onClick={finishRound}>
                ✅ Submit Answer
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function renderShapeSVG(shape, color) {
  const size = 32;
  switch (shape) {
    case 'circle':
      return <svg width={size} height={size} viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill={color} /></svg>;
    case 'square':
      return <svg width={size} height={size} viewBox="0 0 32 32"><rect x="4" y="4" width="24" height="24" rx="3" fill={color} /></svg>;
    case 'triangle':
      return <svg width={size} height={size} viewBox="0 0 32 32"><polygon points="16,2 30,28 2,28" fill={color} /></svg>;
    case 'diamond':
      return <svg width={size} height={size} viewBox="0 0 32 32"><polygon points="16,2 30,16 16,30 2,16" fill={color} /></svg>;
    case 'star':
      return <svg width={size} height={size} viewBox="0 0 32 32"><polygon points="16,2 20,12 30,12 22,19 25,30 16,23 7,30 10,19 2,12 12,12" fill={color} /></svg>;
    case 'hexagon':
      return <svg width={size} height={size} viewBox="0 0 32 32"><polygon points="16,2 28,9 28,23 16,30 4,23 4,9" fill={color} /></svg>;
    case 'removed':
      return <svg width={size} height={size} viewBox="0 0 32 32"><rect x="4" y="4" width="24" height="24" rx="3" fill="transparent" stroke={color} strokeWidth="1.5" strokeDasharray="4,4" opacity="0.3" /></svg>;
    default:
      return null;
  }
}

// ── Memory Test Results Dashboard (Enhanced) ──
function MemoryTestResults({ results }) {
  if (results.length === 0) return null;

  const overallScore = Math.round(
    results.reduce((sum, r) => sum + r.percentage, 0) / results.length
  );

  const getScoreLevel = (pct) => {
    if (pct >= 80) return { label: 'Excellent', color: 'var(--success)', bg: 'var(--success-soft)', emoji: '🟢' };
    if (pct >= 60) return { label: 'Good', color: 'var(--gold)', bg: 'var(--gold-soft)', emoji: '🟡' };
    if (pct >= 40) return { label: 'Fair', color: 'var(--warning)', bg: 'var(--warning-soft)', emoji: '🟠' };
    return { label: 'Needs Attention', color: 'var(--danger)', bg: 'var(--danger-soft)', emoji: '🔴' };
  };

  const overall = getScoreLevel(overallScore);

  const domainInfo = {
    word_recall: { name: 'Word Recall', icon: '📝', domain: 'Verbal Episodic Memory', abbr: 'VEM' },
    sequence_memory: { name: 'Sequence Memory', icon: '🔲', domain: 'Visuospatial Working Memory', abbr: 'VWM' },
    visual_change: { name: 'Visual Change', icon: '🔍', domain: 'Visual Short-Term Memory', abbr: 'VSTM' },
  };

  // Radar chart points calculation
  const radarSize = 200;
  const center = radarSize / 2;
  const maxRadius = 70;
  const angles = results.map((_, i) => (i * 2 * Math.PI) / Math.max(results.length, 3) - Math.PI / 2);
  const radarPoints = results.map((r, i) => {
    const radius = (r.percentage / 100) * maxRadius;
    return `${center + radius * Math.cos(angles[i])},${center + radius * Math.sin(angles[i])}`;
  }).join(' ');
  const bgPoints = [0, 1, 2].map((i) => {
    const angle = (i * 2 * Math.PI) / 3 - Math.PI / 2;
    return `${center + maxRadius * Math.cos(angle)},${center + maxRadius * Math.sin(angle)}`;
  }).join(' ');

  const strengths = results.filter(r => r.percentage >= 70);
  const areas = results.filter(r => r.percentage < 50);

  return (
    <div className="mt-results-dashboard">
      {/* Header */}
      <div className="mt-results-header">
        <div className="mt-results-icon">📊</div>
        <div>
          <h3 className="mt-results-title">Cognitive Assessment Summary</h3>
          <p className="mt-results-sub">{results.length} of 3 domains assessed</p>
        </div>
      </div>

      {/* Score Ring + Radar side by side */}
      <div className="mt-score-overview">
        <div className="mt-overall-score">
          <div className="mt-score-ring" style={{ '--score-pct': overallScore, '--score-color': overall.color }}>
            <svg viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="52" fill="none" stroke="var(--border)" strokeWidth="8" />
              <circle
                cx="60" cy="60" r="52" fill="none"
                stroke={overall.color}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${overallScore * 3.27} 327`}
                transform="rotate(-90 60 60)"
                className="mt-score-circle"
              />
            </svg>
            <div className="mt-score-inner">
              <span className="mt-score-number">{overallScore}%</span>
            </div>
          </div>
          <span className="mt-score-label" style={{ color: overall.color }}>{overall.label}</span>
        </div>

        {/* Radar Chart */}
        {results.length >= 2 && (
          <div className="mt-radar-wrap">
            <svg viewBox={`0 0 ${radarSize} ${radarSize}`} className="mt-radar-svg">
              {/* Background rings */}
              {[0.25, 0.5, 0.75, 1].map((pct) => (
                <polygon key={pct} points={
                  [0, 1, 2].map(i => {
                    const a = (i * 2 * Math.PI) / 3 - Math.PI / 2;
                    const r = maxRadius * pct;
                    return `${center + r * Math.cos(a)},${center + r * Math.sin(a)}`;
                  }).join(' ')
                } fill="none" stroke="var(--border)" strokeWidth="0.5" />
              ))}
              {/* Axis lines */}
              {[0, 1, 2].map(i => {
                const a = (i * 2 * Math.PI) / 3 - Math.PI / 2;
                return <line key={i} x1={center} y1={center} x2={center + maxRadius * Math.cos(a)} y2={center + maxRadius * Math.sin(a)} stroke="var(--border)" strokeWidth="0.5" />;
              })}
              {/* Data polygon */}
              <polygon points={radarPoints} fill="rgba(224, 122, 95, 0.15)" stroke="var(--accent)" strokeWidth="2" />
              {/* Data dots */}
              {results.map((r, i) => {
                const radius = (r.percentage / 100) * maxRadius;
                return <circle key={i} cx={center + radius * Math.cos(angles[i])} cy={center + radius * Math.sin(angles[i])} r="4" fill="var(--accent)" />;
              })}
              {/* Labels */}
              {results.map((r, i) => {
                const info = domainInfo[r.test];
                const labelR = maxRadius + 18;
                const a = angles[i];
                return (
                  <text key={i} x={center + labelR * Math.cos(a)} y={center + labelR * Math.sin(a)} textAnchor="middle" dominantBaseline="middle" fontSize="9" fontWeight="700" fill="var(--text-secondary)">
                    {info?.abbr || r.test}
                  </text>
                );
              })}
            </svg>
          </div>
        )}
      </div>

      {/* Domain Breakdown */}
      <div className="mt-domain-grid">
        {results.map((r, i) => {
          const level = getScoreLevel(r.percentage);
          const info = domainInfo[r.test] || { name: r.test, icon: '🧠', domain: 'Unknown', abbr: '?' };
          return (
            <div key={i} className="mt-domain-card" style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="mt-domain-top">
                <span className="mt-domain-icon">{info.icon}</span>
                <div className="mt-domain-score-pill" style={{ background: level.bg, color: level.color }}>
                  {r.percentage}%
                </div>
              </div>
              <h4 className="mt-domain-name">{info.name}</h4>
              <p className="mt-domain-domain">{info.domain}</p>
              <div className="mt-domain-bar">
                <div className="mt-domain-fill" style={{ width: `${r.percentage}%`, background: level.color }} />
              </div>
              <div className="mt-domain-detail">
                <span>{r.score}/{r.total} correct</span>
                <span className="mt-domain-level" style={{ color: level.color }}>{level.emoji} {level.label}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Cognitive Profile */}
      {results.length >= 2 && (
        <div className="mt-profile-section">
          <h4 className="mt-profile-title">🧠 Cognitive Profile</h4>
          <div className="mt-profile-grid">
            {strengths.length > 0 && (
              <div className="mt-profile-card strengths">
                <div className="mt-profile-card-label">✅ Strengths</div>
                {strengths.map((r) => (
                  <div key={r.test} className="mt-profile-item">
                    <span>{domainInfo[r.test]?.icon}</span>
                    <span>{domainInfo[r.test]?.domain || r.test}</span>
                  </div>
                ))}
              </div>
            )}
            {areas.length > 0 && (
              <div className="mt-profile-card concerns">
                <div className="mt-profile-card-label">⚠️ Areas for Review</div>
                {areas.map((r) => (
                  <div key={r.test} className="mt-profile-item">
                    <span>{domainInfo[r.test]?.icon}</span>
                    <span>{domainInfo[r.test]?.domain || r.test}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="mt-results-disclaimer">
        <div className="mt-disclaimer-icon">⚕️</div>
        <div>
          <strong>Clinical Disclaimer</strong>
          <p>These tests are supplementary cognitive assessments and are not diagnostic tools. For clinical evaluation, consult a qualified neuropsychologist. Results should be interpreted in conjunction with comprehensive clinical data.</p>
        </div>
      </div>
    </div>
  );
}

// ── Main Memory Test Section (Enhanced) ──
function MemoryTestSection() {
  const [activeTest, setActiveTest] = useState(null);
  const [completedTests, setCompletedTests] = useState([]);
  const [testResults, setTestResults] = useState([]);

  const handleTestComplete = (result) => {
    setTestResults((prev) => [...prev, result]);
    setCompletedTests((prev) => [...prev, result.test]);
    setActiveTest(null);
  };

  const resetAll = () => {
    setActiveTest(null);
    setCompletedTests([]);
    setTestResults([]);
  };

  const tests = [
    {
      id: 'word_recall',
      name: 'Word Recall',
      icon: '📝',
      desc: 'Memorize a list of words, then recall them after a distraction period',
      duration: '~2 min',
      domain: 'Verbal Episodic Memory',
      difficulty: 'Easy',
      diffColor: 'var(--success)',
    },
    {
      id: 'sequence_memory',
      name: 'Sequence Memory',
      icon: '🔲',
      desc: 'Repeat growing tile patterns in the correct order with increasing difficulty',
      duration: '~2 min',
      domain: 'Visuospatial Working Memory',
      difficulty: 'Medium',
      diffColor: 'var(--warning)',
    },
    {
      id: 'visual_change',
      name: 'Visual Change Detection',
      icon: '🔍',
      desc: 'Study a grid of shapes, then identify what changed after a blank interval',
      duration: '~3 min',
      domain: 'Visual Short-Term Memory',
      difficulty: 'Hard',
      diffColor: 'var(--danger)',
    },
  ];

  const progress = completedTests.length;

  return (
    <section className="section memory-test-section" id="memory-test">
      <div className="section-inner">
        <div className="section-label reveal">Cognitive Screening</div>
        <h2 className="section-title reveal">Memory Assessment Battery</h2>
        <p className="section-subtitle reveal">
          A digital cognitive screening battery targeting memory domains commonly affected in
          early Huntington's Disease. Complete all three tests for a comprehensive assessment.
        </p>

        {/* Progress tracker */}
        {!activeTest && (
          <div className="mt-progress-tracker reveal">
            <div className="mt-progress-steps">
              {tests.map((test, i) => {
                const isDone = completedTests.includes(test.id);
                return (
                  <Fragment key={test.id}>
                    {i > 0 && <div className={`mt-progress-line ${isDone || completedTests.includes(tests[i - 1]?.id) ? 'active' : ''}`} />}
                    <div className={`mt-progress-step ${isDone ? 'done' : ''}`}>
                      <div className="mt-progress-step-circle">
                        {isDone ? '✓' : i + 1}
                      </div>
                      <span className="mt-progress-step-label">{test.name}</span>
                    </div>
                  </Fragment>
                );
              })}
            </div>
            <div className="mt-progress-text">{progress}/3 completed</div>
          </div>
        )}

        {/* Test Cards */}
        {!activeTest && (
          <div className="mt-selector stagger">
            {tests.map((test) => {
              const isCompleted = completedTests.includes(test.id);
              const result = testResults.find((r) => r.test === test.id);
              return (
                <div
                  key={test.id}
                  className={`mt-selector-card reveal ${isCompleted ? 'completed' : ''}`}
                  onClick={() => !isCompleted && setActiveTest(test.id)}
                >
                  <div className="mt-selector-status">
                    {isCompleted ? <span className="mt-check">✅</span> : <span className="mt-dot" />}
                  </div>
                  <div className="mt-selector-icon">{test.icon}</div>
                  <h4 className="mt-selector-name">{test.name}</h4>
                  <div className="mt-selector-tags">
                    <span className="mt-tag mt-tag-diff" style={{ '--tag-color': test.diffColor }}>{test.difficulty}</span>
                    <span className="mt-tag mt-tag-domain">🧠 {test.domain.split(' ')[0]}</span>
                  </div>
                  <p className="mt-selector-desc">{test.desc}</p>
                  <div className="mt-selector-meta">
                    <span>⏱️ {test.duration}</span>
                    <span>📐 {test.domain}</span>
                  </div>
                  {!isCompleted && (
                    <button className="mt-selector-btn">Start Test →</button>
                  )}
                  {isCompleted && result && (
                    <div className="mt-selector-done">
                      <span className="mt-selector-done-score">{result.percentage}%</span>
                      <span className="mt-selector-done-label">Score achieved</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {activeTest === 'word_recall' && (
          <WordRecallTest onComplete={handleTestComplete} />
        )}
        {activeTest === 'sequence_memory' && (
          <SequenceMemoryTest onComplete={handleTestComplete} />
        )}
        {activeTest === 'visual_change' && (
          <VisualChangeDetectionTest onComplete={handleTestComplete} />
        )}

        <MemoryTestResults results={testResults} />

        {completedTests.length > 0 && !activeTest && (
          <div className="mt-reset-wrap">
            <button className="mt-btn-reset" onClick={resetAll}>
              🔄 Retake All Tests
            </button>
          </div>
        )}

        {/* Section-level disclaimer */}
        {!activeTest && completedTests.length === 0 && (
          <div className="mt-section-disclaimer reveal">
            ⚕️ These tests are supplementary cognitive assessments and are not diagnostic tools. Always consult a qualified healthcare professional for clinical evaluation.
          </div>
        )}
      </div>
    </section>
  );
}
/* ═══════════════════════════════════════════
   METHODOLOGY SECTION
   ═══════════════════════════════════════════ */

function MethodologySection() {
  const methods = [
    { icon: '🔬', title: 'Data Preprocessing', desc: 'T1-weighted MRI scans are skull-stripped, registered to MNI152 standard space, and normalized to 128×128×128 voxel resolution using NiBabel and MONAI transforms.' },
    { icon: '🧠', title: 'Model Architecture', desc: 'A 3D ResNet-50 backbone processes volumetric MRI data while a Bi-LSTM encoder handles longitudinal clinical sequences. Cross-modal attention fuses both streams.' },
    { icon: '⚙️', title: 'Training Strategy', desc: 'Multi-task learning with weighted cross-entropy for classification and MSE for regression. 5-fold cross-validation with early stopping and cosine annealing scheduler.' },
    { icon: '📊', title: 'Evaluation & Explainability', desc: 'Models evaluated on accuracy, F1, AUC-ROC. GradCAM++ heatmaps and SHAP values provide transparent, interpretable explanations for every prediction.' },
  ];

  return (
    <section className="section" id="methodology">
      <div className="section-inner">
        <div className="section-label reveal">Research Approach</div>
        <h2 className="section-title reveal">Methodology</h2>
        <p className="section-subtitle reveal">
          Our multi-modal deep learning pipeline combines neuroimaging and clinical data
          for robust HD stage classification and progression forecasting.
        </p>
        <div className="method-grid stagger">
          {methods.map((m, i) => (
            <div className="method-card reveal" key={i} style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="method-icon">{m.icon}</div>
              <h3 className="method-title">{m.title}</h3>
              <p className="method-desc">{m.desc}</p>
              <div className="method-number">{String(i + 1).padStart(2, '0')}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   TECHNOLOGIES SECTION
   ═══════════════════════════════════════════ */

function TechnologiesSection() {
  const techs = [
    { name: 'PyTorch', role: 'Deep Learning Framework', color: '#ee4c2c' },
    { name: 'MONAI', role: '3D Medical Image Processing', color: '#10b981' },
    { name: 'FastAPI', role: 'REST API Backend', color: '#009688' },
    { name: 'React', role: 'Frontend Interface', color: '#61dafb' },
    { name: 'Vite', role: 'Build Tooling & HMR', color: '#646cff' },
    { name: 'SHAP', role: 'Feature Attribution', color: '#f59e0b' },
    { name: 'GradCAM++', role: 'Visual Explanations', color: '#ef4444' },
    { name: 'NiBabel', role: 'NIfTI File Processing', color: '#8b5cf6' },
  ];

  return (
    <section className="section tech-section" id="technologies">
      <div className="section-inner">
        <div className="section-label reveal">Tech Stack</div>
        <h2 className="section-title reveal">Technologies Used</h2>
        <p className="section-subtitle reveal">
          Built with industry-standard tools for medical AI research, ensuring reproducibility
          and clinical-grade reliability.
        </p>
        <div className="tech-grid stagger">
          {techs.map((t, i) => (
            <div className="tech-card reveal" key={i} style={{ '--tech-color': t.color }}>
              <div className="tech-dot" />
              <h4 className="tech-name">{t.name}</h4>
              <p className="tech-role">{t.role}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   DATASET SECTION
   ═══════════════════════════════════════════ */

function DatasetSection() {
  const stats = [
    { value: '600+', label: 'MRI Scans', icon: '🧠' },
    { value: '5', label: 'Clinical Features', icon: '📋' },
    { value: '3', label: 'HD Stages', icon: '📊' },
    { value: '2', label: 'Forecast Horizons', icon: '📈' },
  ];

  return (
    <section className="section" id="dataset">
      <div className="section-inner">
        <div className="section-label reveal">Data Foundation</div>
        <h2 className="section-title reveal">Dataset Information</h2>
        <p className="section-subtitle reveal">
          NeuroSense is trained on multi-modal data combining structural brain MRI with
          longitudinal clinical assessments for comprehensive HD characterization.
        </p>
        <div className="dataset-stats stagger">
          {stats.map((s, i) => (
            <div className="dataset-stat reveal" key={i}>
              <div className="dataset-stat-icon">{s.icon}</div>
              <div className="dataset-stat-value">{s.value}</div>
              <div className="dataset-stat-label">{s.label}</div>
            </div>
          ))}
        </div>
        <div className="dataset-details reveal">
          <div className="dataset-detail-card">
            <h4>📥 Input Modalities</h4>
            <ul>
              <li>T1-weighted structural MRI (NIfTI format)</li>
              <li>CAG repeat count (genetic biomarker)</li>
              <li>UHDRS Motor & Cognitive scores</li>
              <li>Total Functional Capacity (TFC)</li>
              <li>Patient age at assessment</li>
            </ul>
          </div>
          <div className="dataset-detail-card">
            <h4>📤 Output Predictions</h4>
            <ul>
              <li>HD stage classification (Pre-manifest / Early / Advanced)</li>
              <li>Confidence score with probability distribution</li>
              <li>12-month & 24-month progression forecast</li>
              <li>GradCAM++ brain region heatmaps</li>
              <li>SHAP feature importance attribution</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   AI WORKFLOW VISUALIZATION
   ═══════════════════════════════════════════ */

function AIWorkflowSection() {
  const nodes = [
    { id: 'mri', label: 'MRI Input', sub: '128³ voxels', icon: '🧠', row: 0 },
    { id: 'resnet', label: '3D ResNet-50', sub: 'Feature extraction', icon: '🔲', row: 0 },
    { id: 'fusion', label: 'Cross-Attention', sub: 'Multi-modal fusion', icon: '🔗', row: 0 },
    { id: 'output', label: 'Predictions', sub: 'Stage + Forecast', icon: '📊', row: 0 },
    { id: 'clinical', label: 'Clinical Data', sub: '5 biomarkers', icon: '📋', row: 1 },
    { id: 'bilstm', label: 'Bi-LSTM', sub: 'Temporal encoding', icon: '⚡', row: 1 },
  ];

  return (
    <section className="section" id="ai-workflow">
      <div className="section-inner">
        <div className="section-label reveal">Architecture</div>
        <h2 className="section-title reveal">AI Workflow</h2>
        <p className="section-subtitle reveal">
          Dual-stream architecture with cross-modal attention fusion for robust HD prediction.
        </p>
        <div className="workflow-diagram reveal">
          <div className="workflow-row">
            {nodes.filter(n => n.row === 0).map((node, i) => (
              <Fragment key={node.id}>
                {i > 0 && <div className="workflow-arrow">→</div>}
                <div className="workflow-node">
                  <div className="workflow-node-icon">{node.icon}</div>
                  <div className="workflow-node-label">{node.label}</div>
                  <div className="workflow-node-sub">{node.sub}</div>
                </div>
              </Fragment>
            ))}
          </div>
          <div className="workflow-row workflow-row-bottom">
            {nodes.filter(n => n.row === 1).map((node, i) => (
              <Fragment key={node.id}>
                {i > 0 && <div className="workflow-arrow">→</div>}
                <div className="workflow-node">
                  <div className="workflow-node-icon">{node.icon}</div>
                  <div className="workflow-node-label">{node.label}</div>
                  <div className="workflow-node-sub">{node.sub}</div>
                </div>
              </Fragment>
            ))}
            <div className="workflow-arrow workflow-arrow-up">↑</div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   FAQ SECTION
   ═══════════════════════════════════════════ */

function FAQSection() {
  const [openIndex, setOpenIndex] = useState(null);

  const faqs = [
    { q: 'How accurate is NeuroSense?', a: 'NeuroSense achieves high classification accuracy using multi-modal fusion of MRI and clinical data. However, it is a research tool and should not be used as a sole diagnostic instrument.' },
    { q: 'What MRI format is supported?', a: 'We accept NIfTI format files (.nii and .nii.gz). T1-weighted structural MRI scans are recommended for optimal results.' },
    { q: 'Is my data stored or shared?', a: 'No. All processing happens in-session. MRI data is processed server-side and not stored permanently. Clinical inputs stay in your browser.' },
    { q: 'Can this replace a clinical diagnosis?', a: 'No. NeuroSense is designed for academic research and screening purposes only. Always consult a qualified neurologist for clinical diagnosis.' },
    { q: 'What clinical scores are required?', a: 'The system requires CAG repeat count (36-120), UHDRS Motor Score (0-124), UHDRS Cognitive Score, Total Functional Capacity (0-13), and patient age (18-90).' },
    { q: 'How does the explainability work?', a: 'We use GradCAM++ to generate brain heatmaps showing which regions influenced the prediction, and SHAP values to quantify the contribution of each clinical feature.' },
    { q: 'What is the progression forecast?', a: 'The model predicts the likelihood of stage transition at 12-month and 24-month horizons based on current clinical trajectory and MRI features.' },
  ];

  return (
    <section className="section" id="faq">
      <div className="section-inner">
        <div className="section-label reveal">Common Questions</div>
        <h2 className="section-title reveal">Frequently Asked Questions</h2>
        <div className="faq-list">
          {faqs.map((faq, i) => (
            <div
              key={i}
              className={`faq-item reveal ${openIndex === i ? 'open' : ''}`}
              onClick={() => setOpenIndex(openIndex === i ? null : i)}
            >
              <div className="faq-question">
                <span>{faq.q}</span>
                <span className="faq-toggle">{openIndex === i ? '−' : '+'}</span>
              </div>
              <div className="faq-answer">
                <p>{faq.a}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   ABOUT / CONTACT SECTION
   ═══════════════════════════════════════════ */

function AboutProjectSection() {
  return (
    <section className="section" id="about-project">
      <div className="section-inner">
        <div className="section-label reveal">The Team</div>
        <h2 className="section-title reveal">About This Project</h2>
        <div className="about-project-grid">
          <div className="about-project-card reveal">
            <div className="about-project-icon">🎓</div>
            <h3>Academic Project</h3>
            <p>
              NeuroSense is a major project developed as part of the B.E. Computer Science
              curriculum at SJC Institute of Technology, Chickballapur. It demonstrates the
              application of deep learning in clinical neuroscience.
            </p>
          </div>
          <div className="about-project-card reveal">
            <div className="about-project-icon">🎯</div>
            <h3>Project Motivation</h3>
            <p>
              Early detection of Huntington's Disease can enable proactive interventions
              before irreversible damage occurs. This project explores how multi-modal AI
              can assist clinicians in identifying at-risk individuals.
            </p>
          </div>
          <div className="about-project-card reveal">
            <div className="about-project-icon">👤</div>
            <h3>Developer</h3>
            <p>
              <strong>Suraj S</strong> — SJC Institute of Technology<br />
              Dept. of Computer Science & Engineering<br />
              Specialization: AI/ML in Healthcare
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   ANALYSIS HISTORY HELPERS
   ═══════════════════════════════════════════ */

function saveAnalysis(form, result) {
  try {
    const history = JSON.parse(localStorage.getItem('neurosense-history') || '[]');
    history.unshift({
      id: Date.now(),
      date: new Date().toISOString(),
      form,
      result,
    });
    // Keep last 50
    localStorage.setItem('neurosense-history', JSON.stringify(history.slice(0, 50)));
  } catch (e) { /* storage full */ }
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem('neurosense-history') || '[]');
  } catch { return []; }
}

/* ═══════════════════════════════════════════
   PDF REPORT DOWNLOAD
   ═══════════════════════════════════════════ */

function downloadReport(form, result) {
  const timestamp = new Date().toLocaleString();
  const stage = STAGE_CONFIG[result.prediction]?.label || result.prediction;
  const html = `
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <title>NeuroSense Report — ${timestamp}</title>
    <style>
      body { font-family: 'Inter', Arial, sans-serif; max-width: 700px; margin: 40px auto; color: #1a1a2e; line-height: 1.8; padding: 0 20px; }
      h1 { font-size: 1.5rem; border-bottom: 2px solid #e07a5f; padding-bottom: 10px; }
      h2 { font-size: 1.1rem; margin-top: 28px; color: #e07a5f; }
      table { width: 100%; border-collapse: collapse; margin: 14px 0; }
      th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.9rem; }
      th { background: #faf8f5; font-weight: 600; width: 40%; }
      .badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
      .disclaimer { margin-top: 40px; padding: 14px; background: #fff8f0; border: 1px solid #f0d0b0; border-radius: 8px; font-size: 0.8rem; color: #666; }
      .header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
      .logo { font-size: 1.5rem; }
      @media print { body { margin: 20px; } }
    </style></head><body>
    <div class="header"><span class="logo">🧬</span><h1>NeuroSense — Clinical Report</h1></div>
    <p style="color:#666;font-size:0.85rem;">Generated: ${timestamp}</p>
    <h2>Patient Data</h2>
    <table>
      <tr><th>Age</th><td>${form.age} years</td></tr>
      <tr><th>CAG Repeat Count</th><td>${form.cag_repeat}</td></tr>
      <tr><th>UHDRS Motor Score</th><td>${form.uhdrs_motor}</td></tr>
      <tr><th>UHDRS Cognitive Score</th><td>${form.uhdrs_cognitive}</td></tr>
      <tr><th>TFC Score</th><td>${form.tfc_score}</td></tr>
    </table>
    <h2>Prediction Results</h2>
    <table>
      <tr><th>HD Stage</th><td><span class="badge" style="background:${STAGE_CONFIG[result.prediction]?.color || '#999'}20;color:${STAGE_CONFIG[result.prediction]?.color || '#999'}">${stage}</span></td></tr>
      <tr><th>Confidence</th><td>${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A'}</td></tr>
      ${result.progression_12m ? `<tr><th>12-Month Forecast</th><td>${(result.progression_12m * 100).toFixed(1)}% progression risk</td></tr>` : ''}
      ${result.progression_24m ? `<tr><th>24-Month Forecast</th><td>${(result.progression_24m * 100).toFixed(1)}% progression risk</td></tr>` : ''}
    </table>
    ${result.shap_values ? `<h2>Feature Importance (SHAP)</h2><table>${Object.entries(result.shap_values).map(([k,v]) => `<tr><th>${FEATURE_LABELS[k] || k}</th><td>${typeof v === 'number' ? v.toFixed(4) : v}</td></tr>`).join('')}</table>` : ''}
    <div class="disclaimer"><strong>⚠️ Disclaimer:</strong> This report is generated by NeuroSense, an academic research tool. It is not a substitute for professional medical diagnosis. Consult a qualified neurologist for clinical evaluation.</div>
    </body></html>
  `;
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `NeuroSense_Report_${new Date().toISOString().slice(0, 10)}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ═══════════════════════════════════════════
   ANALYSIS HISTORY PAGE
   ═══════════════════════════════════════════ */

function AnalysisHistoryPage() {
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(null);

  useScrollReveal();

  useEffect(() => {
    window.scrollTo(0, 0);
    setHistory(getHistory());
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('neurosense-history');
    setHistory([]);
    setSelectedId(null);
  };

  const selected = history.find(h => h.id === selectedId);

  return (
    <ActiveSectionContext.Provider value="history">
      <Particles />
      <Navbar />
      <section className="page-hero" id="history-hero">
        <div className="page-hero-inner">
          <div className="hero-badge">📜 Analysis Records</div>
          <h1 className="hero-title">
            Analysis <span className="highlight">History</span>
          </h1>
          <p className="hero-description">
            View your past NeuroSense analyses. All data is stored locally in your browser.
          </p>
          <Link to="/" className="btn btn-secondary">← Back to Home</Link>
        </div>
      </section>

      <section className="section">
        <div className="section-inner">
          {history.length === 0 ? (
            <div className="history-empty reveal">
              <div className="history-empty-icon">📭</div>
              <h3>No analyses yet</h3>
              <p>Run an HD analysis to see results here.</p>
              <Link to="/" className="btn btn-primary" style={{ marginTop: '16px' }}>
                🔬 Run Analysis
              </Link>
            </div>
          ) : (
            <>
              <div className="history-header reveal">
                <span className="history-count">{history.length} analysis record{history.length !== 1 ? 's' : ''}</span>
                <button className="btn-history-clear" onClick={clearHistory}>🗑️ Clear All</button>
              </div>
              <div className="history-grid">
                {history.map((item) => {
                  const stage = STAGE_CONFIG[item.result?.prediction];
                  return (
                    <div
                      key={item.id}
                      className={`history-card reveal ${selectedId === item.id ? 'selected' : ''}`}
                      onClick={() => setSelectedId(selectedId === item.id ? null : item.id)}
                    >
                      <div className="history-card-top">
                        <span className="history-date">
                          {new Date(item.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                        </span>
                        <span className="history-stage-badge" style={{ background: stage?.color ? `${stage.color}20` : '#eee', color: stage?.color || '#999' }}>
                          {stage?.label || item.result?.prediction || '?'}
                        </span>
                      </div>
                      <div className="history-card-details">
                        <span>Age: {item.form?.age}</span>
                        <span>CAG: {item.form?.cag_repeat}</span>
                        <span>Confidence: {item.result?.confidence ? (item.result.confidence * 100).toFixed(0) + '%' : 'N/A'}</span>
                      </div>
                      {selectedId === item.id && (
                        <div className="history-card-expanded">
                          <div className="history-detail-row"><span>UHDRS Motor:</span><span>{item.form?.uhdrs_motor}</span></div>
                          <div className="history-detail-row"><span>UHDRS Cognitive:</span><span>{item.form?.uhdrs_cognitive}</span></div>
                          <div className="history-detail-row"><span>TFC Score:</span><span>{item.form?.tfc_score}</span></div>
                          {item.result?.progression_12m && <div className="history-detail-row"><span>12m Forecast:</span><span>{(item.result.progression_12m * 100).toFixed(1)}%</span></div>}
                          <button className="btn-download-sm" onClick={(e) => { e.stopPropagation(); downloadReport(item.form, item.result); }}>
                            📄 Download Report
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </section>
      <Footer />
    </ActiveSectionContext.Provider>
  );
}

/* ═══════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════ */


function Footer() {
  const navigate = useNavigate();
  const currentYear = new Date().getFullYear();

  const scrollTo = (id) => {
    navigate('/');
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  return (
    <footer className="footer" role="contentinfo">
      <div className="footer-inner">
        {/* Brand column */}
        <div className="footer-col footer-col-brand">
          <div className="footer-brand-block">
            <div className="footer-logo">🧬</div>
            <span className="footer-brand-name">NeuroSense</span>
          </div>
          <p className="footer-brand-desc">
            AI-powered early detection and progression forecasting for Huntington's Disease.
            Combining 3D brain MRI analysis with clinical biomarkers.
          </p>
          <div className="footer-social">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-social-link"
              aria-label="GitHub"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
            </a>
            <a
              href="https://linkedin.com"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-social-link"
              aria-label="LinkedIn"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
            </a>
            <a
              href="mailto:suraj@example.com"
              className="footer-social-link"
              aria-label="Email"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
            </a>
          </div>
        </div>

        {/* Quick Links column */}
        <div className="footer-col">
          <h4 className="footer-col-title">Quick Links</h4>
          <ul className="footer-link-list">
            <li><a href="#hero" onClick={(e) => { e.preventDefault(); scrollTo('hero'); }}>Home</a></li>
            <li><a href="#about" onClick={(e) => { e.preventDefault(); scrollTo('about'); }}>About HD</a></li>
            <li><a href="#how-it-works" onClick={(e) => { e.preventDefault(); scrollTo('how-it-works'); }}>How It Works</a></li>
            <li><a href="#analysis" onClick={(e) => { e.preventDefault(); scrollTo('analysis'); }}>HD Analysis</a></li>
            <li><Link to="/memory-test">Memory Test</Link></li>
          </ul>
        </div>

        {/* Technologies column */}
        <div className="footer-col">
          <h4 className="footer-col-title">Technologies</h4>
          <ul className="footer-link-list footer-tech-list">
            <li><span className="footer-tech-dot pytorch" />PyTorch</li>
            <li><span className="footer-tech-dot monai" />MONAI</li>
            <li><span className="footer-tech-dot fastapi" />FastAPI</li>
            <li><span className="footer-tech-dot react" />React</li>
            <li><span className="footer-tech-dot vite" />Vite</li>
          </ul>
        </div>

        {/* Contact column */}
        <div className="footer-col">
          <h4 className="footer-col-title">Contact</h4>
          <ul className="footer-link-list footer-contact-list">
            <li>
              <span className="footer-contact-icon">👤</span>
              Suraj S
            </li>
            <li>
              <span className="footer-contact-icon">🏫</span>
              SJC Institute of Technology
            </li>
            <li>
              <a href="mailto:suraj@example.com">
                <span className="footer-contact-icon">✉️</span>
                suraj@example.com
              </a>
            </li>
          </ul>
        </div>
      </div>

      {/* Disclaimer bar */}
      <div className="footer-disclaimer-bar">
        <p className="footer-disclaimer">
          <strong>⚠️ Research Use Only</strong> — NeuroSense predictions are for academic research.
          AI outputs are not a substitute for professional medical diagnosis.
        </p>
      </div>

      {/* Bottom bar */}
      <div className="footer-bottom">
        <p className="footer-copy">
          © 2025–{currentYear} Suraj · SJC Institute of Technology
        </p>
        <p className="footer-built">
          Built with ❤️ using PyTorch, MONAI, FastAPI & React
        </p>
      </div>
    </footer>
  );
}

/* ═══════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════ */

/* ═══════════════════════════════════════════
   MEMORY TEST PAGE (standalone)
   ═══════════════════════════════════════════ */

function MemoryTestPage() {
  useScrollReveal();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <ActiveSectionContext.Provider value="memory-test">
      <Particles />
      <Navbar />
      <section className="page-hero" id="memory-test-hero">
        <div className="page-hero-inner">
          <div className="hero-badge">
            🧠 Cognitive Assessment Module
          </div>
          <h1 className="hero-title">
            Memory{' '}
            <span className="highlight">Assessment Battery</span>
          </h1>
          <p className="hero-description">
            A digital cognitive screening battery targeting memory domains commonly
            affected in early Huntington's Disease. Complete all three tests for a
            comprehensive assessment of verbal, visuospatial, and visual memory.
          </p>
          <Link to="/" className="btn btn-secondary">
            ← Back to Home
          </Link>
        </div>
      </section>
      <MemoryTestSection />
      <Footer />
    </ActiveSectionContext.Provider>
  );
}

/* ═══════════════════════════════════════════
   HOME PAGE
   ═══════════════════════════════════════════ */

function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [lastForm, setLastForm] = useState(null);

  const sectionIds = useMemo(() => ['hero', 'about', 'how-it-works', 'methodology', 'technologies', 'dataset', 'ai-workflow', 'analysis', 'faq', 'about-project'], []);
  const activeSection = useActiveSection(sectionIds);

  useScrollReveal();

  const handleSubmit = async (form, mriFile) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setLastForm(form);

    try {
      const fd = new FormData();
      fd.append('cag_repeat', form.cag_repeat);
      fd.append('uhdrs_motor', form.uhdrs_motor);
      fd.append('uhdrs_cognitive', form.uhdrs_cognitive);
      fd.append('tfc_score', form.tfc_score);
      fd.append('age', form.age);
      if (mriFile) fd.append('mri_file', mriFile);

      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.message || err.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      saveAnalysis(form, data);
    } catch (err) {
      setError(err.message || 'Failed to connect to NeuroSense API');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ActiveSectionContext.Provider value={activeSection}>
      <Particles />
      <Navbar />
      <HeroSection />
      <AboutSection />
      <HowItWorksSection />
      <MethodologySection />
      <TechnologiesSection />
      <DatasetSection />
      <AIWorkflowSection />
      <AnalysisSection onSubmit={handleSubmit} isLoading={isLoading} result={result} error={error} lastForm={lastForm} />
      <FAQSection />
      <AboutProjectSection />
      <Footer />
    </ActiveSectionContext.Provider>
  );
}

/* ═══════════════════════════════════════════
   APP ROOT (Router + Theme)
   ═══════════════════════════════════════════ */

export default function App() {
  const theme = useTheme();

  return (
    <ThemeContext.Provider value={theme}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/memory-test" element={<MemoryTestPage />} />
        <Route path="/history" element={<AnalysisHistoryPage />} />
      </Routes>
    </ThemeContext.Provider>
  );
}
