import { useState, useEffect, useCallback, useRef, useMemo, createContext, useContext, Fragment } from 'react';
import { Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom';
import { LanguageProvider, useLanguage, LANGUAGES } from './LanguageContext';
import BrainViewer3D from './BrainViewer3D';
import './App.css';

const API_BASE = 'http://localhost:8000';

const STAGE_CONFIG = {
  pre_manifest: { label: 'Normal / Pre-manifest (Unaffected)', color: 'var(--stage-pre)' },
  early: { label: 'Early HD', color: 'var(--stage-early)' },
  advanced: { label: 'Advanced HD', color: 'var(--stage-advanced)' },
  // DB-stored human-readable labels (from MongoDB predictions)
  "Pre-manifest HD": { label: 'Normal / Pre-manifest (Unaffected)', color: 'var(--stage-pre)' },
  "Early Huntington's": { label: 'Early HD', color: 'var(--stage-early)' },
  "Advanced Huntington's": { label: 'Advanced HD', color: 'var(--stage-advanced)' },
};

const FEATURE_LABELS = {
  cag_repeat: 'CAG Repeat',
  motor_score: 'Motor Performance',
  memory_score: 'Memory & Cognitive',
  age: 'Age',
  symptoms: 'Clinical Symptoms',
};

/* ═══════════════════════════════════════════
   CLINICAL MEDICATIONS & PHARMACOLOGY DATA
   ═══════════════════════════════════════════ */

const MEDICATIONS_DATA = [
  // ─── Normal / Pre-manifest Stage ───
  {
    id: 'coq10',
    name: 'Coenzyme Q10 (Ubiquinone / Idebenone)',
    genericName: 'Coenzyme Q10',
    brandName: 'CoQ10 / Idebenone',
    stage: 'pre_manifest',
    stageLabel: 'Normal / Pre-manifest',
    category: 'neuroprotection',
    categoryLabel: 'Neuroprotection',
    icon: '⚡',
    standardDose: '300 mg – 600 mg PO daily',
    startingDose: '300 mg PO once daily with morning meal',
    titration: 'Increase to 600 mg daily after 4 weeks based on GI tolerance; divide BID (300 mg AM / 300 mg PM) if >300 mg.',
    timings: [
      { time: '8:00 AM', slot: 'Morning', dose: '300 mg', instruction: 'Take with breakfast (fat-containing meal for absorption)' },
    ],
    timingSummary: '🌅 Morning (8:00 AM) with breakfast',
    administration: 'Oral softgel/capsule. Take with dietary fats (e.g., avocado, eggs, nuts) to maximize bioavailability.',
    mechanism: 'Mitochondrial electron transport chain cofactor; scavenges reactive oxygen species (ROS) and buffers striatal bioenergetic decline.',
    primaryObjective: 'Mitochondrial antioxidant support and slowing metabolic neuronal decay in pre-symptomatic gene carriers.',
    sideEffects: ['Mild gastrointestinal upset', 'Nausea', 'Transient insomnia (if taken late)'],
    contraindications: ['Hypersensitivity to CoQ10', 'Concurrent high-dose warfarin (monitor INR)'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Available in chewable wafers or liquid oral suspension.',
  },
  {
    id: 'creatine',
    name: 'Creatine Monohydrate',
    genericName: 'Creatine Monohydrate',
    brandName: 'Micronized Creatine',
    stage: 'pre_manifest',
    stageLabel: 'Normal / Pre-manifest',
    category: 'neuroprotection',
    categoryLabel: 'Neuroprotection',
    icon: '🧬',
    standardDose: '5 g – 10 g PO daily',
    startingDose: '5 g PO once daily dissolved in 250ml water',
    titration: 'No aggressive loading phase needed in HD; maintain continuous 5g/day or titrate to 10g/day (divided 5g BID) after 2 months.',
    timings: [
      { time: '8:00 AM', slot: 'Morning', dose: '5 g', instruction: 'Dissolve powder in water or juice; take with morning routine' },
    ],
    timingSummary: '🌅 Morning (8:00 AM) with plenty of hydration',
    administration: 'Oral powder. Dissolve thoroughly in room-temperature fluid. Ensure minimum 2L daily water intake.',
    mechanism: 'Phosphocreatine energy buffer maintaining intracellular ATP homeostasis and protecting medium spiny neurons against excitotoxicity.',
    primaryObjective: 'Preserving striatal bioenergetic reserves and muscular energy metabolism.',
    sideEffects: ['Mild water retention', 'Stomach cramps if under-hydrated', 'Transient elevation in serum creatinine (non-toxic)'],
    contraindications: ['Pre-existing severe renal insufficiency (eGFR < 30 mL/min)'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Completely soluble powder; easily mixes into thickened liquids, smoothies, or soft purees.',
  },
  {
    id: 'melatonin-pre',
    name: 'Melatonin (Circadian Synchronizer)',
    genericName: 'Melatonin',
    brandName: 'Melatonin / Circadin',
    stage: 'pre_manifest',
    stageLabel: 'Normal / Pre-manifest',
    category: 'sleep',
    categoryLabel: 'Sleep Support',
    icon: '🌙',
    standardDose: '3 mg – 5 mg PO at bedtime',
    startingDose: '3 mg PO 30–60 minutes prior to planned sleep',
    titration: 'Titrate to 5 mg if sleep fragmentation persists. Use sustained-release formulation for middle-of-the-night awakenings.',
    timings: [
      { time: '9:30 PM', slot: 'Bedtime', dose: '3 mg', instruction: 'Take 30–45 min before sleep; minimize screen exposure' },
    ],
    timingSummary: '🌙 Bedtime (9:30 PM) 30 min before sleep',
    administration: 'Oral tablet. Swallow with water; maintain consistent darkness and bedtime routine.',
    mechanism: 'MT1/MT2 receptor agonist restoring suprachiasmatic nucleus circadian rhythms and offering free-radical neuroprotection.',
    primaryObjective: 'Early stabilization of circadian rhythm disturbances characteristic of preclinical Huntington\'s Disease.',
    sideEffects: ['Morning grogginess', 'Vivid dreams', 'Mild headache'],
    contraindications: ['Autoimmune disorders (relative)', 'Concurrent heavy sedative use without monitoring'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Available in fast-dissolving sublingual drops or oral liquid.',
  },
  {
    id: 'omega3',
    name: 'High-EPA Omega-3 Fatty Acids (Ethyl-EPA)',
    genericName: 'Eicosapentaenoic Acid / DHA',
    brandName: 'Vascepa / Pure EPA',
    stage: 'pre_manifest',
    stageLabel: 'Normal / Pre-manifest',
    category: 'neuroprotection',
    categoryLabel: 'Neuroprotection',
    icon: '🌿',
    standardDose: '1,000 mg – 2,000 mg PO daily',
    startingDose: '1,000 mg PO daily with dinner',
    titration: 'Increase to 2,000 mg/day after 2 weeks if tolerated well.',
    timings: [
      { time: '7:00 PM', slot: 'Evening', dose: '1,000 mg', instruction: 'Take with dinner' },
    ],
    timingSummary: '🌆 Evening (7:00 PM) with dinner',
    administration: 'Oral capsules. Ingest whole with food.',
    mechanism: 'Modulates membrane lipid bilayer fluidity, suppresses neuroinflammatory cytokine cascades, and supports cerebral perfusion.',
    primaryObjective: 'Neurovascular membrane stabilization and neuroinflammatory attenuation.',
    sideEffects: ['Fishy aftertaste', 'Mild dyspepsia', 'Minor anti-platelet effect at high doses'],
    contraindications: ['Fish/shellfish severe allergy', 'Active bleeding diathesis'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Liquid emulsion or softgel puncture onto food.',
  },

  // ─── Early HD Stage ───
  {
    id: 'deutetrabenazine',
    name: 'Deutetrabenazine (Austedo)',
    genericName: 'Deutetrabenazine',
    brandName: 'Austedo / Austedo XR',
    stage: 'early',
    stageLabel: 'Early HD Stage',
    category: 'chorea',
    categoryLabel: 'Chorea / Motor Control',
    icon: '⚡',
    standardDose: '12 mg – 48 mg / day (Divided BID)',
    startingDose: '6 mg PO once daily with morning meal (Week 1)',
    titration: 'Titrate weekly by 6 mg/day increments (e.g. Wk 1: 6mg AM; Wk 2: 6mg BID; Wk 3: 9mg BID; up to max 48mg/day) based on chorea reduction and tolerability.',
    timings: [
      { time: '8:00 AM', slot: 'Morning', dose: '12 mg', instruction: 'Take with breakfast (with food)' },
      { time: '7:30 PM', slot: 'Evening', dose: '12 mg', instruction: 'Take with evening meal' },
    ],
    timingSummary: '🌅 Morning (8:00 AM) & 🌆 Evening (7:30 PM) with meals',
    administration: 'Administer with food. Swallow tablets whole; do not chew, crush, or divide extended-release tablets.',
    mechanism: 'Reversible vesicular monoamine transporter 2 (VMAT2) inhibitor; deuteration attenuates CYP2D6 metabolism for smooth, prolonged dopamine depletion without sharp Cmax peaks.',
    primaryObjective: 'First-line FDA-approved chorea suppression with reduced peak-to-trough fluctuations and lower somnolence risk.',
    sideEffects: ['Somnolence / fatigue', 'Diarrhea', 'Dry mouth', 'Insomnia'],
    contraindications: ['Boxed Warning: Suicidal ideation & untreated depression', 'Hepatic impairment', 'Concurrent MAOIs or reserpine', 'CYP2D6 poor metabolizers require lower max dose (36 mg/day)'],
    dysphagiaSafe: false,
    dysphagiaNotes: 'Cannot be crushed. For dysphagic patients, consider transitioning to Tetrabenazine liquid or neuroleptics.',
  },
  {
    id: 'tetrabenazine',
    name: 'Tetrabenazine (Xenazine)',
    genericName: 'Tetrabenazine',
    brandName: 'Xenazine',
    stage: 'early',
    stageLabel: 'Early HD Stage',
    category: 'chorea',
    categoryLabel: 'Chorea / Motor Control',
    icon: '💊',
    standardDose: '25 mg – 75 mg / day (Divided TID)',
    startingDose: '12.5 mg PO once daily in morning (Week 1)',
    titration: 'After 1 week, increase to 12.5 mg BID. Titrate by 12.5 mg weekly to TID dosing as needed; max 50 mg/day (CYP2D6 poor) or 100 mg/day (extensive).',
    timings: [
      { time: '8:00 AM', slot: 'Morning', dose: '25 mg', instruction: 'Take with breakfast' },
      { time: '1:00 PM', slot: 'Midday', dose: '12.5 mg', instruction: 'Take with lunch' },
      { time: '6:30 PM', slot: 'Evening', dose: '25 mg', instruction: 'Take with dinner' },
    ],
    timingSummary: '🌅 8:00 AM, ☀️ 1:00 PM, 🌆 6:30 PM with meals',
    administration: 'Oral tablet. Can be taken with or without food. Maintain regular dosing intervals.',
    mechanism: 'Potent reversible VMAT2 inhibitor causing presynaptic depletion of dopamine, serotonin, and norepinephrine.',
    primaryObjective: 'Rapid and effective chorea suppression in moderate-to-severe involuntary movements.',
    sideEffects: ['Sedation / somnolence', 'Depression / dysphoria', 'Parkinsonism / akathisia', 'Nausea'],
    contraindications: ['Boxed Warning: Depression & Suicidality in HD', 'Concurrent MAOIs (within 14 days)', 'Severe hepatic impairment', 'QTc prolongation (>500 ms)'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Tablets may be crushed and mixed with soft food if immediate-release formulation is utilized.',
  },
  {
    id: 'sertraline',
    name: 'Sertraline (Zoloft)',
    genericName: 'Sertraline Hydrochloride',
    brandName: 'Zoloft',
    stage: 'early',
    stageLabel: 'Early HD Stage',
    category: 'mood',
    categoryLabel: 'Mood & Psychiatric',
    icon: '🧠',
    standardDose: '50 mg – 150 mg PO daily',
    startingDose: '25 mg – 50 mg PO once daily in morning',
    titration: 'Increase by 25–50 mg increments every 2–4 weeks based on clinical depression, irritability, and anxiety response (max 200 mg/day).',
    timings: [
      { time: '8:00 AM', slot: 'Morning', dose: '50 mg', instruction: 'Take with morning meal' },
    ],
    timingSummary: '🌅 Morning (8:00 AM) with breakfast',
    administration: 'Oral tablet or oral solution. Take in morning to prevent nighttime insomnia.',
    mechanism: 'Selective serotonin reuptake inhibitor (SSRI); elevates synaptic 5-HT availability in frontostriatal circuits.',
    primaryObjective: 'First-line management of HD-associated depression, irritability, affective blunting, and obsessive behaviors.',
    sideEffects: ['Nausea (initial 1-2 weeks)', 'Tremor', 'Sexual dysfunction', 'Initial agitation'],
    contraindications: ['Concurrent MAOIs or Pimozide', 'Serotonin syndrome risk with other serotonergic agents'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Available as oral concentrate (20 mg/mL) which must be diluted in 4 oz of water, ginger ale, or orange juice.',
  },
  {
    id: 'trazodone',
    name: 'Trazodone (Desyrel)',
    genericName: 'Trazodone Hydrochloride',
    brandName: 'Desyrel / Oleptro',
    stage: 'early',
    stageLabel: 'Early HD Stage',
    category: 'sleep',
    categoryLabel: 'Sleep Support',
    icon: '🌙',
    standardDose: '50 mg – 100 mg PO at bedtime',
    startingDose: '25 mg – 50 mg PO at bedtime',
    titration: 'Increase to 100 mg at bedtime after 1 week if nocturnal chorea and sleep fragmentation persist.',
    timings: [
      { time: '9:30 PM', slot: 'Bedtime', dose: '50 mg', instruction: 'Take 30 min before bedtime with a small snack' },
    ],
    timingSummary: '🌙 Bedtime (9:30 PM) 30 min before sleep',
    administration: 'Oral tablet. Take shortly after a light snack or with water before sleep.',
    mechanism: 'Serotonin 5-HT2A antagonist and reuptake inhibitor (SARI) with H1-histaminergic blockade inducing restorative slow-wave sleep.',
    primaryObjective: 'Treating sleep maintenance insomnia, nighttime agitation, and nocturnal motor unrest without worsening chorea.',
    sideEffects: ['Orthostatic hypotension', 'Dry mouth', 'Morning grogginess', 'Rare priapism'],
    contraindications: ['Concurrent MAOIs', 'Severe cardiac arrhythmias / recovery phase of MI'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Scored tablets can be cut in half or crushed with applesauce.',
  },

  // ─── Advanced HD Stage ───
  {
    id: 'olanzapine',
    name: 'Olanzapine (Zyprexa / Zydis ODT)',
    genericName: 'Olanzapine',
    brandName: 'Zyprexa / Zydis ODT',
    stage: 'advanced',
    stageLabel: 'Advanced HD Stage',
    category: 'chorea',
    categoryLabel: 'Severe Chorea & Psychosis',
    icon: '🚨',
    standardDose: '5 mg – 15 mg PO daily (Divided or Bedtime)',
    startingDose: '2.5 mg – 5 mg PO at bedtime',
    titration: 'Titrate by 2.5 mg every 1–2 weeks based on severe choreic surges, psychosis, aggression, and appetite stimulation goals (max 20 mg/day).',
    timings: [
      { time: '8:00 AM', slot: 'Morning', dose: '2.5 mg', instruction: 'Take with morning meal' },
      { time: '8:30 PM', slot: 'Bedtime', dose: '5 mg', instruction: 'Take with evening routine' },
    ],
    timingSummary: '🌅 Morning (2.5 mg) & 🌙 Bedtime (5 mg)',
    administration: 'Oral tablet or orally disintegrating tablet (ODT). ODT melts immediately on tongue with saliva; no water required.',
    mechanism: 'Atypical second-generation antipsychotic; potent D2, 5-HT2A, and 5-HT2C receptor antagonist.',
    primaryObjective: 'Dual suppression of refractory chorea and severe neuropsychiatric symptoms (psychosis, severe agitation, weight loss/cachexia).',
    sideEffects: ['Weight gain / metabolic shift', 'Sedation', 'Anticholinergic effects', 'Orthostasis'],
    contraindications: ['Dementia-related psychosis black box warning', 'Severe neutropenia / agranulocytosis risk with clozapine (safe alternative)', 'Severe cardiac decompensation'],
    dysphagiaSafe: true,
    dysphagiaNotes: '⭐ Preferred Zydis ODT formulation: instantly dissolves on tongue, ideal for advanced stage dysphagia and aspiration prevention.',
  },
  {
    id: 'baclofen',
    name: 'Baclofen (Lioresal)',
    genericName: 'Baclofen',
    brandName: 'Lioresal',
    stage: 'advanced',
    stageLabel: 'Advanced HD Stage',
    category: 'rigidity',
    categoryLabel: 'Rigidity & Spasticity',
    icon: '🩺',
    standardDose: '30 mg – 60 mg / day (Divided TID)',
    startingDose: '5 mg PO TID with meals',
    titration: 'Titrate gradually every 3 days by 5 mg/dose (e.g. 10 mg TID -> 15 mg TID -> 20 mg TID) to relieve muscle stiffness and dystonia while preserving trunk stability.',
    timings: [
      { time: '8:00 AM', slot: 'Morning', dose: '10 mg', instruction: 'Take with breakfast' },
      { time: '1:30 PM', slot: 'Midday', dose: '10 mg', instruction: 'Take with lunch' },
      { time: '9:00 PM', slot: 'Bedtime', dose: '15 mg', instruction: 'Take at bedtime' },
    ],
    timingSummary: '🌅 8:00 AM, ☀️ 1:30 PM, 🌙 9:00 PM with meals',
    administration: 'Oral tablet or oral solution. Must be taken consistently; NEVER discontinue abruptly due to risk of rebound spasticity, hallucinations, and seizures.',
    mechanism: 'GABAB receptor agonist at spinal interneuron levels; hyperpolarizes afferent motor terminals suppressing mono- and polysynaptic reflex arcs.',
    primaryObjective: 'Relieving painful limb dystonia, hypertonia, rigidity, and muscle contractures in advanced stage Westphal variant/late HD.',
    sideEffects: ['Drowsiness / sedation', 'Muscle weakness (hypotonia)', 'Dizziness', 'Confusion'],
    contraindications: ['Abrupt cessation warning', 'Severe renal impairment without dose adjustment'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Oral liquid solution (5 mg/5 mL) available; crushed tablets dissolve easily in thickened liquids or PEG feeding tubes.',
  },
  {
    id: 'quetiapine',
    name: 'Quetiapine (Seroquel)',
    genericName: 'Quetiapine Fumarate',
    brandName: 'Seroquel',
    stage: 'advanced',
    stageLabel: 'Advanced HD Stage',
    category: 'mood',
    categoryLabel: 'Mood & Night Agitation',
    icon: '🌙',
    standardDose: '25 mg – 150 mg PO at bedtime',
    startingDose: '25 mg PO at bedtime',
    titration: 'Increase by 25 mg every 3–5 days to control nocturnal delirium, paranoia, and sleep disruption (max 300 mg/day).',
    timings: [
      { time: '9:30 PM', slot: 'Bedtime', dose: '50 mg', instruction: 'Take 30–45 min before sleep' },
    ],
    timingSummary: '🌙 Bedtime (9:30 PM)',
    administration: 'Oral tablet. Immediate-release formulation preferred for bedtime sedation and rapid nighttime calming.',
    mechanism: 'D2 and 5-HT2A antagonist with fast dissociation from D2 receptors, minimizing extrapyramidal symptoms and worsening of parkinsonian features.',
    primaryObjective: 'Managing severe nocturnal agitation, sundowning, hallucinations, and emotional lability with lowest EPS risk.',
    sideEffects: ['Somnolence', 'Orthostatic hypotension', 'Dry mouth', 'Constipation'],
    contraindications: ['Severe hepatic failure', 'Concurrent strong CYP3A4 inhibitors (dose adjust)'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'Immediate-release tablets can be finely crushed and mixed with smooth applesauce or pudding.',
  },
  {
    id: 'clonazepam',
    name: 'Clonazepam (Klonopin)',
    genericName: 'Clonazepam',
    brandName: 'Klonopin / Rivotril',
    stage: 'advanced',
    stageLabel: 'Advanced HD Stage',
    category: 'rigidity',
    categoryLabel: 'Myoclonus & Severe Anxiety',
    icon: '🌿',
    standardDose: '0.5 mg – 2.0 mg / day (Divided BID-TID)',
    startingDose: '0.25 mg – 0.5 mg PO at bedtime',
    titration: 'Increase by 0.25 mg every 3 days. Divide BID (AM/Bedtime) to manage myoclonic jerks and acute panic surges.',
    timings: [
      { time: '8:30 AM', slot: 'Morning', dose: '0.5 mg', instruction: 'Take with morning routine' },
      { time: '9:30 PM', slot: 'Bedtime', dose: '0.5 mg', instruction: 'Take at bedtime' },
    ],
    timingSummary: '🌅 Morning (0.5 mg) & 🌙 Bedtime (0.5 mg)',
    administration: 'Oral tablet or orally disintegrating tablet (ODT). Swallow or allow to dissolve.',
    mechanism: 'High-potency benzodiazepine allosterically enhancing GABA-A receptor conductance, producing potent anticonvulsant, anxiolytic, and anti-myoclonic effects.',
    primaryObjective: 'Controlling myoclonic twitches, sudden choreic surges, extreme panic, and severe sleep disturbances.',
    sideEffects: ['Sedation / ataxia', 'Excess salivation / secretions (monitor swallowing)', 'Physical dependence with prolonged use'],
    contraindications: ['Severe respiratory depression', 'Acute narrow-angle glaucoma', 'Severe sleep apnea'],
    dysphagiaSafe: true,
    dysphagiaNotes: 'ODT formulation (wafer) dissolves in seconds on the tongue without fluid.',
  },
];

/* ═══════════════════════════════════════════
   PERSONALIZED AI PRESCRIPTION GENERATOR
   ═══════════════════════════════════════════ */

function generatePersonalizedPrescription(result, form) {
  if (!result) return null;
  const stage = result.stage || result.prediction || 'early';
  const confidence = result.confidence != null ? Math.round(result.confidence * 100) : 85;
  const risk = result.risk_category || 'medium';
  const motorScore = form?.motor_score != null ? form.motor_score : 70;
  const memoryScore = form?.memory_score != null ? form.memory_score : 75;
  const age = form?.age || 45;

  let primaryMeds = [];
  let secondaryMeds = [];
  let dailyTimeline = [];
  let titrationPlan = [];
  let stageTitle = '';
  let stageColor = 'var(--stage-early)';
  let clinicalNotes = '';

  if (stage === 'pre_manifest' || stage === 'Pre-manifest HD') {
    stageTitle = 'Normal / Pre-manifest (Gene Carrier / At-Risk)';
    stageColor = 'var(--stage-pre)';
    primaryMeds = [
      { ...MEDICATIONS_DATA.find(m => m.id === 'coq10'), customDose: '300 mg PO daily', targetReason: 'Striatal mitochondrial bioenergetic preservation' },
      { ...MEDICATIONS_DATA.find(m => m.id === 'creatine'), customDose: '5 g PO daily', targetReason: 'ATP buffer against excitotoxic neuronal stress' },
    ];
    secondaryMeds = [
      { ...MEDICATIONS_DATA.find(m => m.id === 'melatonin-pre'), customDose: '3 mg PO at bedtime', targetReason: 'Circadian rhythm synchronization' },
      { ...MEDICATIONS_DATA.find(m => m.id === 'omega3'), customDose: '1,000 mg PO daily', targetReason: 'Neurovascular membrane stabilization' },
    ];
    dailyTimeline = [
      { time: '8:00 AM', slot: 'Morning', pills: [{ name: 'Coenzyme Q10 (300 mg)', instruction: 'With breakfast' }, { name: 'Creatine Monohydrate (5 g)', instruction: 'Dissolved in 250ml water' }] },
      { time: '7:00 PM', slot: 'Evening', pills: [{ name: 'Omega-3 EPA/DHA (1,000 mg)', instruction: 'With dinner' }] },
      { time: '9:30 PM', slot: 'Bedtime', pills: [{ name: 'Melatonin (3 mg)', instruction: '30 min before sleep' }] },
    ];
    titrationPlan = [
      { phase: 'Month 1 (Initiation)', detail: 'Start CoQ10 300mg QAM + Creatine 5g/day. Establish baseline sleep hygiene.' },
      { phase: 'Month 2–3 (Optimization)', detail: 'If tolerated with no GI upset, maintain CoQ10 300-600mg daily. Add Melatonin 3mg if circadian fragmentation detected.' },
      { phase: 'Long-term Maintenance', detail: 'Annual motor & cognitive digital tracking; evaluation for active clinical trial enrollment (e.g. HTT-lowering therapeutics).' },
    ];
    clinicalNotes = `Patient is in the pre-symptomatic / pre-manifest phase (Confidence: ${confidence}%). Recommended protocol prioritizes mitochondrial protection, cellular bioenergetics, and lifestyle sleep stabilization without neuroleptic exposure.`;
  } else if (stage === 'advanced' || stage === 'Advanced Huntington\'s') {
    stageTitle = 'Advanced Huntington\'s Disease';
    stageColor = 'var(--stage-advanced)';
    primaryMeds = [
      { ...MEDICATIONS_DATA.find(m => m.id === 'olanzapine'), customDose: '2.5 mg AM / 5 mg Bedtime (Zydis ODT)', targetReason: 'Severe chorea suppression & psychosis management with dysphagia-safe ODT' },
      { ...MEDICATIONS_DATA.find(m => m.id === 'baclofen'), customDose: '10 mg PO TID (Oral Liquid / Crushed)', targetReason: 'Limb hypertonia, rigidity, and painful muscle contractures' },
    ];
    secondaryMeds = [
      { ...MEDICATIONS_DATA.find(m => m.id === 'quetiapine'), customDose: '50 mg PO at bedtime', targetReason: 'Nocturnal agitation & sundowning reduction' },
      { ...MEDICATIONS_DATA.find(m => m.id === 'clonazepam'), customDose: '0.5 mg PO BID (ODT)', targetReason: 'Myoclonic surges and acute panic stabilization' },
    ];
    dailyTimeline = [
      { time: '8:00 AM', slot: 'Morning', pills: [{ name: 'Olanzapine Zydis ODT (2.5 mg)', instruction: 'Dissolve on tongue with morning meal' }, { name: 'Baclofen Liquid (10 mg)', instruction: 'With breakfast' }, { name: 'Clonazepam ODT (0.5 mg)', instruction: 'Oral wafer' }] },
      { time: '1:30 PM', slot: 'Midday', pills: [{ name: 'Baclofen Liquid (10 mg)', instruction: 'With lunch' }] },
      { time: '8:30 PM', slot: 'Evening / Dinner', pills: [{ name: 'Baclofen Liquid (10 mg)', instruction: 'With dinner' }] },
      { time: '9:30 PM', slot: 'Bedtime', pills: [{ name: 'Olanzapine Zydis ODT (5 mg)', instruction: 'At bedtime' }, { name: 'Quetiapine (50 mg)', instruction: 'Crushed with puree' }] },
    ];
    titrationPlan = [
      { phase: 'Week 1 (Initiation)', detail: 'Start Olanzapine ODT 2.5mg Bedtime + Baclofen 5mg TID. Assess swallow safety & aspiration precautions.' },
      { phase: 'Week 2–3 (Dose Escalation)', detail: 'Increase Olanzapine to 2.5mg AM / 5mg Bedtime. Titrate Baclofen by 5mg increments to 10mg TID for rigidity control.' },
      { phase: 'Maintenance & Palliative Care', detail: 'Monitor swallowing mechanics continuously; use thickened liquids or ODT wafers; evaluate PEG tube compatibility.' },
    ];
    clinicalNotes = `Advanced stage presentation (Motor Performance: ${motorScore}%, Progression Risk: ${risk.toUpperCase()}). High priority on aspiration safety, ODT dissolving formulations, rigidity relief, and caregiver-assisted daily scheduling.`;
  } else {
    // Early Stage HD (Default)
    stageTitle = 'Early Huntington\'s Disease';
    stageColor = 'var(--stage-early)';
    primaryMeds = [
      { ...MEDICATIONS_DATA.find(m => m.id === 'deutetrabenazine'), customDose: '12 mg PO BID with meals', targetReason: 'Targeted chorea suppression with reduced dopamine fluctuation peaks' },
      { ...MEDICATIONS_DATA.find(m => m.id === 'sertraline'), customDose: '50 mg PO QAM with breakfast', targetReason: 'Depression, irritability, and executive affective stabilization' },
    ];
    secondaryMeds = [
      { ...MEDICATIONS_DATA.find(m => m.id === 'trazodone'), customDose: '50 mg PO at bedtime', targetReason: 'Nocturnal motor rest and sleep maintenance' },
      { ...MEDICATIONS_DATA.find(m => m.id === 'coq10'), customDose: '300 mg PO daily', targetReason: 'Adjunctive neuroprotection' },
    ];
    dailyTimeline = [
      { time: '8:00 AM', slot: 'Morning', pills: [{ name: 'Deutetrabenazine (12 mg)', instruction: 'Take with breakfast' }, { name: 'Sertraline (50 mg)', instruction: 'Take with morning meal' }] },
      { time: '7:30 PM', slot: 'Evening', pills: [{ name: 'Deutetrabenazine (12 mg)', instruction: 'Take with dinner' }] },
      { time: '9:30 PM', slot: 'Bedtime', pills: [{ name: 'Trazodone (50 mg)', instruction: '30 min before sleep' }] },
    ];
    titrationPlan = [
      { phase: 'Week 1 (Initiation)', detail: 'Deutetrabenazine 6mg PO daily with breakfast + Sertraline 25mg daily. Monitor mood and depression scores.' },
      { phase: 'Week 2–3 (Titration)', detail: 'Increase Deutetrabenazine to 6mg BID, then 12mg BID as tolerated for chorea control. Sertraline up to 50mg daily.' },
      { phase: 'Maintenance (Month 1+)', detail: 'Target maintenance at 24-36mg/day divided BID. Quarterly review of motor score and psychiatric symptoms.' },
    ];
    clinicalNotes = `Early HD presentation (Confidence: ${confidence}%, Motor Score: ${motorScore}%). Protocol targets involuntary chorea suppression with VMAT2 inhibition while supporting mood and cognitive resilience.`;
  }

  return {
    stage,
    stageTitle,
    stageColor,
    confidence,
    risk,
    motorScore,
    memoryScore,
    age,
    primaryMeds,
    secondaryMeds,
    dailyTimeline,
    titrationPlan,
    clinicalNotes,
  };
}

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
  const [langOpen, setLangOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === '/';
  const isMemoryTest = location.pathname === '/memory-test';
  const activeSection = useContext(ActiveSectionContext);
  const { isDark, toggle: toggleTheme } = useContext(ThemeContext);
  const { language, setLanguage, t } = useLanguage();
  const { user, logout } = useAuth();
  const langRef = useRef(null);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef(null);

  const currentLang = LANGUAGES.find((l) => l.code === language) || LANGUAGES[0];

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

  // Close language dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (langRef.current && !langRef.current.contains(e.target)) {
        setLangOpen(false);
      }
      if (userMenuRef.current && !userMenuRef.current.contains(e.target)) {
        setUserMenuOpen(false);
      }
    };
    if (langOpen || userMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [langOpen, userMenuOpen]);

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
    { id: 'about', label: t('nav.aboutHD'), section: true },
    { id: 'analysis', label: t('nav.analysis'), section: true },
  ];

  const handleLangSelect = (code) => {
    setLanguage(code);
    setLangOpen(false);
  };

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
                {t('nav.memoryTest')}
              </Link>
            </li>
            <li>
              <Link
                to="/motor-test"
                className={`navbar-link-page ${location.pathname === '/motor-test' ? 'active' : ''}`}
              >
                {t('nav.motorTest')}
              </Link>
            </li>
            <li>
              <Link
                to="/progression"
                className={`navbar-link-page ${location.pathname === '/progression' ? 'active' : ''}`}
              >
                {t('nav.progression')}
              </Link>
            </li>
            <li>
              <Link
                to="/medications"
                className={`navbar-link-page ${location.pathname === '/medications' ? 'active' : ''}`}
              >
                {t('nav.medications')}
              </Link>
            </li>
            <li>
              <Link
                to="/history"
                className={`navbar-link-page ${location.pathname === '/history' ? 'active' : ''}`}
              >
                {t('nav.history')}
              </Link>
            </li>
            <li>
              <div className={`lang-switcher ${langOpen ? 'open' : ''}`} ref={langRef}>
                <button
                  className="lang-btn"
                  onClick={() => setLangOpen(!langOpen)}
                  aria-label={t('nav.language')}
                  aria-expanded={langOpen}
                >
                  <span className="lang-btn-flag">{currentLang.flag}</span>
                  <span className="lang-btn-code">{currentLang.code}</span>
                  <span className="lang-btn-arrow">▼</span>
                </button>
                <div className="lang-dropdown">
                  {LANGUAGES.map((lang) => (
                    <button
                      key={lang.code}
                      className={`lang-option ${language === lang.code ? 'active' : ''}`}
                      onClick={() => handleLangSelect(lang.code)}
                    >
                      <span className="lang-option-flag">{lang.flag}</span>
                      <span className="lang-option-text">
                        <span className="lang-option-name">{lang.name}</span>
                        <span className="lang-option-native">{lang.native}</span>
                      </span>
                      <span className="lang-option-check">✓</span>
                    </button>
                  ))}
                </div>
              </div>
            </li>
            <li>
              <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle dark mode">
                {isDark ? '☀️' : '🌙'}
              </button>
            </li>
            <li>
              {user ? (
                <div className={`navbar-user-menu ${userMenuOpen ? 'open' : ''}`} ref={userMenuRef}>
                  <button className="navbar-user-btn" onClick={() => setUserMenuOpen(!userMenuOpen)}>
                    <div className="navbar-user-avatar">
                      {user.name ? user.name.charAt(0).toUpperCase() : '?'}
                    </div>
                    <span className="navbar-user-name">{user.name?.split(' ')[0]}</span>
                    <span className="navbar-user-arrow">▼</span>
                  </button>
                  <div className="navbar-user-dropdown">
                    <div className="navbar-user-info">
                      <div className="navbar-user-info-name">{user.name}</div>
                      <div className="navbar-user-info-email">{user.email}</div>
                    </div>
                    <div className="navbar-user-divider" />
                    <button className="navbar-user-option" onClick={() => { navigate('/history'); setUserMenuOpen(false); }}>
                      📊 Analysis History
                    </button>
                    <button className="navbar-user-option navbar-user-logout" onClick={() => { logout(); setUserMenuOpen(false); }}>
                      🚪 Sign Out
                    </button>
                  </div>
                </div>
              ) : (
                <Link to="/login" className="navbar-cta">
                  Sign In
                </Link>
              )}
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
              {t('nav.memoryTest')}
            </Link>
          </li>
          <li>
            <Link
              to="/motor-test"
              className={location.pathname === '/motor-test' ? 'active' : ''}
              onClick={() => setMobileOpen(false)}
            >
              {t('nav.motorTest')}
            </Link>
          </li>
          <li>
            <Link
              to="/progression"
              className={location.pathname === '/progression' ? 'active' : ''}
              onClick={() => setMobileOpen(false)}
            >
              {t('nav.progression')}
            </Link>
          </li>
          <li>
            <Link
              to="/medications"
              className={location.pathname === '/medications' ? 'active' : ''}
              onClick={() => setMobileOpen(false)}
            >
              {t('nav.medications')}
            </Link>
          </li>
          <li>
            <Link
              to="/history"
              className={location.pathname === '/history' ? 'active' : ''}
              onClick={() => setMobileOpen(false)}
            >
              {t('nav.history')}
            </Link>
          </li>
        </ul>

        {/* Mobile language selector */}
        <div className="mobile-lang-section">
          <div className="mobile-lang-label">{t('nav.language')}</div>
          <div className="mobile-lang-grid">
            {LANGUAGES.map((lang) => (
              <button
                key={lang.code}
                className={`mobile-lang-option ${language === lang.code ? 'active' : ''}`}
                onClick={() => { setLanguage(lang.code); }}
              >
                <span className="mobile-lang-option-flag">{lang.flag}</span>
                <span>{lang.native}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mobile-drawer-cta">
          {user ? (
            <>
              <div className="mobile-user-info">
                <div className="navbar-user-avatar">{user.name ? user.name.charAt(0).toUpperCase() : '?'}</div>
                <div>
                  <div className="mobile-user-name">{user.name}</div>
                  <div className="mobile-user-email">{user.email}</div>
                </div>
              </div>
              <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => { logout(); setMobileOpen(false); }}>
                🚪 Sign Out
              </button>
            </>
          ) : (
            <Link to="/login" className="btn btn-primary" onClick={() => setMobileOpen(false)}>
              🔐 Sign In / Register
            </Link>
          )}
        </div>
      </aside>
    </>
  );
}

/* ═══════════════════════════════════════════
   HERO SECTION
   ═══════════════════════════════════════════ */

function HeroSection() {
  const navigate = useNavigate();
  const { t } = useLanguage();

  const features = [
    t('hero.feat1'),
    t('hero.feat2'),
    t('hero.feat3'),
    t('hero.feat4'),
  ];

  const trustBadges = [
    { icon: '⚡', label: t('hero.aiPowered') },
    { icon: '📚', label: t('hero.researchBased') },
    { icon: '🏥', label: t('hero.clinicalWorkflow') },
  ];

  const heroSlides = [
    { src: '/hero-slide-1.png', alt: 'AI laboratory analyzing a glowing 3D brain with holographic dashboard', tag: t('slide.aiAnalysis'), title: t('slide.deepLearning'), time: t('slide.realtime'), link: '#analysis' },
    { 
      isFact: true, 
      tag: t('slide.genetics'), 
      icon: '🧬', 
      number: '50%', 
      label: t('slide.inheritanceRisk'), 
      description: t('slide.inheritanceDesc'),
      footerText: t('slide.geneticPenetrance'),
      link: '#about' 
    },
    { src: '/hero-slide-2.png', alt: 'Patient undergoing brain MRI scan with healthcare professionals monitoring', tag: t('slide.mriScan'), title: t('slide.advancedMRI'), time: t('slide.3dVolumetric'), link: '#analysis' },
    { 
      isFact: true, 
      tag: t('slide.prevalence'), 
      icon: '📊', 
      number: '13 / 100k', 
      label: t('slide.westernPop'), 
      description: t('slide.prevalenceDesc'),
      footerText: t('slide.worldwide'),
      link: '#about' 
    },
    { src: '/hero-slide-3.png', alt: 'Neurologist examining brain MRI scans on holographic displays', tag: t('slide.diagnostics'), title: t('slide.clinicalDecision'), time: t('slide.evidenceBased'), link: '#how-it-works' },
    { 
      isFact: true, 
      tag: t('slide.biomarkers'), 
      icon: '🔬', 
      number: '≥36 CAG', 
      label: t('slide.trinucleotide'), 
      description: t('slide.biomarkersDesc'),
      footerText: t('slide.higherCounts'),
      link: '#about' 
    },
    { src: '/hero-slide-4.png', alt: 'Doctor consulting patient about brain scan results', tag: t('slide.patientCare'), title: t('slide.actionableInsights'), time: t('slide.personalized'), link: '/memory-test' },
    { 
      isFact: true, 
      tag: t('slide.progression'), 
      icon: '⏳', 
      number: '15-20 Yrs', 
      label: t('slide.postOnset'), 
      description: t('slide.progressionDesc'),
      footerText: t('slide.earlyAI'),
      link: '#about' 
    },
    { src: '/hero-slide-5.png', alt: 'Disease progression visualization from healthy to Huntington\'s stages', tag: t('slide.progression'), title: t('slide.hdStage'), time: t('slide.3stageModel'), link: '#about' },
    { src: '/hero-slide-6.png', alt: 'AI healthcare platform with holographic brain model and analytics', tag: t('slide.platform'), title: t('slide.explainableDash'), time: t('slide.transparent'), link: '#how-it-works' },
  ];

  const handleExplore = (link) => {
    if (link.startsWith('#')) {
      const id = link.substring(1);
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    } else {
      navigate(link);
    }
  };

  const numSlides = heroSlides.length;
  const prefixCount = 2;
  const suffixCount = 2;
  const paddedSlides = [
    ...heroSlides.slice(-prefixCount),
    ...heroSlides,
    ...heroSlides.slice(0, suffixCount),
  ];

  const [currentIndex, setCurrentIndex] = useState(2); // Start at S0 (index 2)
  const [isTransitioning, setIsTransitioning] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragOffset, setDragOffset] = useState(0);

  // Logical index for dots and matching active state
  const logicalIndex = (currentIndex - prefixCount + numSlides) % numSlides;

  // Auto-advance carousel
  useEffect(() => {
    if (isDragging) return;
    const interval = setInterval(() => {
      setCurrentIndex((prev) => prev + 1);
    }, 4000);
    return () => clearInterval(interval);
  }, [isDragging]);

  // Restore transition mode after instant jump
  useEffect(() => {
    if (!isTransitioning) {
      const raf = requestAnimationFrame(() => {
        setIsTransitioning(true);
      });
      return () => cancelAnimationFrame(raf);
    }
  }, [isTransitioning]);

  const handlePrev = () => {
    if (currentIndex <= 0) return;
    setIsTransitioning(true);
    setCurrentIndex((prev) => prev - 1);
  };

  const handleNext = () => {
    if (currentIndex >= paddedSlides.length - 1) return;
    setIsTransitioning(true);
    setCurrentIndex((prev) => prev + 1);
  };

  const handleTransitionEnd = () => {
    if (currentIndex <= 1) {
      setIsTransitioning(false);
      setCurrentIndex(currentIndex + numSlides);
    } else if (currentIndex >= numSlides + prefixCount) {
      setIsTransitioning(false);
      setCurrentIndex(currentIndex - numSlides);
    }
  };

  // Drag handlers
  const handleDragStart = (e) => {
    setIsDragging(true);
    setDragStartX(e.type === 'touchstart' ? e.touches[0].clientX : e.clientX);
    setDragOffset(0);
  };

  const handleDragMove = (e) => {
    if (!isDragging) return;
    const currentX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
    setDragOffset(currentX - dragStartX);
  };

  const handleDragEnd = () => {
    if (!isDragging) return;
    setIsDragging(false);
    const threshold = 60;
    if (dragOffset < -threshold) {
      handleNext();
    } else if (dragOffset > threshold) {
      handlePrev();
    }
    setDragOffset(0);
  };

  return (
    <section className="hero" id="hero">
      {/* ─── Coverflow Card Carousel ─── */}
      <div
        className="hero-carousel-fullwidth"
        onMouseDown={handleDragStart}
        onMouseMove={handleDragMove}
        onMouseUp={handleDragEnd}
        onMouseLeave={handleDragEnd}
        onTouchStart={handleDragStart}
        onTouchMove={handleDragMove}
        onTouchEnd={handleDragEnd}
      >
        <div
          className="hero-carousel-track"
          style={{
            transform: `translateX(calc(50% - (var(--hero-card-width) / 2) - (${currentIndex} * (var(--hero-card-width) + var(--hero-card-gap))) + ${dragOffset}px))`,
            transition: (!isTransitioning || isDragging) ? 'none' : 'transform 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
          }}
          onTransitionEnd={handleTransitionEnd}
        >
          {paddedSlides.map((slide, i) => {
            const isCenter = i === currentIndex;
            const scale = isCenter ? 1 : 0.9;
            const opacity = isCenter ? 1 : 0.6;

            return (
              <div
                key={i}
                className={`hero-card hero-carousel-card ${slide.isFact ? 'hero-fact-card' : ''} ${isCenter ? 'active' : ''}`}
                style={{
                  transform: `scale(${scale})`,
                  opacity,
                  transition: isDragging ? 'none' : 'transform 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94), opacity 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                }}
                onClick={() => {
                  if (isCenter) {
                    handleExplore(slide.link);
                  } else {
                    setCurrentIndex(i);
                  }
                }}
              >
                <div className="hero-card-tag">{slide.tag}</div>
                {slide.isFact ? (
                  <div className="hero-fact-card-content">
                    <div className="hero-fact-icon">{slide.icon}</div>
                    <div className="hero-fact-number">{slide.number}</div>
                    <div className="hero-fact-label">{slide.label}</div>
                    <p className="hero-fact-description">{slide.description}</p>
                    <div className="hero-fact-footer">{slide.footerText}</div>
                  </div>
                ) : (
                  <>
                    <img src={slide.src} alt={slide.alt} className="hero-card-img" draggable="false" />
                    <div className="hero-card-overlay">
                      <h3 className="hero-card-title">{slide.title}</h3>
                      <span className="hero-card-time">{slide.time}</span>
                    </div>
                  </>
                )}
                <div
                  className="hero-card-readmore"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleExplore(slide.link);
                  }}
                >
                  <span className="hero-card-readmore-icon">→</span>
                  <span>{t('hero.explore')}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Progress dots */}
        <div className="hero-carousel-dots">
          {heroSlides.map((_, i) => (
            <button
              key={i}
              className={`hero-carousel-dot ${i === logicalIndex ? 'active' : ''}`}
              onClick={() => {
                setIsTransitioning(true);
                setCurrentIndex(i + prefixCount);
              }}
              aria-label={`Go to slide ${i + 1}`}
            />
          ))}
        </div>

        {/* Navigation arrows */}
        <button
          className="hero-carousel-arrow hero-carousel-arrow-left"
          onClick={handlePrev}
          aria-label="Previous slide"
        >‹</button>
        <button
          className="hero-carousel-arrow hero-carousel-arrow-right"
          onClick={handleNext}
          aria-label="Next slide"
        >›</button>
      </div>

      {/* ─── Hero Text Content (centered below carousel) ─── */}
      <div className="hero-text-below">
        <div className="hero-badge">
          {t('hero.badge')}
        </div>
        <h1 className="hero-title">
          {t('hero.title1')}{' '}
          <span className="highlight">{t('hero.titleHighlight')}</span>{' '}
          {t('hero.title2')}
        </h1>
        <p className="hero-description">
          {t('hero.desc')}
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
            {t('hero.startAnalysis')}
            <span className="btn-arrow">→</span>
          </a>
          <a href="#about" className="btn btn-secondary hero-btn" onClick={(e) => { e.preventDefault(); document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' }); }}>
            {t('hero.learnAboutHD')}
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
            <div className="hero-stat-label">{t('hero.targetAUC')}</div>
          </div>
          <div className="hero-stat">
            <div className="hero-stat-value">3-Stage</div>
            <div className="hero-stat-label">{t('hero.hdClass')}</div>
          </div>
          <div className="hero-stat">
            <div className="hero-stat-value">24-Mo</div>
            <div className="hero-stat-label">{t('hero.progForecast')}</div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   ABOUT HD SECTION
   ═══════════════════════════════════════════ */

function AboutSection() {
  const { t } = useLanguage();
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
                {t('about.badgeText')}
              </div>
            </div>
          </div>
          <div>
            <div className="section-label reveal">{t('about.label')}</div>
            <h2 className="section-title reveal">
              {t('about.title')}
            </h2>
            <p className="section-subtitle reveal" style={{ maxWidth: '520px' }}>
              {t('about.desc')}
            </p>
            <div className="about-facts stagger">
              <div className="fact-card reveal">
                <div className="fact-icon">🧠</div>
                <div className="fact-content">
                  <h4>{t('about.fact1Title')}</h4>
                  <p>{t('about.fact1Desc')}</p>
                </div>
              </div>
              <div className="fact-card reveal">
                <div className="fact-icon">🔗</div>
                <div className="fact-content">
                  <h4>{t('about.fact2Title')}</h4>
                  <p>{t('about.fact2Desc')}</p>
                </div>
              </div>
              <div className="fact-card reveal">
                <div className="fact-icon">📊</div>
                <div className="fact-content">
                  <h4>{t('about.fact3Title')}</h4>
                  <p>{t('about.fact3Desc')}</p>
                </div>
              </div>
              <div className="fact-card reveal">
                <div className="fact-icon">⏱️</div>
                <div className="fact-content">
                  <h4>{t('about.fact4Title')}</h4>
                  <p>{t('about.fact4Desc')}</p>
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
  const { t } = useLanguage();
  const steps = [
    {
      num: '01',
      icon: '🧠',
      title: t('hiw.step1Title'),
      desc: t('hiw.step1Desc'),
    },
    {
      num: '02',
      icon: '⚡',
      title: t('hiw.step2Title'),
      desc: t('hiw.step2Desc'),
    },
    {
      num: '03',
      icon: '📋',
      title: t('hiw.step3Title'),
      desc: t('hiw.step3Desc'),
    },
  ];

  return (
    <section className="section" id="how-it-works">
      <div className="section-inner">
        <div className="section-label reveal">{t('hiw.label')}</div>
        <h2 className="section-title reveal">{t('hiw.title')}</h2>
        <p className="section-subtitle reveal">
          {t('hiw.subtitle')}
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
   MRI QUALITY ASSESSMENT
   ═══════════════════════════════════════════ */

function analyseMRIQuality(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const w = img.width;
        const h = img.height;
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data;
        const totalPixels = w * h;

        // 1. Resolution score
        const minDim = Math.min(w, h);
        const resScore = minDim >= 256 ? 100 : minDim >= 150 ? 75 : minDim >= 80 ? 50 : 25;

        // 2. Brightness (average luminance)
        let totalLum = 0;
        for (let i = 0; i < data.length; i += 4) {
          totalLum += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        }
        const avgBrightness = totalLum / totalPixels;
        const brightScore = avgBrightness >= 40 && avgBrightness <= 200
          ? Math.min(100, 100 - Math.abs(avgBrightness - 120) * 0.5)
          : Math.max(20, 60 - Math.abs(avgBrightness - 120) * 0.3);

        // 3. Contrast (standard deviation of luminance)
        let sumSqDiff = 0;
        for (let i = 0; i < data.length; i += 4) {
          const lum = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          sumSqDiff += (lum - avgBrightness) ** 2;
        }
        const stdDev = Math.sqrt(sumSqDiff / totalPixels);
        const contrastScore = stdDev >= 40 ? 100 : stdDev >= 25 ? 80 : stdDev >= 15 ? 60 : 35;

        // 4. Sharpness (Laplacian variance on luminance)
        const lumArray = [];
        for (let i = 0; i < data.length; i += 4) {
          lumArray.push(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
        }
        let laplacianVar = 0;
        let lapCount = 0;
        for (let y = 1; y < h - 1; y++) {
          for (let x = 1; x < w - 1; x++) {
            const idx = y * w + x;
            const lap = -4 * lumArray[idx] + lumArray[idx - 1] + lumArray[idx + 1] + lumArray[idx - w] + lumArray[idx + w];
            laplacianVar += lap * lap;
            lapCount++;
          }
        }
        laplacianVar = lapCount > 0 ? laplacianVar / lapCount : 0;
        const sharpScore = laplacianVar >= 500 ? 100 : laplacianVar >= 200 ? 85 : laplacianVar >= 80 ? 65 : laplacianVar >= 30 ? 45 : 25;

        // 5. Noise estimation (local variance in uniform regions)
        const blockSize = 8;
        let noiseSum = 0;
        let noiseBlocks = 0;
        for (let by = 0; by < h - blockSize; by += blockSize) {
          for (let bx = 0; bx < w - blockSize; bx += blockSize) {
            let blockMean = 0;
            for (let dy = 0; dy < blockSize; dy++) {
              for (let dx = 0; dx < blockSize; dx++) {
                blockMean += lumArray[(by + dy) * w + (bx + dx)];
              }
            }
            blockMean /= (blockSize * blockSize);
            let blockVar = 0;
            for (let dy = 0; dy < blockSize; dy++) {
              for (let dx = 0; dx < blockSize; dx++) {
                const diff = lumArray[(by + dy) * w + (bx + dx)] - blockMean;
                blockVar += diff * diff;
              }
            }
            blockVar /= (blockSize * blockSize);
            if (blockVar < 100) {
              noiseSum += blockVar;
              noiseBlocks++;
            }
          }
        }
        const avgNoise = noiseBlocks > 0 ? noiseSum / noiseBlocks : 0;
        const noiseScore = avgNoise <= 5 ? 100 : avgNoise <= 15 ? 80 : avgNoise <= 30 ? 60 : 35;

        // 6. Anatomical Brain Verification (Bilateral symmetry & cranial perimeter)
        let borderLum = 0;
        let borderCount = 0;
        const bThickness = Math.max(2, Math.floor(Math.min(w, h) * 0.06));
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            if (y < bThickness || y >= h - bThickness || x < bThickness || x >= w - bThickness) {
              const idx = (y * w + x) * 4;
              borderLum += data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
              borderCount++;
            }
          }
        }
        const avgBorderLum = borderCount > 0 ? borderLum / borderCount : 0;

        let symDiff = 0;
        let symTotal = 0;
        const halfW = Math.floor(w / 2);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < halfW; x++) {
            const lIdx = (y * w + x) * 4;
            const rIdx = (y * w + (w - 1 - x)) * 4;
            const lLum = data[lIdx] * 0.299 + data[lIdx + 1] * 0.587 + data[lIdx + 2] * 0.114;
            const rLum = data[rIdx] * 0.299 + data[rIdx + 1] * 0.587 + data[rIdx + 2] * 0.114;
            symDiff += Math.abs(lLum - rLum);
            symTotal += Math.max(lLum, rLum, 10);
          }
        }
        const symmetryScore = Math.max(0, Math.min(100, Math.round((1 - (symDiff / (symTotal + 1e-5))) * 100)));
        const aspect = w / h;
        const isBrainMRI = (avgBorderLum < 65) && (symmetryScore >= 52) && (aspect >= 0.60 && aspect <= 1.65);

        const overall = Math.round(resScore * 0.15 + brightScore * 0.2 + contrastScore * 0.25 + sharpScore * 0.25 + noiseScore * 0.15);

        resolve({
          resolution: { score: resScore, value: `${w}×${h}`, pass: resScore >= 50 },
          brightness: { score: Math.round(brightScore), value: Math.round(avgBrightness), pass: brightScore >= 50 },
          contrast: { score: Math.round(contrastScore), value: Math.round(stdDev), pass: contrastScore >= 50 },
          sharpness: { score: Math.round(sharpScore), value: Math.round(laplacianVar), pass: sharpScore >= 50 },
          noise: { score: Math.round(noiseScore), value: Math.round(avgNoise), pass: noiseScore >= 50 },
          anatomy: {
            isBrain: isBrainMRI,
            symmetry: symmetryScore,
            borderLuminance: Math.round(avgBorderLum),
            label: isBrainMRI ? 'Cranial / Brain Scan (Verified)' : 'Non-Brain / Extremity Scan Detected',
          },
          overall,
        });
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function MRIQualityAssessment({ file }) {
  const [quality, setQuality] = useState(null);
  const [isAnalysing, setIsAnalysing] = useState(false);
  const { t } = useLanguage();

  useEffect(() => {
    if (!file) { setQuality(null); return; }
    const fname = file.name.toLowerCase();
    if (!(fname.endsWith('.png') || fname.endsWith('.jpg') || fname.endsWith('.jpeg'))) {
      setQuality(null);
      return;
    }
    setIsAnalysing(true);
    analyseMRIQuality(file).then((result) => {
      setQuality(result);
      setIsAnalysing(false);
    });
  }, [file]);

  if (!file || (!quality && !isAnalysing)) return null;

  if (isAnalysing) {
    return (
      <div className="mri-qa-card mri-qa-loading">
        <div className="mri-qa-header">
          <span className="mri-qa-icon">🔍</span>
          <span className="mri-qa-title">{t('mriQ.title')}</span>
        </div>
        <div className="mri-qa-spinner-wrap">
          <div className="mri-qa-spinner" />
          <span>{t('mriQ.analysing')}</span>
        </div>
      </div>
    );
  }

  if (!quality) return null;

  const getLevel = (score) => {
    if (score >= 80) return { label: t('mriQ.excellent'), color: 'var(--success, #3d9970)', emoji: '🟢' };
    if (score >= 60) return { label: t('mriQ.good'), color: 'var(--gold, #e09f3e)', emoji: '🟡' };
    if (score >= 45) return { label: t('mriQ.acceptable'), color: 'var(--warning, #e07a5f)', emoji: '🟠' };
    return { label: t('mriQ.poor'), color: 'var(--danger, #d64545)', emoji: '🔴' };
  };

  const overall = getLevel(quality.overall);
  const metrics = [
    { key: 'resolution', label: t('mriQ.resolution'), desc: t('mriQ.resDesc'), ...quality.resolution },
    { key: 'brightness', label: t('mriQ.brightness'), desc: t('mriQ.brightDesc'), ...quality.brightness },
    { key: 'contrast', label: t('mriQ.contrast'), desc: t('mriQ.contDesc'), ...quality.contrast },
    { key: 'sharpness', label: t('mriQ.sharpness'), desc: t('mriQ.sharpDesc'), ...quality.sharpness },
    { key: 'noise', label: t('mriQ.noiseLevel'), desc: t('mriQ.noiseDesc'), ...quality.noise },
  ];

  const overallMsg = quality.overall >= 70 ? t('mriQ.suitableMsg')
    : quality.overall >= 45 ? t('mriQ.cautionMsg')
    : t('mriQ.unsuitableMsg');

  return (
    <div className="mri-qa-card">
      <div className="mri-qa-header">
        <span className="mri-qa-icon">🔍</span>
        <span className="mri-qa-title">{t('mriQ.title')}</span>
      </div>

      {/* Anatomy Validation Banner */}
      {quality.anatomy && !quality.anatomy.isBrain && (
        <div className="mri-qa-anatomy-alert">
          <div className="mri-qa-anatomy-icon">⚠️</div>
          <div className="mri-qa-anatomy-content">
            <strong>Non-Brain Scan Detected</strong>
            <p>The uploaded image does not match the anatomical characteristics of a Brain MRI (e.g. limb, bone, or non-cranial view). Please upload a Brain MRI scan.</p>
          </div>
        </div>
      )}

      {quality.anatomy && quality.anatomy.isBrain && (
        <div className="mri-qa-anatomy-pass">
          <span>🧠 <strong>Anatomy Verified:</strong> Cranial Brain MRI</span>
        </div>
      )}

      {/* Overall Score */}
      <div className="mri-qa-overall" style={{ '--qa-color': overall.color }}>
        <div className="mri-qa-score-ring">
          <svg viewBox="0 0 80 80">
            <circle cx="40" cy="40" r="34" fill="none" stroke="var(--border, #e0d6cc)" strokeWidth="6" />
            <circle
              cx="40" cy="40" r="34" fill="none"
              stroke={overall.color}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${quality.overall * 2.14} 214`}
              transform="rotate(-90 40 40)"
              className="mri-qa-score-arc"
            />
          </svg>
          <div className="mri-qa-score-value">{quality.overall}%</div>
        </div>
        <div className="mri-qa-overall-info">
          <span className="mri-qa-overall-label" style={{ color: overall.color }}>{overall.emoji} {overall.label}</span>
          <span className="mri-qa-overall-msg">{overallMsg}</span>
        </div>
      </div>

      {/* Metric Bars */}
      <div className="mri-qa-metrics">
        {metrics.map((m) => {
          const level = getLevel(m.score);
          return (
            <div className="mri-qa-metric" key={m.key}>
              <div className="mri-qa-metric-top">
                <span className="mri-qa-metric-label">{m.label}</span>
                <span className={`mri-qa-badge ${m.pass ? 'pass' : 'fail'}`}>
                  {m.pass ? t('mriQ.pass') : t('mriQ.fail')}
                </span>
              </div>
              <div className="mri-qa-bar-wrap">
                <div className="mri-qa-bar-fill" style={{ width: `${m.score}%`, background: level.color }} />
              </div>
              <div className="mri-qa-metric-bottom">
                <span className="mri-qa-metric-desc">{m.desc}</span>
                <span className="mri-qa-metric-score">{m.score}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════
   CLINICAL FORM (Enhanced Validation & UX)
   ═══════════════════════════════════════════ */

const FIELD_CONFIG = {
  cag_repeat: { label: 'CAG Repeat Count (Optional)', min: 10, max: 120, step: '1', placeholder: 'e.g. 18 (Normal) or 42 (HD)', section: 'genetic', optional: true, hint: 'Normal: 10–26 | Intermediate: 27–35 | HD: 36–120' },
  motor_score: { label: 'Motor Assessment Score', min: 0, max: 100, step: '1', placeholder: 'e.g. 85', section: 'assessment', hint: '0–100% (Reaction, Tapping, Coordination)' },
  memory_score: { label: 'Memory & Cognitive Score', min: 0, max: 100, step: '1', placeholder: 'e.g. 88', section: 'assessment', hint: '0–100% (Recall, Sequence, Visual)' },
  age: { label: 'Patient Age', min: 18, max: 90, step: '1', placeholder: 'e.g. 42', section: 'assessment', hint: 'Must be between 18–90' },
};

const MAX_FILE_SIZE_MB = 500;

const SYMPTOM_CONFIG = [
  {
    id: 'movement',
    icon: '🏃',
    titleKey: 'analysis.symptomsMovement',
    symptoms: [
      'chorea', 'dystonia', 'bradykinesia', 'impaired_gait',
      'difficulty_swallowing', 'slurred_speech', 'abnormal_eye_movements',
    ],
  },
  {
    id: 'cognitive',
    icon: '🧠',
    titleKey: 'analysis.symptomsCognitive',
    symptoms: [
      'difficulty_organizing', 'slow_processing', 'difficulty_learning',
      'perseveration', 'lack_of_awareness', 'poor_impulse_control',
    ],
  },
  {
    id: 'psychiatric',
    icon: '💭',
    titleKey: 'analysis.symptomsPsychiatric',
    symptoms: [
      'depression', 'irritability', 'apathy', 'anxiety',
      'social_withdrawal', 'insomnia', 'weight_loss_fatigue',
    ],
  },
];

function ClinicalForm({ onSubmit, isLoading }) {
  const [form, setForm] = useState(() => {
    const savedMotor = localStorage.getItem('neurosense_motor_score');
    const savedMemory = localStorage.getItem('neurosense_memory_score');
    return {
      cag_repeat: '',
      motor_score: savedMotor || '',
      memory_score: savedMemory || '',
      age: '',
    };
  });
  const [mriFile, setMriFile] = useState(null);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [dragOver, setDragOver] = useState(false);
  const [fileError, setFileError] = useState(null);
  const [symptoms, setSymptoms] = useState([]);
  const [symptomsOpen, setSymptomsOpen] = useState(false);

  // Sync test scores when returning from test pages
  useEffect(() => {
    const handleStorageChange = () => {
      const savedMotor = localStorage.getItem('neurosense_motor_score');
      const savedMemory = localStorage.getItem('neurosense_memory_score');
      setForm((prev) => ({
        ...prev,
        ...(savedMotor && !prev.motor_score ? { motor_score: savedMotor } : {}),
        ...(savedMemory && !prev.memory_score ? { memory_score: savedMemory } : {}),
      }));
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const savedMotor = localStorage.getItem('neurosense_motor_score');
  const savedMemory = localStorage.getItem('neurosense_memory_score');

  const validateField = (key, value) => {
    const cfg = FIELD_CONFIG[key];
    if (!cfg) return null;
    if (cfg.optional && (value === '' || value === undefined || value === null)) return null;
    if (value === '' || value === undefined) return 'This field is required';
    const num = parseFloat(value);
    if (isNaN(num)) return 'Please enter a valid number';
    if (num < cfg.min) return `Minimum value is ${cfg.min}`;
    if (num > cfg.max) return `Maximum value is ${cfg.max}`;
    return null;
  };

  const isFieldValid = (key) => {
    const cfg = FIELD_CONFIG[key];
    if (cfg?.optional) {
      return !validateField(key, form[key]);
    }
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

  const requiredKeys = ['motor_score', 'memory_score', 'age'];
  const allRequiredValid = requiredKeys.every((k) => isFieldValid(k)) && isFieldValid('cag_repeat');

  const validCount = ['cag_repeat', 'motor_score', 'memory_score', 'age'].filter(
    (k) => isFieldValid(k)
  ).length;

  const handleFile = (file) => {
    setFileError(null);
    if (!file) { setMriFile(null); return; }
    const validExts = ['.nii', '.nii.gz', '.gz', '.png', '.jpg', '.jpeg'];
    const name = file.name.toLowerCase();
    if (!validExts.some((ext) => name.endsWith(ext))) {
      setFileError('Invalid format. Please upload a brain MRI image (.png, .jpg) or NIfTI file (.nii, .nii.gz)');
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
    onSubmit(form, mriFile, symptoms);
  };

  const renderField = (key) => {
    const cfg = FIELD_CONFIG[key];
    const hasError = touched[key] && errors[key];
    const valid = isFieldValid(key);
    const rangeText = `${cfg.min}–${cfg.max}${key.includes('score') ? '%' : ''}`;

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

        {/* Quick test launchers and auto-fill status */}
        {key === 'motor_score' && (
          <div className="form-field-helper">
            {savedMotor ? (
              <>
                <span className="autofill-badge">✓ Test Result: {savedMotor}%</span>
                {form.motor_score !== savedMotor && (
                  <button type="button" className="btn-test-quick" onClick={() => set('motor_score', savedMotor)}>
                    Fill {savedMotor}%
                  </button>
                )}
                <Link to="/motor-test" className="btn-test-quick">Retake Test</Link>
              </>
            ) : (
              <>
                <span className="form-range">Don't know your score?</span>
                <Link to="/motor-test" className="btn-test-quick">⚡ Take Motor Test</Link>
              </>
            )}
          </div>
        )}

        {key === 'memory_score' && (
          <div className="form-field-helper">
            {savedMemory ? (
              <>
                <span className="autofill-badge">✓ Test Result: {savedMemory}%</span>
                {form.memory_score !== savedMemory && (
                  <button type="button" className="btn-test-quick" onClick={() => set('memory_score', savedMemory)}>
                    Fill {savedMemory}%
                  </button>
                )}
                <Link to="/memory-test" className="btn-test-quick">Retake Test</Link>
              </>
            ) : (
              <>
                <span className="form-range">Don't know your score?</span>
                <Link to="/memory-test" className="btn-test-quick">🧠 Take Memory Test</Link>
              </>
            )}
          </div>
        )}

        {hasError && <p className="form-error-msg">{errors[key]}</p>}
      </div>
    );
  };

  const { t } = useLanguage();

  return (
    <div className="form-card">
      <h3 className="form-card-title">{t('analysis.patientAssessment')}</h3>
      <form onSubmit={handleSubmit} noValidate>
        {/* MRI Upload */}
        <div className="form-section-label">{t('analysis.neuroimaging')}</div>
        <div
          className={`file-drop ${mriFile ? 'has-file' : ''} ${dragOver ? 'drag-over' : ''} ${fileError ? 'has-error' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".nii,.nii.gz,.gz,.png,.jpg,.jpeg"
            onChange={(e) => handleFile(e.target.files?.[0] || null)}
          />
          {mriFile ? (
            <>
              <div className="file-drop-icon">✅</div>
              <p className="file-drop-text">
                <strong>{mriFile.name}</strong>
              </p>
              <p className="file-drop-meta">
                {(mriFile.size / 1e6).toFixed(1)} MB • {mriFile.name.match(/\.(png|jpg|jpeg)$/i) ? 'Brain MRI Image' : 'NIfTI format'}
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
                {dragOver ? 'Drop file here' : <>Drop Brain MRI or <strong>browse</strong></>}
              </p>
              <p className="file-drop-hint">Brain MRI image (.png, .jpg) or NIfTI (.nii, .nii.gz) • Max {MAX_FILE_SIZE_MB} MB</p>
            </>
          )}
        </div>
        {fileError && <p className="form-error-msg file-error">{fileError}</p>}

        {/* MRI Quality Assessment */}
        <MRIQualityAssessment file={mriFile} />

        {/* Genetic */}
        <div className="form-section-label">{t('analysis.genetic')}</div>
        {renderField('cag_repeat')}

        {/* Clinical Symptoms (Optional) */}
        <div className={`symptoms-section ${symptomsOpen ? 'open' : ''} ${symptoms.length > 0 ? 'has-selections' : ''}`}>
          <button
            type="button"
            className="symptoms-toggle"
            onClick={() => setSymptomsOpen((o) => !o)}
          >
            <span className="symptoms-toggle-left">
              <span className="symptoms-toggle-icon">🩺</span>
              {t('analysis.symptomsTitle')} <span style={{ opacity: 0.5, fontWeight: 400 }}>{t('analysis.symptomsOptional')}</span>
              {symptoms.length > 0 && <span className="symptoms-badge">{symptoms.length}</span>}
            </span>
            <span className="symptoms-chevron">▼</span>
          </button>
          <div className="symptoms-body">
            <p className="symptoms-hint">{t('analysis.symptomsHint')}</p>
            {SYMPTOM_CONFIG.map((cat) => (
              <div className="symptoms-category" key={cat.id}>
                <div className="symptoms-category-header">
                  <span className="symptoms-category-icon">{cat.icon}</span>
                  {t(cat.titleKey)}
                </div>
                <div className="symptoms-chips">
                  {cat.symptoms.map((sym) => {
                    const isSelected = symptoms.includes(sym);
                    return (
                      <button
                        type="button"
                        key={sym}
                        className={`symptom-chip ${isSelected ? 'selected' : ''}`}
                        onClick={() => {
                          setSymptoms((prev) =>
                            isSelected ? prev.filter((s) => s !== sym) : [...prev, sym]
                          );
                        }}
                      >
                        <span className="symptom-chip-check">✓</span>
                        {t(`analysis.sym.${sym}`)}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
            {symptoms.length > 0 && (
              <button type="button" className="symptoms-clear" onClick={() => setSymptoms([])}>
                ✕ {t('analysis.symptomsClear')}
              </button>
            )}
          </div>
        </div>

        {/* Digital Assessment */}
        <div className="form-section-label">Motor & Cognitive Assessments</div>
        {renderField('motor_score')}
        {renderField('memory_score')}
        {renderField('age')}

        {/* Validation summary */}
        <div className="form-validation-bar">
          <div className="form-validation-dots">
            {['cag_repeat', 'motor_score', 'memory_score', 'age'].map((k) => (
              <span key={k} className={`form-val-dot ${isFieldValid(k) ? 'valid' : touched[k] && errors[k] ? 'error' : ''}`} />
            ))}
          </div>
          <span className="form-validation-text">{validCount}/4 {t('analysis.fieldsCompleted')}</span>
        </div>

        <button type="submit" className="btn-analyze" disabled={isLoading || !allRequiredValid}>
          {isLoading && <span className="spinner" />}
          {isLoading ? t('analysis.analysing') : allRequiredValid ? t('analysis.runHD') : `${t('analysis.completeFields')} (${validCount}/4)`}
        </button>
      </form>
    </div>
  );
}

/* ═══════════════════════════════════════════
   AI RECOMMENDATION ENGINE
   ═══════════════════════════════════════════ */

function getRecommendations(stage, riskCategory, t) {
  const stageKey = stage === 'pre_manifest' ? 'pre' : stage === 'early' ? 'early' : 'adv';
  const summaryKey = stage === 'pre_manifest' ? 'preManifestSummary' : stage === 'early' ? 'earlySummary' : 'advancedSummary';

  const categories = [
    {
      id: 'immediate',
      icon: '🚨',
      title: t('rec.immediateActions'),
      urgency: 'high',
      color: '#d64545',
      items: [1, 2, 3].map(n => t(`rec.${stageKey}.imm${n}`)),
    },
    {
      id: 'medical',
      icon: '🩺',
      title: t('rec.medicalCare'),
      urgency: 'high',
      color: '#e07a5f',
      items: stage === 'early'
        ? [1, 2, 3, 4, 5].map(n => t(`rec.${stageKey}.med${n}`))
        : [1, 2, 3, 4].map(n => t(`rec.${stageKey}.med${n}`)),
    },
    {
      id: 'lifestyle',
      icon: '🌿',
      title: t('rec.lifestyle'),
      urgency: 'medium',
      color: '#3d9970',
      items: stage === 'advanced'
        ? [1, 2, 3, 4].map(n => t(`rec.${stageKey}.life${n}`))
        : [1, 2, 3, 4, 5].map(n => t(`rec.${stageKey}.life${n}`)),
    },
    {
      id: 'monitoring',
      icon: '📋',
      title: t('rec.monitoring'),
      urgency: 'medium',
      color: '#e09f3e',
      items: stage === 'early'
        ? [1, 2, 3, 4].map(n => t(`rec.${stageKey}.mon${n}`))
        : [1, 2, 3].map(n => t(`rec.${stageKey}.mon${n}`)),
    },
    {
      id: 'support',
      icon: '🤝',
      title: t('rec.supportResources'),
      urgency: 'low',
      color: '#435285',
      items: [1, 2, 3].map(n => t(`rec.${stageKey}.sup${n}`)),
    },
  ];

  return {
    summary: t(`rec.${summaryKey}`),
    categories,
    riskLevel: riskCategory || 'low',
  };
}

function AIRecommendations({ result, lastForm }) {
  const [expanded, setExpanded] = useState(false);
  const [activeCategory, setActiveCategory] = useState('immediate');
  const { t } = useLanguage();

  if (!result || !result.stage) return null;

  const recs = getRecommendations(result.stage, result.risk_category, t);
  const stageColor = STAGE_CONFIG[result.stage]?.color || 'var(--accent)';
  const visibleCategories = expanded ? recs.categories : recs.categories.slice(0, 3);

  const urgencyConfig = {
    high: { label: t('rec.urgencyHigh'), color: '#d64545', icon: '⚡' },
    medium: { label: t('rec.urgencyMedium'), color: '#e09f3e', icon: '✦' },
    low: { label: t('rec.urgencyLow'), color: '#3d9970', icon: '○' },
  };

  return (
    <div className="result-card rec-card">
      <div className="result-header">
        <span>💡</span>
        <span className="result-title">{t('rec.title')}</span>
        <span className="rec-ai-badge">AI</span>
      </div>

      {/* Summary */}
      <div className="rec-summary" style={{ borderLeftColor: stageColor }}>
        <p>{recs.summary}</p>
      </div>

      {/* Category Tabs */}
      <div className="rec-tabs">
        {recs.categories.map(cat => (
          <button
            key={cat.id}
            className={`rec-tab ${activeCategory === cat.id ? 'active' : ''}`}
            onClick={() => setActiveCategory(cat.id)}
            style={{ '--tab-color': cat.color }}
          >
            <span className="rec-tab-icon">{cat.icon}</span>
            <span className="rec-tab-label">{cat.title}</span>
          </button>
        ))}
      </div>

      {/* Active Category Content */}
      {recs.categories
        .filter(cat => cat.id === activeCategory)
        .map(cat => {
          const urgency = urgencyConfig[cat.urgency];
          return (
            <div key={cat.id} className="rec-category-panel" style={{ '--cat-color': cat.color }}>
              <div className="rec-category-header">
                <div className="rec-category-title-row">
                  <span className="rec-category-icon">{cat.icon}</span>
                  <h4 className="rec-category-title">{cat.title}</h4>
                </div>
                <span className="rec-urgency-badge" style={{ background: `${urgency.color}15`, color: urgency.color, border: `1px solid ${urgency.color}30` }}>
                  {urgency.icon} {urgency.label}
                </span>
              </div>
              <ul className="rec-list">
                {cat.items.map((item, idx) => (
                  <li key={idx} className="rec-item">
                    <div className="rec-item-number" style={{ background: `${cat.color}15`, color: cat.color }}>
                      {idx + 1}
                    </div>
                    <span className="rec-item-text">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}

      {/* Disclaimer */}
      <div className="rec-disclaimer">
        <span className="rec-disclaimer-icon">⚠️</span>
        <p>{t('rec.disclaimer')}</p>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════
   PROGRESSION FORECAST CARD & TRAJECTORY ENGINE
   ═══════════════════════════════════════════ */

function ProgressionForecastCard({ result, lastForm }) {
  const [activeMetric, setActiveMetric] = useState('motor');
  const { t } = useLanguage();

  if (!result) return null;

  const motor0 = parseFloat(lastForm?.motor_score) || 85.0;
  const memory0 = parseFloat(lastForm?.memory_score) || 85.0;
  const prog12 = result.progression_12mo ?? 1.7;
  const prog24 = result.progression_24mo ?? 3.1;
  const risk = result.risk_category || 'low';

  // Calculate monthly trajectory points: 0, 6, 12, 18, 24, 36 months
  const trajectoryPoints = [
    { month: 0, label: 'Baseline (Now)', motor: motor0, memory: memory0 },
    { month: 6, label: '6 Months', motor: Math.max(5, motor0 - prog12 * 0.45), memory: Math.max(5, memory0 - prog12 * 0.3) },
    { month: 12, label: '12 Months (1 Yr)', motor: Math.max(5, motor0 - prog12), memory: Math.max(5, memory0 - prog12 * 0.65) },
    { month: 18, label: '18 Months', motor: Math.max(5, motor0 - prog12 * 1.45), memory: Math.max(5, memory0 - prog12 * 0.95) },
    { month: 24, label: '24 Months (2 Yrs)', motor: Math.max(5, motor0 - prog24), memory: Math.max(5, memory0 - prog24 * 0.65) },
    { month: 36, label: '36 Months (3 Yrs)', motor: Math.max(5, motor0 - prog24 * 1.42), memory: Math.max(5, memory0 - prog24 * 0.92) },
  ];

  // SVG dimensions
  const svgWidth = 520;
  const svgHeight = 170;
  const paddingX = 40;
  const paddingY = 25;
  const chartW = svgWidth - paddingX * 2;
  const chartH = svgHeight - paddingY * 2;

  const getMetricVal = (p) => {
    if (activeMetric === 'motor') return p.motor;
    if (activeMetric === 'cognitive') return p.memory;
    return (p.motor * 0.55 + p.memory * 0.45);
  };

  const metricColor = activeMetric === 'motor' ? '#e09f3e' : activeMetric === 'cognitive' ? '#3b82f6' : '#a855f7';

  // Map points to SVG coordinates
  const coords = trajectoryPoints.map((pt, idx) => {
    const x = paddingX + (idx / (trajectoryPoints.length - 1)) * chartW;
    const val = getMetricVal(pt);
    const y = paddingY + chartH - ((val / 100) * chartH);
    return { x, y, ...pt, val: Math.round(val * 10) / 10 };
  });

  // Build SVG smooth path
  const pathD = coords.reduce((acc, pt, i, arr) => {
    if (i === 0) return `M ${pt.x} ${pt.y}`;
    const prev = arr[i - 1];
    const cpX = (prev.x + pt.x) / 2;
    return `${acc} C ${cpX} ${prev.y}, ${cpX} ${pt.y}, ${pt.x} ${pt.y}`;
  }, '');

  const areaD = `${pathD} L ${coords[coords.length - 1].x} ${paddingY + chartH} L ${coords[0].x} ${paddingY + chartH} Z`;

  // Domain metrics
  const motor12 = trajectoryPoints[2].motor.toFixed(1);
  const motor24 = trajectoryPoints[4].motor.toFixed(1);
  const mem12 = trajectoryPoints[2].memory.toFixed(1);
  const mem24 = trajectoryPoints[4].memory.toFixed(1);

  // Clinical risk narrative
  const narrative = risk === 'low'
    ? {
        badge: '🟢 Stable Prognostic Baseline',
        title: 'Minimal Prospective Progression',
        desc: `Over the next 12 to 24 months, motor coordination and cognitive functions are forecasted to remain within normal physiological stability (< 2.0 Δ/yr). Annual digital check-ins are recommended to maintain baseline.`,
        checkup: '12 Months (Routine Check-In)',
      }
    : risk === 'medium'
    ? {
        badge: '🟡 Moderate Progression Trajectory',
        title: 'Anticipated Gradual Evolution',
        desc: `Projected annual change rate is +${prog12.toFixed(1)} Δ. Recommended semi-annual motor tests, targeted physical therapy, and proactive neurological consult to preserve agility.`,
        checkup: '6 Months (Semi-Annual Reassessment)',
      }
    : {
        badge: '🔴 Active Progression Trajectory',
        title: 'Accelerated Progression Outlook',
        desc: `Expected 12-month change is +${prog12.toFixed(1)} Δ with significant motor chorea and functional shifts. Early physical/speech rehabilitation and supportive assistive care are strongly recommended.`,
        checkup: '3 Months (Active Clinical Monitoring)',
      };

  return (
    <div className="result-card prog-forecast-card">
      <div className="result-header">
        <span>📈</span>
        <span className="result-title">{t('results.progressionForecast')}</span>
        <span className={`prog-risk-badge risk-${risk}`}>{risk.toUpperCase()} RISK</span>
      </div>

      {/* Interactive Trajectory Chart */}
      <div className="prog-chart-section">
        <div className="prog-chart-header">
          <span className="prog-chart-title">36-Month Longitudinal Trajectory Horizon</span>
          <div className="prog-tabs">
            <button className={`prog-tab ${activeMetric === 'motor' ? 'active' : ''}`} onClick={() => setActiveMetric('motor')}>⚡ Motor Trajectory</button>
            <button className={`prog-tab ${activeMetric === 'cognitive' ? 'active' : ''}`} onClick={() => setActiveMetric('cognitive')}>🧠 Memory & Cognitive</button>
            <button className={`prog-tab ${activeMetric === 'composite' ? 'active' : ''}`} onClick={() => setActiveMetric('composite')}>🌐 Composite Outlook</button>
          </div>
        </div>

        {/* SVG Graph Container */}
        <div className="prog-svg-wrap">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="prog-svg">
            <defs>
              <linearGradient id="progGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={metricColor} stopOpacity="0.35" />
                <stop offset="100%" stopColor={metricColor} stopOpacity="0.02" />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            <line x1={paddingX} y1={paddingY} x2={svgWidth - paddingX} y2={paddingY} stroke="var(--border)" strokeDasharray="3 3" opacity="0.4" />
            <line x1={paddingX} y1={paddingY + chartH / 2} x2={svgWidth - paddingX} y2={paddingY + chartH / 2} stroke="var(--border)" strokeDasharray="3 3" opacity="0.4" />
            <line x1={paddingX} y1={paddingY + chartH} x2={svgWidth - paddingX} y2={paddingY + chartH} stroke="var(--border)" opacity="0.7" />

            {/* Area Fill */}
            <path d={areaD} fill="url(#progGrad)" />

            {/* Trajectory Line */}
            <path d={pathD} fill="none" stroke={metricColor} strokeWidth="3" strokeLinecap="round" />

            {/* Milestone Pins */}
            {coords.map((pt, i) => (
              <g key={i} className="prog-pin-group">
                <circle cx={pt.x} cy={pt.y} r="5" fill={metricColor} stroke="var(--bg-card, #1e1e24)" strokeWidth="2.5" />
                <text x={pt.x} y={pt.y - 10} textAnchor="middle" fill="var(--text-primary, #fff)" fontSize="10" fontWeight="700" fontFamily="var(--font-mono)">
                  {pt.val}%
                </text>
                <text x={pt.x} y={paddingY + chartH + 16} textAnchor="middle" fill="var(--text-muted, #888)" fontSize="9.5">
                  {pt.month === 0 ? 'Now' : `${pt.month}m`}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Domain Projection Cards (2 Columns: Motor & Cognitive) */}
      <div className="prog-domain-grid">
        <div className="prog-domain-card">
          <div className="prog-domain-header">
            <span>⚡</span>
            <strong>Motor Performance Horizon</strong>
          </div>
          <div className="prog-domain-values">
            <span>Now: <strong>{motor0}%</strong></span>
            <span>12m: <strong>{motor12}%</strong></span>
            <span>24m: <strong>{motor24}%</strong></span>
          </div>
          <p className="prog-domain-status">
            {risk === 'low' ? 'Normal dexterity & reaction stability anticipated.' : 'Gradual choreic slowing expected.'}
          </p>
        </div>

        <div className="prog-domain-card">
          <div className="prog-domain-header">
            <span>🧠</span>
            <strong>Memory & Cognitive Horizon</strong>
          </div>
          <div className="prog-domain-values">
            <span>Now: <strong>{memory0}%</strong></span>
            <span>12m: <strong>{mem12}%</strong></span>
            <span>24m: <strong>{mem24}%</strong></span>
          </div>
          <p className="prog-domain-status">
            {risk === 'low' ? 'Visuospatial and verbal recall well preserved.' : 'Mild working memory drift anticipated.'}
          </p>
        </div>
      </div>

      {/* Clinical Guidance Box */}
      <div className="prog-narrative-box">
        <div className="prog-narrative-header">
          <span className="prog-narrative-badge">{narrative.badge}</span>
          <span className="prog-narrative-title">{narrative.title}</span>
        </div>
        <p className="prog-narrative-desc">{narrative.desc}</p>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════
   MEDICATION PRESCRIPTION CARD (Results Section)
   ═══════════════════════════════════════════ */

function MedicationPrescriptionCard({ result, lastForm }) {
  const navigate = useNavigate();
  const { t } = useLanguage();
  if (!result || (!result.stage && !result.prediction)) return null;

  const rx = generatePersonalizedPrescription(result, lastForm);
  if (!rx) return null;

  const stage = result.stage || result.prediction;
  const sc = STAGE_CONFIG[stage] || STAGE_CONFIG.early;

  return (
    <div className="result-card rx-summary-card">
      <div className="result-header">
        <span>💊</span>
        <span className="result-title">{t('med.patientRxTitle')}</span>
        <span className="rx-ai-badge">AI Prescription</span>
      </div>

      <div className="rx-summary-banner" style={{ borderLeftColor: sc.color }}>
        <div className="rx-banner-top">
          <span className="rx-stage-pill" style={{ background: `${sc.color}20`, color: sc.color, borderColor: `${sc.color}40` }}>
            🎯 {rx.stageTitle}
          </span>
          <span className="rx-confidence-pill">
            Confidence: <strong>{rx.confidence}%</strong>
          </span>
        </div>
        <p className="rx-summary-note">{rx.clinicalNotes}</p>
      </div>

      {/* Primary Prescribed Medications */}
      <div className="rx-meds-preview-list">
        <h4 className="rx-subheading">Core Prescribed Pharmacotherapy:</h4>
        <div className="rx-meds-grid">
          {rx.primaryMeds.map((med) => (
            <div key={med.id} className="rx-med-box">
              <div className="rx-med-box-top">
                <span className="rx-med-icon">{med.icon}</span>
                <div>
                  <div className="rx-med-name">{med.name}</div>
                  <div className="rx-med-target">{med.targetReason}</div>
                </div>
              </div>
              <div className="rx-med-details-row">
                <span className="rx-dose-tag"><strong>Dose:</strong> {med.customDose || med.standardDose}</span>
                <span className="rx-timing-tag"><strong>Timing:</strong> {med.timingSummary}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Daily Pill Clock Schedule Preview */}
      <div className="rx-timeline-preview">
        <h4 className="rx-subheading">Daily Administration Schedule:</h4>
        <div className="rx-timeline-slots">
          {rx.dailyTimeline.map((slot, i) => (
            <div key={i} className="rx-slot-box">
              <div className="rx-slot-header">
                <span className="rx-slot-time">{slot.time}</span>
                <span className="rx-slot-name">{slot.slot}</span>
              </div>
              <ul className="rx-slot-pills">
                {slot.pills.map((pill, pIdx) => (
                  <li key={pIdx}>
                    <span className="rx-pill-bullet">💊</span>
                    <div>
                      <strong>{pill.name}</strong>
                      <span className="rx-pill-inst"> ({pill.instruction})</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Deep Link Action Button to Full Medications Page */}
      <div className="rx-card-actions">
        <button
          type="button"
          className="btn-open-medications"
          onClick={() => {
            navigate(`/medications?stage=${stage}&source=prediction`);
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
        >
          <span>💊 View Complete Medication Plan & Dosing Guide</span>
          <span className="btn-arrow">→</span>
        </button>
      </div>

      <div className="rec-disclaimer">
        <span className="rec-disclaimer-icon">⚠️</span>
        <p>{t('med.disclaimer')}</p>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════
   RESULTS PANEL (3-state: idle / loading / results)
   ═══════════════════════════════════════════ */


function ResultsPanel({ result, error, isLoading, lastForm }) {
  const [loadingStep, setLoadingStep] = useState(0);
  const { t } = useLanguage();

  const loadingSteps = [
    { icon: '📤', label: t('loading.step1'), detail: t('loading.step1d') },
    { icon: '🧠', label: t('loading.step2'), detail: t('loading.step2d') },
    { icon: '⚡', label: t('loading.step3'), detail: t('loading.step3d') },
    { icon: '📊', label: t('loading.step4'), detail: t('loading.step4d') },
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
            <h3 className="dash-loading-title">{t('loading.title')}</h3>
            <p className="dash-loading-sub">{t('loading.subtitle')}</p>
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
          <span>{t('loading.footer')}</span>
        </div>
      </div>
    );
  }

  /* ─── Idle / placeholder state ─── */
  if (!result) {
    const workflowSteps = [
      { icon: '📤', label: t('results.upload'), desc: t('results.uploadDesc') },
      { icon: '⚙️', label: t('results.process'), desc: t('results.processDesc') },
      { icon: '🎯', label: t('results.predict'), desc: t('results.predictDesc') },
      { icon: '📊', label: t('results.explain'), desc: t('results.explainDesc') },
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
                <span className="dash-mri-overlay-text">{t('results.sampleMRI')}</span>
              </div>
            </div>
          </div>
          <div className="dash-preview-info">
            <h3 className="dash-preview-title">{t('results.readyTitle')}</h3>
            <p className="dash-preview-desc">
              {t('results.readyDesc')}
            </p>
          </div>
        </div>

        {/* Workflow pipeline */}
        <div className="dash-workflow">
          <div className="dash-workflow-label">
            <span className="dash-workflow-dot" />
            {t('results.analysisPipeline')}
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
            <div className="dash-output-label">{t('results.stageClass')}</div>
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
            <div className="dash-output-label">{t('results.progForecast')}</div>
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
            <div className="dash-output-label">{t('results.gradcamHeatmap')}</div>
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
            <span>{t('results.cap1')}</span>
          </div>
          <div className="dash-cap-item">
            <span className="dash-cap-check">✓</span>
            <span>{t('results.cap2')}</span>
          </div>
          <div className="dash-cap-item">
            <span className="dash-cap-check">✓</span>
            <span>{t('results.cap3')}</span>
          </div>
          <div className="dash-cap-item">
            <span className="dash-cap-check">✓</span>
            <span>{t('results.cap4')}</span>
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
        <span className="dash-results-header-label">{t('results.analysisComplete')}</span>
        <span className="dash-results-header-time">
          {result.processing_time_s ? `${result.processing_time_s.toFixed(2)}s` : '—'}
        </span>
      </div>

      {/* Stage Classification */}
      <div className="result-card">
        <div className="result-header">
          <span>🎯</span>
          <span className="result-title">{t('results.stageClassification')}</span>
        </div>
        <div className="stage-display">
          <div className={`stage-badge ${result.stage}`}>{sc.label}</div>
          <div className="confidence-block">
            <p className="confidence-label">{t('results.confidence')}</p>
            <p className="confidence-number" style={{ color: sc.color }}>{(result.confidence * 100).toFixed(1)}%</p>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: `${result.confidence * 100}%`, background: sc.color }} />
            </div>
          </div>
        </div>
        {result.stage_probabilities && (
          <div className="prob-list" style={{ marginTop: 20 }}>
            {[
              { name: t('results.preManifest'), val: result.stage_probabilities.pre_manifest, color: 'var(--stage-pre)' },
              { name: t('results.earlyHD'), val: result.stage_probabilities.early, color: 'var(--stage-early)' },
              { name: t('results.advancedHD'), val: result.stage_probabilities.advanced, color: 'var(--stage-advanced)' },
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
      <ProgressionForecastCard result={result} lastForm={lastForm} />

      {/* AI Prescribed Medication Regimen */}
      <MedicationPrescriptionCard result={result} lastForm={lastForm} />

      {/* SHAP Feature Attribution */}
      {result.shap_features?.length > 0 && (() => {
        const canonicalFeatures = result.shap_features
          .filter((f) => FEATURE_LABELS[f.name])
          .filter((f, idx, arr) => arr.findIndex((x) => x.name === f.name) === idx);

        if (canonicalFeatures.length === 0) return null;

        const totalAbsImpact = canonicalFeatures.reduce((sum, f) => sum + Math.abs(f.impact), 0) || 1.0;
        const maxAbsImpact = Math.max(...canonicalFeatures.map((x) => Math.abs(x.impact)), 0.01);

        const featureIcons = {
          cag_repeat: '🧬',
          motor_score: '⚡',
          memory_score: '🧠',
          age: '🎂',
          symptoms: '🩺',
        };

        const getClinicalBadge = (f) => {
          const isRisk = f.impact > 0;
          const sharePct = Math.round((Math.abs(f.impact) / totalAbsImpact) * 100);

          if (isRisk) {
            if (sharePct >= 35) {
              return { label: 'High Risk Driver', type: 'risk-high' };
            }
            if (sharePct >= 15) {
              return { label: 'Moderate Driver', type: 'risk-medium' };
            }
            return { label: 'Minor Influence', type: 'risk-low' };
          }
          return { label: 'Protective / Normal', type: 'protective' };
        };

        return (
          <div className="result-card">
            <div className="result-header">
              <span>📊</span>
              <span className="result-title">{t('results.featureAttribution')}</span>
              <span className="shap-legend-hint">Clinical Influence</span>
            </div>

            <div className="shap-list">
              {canonicalFeatures.map((f) => {
                const w = Math.min((Math.abs(f.impact) / maxAbsImpact) * 100, 100);
                const pos = f.impact >= 0;
                const badge = getClinicalBadge(f);
                const icon = featureIcons[f.name] || '📌';

                return (
                  <div className="shap-row" key={f.name}>
                    <div className="shap-feature-info">
                      <span className="shap-feature-icon">{icon}</span>
                      <span className="shap-name">{FEATURE_LABELS[f.name] || f.name}</span>
                    </div>

                    <div className="shap-bar-track">
                      <div
                        className={`shap-bar-fill ${pos ? 'pos' : 'neg'}`}
                        style={{ width: `${Math.max(8, w)}%` }}
                      />
                    </div>

                    <div className={`shap-pill-badge ${badge.type}`}>
                      <span className="shap-pill-dot" />
                      <span className="shap-pill-text">{badge.label}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}

      {/* GradCAM */}
      <div className="result-card">
        <div className="result-header">
          <span>🔥</span>
          <span className="result-title">{t('results.gradcam')}</span>
        </div>
        {result.gradcam_url ? (
          <div className="gradcam-wrap">
            <img src={`${API_BASE}${result.gradcam_url}`} alt="GradCAM++ heatmap overlay on axial MRI slices" />
          </div>
        ) : (
          <div className="gradcam-empty">
            <div className="gradcam-empty-icon">🧠</div>
            <p>{t('results.uploadMRI')}</p>
          </div>
        )}
      </div>

      {/* 3D Brain Viewer */}
      <BrainViewer3D stage={result.stage} visible={true} />

      {/* Processing info */}
      <div className="proc-info">
        <span>{t('results.request')}: <span>{result.request_id || '—'}</span></span>
        <span>{t('results.processedIn')} <span>{result.processing_time_s?.toFixed(2)}s</span></span>
      </div>

      {/* Download Report */}
      {lastForm && (
        <button className="btn-download-report" onClick={() => downloadReport(lastForm, result)}>
          {t('results.downloadReport')}
        </button>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════
   ANALYSIS SECTION
   ═══════════════════════════════════════════ */

function AnalysisSection({ onSubmit, isLoading, result, error, lastForm }) {
  const { t } = useLanguage();
  return (
    <section className="section analysis-section" id="analysis">
      <div className="section-inner">
        <div className="section-label reveal">{t('analysis.label')}</div>
        <h2 className="section-title reveal">{t('analysis.title')}</h2>
        <p className="section-subtitle reveal">
          {t('analysis.subtitle')}
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

const SEQUENCE_ROUNDS = [
  { round: 1, length: 3, speedMs: 750, activeMs: 450, difficulty: 'Easy', color: 'var(--success)' },
  { round: 2, length: 4, speedMs: 650, activeMs: 400, difficulty: 'Moderate', color: '#4eca88' },
  { round: 3, length: 5, speedMs: 550, activeMs: 350, difficulty: 'Intermediate', color: 'var(--gold)' },
  { round: 4, length: 6, speedMs: 460, activeMs: 290, difficulty: 'Challenging', color: 'var(--warning)' },
  { round: 5, length: 7, speedMs: 380, activeMs: 240, difficulty: 'Advanced', color: 'var(--danger)' },
];

function SequenceMemoryTest({ onComplete }) {
  const [phase, setPhase] = useState('intro'); // intro, watch, input, result, done
  const [roundIndex, setRoundIndex] = useState(0);
  const [sequence, setSequence] = useState([]);
  const [playerSequence, setPlayerSequence] = useState([]);
  const [activeTile, setActiveTile] = useState(null);
  const [showingSequence, setShowingSequence] = useState(false);
  const [isCorrect, setIsCorrect] = useState(null);
  const [roundResults, setRoundResults] = useState([]);
  const GRID_SIZE = 9;

  const currentCfg = SEQUENCE_ROUNDS[roundIndex] || SEQUENCE_ROUNDS[0];

  const generateSequence = useCallback((len) => {
    const seq = [];
    let prev = -1;
    for (let i = 0; i < len; i++) {
      let next;
      do {
        next = Math.floor(Math.random() * GRID_SIZE);
      } while (next === prev && GRID_SIZE > 1);
      seq.push(next);
      prev = next;
    }
    return seq;
  }, []);

  const playSequence = useCallback((seq, speedMs, activeMs) => {
    setShowingSequence(true);
    setActiveTile(null);
    let i = 0;
    const interval = setInterval(() => {
      if (i < seq.length) {
        setActiveTile(seq[i]);
        setTimeout(() => setActiveTile(null), activeMs);
        i++;
      } else {
        clearInterval(interval);
        setShowingSequence(false);
        setPhase('input');
      }
    }, speedMs);
  }, []);

  const startRound = useCallback((index) => {
    const cfg = SEQUENCE_ROUNDS[index];
    const seq = generateSequence(cfg.length);
    setSequence(seq);
    setPlayerSequence([]);
    setIsCorrect(null);
    setPhase('watch');
    setTimeout(() => playSequence(seq, cfg.speedMs, cfg.activeMs), 600);
  }, [generateSequence, playSequence]);

  const finishTest = useCallback((finalResults) => {
    const passedCount = finalResults.filter((r) => r.passed).length;
    const pct = Math.round((passedCount / SEQUENCE_ROUNDS.length) * 100);
    const maxLenPassed = finalResults
      .filter((r) => r.passed)
      .reduce((max, r) => Math.max(max, r.length), 0);

    setPhase('done');
    onComplete({
      test: 'sequence_memory',
      score: passedCount,
      total: SEQUENCE_ROUNDS.length,
      percentage: pct,
      maxSequenceLength: maxLenPassed,
      rounds: SEQUENCE_ROUNDS.length,
      roundDetails: finalResults,
    });
  }, [onComplete]);

  const handleTileClick = (index) => {
    if (showingSequence || phase !== 'input') return;

    const newPlayerSeq = [...playerSequence, index];
    setPlayerSequence(newPlayerSeq);
    setActiveTile(index);
    setTimeout(() => setActiveTile(null), 200);

    const currentPos = newPlayerSeq.length - 1;

    if (newPlayerSeq[currentPos] !== sequence[currentPos]) {
      // Mistake made in this round
      setIsCorrect(false);
      setPhase('result');
      const updatedResults = [...roundResults, { round: currentCfg.round, length: currentCfg.length, passed: false }];
      setRoundResults(updatedResults);

      setTimeout(() => {
        if (roundIndex + 1 < SEQUENCE_ROUNDS.length) {
          const nextIdx = roundIndex + 1;
          setRoundIndex(nextIdx);
          startRound(nextIdx);
        } else {
          finishTest(updatedResults);
        }
      }, 1300);
    } else if (newPlayerSeq.length === sequence.length) {
      // Round completed successfully!
      setIsCorrect(true);
      setPhase('result');
      const updatedResults = [...roundResults, { round: currentCfg.round, length: currentCfg.length, passed: true }];
      setRoundResults(updatedResults);

      setTimeout(() => {
        if (roundIndex + 1 < SEQUENCE_ROUNDS.length) {
          const nextIdx = roundIndex + 1;
          setRoundIndex(nextIdx);
          startRound(nextIdx);
        } else {
          finishTest(updatedResults);
        }
      }, 1300);
    }
  };

  const stopTest = () => {
    finishTest(roundResults);
  };

  if (phase === 'intro') {
    return (
      <div className="mt-test-card">
        <div className="mt-test-icon">🔲</div>
        <h4 className="mt-test-name">Sequence Memory Test</h4>
        <p className="mt-test-desc">
          Watch the tiles light up in a pattern, then repeat the sequence by clicking them in the exact order.
          Complete <strong>5 progressive rounds</strong> from 3 up to 7 steps with increasing speed.
        </p>
        <div className="mt-round-preview">
          {SEQUENCE_ROUNDS.map((r) => (
            <div key={r.round} className="mt-round-badge">
              <span className="mt-round-num">R{r.round}</span>
              <span className="mt-round-steps">{r.length} steps</span>
            </div>
          ))}
        </div>
        <button className="mt-btn-start" onClick={() => startRound(0)}>
          🎯 Begin Test (5 Rounds)
        </button>
      </div>
    );
  }

  if (phase === 'done') return null;

  return (
    <div className="mt-test-card mt-active">
      <div className="mt-phase-header">
        <span className="mt-phase-badge study">
          {phase === 'watch' ? '👁️ Watch Pattern' : phase === 'input' ? '👆 Your Turn: Repeat Pattern' : isCorrect ? '✅ Round Passed!' : '❌ Round Missed'}
        </span>
        <div className="mt-seq-info">
          <span className="mt-level" style={{ color: currentCfg.color }}>
            Round {currentCfg.round}/5 · {currentCfg.difficulty} ({currentCfg.length} steps)
          </span>
        </div>
      </div>

      {/* 5-Round Progress Indicator */}
      <div className="mt-round-indicator-bar">
        {SEQUENCE_ROUNDS.map((r, i) => {
          const res = roundResults.find((resItem) => resItem.round === r.round);
          const isCurrent = i === roundIndex;
          return (
            <div
              key={r.round}
              className={`mt-round-pill ${res ? (res.passed ? 'passed' : 'failed') : isCurrent ? 'current' : 'pending'}`}
            >
              <span>R{r.round}</span>
              <small>{res ? (res.passed ? '✓' : '✕') : `${r.length}s`}</small>
            </div>
          );
        })}
      </div>

      {phase === 'result' && (
        <div className={`mt-seq-result ${isCorrect ? 'correct' : 'incorrect'}`}>
          {isCorrect
            ? `✅ Round ${currentCfg.round} of 5 completed!`
            : `❌ Missed sequence for Round ${currentCfg.round}. Moving to next round...`}
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
        🛑 End Test Early
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
  const navigate = useNavigate();
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

      {/* Transfer score to HD Analysis */}
      <button
        type="button"
        className="btn-use-score"
        onClick={() => {
          localStorage.setItem('neurosense_memory_score', overallScore);
          window.dispatchEvent(new Event('storage'));
          navigate('/');
          setTimeout(() => {
            document.getElementById('analysis')?.scrollIntoView({ behavior: 'smooth' });
          }, 150);
        }}
      >
        🚀 Use Memory Score ({overallScore}%) in HD Analysis
      </button>

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
  const { t } = useLanguage();
  const { user } = useAuth();

  const handleTestComplete = async (result) => {
    setTestResults((prev) => [...prev, result]);
    setCompletedTests((prev) => [...prev, result.test]);
    setActiveTest(null);

    // Save to MongoDB under user document
    if (user?._id) {
      try {
        await fetch(`${API_BASE}/users/${user._id}/tests`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            testType: 'memory',
            testName: result.test,
            score: result.score,
            total: result.total,
            percentage: result.percentage,
            details: {
              ...(result.wordsShown && { wordsShown: result.wordsShown }),
              ...(result.wordsRecalled && { wordsRecalled: result.wordsRecalled }),
              ...(result.rounds && { rounds: result.rounds }),
            },
          }),
        });
      } catch (e) { /* silent fail — local results still shown */ }
    }
  };

  const resetAll = () => {
    setActiveTest(null);
    setCompletedTests([]);
    setTestResults([]);
  };

  const tests = [
    {
      id: 'word_recall',
      name: t('mem.wordRecall'),
      icon: '📝',
      desc: t('mem.wordRecallDesc'),
      duration: '~2 min',
      domain: t('mem.verbalMemory'),
      difficulty: t('mem.easy'),
      diffColor: 'var(--success)',
    },
    {
      id: 'sequence_memory',
      name: t('mem.sequenceMemory'),
      icon: '🔲',
      desc: t('mem.sequenceDesc'),
      duration: '~2 min',
      domain: t('mem.visuospatialMemory'),
      difficulty: t('mem.medium'),
      diffColor: 'var(--warning)',
    },
    {
      id: 'visual_change',
      name: t('mem.visualChange'),
      icon: '🔍',
      desc: t('mem.visualDesc'),
      duration: '~3 min',
      domain: t('mem.visualSTM'),
      difficulty: t('mem.hard'),
      diffColor: 'var(--danger)',
    },
  ];

  const progress = completedTests.length;

  return (
    <section className="section memory-test-section" id="memory-test">
      <div className="section-inner">
        <div className="section-label reveal">{t('mem.label')}</div>
        <h2 className="section-title reveal">{t('mem.title')}</h2>
        <p className="section-subtitle reveal">
          {t('mem.subtitle')}
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
            <div className="mt-progress-text">{progress}/3 {t('mem.completed')}</div>
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
                    <button className="mt-selector-btn">{t('mem.startTest')}</button>
                  )}
                  {isCompleted && result && (
                    <div className="mt-selector-done">
                      <span className="mt-selector-done-score">{result.percentage}%</span>
                      <span className="mt-selector-done-label">{t('mem.scoreAchieved')}</span>
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
              {t('mem.retakeAll')}
            </button>
          </div>
        )}

        {/* Section-level disclaimer */}
        {!activeTest && completedTests.length === 0 && (
          <div className="mt-section-disclaimer reveal">
            {t('mem.disclaimer')}
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
  const { t } = useLanguage();
  const methods = [
    { icon: '🔬', title: t('method.1title'), desc: t('method.1desc') },
    { icon: '🧠', title: t('method.2title'), desc: t('method.2desc') },
    { icon: '⚙️', title: t('method.3title'), desc: t('method.3desc') },
    { icon: '📊', title: t('method.4title'), desc: t('method.4desc') },
  ];

  return (
    <section className="section" id="methodology">
      <div className="section-inner">
        <div className="section-label reveal">{t('method.label')}</div>
        <h2 className="section-title reveal">{t('method.title')}</h2>
        <p className="section-subtitle reveal">
          {t('method.subtitle')}
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
  const { t } = useLanguage();
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
        <div className="section-label reveal">{t('tech.label')}</div>
        <h2 className="section-title reveal">{t('tech.title')}</h2>
        <p className="section-subtitle reveal">
          {t('tech.subtitle')}
        </p>
        <div className="tech-grid stagger">
          {techs.map((tech, i) => (
            <div className="tech-card reveal" key={i} style={{ '--tech-color': tech.color }}>
              <div className="tech-dot" />
              <h4 className="tech-name">{tech.name}</h4>
              <p className="tech-role">{tech.role}</p>
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
  const { t } = useLanguage();
  const stats = [
    { value: '600+', label: t('data.mriScans'), icon: '🧠' },
    { value: '5', label: t('data.clinicalFeatures'), icon: '📋' },
    { value: '3', label: t('data.hdStages'), icon: '📊' },
    { value: '2', label: t('data.forecastHorizons'), icon: '📈' },
  ];

  return (
    <section className="section" id="dataset">
      <div className="section-inner">
        <div className="section-label reveal">{t('data.label')}</div>
        <h2 className="section-title reveal">{t('data.title')}</h2>
        <p className="section-subtitle reveal">
          {t('data.subtitle')}
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
            <h4>{t('data.inputTitle')}</h4>
            <ul>
              <li>{t('data.input1')}</li>
              <li>{t('data.input2')}</li>
              <li>{t('data.input3')}</li>
              <li>{t('data.input4')}</li>
              <li>{t('data.input5')}</li>
            </ul>
          </div>
          <div className="dataset-detail-card">
            <h4>{t('data.outputTitle')}</h4>
            <ul>
              <li>{t('data.output1')}</li>
              <li>{t('data.output2')}</li>
              <li>{t('data.output3')}</li>
              <li>{t('data.output4')}</li>
              <li>{t('data.output5')}</li>
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
  const { t } = useLanguage();
  const topNodes = [
    { id: 'mri', label: t('aiw.mriInput'), sub: t('aiw.mriSub'), icon: '🧠' },
    { id: 'resnet', label: t('aiw.resnet'), sub: t('aiw.resnetSub'), icon: '🔲' },
    { id: 'fusion', label: t('aiw.crossAttn'), sub: t('aiw.crossAttnSub'), icon: '🔗' },
    { id: 'output', label: t('aiw.predictions'), sub: t('aiw.predSub'), icon: '📊' },
  ];
  const bottomNodes = [
    { id: 'clinical', label: t('aiw.clinicalData'), sub: t('aiw.clinicalSub'), icon: '📋' },
    { id: 'bilstm', label: t('aiw.bilstm'), sub: t('aiw.bilstmSub'), icon: '⚡' },
  ];

  return (
    <section className="section" id="ai-workflow">
      <div className="section-inner">
        <div className="section-label reveal">{t('aiw.label')}</div>
        <h2 className="section-title reveal">{t('aiw.title')}</h2>
        <p className="section-subtitle reveal">
          {t('aiw.subtitle')}
        </p>
        <div className="workflow-diagram reveal">
          {/* Top row: MRI stream */}
          <div className="workflow-row">
            {topNodes.map((node, i) => (
              <Fragment key={node.id}>
                {i > 0 && (
                  <div className="workflow-connector">
                    <svg viewBox="0 0 40 20" className="workflow-svg-arrow">
                      <defs>
                        <marker id={`arrowH-${i}`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                          <path d="M0,0 L8,4 L0,8" fill="var(--accent)" />
                        </marker>
                      </defs>
                      <line x1="0" y1="10" x2="32" y2="10" stroke="var(--accent)" strokeWidth="2" markerEnd={`url(#arrowH-${i})`} strokeDasharray="4 3" className="workflow-line-anim" />
                    </svg>
                  </div>
                )}
                <div className={`workflow-node ${node.id === 'fusion' ? 'workflow-node-fusion' : ''}`}>
                  <div className="workflow-node-icon">{node.icon}</div>
                  <div className="workflow-node-label">{node.label}</div>
                  <div className="workflow-node-sub">{node.sub}</div>
                </div>
              </Fragment>
            ))}
          </div>

          {/* Bottom row: Clinical stream + merge connector */}
          <div className="workflow-bottom-area">
            <div className="workflow-row workflow-row-bottom">
              {bottomNodes.map((node, i) => (
                <Fragment key={node.id}>
                  {i > 0 && (
                    <div className="workflow-connector">
                      <svg viewBox="0 0 40 20" className="workflow-svg-arrow">
                        <defs>
                          <marker id={`arrowHB-${i}`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                            <path d="M0,0 L8,4 L0,8" fill="var(--accent)" />
                          </marker>
                        </defs>
                        <line x1="0" y1="10" x2="32" y2="10" stroke="var(--accent)" strokeWidth="2" markerEnd={`url(#arrowHB-${i})`} strokeDasharray="4 3" className="workflow-line-anim" />
                      </svg>
                    </div>
                  )}
                  <div className={`workflow-node ${node.id === 'bilstm' ? 'workflow-node-fusion' : ''}`}>
                    <div className="workflow-node-icon">{node.icon}</div>
                    <div className="workflow-node-label">{node.label}</div>
                    <div className="workflow-node-sub">{node.sub}</div>
                  </div>
                </Fragment>
              ))}
            </div>

            {/* CSS-based L-connector: Bi-LSTM → Cross-Attention */}
            <div className="workflow-merge-line">
              <div className="workflow-merge-vert"></div>
              <div className="workflow-merge-horiz"></div>
              <div className="workflow-merge-arrow">▲</div>
            </div>
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
  const { t } = useLanguage();

  const faqs = [
    { q: t('faq.q1'), a: t('faq.a1') },
    { q: t('faq.q2'), a: t('faq.a2') },
    { q: t('faq.q3'), a: t('faq.a3') },
    { q: t('faq.q4'), a: t('faq.a4') },
    { q: t('faq.q5'), a: t('faq.a5') },
    { q: t('faq.q6'), a: t('faq.a6') },
    { q: t('faq.q7'), a: t('faq.a7') },
  ];

  return (
    <section className="section" id="faq">
      <div className="section-inner">
        <div className="section-label reveal">{t('faq.label')}</div>
        <h2 className="section-title reveal">{t('faq.title')}</h2>
        <div className="faq-list reveal">
          {faqs.map((faq, i) => (
            <div
              key={i}
              className={`faq-item ${openIndex === i ? 'open' : ''}`}
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
  const { t } = useLanguage();
  return (
    <section className="section" id="about-project">
      <div className="section-inner">
        <div className="section-label reveal">{t('proj.label')}</div>
        <h2 className="section-title reveal">{t('proj.title')}</h2>
        <div className="about-project-grid">
          <div className="about-project-card reveal">
            <div className="about-project-icon">🎓</div>
            <h3>{t('proj.academicTitle')}</h3>
            <p>{t('proj.academicDesc')}</p>
          </div>
          <div className="about-project-card reveal">
            <div className="about-project-icon">🎯</div>
            <h3>{t('proj.motivationTitle')}</h3>
            <p>{t('proj.motivationDesc')}</p>
          </div>
          <div className="about-project-card reveal">
            <div className="about-project-icon">👤</div>
            <h3>{t('proj.developerTitle')}</h3>
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

function _historyKey(userId) {
  return userId ? `neurosense-history-${userId}` : 'neurosense-history-anonymous';
}

function saveAnalysis(form, result, userId) {
  try {
    const key = _historyKey(userId);
    const history = JSON.parse(localStorage.getItem(key) || '[]');
    history.unshift({
      id: Date.now(),
      date: new Date().toISOString(),
      form,
      result,
    });
    // Keep last 50
    localStorage.setItem(key, JSON.stringify(history.slice(0, 50)));
  } catch (e) { /* storage full */ }
}

function getHistory(userId) {
  try {
    const key = _historyKey(userId);
    return JSON.parse(localStorage.getItem(key) || '[]');
  } catch { return []; }
}

function clearLocalHistory(userId) {
  localStorage.removeItem(_historyKey(userId));
}

async function getHistoryFromDB(userId) {
  try {
    // If user is logged in, fetch only their predictions; otherwise fetch nothing
    if (!userId) return [];
    const res = await fetch(`${API_BASE}/users/${userId}/predictions?limit=100`);
    if (!res.ok) return [];
    const data = await res.json();
    // Transform MongoDB predictions to match the localStorage format
    return (data.predictions || []).map((p) => ({
      id: p._id,
      date: p.createdAt,
      fromDB: true,
      form: p.clinicalInputs ? {
        cag_repeat: p.clinicalInputs.cag_repeat,
        motor_score: p.clinicalInputs.motor_score ?? (p.clinicalInputs.uhdrs_motor != null ? 100 - (p.clinicalInputs.uhdrs_motor / 1.24) : null),
        memory_score: p.clinicalInputs.memory_score ?? (p.clinicalInputs.uhdrs_cognitive != null ? p.clinicalInputs.uhdrs_cognitive / 2 : null),
        functional_score: p.clinicalInputs.functional_score ?? (p.clinicalInputs.tfc_score != null ? p.clinicalInputs.tfc_score / 0.13 : null),
        age: p.clinicalInputs.age,
      } : {},
      result: {
        prediction: p.prediction,
        confidence: p.confidence / 100,
        risk_category: p.riskLevel,
        stage_probabilities: p.stageProbabilities ? {
          pre_manifest: p.stageProbabilities.pre_manifest,
          early: p.stageProbabilities.early,
          advanced: p.stageProbabilities.advanced,
        } : {},
        progression_12mo: p.progression12mo || 0,
        progression_24mo: p.progression24mo || 0,
      },
    }));
  } catch {
    return [];
  }
}

/* ═══════════════════════════════════════════
   PDF REPORT DOWNLOAD
   ═══════════════════════════════════════════ */

function downloadReport(form, result) {
  const timestamp = new Date().toLocaleString();
  const stage = STAGE_CONFIG[result.prediction]?.label || result.prediction;
  const motorDisplay = form.motor_score != null ? `${form.motor_score}%` : form.uhdrs_motor != null ? `${form.uhdrs_motor}` : '—';
  const memoryDisplay = form.memory_score != null ? `${form.memory_score}%` : form.uhdrs_cognitive != null ? `${form.uhdrs_cognitive}` : '—';
  const funcDisplay = form.functional_score != null ? `${form.functional_score}%` : form.tfc_score != null ? `${form.tfc_score}` : '—';

  const rx = generatePersonalizedPrescription(result, form);

  const html = `
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <title>NeuroSense Report — ${timestamp}</title>
    <style>
      body { font-family: 'Inter', Arial, sans-serif; max-width: 760px; margin: 40px auto; color: #1a1a2e; line-height: 1.8; padding: 0 20px; }
      h1 { font-size: 1.5rem; border-bottom: 2px solid #e07a5f; padding-bottom: 10px; }
      h2 { font-size: 1.15rem; margin-top: 28px; color: #e07a5f; border-left: 4px solid #e07a5f; padding-left: 10px; }
      h3 { font-size: 1rem; margin-top: 20px; color: #2d3561; }
      table { width: 100%; border-collapse: collapse; margin: 14px 0; }
      th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.9rem; }
      th { background: #faf8f5; font-weight: 600; width: 35%; }
      .badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
      .disclaimer { margin-top: 40px; padding: 14px; background: #fff8f0; border: 1px solid #f0d0b0; border-radius: 8px; font-size: 0.8rem; color: #666; }
      .header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
      .logo { font-size: 1.5rem; }
      .rx-notes { background: #fdf5ef; padding: 12px 16px; border-radius: 8px; font-size: 0.88rem; margin: 12px 0; border: 1px solid #f4a68e40; }
      @media print { body { margin: 20px; } }
    </style></head><body>
    <div class="header"><span class="logo">🧬</span><h1>NeuroSense — Assessment & Prediction Report</h1></div>
    <p style="color:#666;font-size:0.85rem;">Generated: ${timestamp}</p>
    <h2>Patient & Assessment Data</h2>
    <table>
      <tr><th>Age</th><td>${form.age} years</td></tr>
      <tr><th>CAG Repeat Count</th><td>${form.cag_repeat || 'N/A'}</td></tr>
      <tr><th>Motor Assessment Score</th><td>${motorDisplay}</td></tr>
      <tr><th>Memory & Cognitive Score</th><td>${memoryDisplay}</td></tr>
      <tr><th>Daily Functional Capacity</th><td>${funcDisplay}</td></tr>
    </table>
    <h2>Prediction Results</h2>
    <table>
      <tr><th>HD Stage</th><td><span class="badge" style="background:${STAGE_CONFIG[result.prediction]?.color || '#999'}20;color:${STAGE_CONFIG[result.prediction]?.color || '#999'}">${stage}</span></td></tr>
      <tr><th>Confidence</th><td>${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A'}</td></tr>
      ${result.progression_12m ? `<tr><th>12-Month Forecast</th><td>${(result.progression_12m * 100).toFixed(1)}% progression risk</td></tr>` : ''}
      ${result.progression_24m ? `<tr><th>24-Month Forecast</th><td>${(result.progression_24m * 100).toFixed(1)}% progression risk</td></tr>` : ''}
    </table>
    ${result.shap_values ? `<h2>Feature Importance (SHAP)</h2><table>${Object.entries(result.shap_values).map(([k,v]) => `<tr><th>${FEATURE_LABELS[k] || k}</th><td>${typeof v === 'number' ? v.toFixed(4) : v}</td></tr>`).join('')}</table>` : ''}
    
    ${rx ? `
      <h2>AI Prescribed Medication Regimen & Schedule</h2>
      <div class="rx-notes"><strong>Clinical Strategy:</strong> ${rx.clinicalNotes}</div>
      <table>
        <thead>
          <tr style="background:#faf8f5;">
            <th style="width:28%;">Medication</th>
            <th style="width:22%;">Prescribed Dose</th>
            <th style="width:25%;">Daily Timing</th>
            <th style="width:25%;">Primary Objective</th>
          </tr>
        </thead>
        <tbody>
          ${rx.primaryMeds.concat(rx.secondaryMeds).map(m => `
            <tr>
              <td><strong>${m.name}</strong><br><span style="font-size:0.78rem; color:#777;">${m.mechanism}</span></td>
              <td><span class="badge" style="background:#e07a5f15; color:#c9623f;">${m.customDose || m.standardDose}</span></td>
              <td>${m.timingSummary}</td>
              <td>${m.targetReason || m.primaryObjective}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      <h3>Daily Dosing Schedule</h3>
      <table>
        ${rx.dailyTimeline.map(slot => `
          <tr>
            <th>${slot.time} (${slot.slot})</th>
            <td>${slot.pills.map(p => `<strong>${p.name}</strong> — <span style="color:#555;">${p.instruction}</span>`).join('<br>')}</td>
          </tr>
        `).join('')}
      </table>
    ` : ''}

    <div class="disclaimer"><strong>⚠️ Disclaimer:</strong> This report is generated by NeuroSense, an AI-powered detection and monitoring platform for individuals without direct clinical access. It is not a substitute for formal neurological diagnosis. Consult a qualified neurologist for personal healthcare decisions.</div>
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
   DISEASE PROGRESSION TIMELINE PAGE
   ═══════════════════════════════════════════ */

const PROGRESSION_STAGES = [
  {
    id: 0,
    color: '#3d9970',
    icon: '🧠',
    image: '/images/progression/brain_healthy.png',
    barWidth: 5,
  },
  {
    id: 1,
    color: '#4eca88',
    icon: '🔬',
    image: '/images/progression/brain_premanifest.png',
    barWidth: 25,
  },
  {
    id: 2,
    color: '#e09f3e',
    icon: '⚡',
    image: '/images/progression/brain_early_hd.png',
    barWidth: 50,
  },
  {
    id: 3,
    color: '#e07a5f',
    icon: '🔥',
    image: '/images/progression/brain_moderate_hd.png',
    barWidth: 75,
  },
  {
    id: 4,
    color: '#d64545',
    icon: '🚨',
    image: '/images/progression/brain_advanced_hd.png',
    barWidth: 100,
  },
];

function DiseaseProgressionPage() {
  const [activeStage, setActiveStage] = useState(null);
  const [lightboxImage, setLightboxImage] = useState(null);
  const [animatedBar, setAnimatedBar] = useState(0);
  const { t } = useLanguage();

  useScrollReveal();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  // Animate the progress bar when a stage is selected
  useEffect(() => {
    if (activeStage !== null) {
      setAnimatedBar(0);
      const target = PROGRESSION_STAGES[activeStage].barWidth;
      let frame;
      let current = 0;
      const step = () => {
        current += (target - current) * 0.08;
        if (Math.abs(current - target) < 0.5) {
          setAnimatedBar(target);
        } else {
          setAnimatedBar(current);
          frame = requestAnimationFrame(step);
        }
      };
      frame = requestAnimationFrame(step);
      return () => cancelAnimationFrame(frame);
    } else {
      setAnimatedBar(0);
    }
  }, [activeStage]);

  const stage = activeStage !== null ? PROGRESSION_STAGES[activeStage] : null;

  return (
    <ActiveSectionContext.Provider value="progression">
      <Particles />
      <Navbar />

      {/* Hero Section */}
      <section className="page-hero" id="progression-hero">
        <div className="page-hero-inner">
          <div className="hero-badge">{t('prog.badge')}</div>
          <h1 className="hero-title">
            {t('prog.title1')} <span className="highlight">{t('prog.titleHighlight')}</span>
          </h1>
          <p className="hero-description">{t('prog.desc')}</p>
          <Link to="/" className="btn btn-secondary">{t('prog.backHome')}</Link>
        </div>
      </section>

      {/* Interactive Timeline Section */}
      <section className="section prog-section" id="progression-timeline">
        <div className="section-inner">
          <div className="prog-section-header reveal">
            <h2 className="section-title">{t('prog.timelineTitle')}</h2>
            <p className="section-subtitle">{t('prog.timelineSubtitle')}</p>
          </div>

          {/* Timeline Track */}
          <div className="prog-timeline-track reveal">
            <div className="prog-track-line">
              <div
                className="prog-track-fill"
                style={{
                  width: `${animatedBar}%`,
                  background: stage ? `linear-gradient(90deg, #3d9970, ${stage.color})` : 'var(--accent)',
                }}
              />
            </div>
            <div className="prog-track-nodes">
              {PROGRESSION_STAGES.map((s, i) => (
                <button
                  key={s.id}
                  className={`prog-node ${activeStage === i ? 'active' : ''} ${activeStage !== null && activeStage > i ? 'passed' : ''}`}
                  style={{
                    '--node-color': s.color,
                    left: `${s.barWidth}%`,
                  }}
                  onClick={() => setActiveStage(activeStage === i ? null : i)}
                  aria-label={t(`prog.stage${i}Label`)}
                >
                  <span className="prog-node-dot">{s.icon}</span>
                  <span className="prog-node-label">{t(`prog.stage${i}Label`)}</span>
                  <span className="prog-node-years">{t(`prog.stage${i}Years`)}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Expanded Stage Detail Card */}
          {activeStage !== null && stage && (
            <div className="prog-stage-detail reveal" style={{ '--stage-color': stage.color }}>
              <div className="prog-detail-header">
                <div className="prog-detail-icon" style={{ background: `${stage.color}18`, color: stage.color }}>
                  {stage.icon}
                </div>
                <div className="prog-detail-titles">
                  <h3 className="prog-detail-title">{t(`prog.stage${activeStage}Label`)}</h3>
                  <span className="prog-detail-subtitle">{t(`prog.stage${activeStage}Subtitle`)}</span>
                </div>
                <div className="prog-detail-years-badge" style={{ background: `${stage.color}15`, color: stage.color, border: `1px solid ${stage.color}30` }}>
                  {t(`prog.stage${activeStage}Years`)}
                </div>
              </div>

              <p className="prog-detail-desc">{t(`prog.stage${activeStage}Desc`)}</p>

              <div className="prog-detail-grid">
                {/* Brain Changes Column */}
                <div className="prog-detail-column">
                  <h4 className="prog-column-title">
                    <span className="prog-column-icon">🧠</span>
                    {t('prog.brainChanges')}
                  </h4>
                  <ul className="prog-detail-list">
                    {[1, 2, 3, 4].map((n) => (
                      <li key={n} className="prog-detail-item">
                        <span className="prog-item-dot" style={{ background: stage.color }} />
                        {t(`prog.stage${activeStage}BrainChange${n}`)}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Brain Scan Image */}
                <div className="prog-detail-image-col">
                  <div
                    className="prog-brain-scan"
                    onClick={() => setLightboxImage(stage.image)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') setLightboxImage(stage.image); }}
                    aria-label={t('prog.viewBrain')}
                  >
                    <img src={stage.image} alt={t(`prog.stage${activeStage}Label`)} />
                    <div className="prog-scan-overlay">
                      <span>🔍 {t('prog.viewBrain')}</span>
                    </div>
                    <div className="prog-scan-glow" style={{ background: `radial-gradient(circle, ${stage.color}20 0%, transparent 70%)` }} />
                  </div>
                </div>

                {/* Symptoms Column */}
                <div className="prog-detail-column">
                  <h4 className="prog-column-title">
                    <span className="prog-column-icon">⚕️</span>
                    {t('prog.symptoms')}
                  </h4>
                  <ul className="prog-detail-list">
                    {[1, 2, 3, 4].map((n) => (
                      <li key={n} className="prog-detail-item">
                        <span className="prog-item-dot" style={{ background: stage.color }} />
                        {t(`prog.stage${activeStage}Symptom${n}`)}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Clinical Markers Bar */}
              <div className="prog-clinical-markers">
                <h4 className="prog-column-title">
                  <span className="prog-column-icon">📋</span>
                  {t('prog.clinicalMarkers')}
                </h4>
                <div className="prog-markers-row">
                  {[1, 2, 3].map((n) => (
                    <div key={n} className="prog-marker-chip" style={{ background: `${stage.color}10`, border: `1px solid ${stage.color}25` }}>
                      {t(`prog.stage${activeStage}Marker${n}`)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Stage-by-Stage Brain Comparison */}
      <section className="section prog-comparison-section" id="progression-comparison">
        <div className="section-inner">
          <div className="prog-section-header reveal">
            <h2 className="section-title">{t('prog.comparisonTitle')}</h2>
            <p className="section-subtitle">{t('prog.comparisonSubtitle')}</p>
          </div>
          <div className="prog-comparison-grid reveal">
            {PROGRESSION_STAGES.map((s, i) => (
              <div
                key={s.id}
                className={`prog-comparison-card ${activeStage === i ? 'highlight' : ''}`}
                onClick={() => { setActiveStage(i); document.getElementById('progression-timeline')?.scrollIntoView({ behavior: 'smooth' }); }}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === 'Enter') { setActiveStage(i); document.getElementById('progression-timeline')?.scrollIntoView({ behavior: 'smooth' }); } }}
              >
                <div className="prog-comp-img-wrap">
                  <img src={s.image} alt={t(`prog.stage${i}Label`)} loading="lazy" />
                  <div className="prog-comp-stage-indicator" style={{ background: s.color }}>
                    {s.icon}
                  </div>
                </div>
                <div className="prog-comp-info">
                  <h4 className="prog-comp-title" style={{ color: s.color }}>{t(`prog.stage${i}Label`)}</h4>
                  <p className="prog-comp-subtitle">{t(`prog.stage${i}Subtitle`)}</p>
                  <span className="prog-comp-years">{t(`prog.stage${i}Years`)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Key Clinical Insights */}
      <section className="section prog-insights-section" id="progression-insights">
        <div className="section-inner">
          <div className="prog-section-header reveal">
            <h2 className="section-title">{t('prog.keyInsights')}</h2>
          </div>
          <div className="prog-insights-grid reveal">
            {[1, 2, 3].map((n) => {
              const colors = ['#3d9970', '#e09f3e', '#e07a5f'];
              const icons = ['🎯', '🧬', '💊'];
              return (
                <div key={n} className="prog-insight-card" style={{ '--insight-color': colors[n - 1] }}>
                  <div className="prog-insight-icon">{icons[n - 1]}</div>
                  <h4 className="prog-insight-title">{t(`prog.insight${n}Title`)}</h4>
                  <p className="prog-insight-desc">{t(`prog.insight${n}Desc`)}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <Footer />

      {/* Brain Scan Lightbox */}
      {lightboxImage && (
        <div
          className="prog-lightbox"
          onClick={() => setLightboxImage(null)}
          role="dialog"
          aria-label={t('prog.closeBrain')}
        >
          <div className="prog-lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <img src={lightboxImage} alt="Brain scan detail" />
            <button className="prog-lightbox-close" onClick={() => setLightboxImage(null)}>✕</button>
          </div>
        </div>
      )}
    </ActiveSectionContext.Provider>
  );
}

/* ═══════════════════════════════════════════
   ANALYSIS HISTORY PAGE
   ═══════════════════════════════════════════ */


function AnalysisHistoryPage() {
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const { t } = useLanguage();
  const { user } = useAuth();
  const userId = user?._id || null;

  useScrollReveal();

  useEffect(() => {
    window.scrollTo(0, 0);
    // Load from localStorage immediately, then merge with DB
    const localHistory = getHistory(userId);
    setHistory(localHistory);
    setIsLoading(true);

    getHistoryFromDB(userId).then((dbHistory) => {
      // Merge: use DB records as the primary source, add any localStorage-only entries
      // Deduplicate by checking if a DB record's timestamp closely matches a local one
      const dbDates = new Set(dbHistory.map((h) => new Date(h.date).getTime()));
      const localOnly = localHistory.filter((h) => {
        const ts = new Date(h.date).getTime();
        // If no DB record within 5 seconds of this local record, keep it
        for (const dbTs of dbDates) {
          if (Math.abs(ts - dbTs) < 5000) return false;
        }
        return true;
      });
      const merged = [...dbHistory, ...localOnly].sort(
        (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
      );
      setHistory(merged);
      setIsLoading(false);
    });
  }, [userId]);

  const clearHistory = () => {
    // Only clear localStorage for this user — DB records are preserved
    clearLocalHistory(userId);
    // Reload from DB only
    setIsLoading(true);
    getHistoryFromDB(userId).then((dbHistory) => {
      setHistory(dbHistory);
      setSelectedId(null);
      setIsLoading(false);
    });
  };

  const selected = history.find(h => h.id === selectedId);

  return (
    <ActiveSectionContext.Provider value="history">
      <Particles />
      <Navbar />
      <section className="page-hero" id="history-hero">
        <div className="page-hero-inner">
          <div className="hero-badge">{t('hist.badge')}</div>
          <h1 className="hero-title">
            {t('hist.title1')} <span className="highlight">{t('hist.titleHighlight')}</span>
          </h1>
          <p className="hero-description">
            {t('hist.desc')}
          </p>
          <Link to="/" className="btn btn-secondary">{t('hist.backHome')}</Link>
        </div>
      </section>

      <section className="section">
        <div className="section-inner">
          {history.length === 0 ? (
            <div className="history-empty reveal">
              <div className="history-empty-icon">📭</div>
              <h3>{t('hist.noAnalyses')}</h3>
              <p>{t('hist.runAnalysis')}</p>
              <Link to="/" className="btn btn-primary" style={{ marginTop: '16px' }}>
                {t('hist.runBtn')}
              </Link>
            </div>
          ) : (
            <>
              <div className="history-header reveal">
                <span className="history-count">{history.length} {t('hist.analysisRecords')}</span>
                <button className="btn-history-clear" onClick={clearHistory}>{t('hist.clearAll')}</button>
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
                          <div className="history-detail-row"><span>Motor Score:</span><span>{item.form?.motor_score != null ? `${item.form.motor_score}%` : item.form?.uhdrs_motor != null ? `${item.form.uhdrs_motor}` : '—'}</span></div>
                          <div className="history-detail-row"><span>Memory Score:</span><span>{item.form?.memory_score != null ? `${item.form.memory_score}%` : item.form?.uhdrs_cognitive != null ? `${item.form.uhdrs_cognitive}` : '—'}</span></div>
                          <div className="history-detail-row"><span>Daily Function:</span><span>{item.form?.functional_score != null ? `${item.form.functional_score}%` : item.form?.tfc_score != null ? `${item.form.tfc_score}` : '—'}</span></div>
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
   MEDICATIONS & PRESCRIPTION PAGE
   ═══════════════════════════════════════════ */

function MedicationsPage() {
  const { t } = useLanguage();
  const navigate = useNavigate();
  const location = useLocation();

  const queryParams = new URLSearchParams(location.search);
  const stageParam = queryParams.get('stage');
  const sourceParam = queryParams.get('source');

  const [selectedStageTab, setSelectedStageTab] = useState(() => {
    if (stageParam === 'pre_manifest' || stageParam === 'Pre-manifest HD') return 'pre_manifest';
    if (stageParam === 'advanced' || stageParam === 'Advanced Huntington\'s') return 'advanced';
    if (stageParam === 'early' || stageParam === 'Early Huntington\'s') return 'early';
    return 'all';
  });

  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCardId, setExpandedCardId] = useState(null);

  // Titration calculator state
  const [calcMedId, setCalcMedId] = useState('deutetrabenazine');
  const [calcSymptomLevel, setCalcSymptomLevel] = useState('moderate');

  // Load last analysis result from localStorage if available
  const [lastAnalysis, setLastAnalysis] = useState(() => {
    try {
      const historyStr = localStorage.getItem('neurosense-history-anonymous');
      if (historyStr) {
        const history = JSON.parse(historyStr);
        if (Array.isArray(history) && history.length > 0) {
          return history[0];
        }
      }
    } catch { /* no-op */ }
    return null;
  });

  useScrollReveal();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    if (stageParam) {
      if (stageParam === 'pre_manifest' || stageParam === 'Pre-manifest HD') setSelectedStageTab('pre_manifest');
      else if (stageParam === 'advanced' || stageParam === 'Advanced Huntington\'s') setSelectedStageTab('advanced');
      else if (stageParam === 'early' || stageParam === 'Early Huntington\'s') setSelectedStageTab('early');
    }
  }, [stageParam]);

  // Determine active patient prescription
  const effectiveStage = selectedStageTab !== 'all' ? selectedStageTab : (lastAnalysis?.result?.stage || lastAnalysis?.result?.prediction || 'early');
  const simulatedResult = {
    stage: effectiveStage,
    confidence: lastAnalysis?.result?.confidence || 0.88,
    risk_category: lastAnalysis?.result?.risk_category || 'medium',
  };
  const simulatedForm = {
    motor_score: lastAnalysis?.form?.motor_score || 72,
    memory_score: lastAnalysis?.form?.memory_score || 80,
    age: lastAnalysis?.form?.age || 48,
  };

  const patientRx = useMemo(() => {
    return generatePersonalizedPrescription(simulatedResult, simulatedForm);
  }, [effectiveStage, lastAnalysis]);

  // Filtered medications
  const filteredMeds = useMemo(() => {
    return MEDICATIONS_DATA.filter((med) => {
      if (selectedStageTab !== 'all' && med.stage !== selectedStageTab) {
        return false;
      }
      if (selectedCategory !== 'all' && med.category !== selectedCategory) {
        return false;
      }
      if (searchTerm.trim()) {
        const q = searchTerm.toLowerCase();
        const matchName = med.name.toLowerCase().includes(q);
        const matchGen = med.genericName.toLowerCase().includes(q);
        const matchBrand = med.brandName.toLowerCase().includes(q);
        const matchCat = med.categoryLabel.toLowerCase().includes(q);
        const matchMech = med.mechanism.toLowerCase().includes(q);
        const matchObj = med.primaryObjective.toLowerCase().includes(q);
        return matchName || matchGen || matchBrand || matchCat || matchMech || matchObj;
      }
      return true;
    });
  }, [selectedStageTab, selectedCategory, searchTerm]);

  const stageCounts = useMemo(() => {
    return {
      all: MEDICATIONS_DATA.length,
      pre_manifest: MEDICATIONS_DATA.filter(m => m.stage === 'pre_manifest').length,
      early: MEDICATIONS_DATA.filter(m => m.stage === 'early').length,
      advanced: MEDICATIONS_DATA.filter(m => m.stage === 'advanced').length,
    };
  }, []);

  const handlePrint = () => {
    window.print();
  };

  // Titration calculator data
  const selectedCalcMed = useMemo(() => {
    return MEDICATIONS_DATA.find(m => m.id === calcMedId) || MEDICATIONS_DATA[4];
  }, [calcMedId]);

  const calcSchedule = useMemo(() => {
    if (calcMedId === 'deutetrabenazine') {
      return {
        starting: '6 mg PO once daily in morning with food',
        step1: '6 mg PO BID (12 mg/day) with breakfast & dinner',
        step2: '9 mg PO BID (18 mg/day) with breakfast & dinner',
        maintenance: calcSymptomLevel === 'severe' ? '18 mg – 24 mg BID (36–48 mg/day)' : calcSymptomLevel === 'moderate' ? '12 mg – 18 mg BID (24–36 mg/day)' : '6 mg – 12 mg BID (12–24 mg/day)',
        food: 'Must take with meals; do not chew or crush',
        maxLimit: '48 mg/day (36 mg/day in CYP2D6 poor metabolizers)',
        monitoring: 'Weekly depression score & suicidal ideation check; QTc monitoring if combined with neuroleptics',
      };
    }
    if (calcMedId === 'tetrabenazine') {
      return {
        starting: '12.5 mg PO once daily in morning',
        step1: '12.5 mg PO BID (25 mg/day)',
        step2: '12.5 mg PO TID (37.5 mg/day)',
        maintenance: calcSymptomLevel === 'severe' ? '25 mg TID (75 mg/day)' : '12.5 mg – 25 mg TID (37.5–50 mg/day)',
        food: 'With or without food; maintain consistent meal schedule',
        maxLimit: '50 mg/day (poor metabolizer) / 100 mg/day (extensive)',
        monitoring: 'Akathisia, parkinsonism, mood changes, insomnia',
      };
    }
    if (calcMedId === 'olanzapine') {
      return {
        starting: '2.5 mg PO at bedtime (Zydis ODT preferred for dysphagia)',
        step1: '2.5 mg AM / 2.5 mg Bedtime (5 mg/day)',
        step2: '2.5 mg AM / 5.0 mg Bedtime (7.5 mg/day)',
        maintenance: calcSymptomLevel === 'severe' ? '5 mg AM / 10 mg Bedtime (15 mg/day)' : '2.5 mg AM / 5 mg Bedtime (7.5 mg/day)',
        food: 'ODT dissolves instantly on tongue without water',
        maxLimit: '20 mg/day',
        monitoring: 'Fasting glucose, lipid panel, weight, orthostasis',
      };
    }
    if (calcMedId === 'baclofen') {
      return {
        starting: '5 mg PO TID with meals (15 mg/day)',
        step1: '10 mg PO TID with meals (30 mg/day)',
        step2: '15 mg PO TID with meals (45 mg/day)',
        maintenance: calcSymptomLevel === 'severe' ? '20 mg PO TID (60 mg/day)' : '10 mg – 15 mg PO TID (30–45 mg/day)',
        food: 'With food to minimize GI upset; oral liquid available for dysphagia',
        maxLimit: '80 mg/day',
        monitoring: 'Avoid abrupt discontinuation (seizure risk); monitor muscle tone & trunk stability',
      };
    }
    // Default Sertraline
    return {
      starting: '25 mg – 50 mg PO once daily in morning',
      step1: '50 mg PO once daily (after 2 weeks)',
      step2: '75 mg PO once daily (after 4 weeks)',
      maintenance: '50 mg – 150 mg PO once daily in morning',
      food: 'Take with morning breakfast',
      maxLimit: '200 mg/day',
      monitoring: 'Initial 2-week GI distress, mood swings, serotonin syndrome watch',
    };
  }, [calcMedId, calcSymptomLevel]);

  return (
    <ActiveSectionContext.Provider value="medications">
      <Particles />
      <Navbar />

      {/* Page Hero */}
      <section className="page-hero" id="medications-hero">
        <div className="page-hero-inner">
          <div className="hero-badge">{t('med.badge')}</div>
          <h1 className="hero-title">
            {t('med.title1')} <span className="highlight">{t('med.titleHighlight')}</span>
          </h1>
          <p className="hero-description">{t('med.desc')}</p>
          <div className="med-hero-actions">
            <Link to="/" className="btn btn-secondary">{t('med.backHome')}</Link>
            <button className="btn btn-primary" onClick={handlePrint}>
              {t('med.printPrescription')}
            </button>
            {sourceParam === 'prediction' && (
              <a href="#active-prescription" className="btn btn-secondary" style={{ borderColor: 'var(--accent)', color: 'var(--accent)' }}>
                🎯 View Active AI Prescription
              </a>
            )}
          </div>
        </div>
      </section>

      {/* Main Medications Content */}
      <section className="section med-section" id="medications-content">
        <div className="section-inner">

          {/* Active Patient Prescription Spotlight */}
          {patientRx && (
            <div className="med-patient-banner reveal" id="active-prescription" style={{ '--accent-glow': patientRx.stageColor }}>
              <div className="med-patient-header">
                <div className="med-patient-title-wrap">
                  <span className="med-patient-icon">📋</span>
                  <div>
                    <h2 className="med-patient-title">{t('med.patientRxTitle')}</h2>
                    <p className="med-patient-sub">{t('med.patientRxSub')}</p>
                  </div>
                </div>
                <div className="med-patient-meta-badges">
                  <span className="med-patient-badge" style={{ background: `${patientRx.stageColor}20`, color: patientRx.stageColor, border: `1px solid ${patientRx.stageColor}40` }}>
                    🎯 {patientRx.stageTitle}
                  </span>
                  <span className="med-patient-badge confidence">
                    AI Confidence: {patientRx.confidence}%
                  </span>
                  <span className={`med-patient-badge risk-${patientRx.risk}`}>
                    Risk: {patientRx.risk.toUpperCase()}
                  </span>
                </div>
              </div>

              <div className="med-patient-clinical-note">
                <strong>Clinical Assessment Summary:</strong> {patientRx.clinicalNotes}
              </div>

              {/* Prescribed Drug Grid */}
              <div className="med-prescribed-grid">
                <div className="med-prescribed-col">
                  <h3 className="med-col-title">Primary Therapeutic Regimen:</h3>
                  <div className="med-prescribed-cards">
                    {patientRx.primaryMeds.map((med) => (
                      <div key={med.id} className="med-prescribed-card primary">
                        <div className="med-pcard-header">
                          <span className="med-pcard-icon">{med.icon}</span>
                          <div>
                            <div className="med-pcard-name">{med.name}</div>
                            <div className="med-pcard-brand">{med.brandName} · <span className="med-pcard-cat">{med.categoryLabel}</span></div>
                          </div>
                        </div>
                        <div className="med-pcard-body">
                          <div className="med-pcard-row">
                            <span className="med-pcard-label">Prescribed Dose:</span>
                            <span className="med-pcard-value dose-highlight">{med.customDose || med.standardDose}</span>
                          </div>
                          <div className="med-pcard-row">
                            <span className="med-pcard-label">Daily Timing:</span>
                            <span className="med-pcard-value">{med.timingSummary}</span>
                          </div>
                          <div className="med-pcard-row">
                            <span className="med-pcard-label">Target Rationale:</span>
                            <span className="med-pcard-value note">{med.targetReason}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="med-prescribed-col">
                  <h3 className="med-col-title">Adjunctive & Supportive Regimen:</h3>
                  <div className="med-prescribed-cards">
                    {patientRx.secondaryMeds.map((med) => (
                      <div key={med.id} className="med-prescribed-card secondary">
                        <div className="med-pcard-header">
                          <span className="med-pcard-icon">{med.icon}</span>
                          <div>
                            <div className="med-pcard-name">{med.name}</div>
                            <div className="med-pcard-brand">{med.brandName} · <span className="med-pcard-cat">{med.categoryLabel}</span></div>
                          </div>
                        </div>
                        <div className="med-pcard-body">
                          <div className="med-pcard-row">
                            <span className="med-pcard-label">Prescribed Dose:</span>
                            <span className="med-pcard-value">{med.customDose || med.standardDose}</span>
                          </div>
                          <div className="med-pcard-row">
                            <span className="med-pcard-label">Daily Timing:</span>
                            <span className="med-pcard-value">{med.timingSummary}</span>
                          </div>
                          <div className="med-pcard-row">
                            <span className="med-pcard-label">Target Rationale:</span>
                            <span className="med-pcard-value note">{med.targetReason}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Daily Pill Clock Schedule Visualizer */}
              <div className="med-daily-clock-section">
                <h3 className="med-section-heading">
                  <span>⏰</span> {t('med.dailySchedule')}
                </h3>
                <div className="med-clock-timeline">
                  {patientRx.dailyTimeline.map((slot, sIdx) => (
                    <div key={sIdx} className="med-clock-slot">
                      <div className="med-clock-slot-badge">
                        <span className="med-clock-time">{slot.time}</span>
                        <span className="med-clock-name">{slot.slot}</span>
                      </div>
                      <div className="med-clock-pills">
                        {slot.pills.map((pill, pIdx) => (
                          <div key={pIdx} className="med-clock-pill-item">
                            <span className="med-clock-pill-icon">💊</span>
                            <div className="med-clock-pill-text">
                              <span className="med-clock-pill-name">{pill.name}</span>
                              <span className="med-clock-pill-inst">{pill.instruction}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Titration Roadmap for Active Prescription */}
              <div className="med-titration-section">
                <h3 className="med-section-heading">
                  <span>📈</span> {t('med.titrationRoadmap')}
                </h3>
                <div className="med-titration-steps">
                  {patientRx.titrationPlan.map((step, tIdx) => (
                    <div key={tIdx} className="med-titration-step">
                      <div className="med-tstep-num">{tIdx + 1}</div>
                      <div className="med-tstep-content">
                        <div className="med-tstep-phase">{step.phase}</div>
                        <div className="med-tstep-detail">{step.detail}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action buttons inside prescription banner */}
              <div className="med-patient-actions">
                <button className="btn btn-primary" onClick={handlePrint}>
                  🖨️ Print Patient Prescription
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    navigate('/');
                    setTimeout(() => {
                      document.getElementById('analysis')?.scrollIntoView({ behavior: 'smooth' });
                    }, 150);
                  }}
                >
                  🔬 Return to HD Analysis Dashboard
                </button>
              </div>
            </div>
          )}

          {/* Search and Filter Controls */}
          <div className="med-controls-bar reveal">
            {/* Search Input */}
            <div className="med-search-wrap">
              <span className="med-search-icon">🔍</span>
              <input
                type="text"
                className="med-search-input"
                placeholder={t('med.searchPlaceholder')}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              {searchTerm && (
                <button className="med-search-clear" onClick={() => setSearchTerm('')}>✕</button>
              )}
            </div>

            {/* Stage Filter Tabs */}
            <div className="med-stage-tabs">
              <button
                className={`med-stage-tab ${selectedStageTab === 'all' ? 'active' : ''}`}
                onClick={() => setSelectedStageTab('all')}
              >
                <span>🌐 {t('med.tabsAll')}</span>
                <span className="med-tab-count">{stageCounts.all}</span>
              </button>
              <button
                className={`med-stage-tab pre ${selectedStageTab === 'pre_manifest' ? 'active' : ''}`}
                onClick={() => setSelectedStageTab('pre_manifest')}
              >
                <span>🟢 {t('med.tabsPre')}</span>
                <span className="med-tab-count">{stageCounts.pre_manifest}</span>
              </button>
              <button
                className={`med-stage-tab early ${selectedStageTab === 'early' ? 'active' : ''}`}
                onClick={() => setSelectedStageTab('early')}
              >
                <span>🟡 {t('med.tabsEarly')}</span>
                <span className="med-tab-count">{stageCounts.early}</span>
              </button>
              <button
                className={`med-stage-tab advanced ${selectedStageTab === 'advanced' ? 'active' : ''}`}
                onClick={() => setSelectedStageTab('advanced')}
              >
                <span>🔴 {t('med.tabsAdv')}</span>
                <span className="med-tab-count">{stageCounts.advanced}</span>
              </button>
            </div>

            {/* Symptom Filter Pills */}
            <div className="med-category-pills">
              <span className="med-cat-label">{t('med.filterSymptom')}</span>
              {[
                { id: 'all', label: t('med.symptomAll'), icon: '✨' },
                { id: 'chorea', label: t('med.symptomChorea'), icon: '⚡' },
                { id: 'mood', label: t('med.symptomMood'), icon: '🧠' },
                { id: 'sleep', label: t('med.symptomSleep'), icon: '🌙' },
                { id: 'rigidity', label: t('med.symptomRigidity'), icon: '🩺' },
                { id: 'neuroprotection', label: t('med.symptomNeuro'), icon: '🌿' },
              ].map((cat) => (
                <button
                  key={cat.id}
                  className={`med-cat-pill ${selectedCategory === cat.id ? 'active' : ''}`}
                  onClick={() => setSelectedCategory(cat.id)}
                >
                  <span>{cat.icon}</span>
                  <span>{cat.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Directory Count Header */}
          <div className="med-directory-header reveal">
            <h2 className="section-title">Clinical Medications Directory</h2>
            <p className="section-subtitle">
              Showing <strong>{filteredMeds.length}</strong> evidence-based medication protocols for Huntington's Disease management.
            </p>
          </div>

          {/* Medication Cards Grid */}
          <div className="med-cards-grid reveal">
            {filteredMeds.map((med) => {
              const isExpanded = expandedCardId === med.id;
              const stageBadgeClass = med.stage === 'pre_manifest' ? 'stage-pre' : med.stage === 'advanced' ? 'stage-advanced' : 'stage-early';

              return (
                <div key={med.id} className={`med-card ${isExpanded ? 'expanded' : ''}`}>
                  {/* Card Top */}
                  <div className="med-card-top">
                    <div className="med-card-icon-wrap">
                      <span className="med-card-icon">{med.icon}</span>
                    </div>
                    <div className="med-card-title-block">
                      <h3 className="med-card-title">{med.name}</h3>
                      <div className="med-card-brand-row">
                        <span className="med-card-brand">{med.brandName}</span>
                        <span className="med-card-cat-badge">{med.categoryLabel}</span>
                      </div>
                    </div>
                    <div className="med-card-stage-wrap">
                      <span className={`med-stage-indicator ${stageBadgeClass}`}>
                        {med.stageLabel}
                      </span>
                    </div>
                  </div>

                  {/* Objective Summary */}
                  <p className="med-card-objective">{med.primaryObjective}</p>

                  {/* Key Metrics Row */}
                  <div className="med-card-metrics">
                    <div className="med-metric-box">
                      <span className="med-metric-label">{t('med.standardDose')}</span>
                      <span className="med-metric-val highlight">{med.standardDose}</span>
                    </div>
                    <div className="med-metric-box">
                      <span className="med-metric-label">{t('med.timing')}</span>
                      <span className="med-metric-val">{med.timingSummary}</span>
                    </div>
                  </div>

                  {/* Daily Timing Schedule Pills */}
                  <div className="med-timings-pills">
                    <span className="med-timings-label">Daily Times:</span>
                    <div className="med-timings-list">
                      {med.timings.map((tm, idx) => (
                        <span key={idx} className="med-timing-badge">
                          <strong>{tm.time}</strong> ({tm.slot})
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Expandable Accordion */}
                  {isExpanded && (
                    <div className="med-card-details-accordion">
                      {/* Starting & Titration */}
                      <div className="med-detail-section">
                        <h4 className="med-detail-title">📋 {t('med.titrationRoadmap')}</h4>
                        <div className="med-detail-box">
                          <p><strong>Starting Dose:</strong> {med.startingDose}</p>
                          <p><strong>Titration Schedule:</strong> {med.titration}</p>
                        </div>
                      </div>

                      {/* Administration & Food */}
                      <div className="med-detail-section">
                        <h4 className="med-detail-title">🍽️ {t('med.administration')}</h4>
                        <div className="med-detail-box">
                          <p>{med.administration}</p>
                        </div>
                      </div>

                      {/* Mechanism */}
                      <div className="med-detail-section">
                        <h4 className="med-detail-title">🔬 {t('med.mechanism')}</h4>
                        <div className="med-detail-box">
                          <p>{med.mechanism}</p>
                        </div>
                      </div>

                      {/* Side Effects & Precautions */}
                      <div className="med-detail-section">
                        <h4 className="med-detail-title">⚠️ {t('med.adverseEffects')}</h4>
                        <div className="med-detail-box warning">
                          <p><strong>Common Side Effects:</strong> {med.sideEffects.join(', ')}</p>
                          <p><strong>Safety Alerts:</strong> {med.contraindications.join('; ')}</p>
                        </div>
                      </div>

                      {/* Dysphagia Guidance */}
                      <div className="med-detail-section">
                        <h4 className="med-detail-title">🥄 Dysphagia & Swallowing Guidance</h4>
                        <div className={`med-detail-box ${med.dysphagiaSafe ? 'safe' : 'caution'}`}>
                          <p>{med.dysphagiaNotes}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Expand/Collapse Toggle Button */}
                  <button
                    className="med-card-expand-btn"
                    onClick={() => setExpandedCardId(isExpanded ? null : med.id)}
                  >
                    <span>{isExpanded ? 'Hide Pharmacology Details ▲' : 'View Full Pharmacology & Dosing Guide ▼'}</span>
                  </button>
                </div>
              );
            })}
          </div>

          {filteredMeds.length === 0 && (
            <div className="med-empty-state reveal">
              <span className="med-empty-icon">🔍</span>
              <h3>No medications found matching your criteria</h3>
              <p>Try clearing your search or switching stage and symptom filters.</p>
              <button
                className="btn btn-secondary"
                onClick={() => { setSelectedStageTab('all'); setSelectedCategory('all'); setSearchTerm(''); }}
              >
                Reset All Filters
              </button>
            </div>
          )}

          {/* Interactive Titration Calculator Section */}
          <div className="med-calc-section reveal">
            <div className="med-calc-header">
              <div className="med-calc-icon">🧮</div>
              <div>
                <h2 className="section-title">{t('med.calculatorTitle')}</h2>
                <p className="section-subtitle">
                  Simulate step-by-step weekly dose escalation and safety monitoring for top Huntington's Disease medications.
                </p>
              </div>
            </div>

            <div className="med-calc-panel">
              <div className="med-calc-controls">
                <div className="med-calc-field">
                  <label className="med-calc-label">Select Medication:</label>
                  <select
                    className="med-calc-select"
                    value={calcMedId}
                    onChange={(e) => setCalcMedId(e.target.value)}
                  >
                    <option value="deutetrabenazine">Deutetrabenazine (Austedo) — Chorea</option>
                    <option value="tetrabenazine">Tetrabenazine (Xenazine) — Chorea</option>
                    <option value="olanzapine">Olanzapine (Zyprexa / Zydis) — Severe Chorea & Psychosis</option>
                    <option value="baclofen">Baclofen (Lioresal) — Rigidity & Spasticity</option>
                    <option value="sertraline">Sertraline (Zoloft) — Depression & Mood</option>
                  </select>
                </div>

                <div className="med-calc-field">
                  <label className="med-calc-label">Symptom Severity Level:</label>
                  <div className="med-calc-radio-group">
                    {['mild', 'moderate', 'severe'].map((lvl) => (
                      <button
                        key={lvl}
                        className={`med-calc-radio ${calcSymptomLevel === lvl ? 'active' : ''}`}
                        onClick={() => setCalcSymptomLevel(lvl)}
                      >
                        {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Titration Output Roadmap */}
              <div className="med-calc-results">
                <h3 className="med-calc-res-title">
                  Simulated Escalation Protocol for <strong>{selectedCalcMed.name}</strong> ({calcSymptomLevel.toUpperCase()} severity):
                </h3>

                <div className="med-calc-step-grid">
                  <div className="med-cstep-card">
                    <div className="med-cstep-header week1">Week 1 (Initiation)</div>
                    <div className="med-cstep-dose">{calcSchedule.starting}</div>
                    <div className="med-cstep-note">Establish tolerability & baseline vitals.</div>
                  </div>

                  <div className="med-cstep-card">
                    <div className="med-cstep-header week2">Week 2 (Step 1)</div>
                    <div className="med-cstep-dose">{calcSchedule.step1}</div>
                    <div className="med-cstep-note">Assess early symptom response.</div>
                  </div>

                  <div className="med-cstep-card">
                    <div className="med-cstep-header week3">Week 3 (Step 2)</div>
                    <div className="med-cstep-dose">{calcSchedule.step2}</div>
                    <div className="med-cstep-note">Monitor for somnolence or mood shifts.</div>
                  </div>

                  <div className="med-cstep-card maintenance">
                    <div className="med-cstep-header target">Target Maintenance</div>
                    <div className="med-cstep-dose">{calcSchedule.maintenance}</div>
                    <div className="med-cstep-note">Long-term symptom control dose.</div>
                  </div>
                </div>

                <div className="med-calc-safety-box">
                  <div className="med-safety-item">
                    <span className="med-safety-icon">🍽️</span>
                    <div><strong>Administration:</strong> {calcSchedule.food}</div>
                  </div>
                  <div className="med-safety-item">
                    <span className="med-safety-icon">🛑</span>
                    <div><strong>Max Daily Limit:</strong> {calcSchedule.maxLimit}</div>
                  </div>
                  <div className="med-safety-item">
                    <span className="med-safety-icon">🩺</span>
                    <div><strong>Safety Monitoring:</strong> {calcSchedule.monitoring}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Dysphagia & Swallowing Safety Guide */}
          <div className="med-dysphagia-section reveal">
            <div className="med-dysphagia-header">
              <span className="med-dysphagia-icon">🥣</span>
              <div>
                <h2 className="section-title">{t('med.dysphagiaGuide')}</h2>
                <p className="section-subtitle">
                  Critical administration protocols for patients experiencing dysphagia and swallowing difficulties in Moderate to Advanced HD stages.
                </p>
              </div>
            </div>

            <div className="med-dysphagia-grid">
              <div className="med-dysphagia-card">
                <div className="med-dcard-icon">⚡</div>
                <h4>Orally Disintegrating Tablets (ODT)</h4>
                <p>
                  Use formulations like Olanzapine Zydis ODT or Clonazepam ODT that instantly dissolve on the tongue with saliva, requiring zero water swallowing.
                </p>
              </div>

              <div className="med-dysphagia-card">
                <div className="med-dcard-icon">🧪</div>
                <h4>Oral Liquid Solutions</h4>
                <p>
                  Utilize liquid solutions (e.g., Baclofen 5mg/5mL, Sertraline concentrate 20mg/mL) diluted into 4 oz of juice or water to eliminate choking risk.
                </p>
              </div>

              <div className="med-dysphagia-card">
                <div className="med-dcard-icon">⚠️</div>
                <h4>Crushing Restrictions</h4>
                <p>
                  Never crush extended-release tablets (e.g., Austedo XR). Immediate-release tablets (Baclofen, Quetiapine) may be finely crushed into smooth applesauce.
                </p>
              </div>

              <div className="med-dysphagia-card">
                <div className="med-dcard-icon">🪑</div>
                <h4>90° Upright Posture</h4>
                <p>
                  Patients must sit completely upright (90 degrees) during medication intake and remain upright for at least 30 minutes post-dose to prevent aspiration pneumonia.
                </p>
              </div>
            </div>
          </div>

          {/* Medical Disclaimer Card */}
          <div className="med-disclaimer-card reveal">
            <span className="med-disclaimer-icon">⚕️</span>
            <div>
              <h4>Medical Decision Support Notice</h4>
              <p>{t('med.disclaimer')}</p>
            </div>
          </div>

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
  const { t } = useLanguage();

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
            {t('footer.brandDesc')}
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
          <h4 className="footer-col-title">{t('footer.quickLinks')}</h4>
          <ul className="footer-link-list">
            <li><a href="#hero" onClick={(e) => { e.preventDefault(); scrollTo('hero'); }}>{t('footer.home')}</a></li>
            <li><a href="#about" onClick={(e) => { e.preventDefault(); scrollTo('about'); }}>{t('nav.aboutHD')}</a></li>
            <li><a href="#how-it-works" onClick={(e) => { e.preventDefault(); scrollTo('how-it-works'); }}>{t('nav.howItWorks')}</a></li>
            <li><a href="#analysis" onClick={(e) => { e.preventDefault(); scrollTo('analysis'); }}>{t('nav.analysis')}</a></li>
            <li><Link to="/medications">{t('nav.medications')}</Link></li>
            <li><Link to="/progression">{t('nav.progression')}</Link></li>
            <li><Link to="/memory-test">{t('nav.memoryTest')}</Link></li>
            <li><Link to="/motor-test">{t('nav.motorTest')}</Link></li>
            <li><Link to="/history">{t('nav.history')}</Link></li>
          </ul>
        </div>

        {/* Technologies column */}
        <div className="footer-col">
          <h4 className="footer-col-title">{t('footer.technologies')}</h4>
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
          <h4 className="footer-col-title">{t('footer.contact')}</h4>
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
          {t('footer.disclaimer')}
        </p>
      </div>

      {/* Bottom bar */}
      <div className="footer-bottom">
        <p className="footer-copy">
          © 2025–{currentYear} Suraj · SJC Institute of Technology
        </p>
        <p className="footer-built">
          {t('footer.builtWith')}
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
  const { t } = useLanguage();
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
            {t('memPage.badge')}
          </div>
          <h1 className="hero-title">
            {t('memPage.title1')}{' '}
            <span className="highlight">{t('memPage.titleHighlight')}</span>
          </h1>
          <p className="hero-description">
            {t('memPage.desc')}
          </p>
          <Link to="/" className="btn btn-secondary">
            {t('memPage.backHome')}
          </Link>
        </div>
      </section>
      <MemoryTestSection />
      <Footer />
    </ActiveSectionContext.Provider>
  );
}

/* ═══════════════════════════════════════════
   MOTOR FUNCTION ASSESSMENT
   ═══════════════════════════════════════════ */

function ReactionTimeTest({ onComplete }) {
  const [phase, setPhase] = useState('intro'); // intro, wait, ready, tap, results, done
  const [trials, setTrials] = useState([]);
  const [trialNum, setTrialNum] = useState(0);
  const [startTime, setStartTime] = useState(null);
  const [isTooEarly, setIsTooEarly] = useState(false);
  const timeoutRef = useRef(null);
  const TOTAL_TRIALS = 5;

  const startTrial = () => {
    setIsTooEarly(false);
    setPhase('wait');
    const delay = 1500 + Math.random() * 3000;
    timeoutRef.current = setTimeout(() => {
      setStartTime(Date.now());
      setPhase('ready');
    }, delay);
  };

  const handleTap = () => {
    if (phase === 'wait') {
      clearTimeout(timeoutRef.current);
      setIsTooEarly(true);
      setPhase('tap');
      setTimeout(() => startTrial(), 1200);
    } else if (phase === 'ready') {
      const rt = Date.now() - startTime;
      const newTrials = [...trials, rt];
      setTrials(newTrials);
      setTrialNum((n) => n + 1);
      setPhase('tap');

      if (newTrials.length >= TOTAL_TRIALS) {
        setTimeout(() => {
          const avg = Math.round(newTrials.reduce((s, t) => s + t, 0) / newTrials.length);
          const best = Math.min(...newTrials);
          const score = avg <= 250 ? 100 : avg <= 350 ? 80 : avg <= 500 ? 60 : avg <= 700 ? 40 : 20;
          setPhase('done');
          onComplete({
            test: 'reaction_time',
            score,
            total: 100,
            percentage: score,
            avgMs: avg,
            bestMs: best,
            trials: newTrials,
          });
        }, 800);
      } else {
        setTimeout(() => startTrial(), 800);
      }
    }
  };

  useEffect(() => {
    return () => clearTimeout(timeoutRef.current);
  }, []);

  if (phase === 'intro') {
    return (
      <div className="mt-test-card">
        <div className="mt-test-icon">⚡</div>
        <h4 className="mt-test-name">Reaction Time Test</h4>
        <p className="mt-test-desc">
          A green target will appear after a random delay. Tap/click as fast as possible when it turns <strong>green</strong>.
          Wait for the signal — tapping too early resets the trial. <strong>{TOTAL_TRIALS} trials</strong>.
        </p>
        <button className="mt-btn-start" onClick={startTrial}>
          ⚡ Begin Test
        </button>
      </div>
    );
  }

  if (phase === 'done') return null;

  return (
    <div
      className={`mt-test-card mt-active mt-reaction-area ${phase === 'ready' ? 'mt-reaction-go' : phase === 'wait' ? 'mt-reaction-wait' : ''}`}
      onClick={handleTap}
      style={{ cursor: 'pointer', userSelect: 'none' }}
    >
      <div className="mt-phase-header">
        <span className="mt-phase-badge study">
          {phase === 'wait' ? '🔴 Wait...' : phase === 'ready' ? '🟢 TAP NOW!' : isTooEarly ? '⚠️ Too early!' : `✅ ${trials[trials.length - 1]}ms`}
        </span>
        <span className="mt-level">Trial {trialNum + 1}/{TOTAL_TRIALS}</span>
      </div>

      <div className={`mt-reaction-target ${phase === 'ready' ? 'go' : phase === 'wait' ? 'waiting' : ''}`}>
        {phase === 'wait' && <span className="mt-reaction-text">Wait for green...</span>}
        {phase === 'ready' && <span className="mt-reaction-text">TAP!</span>}
        {phase === 'tap' && !isTooEarly && <span className="mt-reaction-text">{trials[trials.length - 1]}ms</span>}
        {phase === 'tap' && isTooEarly && <span className="mt-reaction-text">Too early!</span>}
      </div>

      {trials.length > 0 && (
        <div className="mt-reaction-trials">
          {trials.map((rt, i) => (
            <span key={i} className={`mt-reaction-trial-chip ${rt <= 300 ? 'fast' : rt <= 500 ? 'normal' : 'slow'}`}>
              {rt}ms
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function FingerTappingTest({ onComplete }) {
  const [phase, setPhase] = useState('intro'); // intro, tapping, done
  const [count, setCount] = useState(0);
  const [timer, setTimer] = useState(10);
  const [tapHistory, setTapHistory] = useState([]);
  const timerRef = useRef(null);
  const startRef = useRef(null);
  const DURATION = 10;

  const startTest = () => {
    setCount(0);
    setTapHistory([]);
    setTimer(DURATION);
    setPhase('tapping');
    startRef.current = Date.now();

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
  };

  useEffect(() => {
    if (phase === 'done') {
      const tapsPerSec = count / DURATION;
      const score = tapsPerSec >= 7 ? 100 : tapsPerSec >= 5.5 ? 80 : tapsPerSec >= 4 ? 60 : tapsPerSec >= 2.5 ? 40 : 20;
      onComplete({
        test: 'finger_tapping',
        score,
        total: 100,
        percentage: score,
        totalTaps: count,
        tapsPerSecond: Math.round(tapsPerSec * 10) / 10,
      });
    }
  }, [phase]);

  useEffect(() => {
    return () => clearInterval(timerRef.current);
  }, []);

  const handleTap = () => {
    if (phase !== 'tapping') return;
    setCount((c) => c + 1);
    setTapHistory((h) => [...h, Date.now() - startRef.current]);
  };

  if (phase === 'intro') {
    return (
      <div className="mt-test-card">
        <div className="mt-test-icon">👆</div>
        <h4 className="mt-test-name">Finger Tapping Test</h4>
        <p className="mt-test-desc">
          Tap the button as many times as possible in <strong>{DURATION} seconds</strong>.
          This measures your motor speed, analogous to the UHDRS finger tapping sub-score.
        </p>
        <button className="mt-btn-start" onClick={startTest}>
          👆 Begin Test
        </button>
      </div>
    );
  }

  if (phase === 'done') return null;

  return (
    <div className="mt-test-card mt-active">
      <div className="mt-phase-header">
        <span className="mt-phase-badge study">👆 Tap Rapidly</span>
        <span className="mt-timer">{timer}s</span>
      </div>
      <div className="mt-timer-bar">
        <div className="mt-timer-fill" style={{ width: `${(timer / DURATION) * 100}%` }} />
      </div>

      <div className="mt-tapping-area">
        <button className="mt-tap-button" onClick={handleTap} aria-label="Tap">
          <span className="mt-tap-count">{count}</span>
          <span className="mt-tap-label">TAPS</span>
        </button>
      </div>

      <div className="mt-tapping-stats">
        <span>Rate: <strong>{timer < DURATION ? ((count / (DURATION - timer)) || 0).toFixed(1) : '0.0'}</strong> taps/sec</span>
      </div>
    </div>
  );
}

function TrackingTest({ onComplete }) {
  const [phase, setPhase] = useState('intro'); // intro, tracking, done
  const [target, setTarget] = useState({ x: 50, y: 50 });
  const [cursor, setCursor] = useState({ x: 50, y: 50 });
  const [distances, setDistances] = useState([]);
  const [timer, setTimer] = useState(15);
  const areaRef = useRef(null);
  const frameRef = useRef(null);
  const timerRef = useRef(null);
  const targetRef = useRef({ x: 50, y: 50, vx: 1.5, vy: 1.2 });
  const DURATION = 15;

  const startTest = () => {
    setPhase('tracking');
    setDistances([]);
    setTimer(DURATION);

    timerRef.current = setInterval(() => {
      setTimer((t) => {
        if (t <= 1) {
          clearInterval(timerRef.current);
          cancelAnimationFrame(frameRef.current);
          setPhase('done');
          return 0;
        }
        return t - 1;
      });
    }, 1000);

    const animate = () => {
      const t = targetRef.current;
      t.x += t.vx;
      t.y += t.vy;

      if (t.x <= 8 || t.x >= 92) t.vx *= -1;
      if (t.y <= 8 || t.y >= 92) t.vy *= -1;

      // Slight random perturbation for organic movement
      t.vx += (Math.random() - 0.5) * 0.3;
      t.vy += (Math.random() - 0.5) * 0.3;
      t.vx = Math.max(-3, Math.min(3, t.vx));
      t.vy = Math.max(-3, Math.min(3, t.vy));

      setTarget({ x: t.x, y: t.y });
      frameRef.current = requestAnimationFrame(animate);
    };
    frameRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    if (phase === 'done') {
      const avgDist = distances.length > 0
        ? distances.reduce((s, d) => s + d, 0) / distances.length
        : 50;
      const score = avgDist <= 5 ? 100 : avgDist <= 10 ? 85 : avgDist <= 18 ? 65 : avgDist <= 30 ? 45 : 20;
      onComplete({
        test: 'tracking',
        score,
        total: 100,
        percentage: score,
        avgDistance: Math.round(avgDist * 10) / 10,
        samples: distances.length,
      });
    }
  }, [phase]);

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      cancelAnimationFrame(frameRef.current);
    };
  }, []);

  const handleMouseMove = (e) => {
    if (phase !== 'tracking' || !areaRef.current) return;
    const rect = areaRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setCursor({ x, y });

    const dx = x - target.x;
    const dy = y - target.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    setDistances((prev) => [...prev, dist]);
  };

  const handleTouchMove = (e) => {
    if (phase !== 'tracking' || !areaRef.current) return;
    e.preventDefault();
    const touch = e.touches[0];
    const rect = areaRef.current.getBoundingClientRect();
    const x = ((touch.clientX - rect.left) / rect.width) * 100;
    const y = ((touch.clientY - rect.top) / rect.height) * 100;
    setCursor({ x, y });

    const dx = x - target.x;
    const dy = y - target.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    setDistances((prev) => [...prev, dist]);
  };

  if (phase === 'intro') {
    return (
      <div className="mt-test-card">
        <div className="mt-test-icon">🎯</div>
        <h4 className="mt-test-name">Target Tracking Test</h4>
        <p className="mt-test-desc">
          Follow the moving target with your cursor (or finger on mobile) for <strong>{DURATION} seconds</strong>.
          Stay as close to the target as possible. Measures coordination and hand-eye accuracy.
        </p>
        <button className="mt-btn-start" onClick={startTest}>
          🎯 Begin Test
        </button>
      </div>
    );
  }

  if (phase === 'done') return null;

  const currentDist = distances.length > 0 ? distances[distances.length - 1] : 0;
  const isClose = currentDist < 12;

  return (
    <div className="mt-test-card mt-active">
      <div className="mt-phase-header">
        <span className="mt-phase-badge study">🎯 Track the Target</span>
        <span className="mt-timer">{timer}s</span>
      </div>
      <div className="mt-timer-bar">
        <div className="mt-timer-fill recall" style={{ width: `${(timer / DURATION) * 100}%` }} />
      </div>

      <div
        className="mt-tracking-area"
        ref={areaRef}
        onMouseMove={handleMouseMove}
        onTouchMove={handleTouchMove}
      >
        {/* Target */}
        <div
          className={`mt-tracking-target ${isClose ? 'close' : ''}`}
          style={{ left: `${target.x}%`, top: `${target.y}%` }}
        >
          <div className="mt-tracking-target-inner" />
          <div className="mt-tracking-target-ring" />
        </div>

        {/* Cursor indicator */}
        <div
          className="mt-tracking-cursor"
          style={{ left: `${cursor.x}%`, top: `${cursor.y}%` }}
        />

        {/* Distance line */}
        <svg className="mt-tracking-line-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
          <line
            x1={cursor.x} y1={cursor.y}
            x2={target.x} y2={target.y}
            stroke={isClose ? 'rgba(61,153,112,0.4)' : 'rgba(224,122,95,0.3)'}
            strokeWidth="0.3"
            strokeDasharray="1 1"
          />
        </svg>
      </div>

      <div className="mt-tracking-stats">
        <span className={`mt-tracking-dist ${isClose ? 'close' : 'far'}`}>
          Distance: {currentDist.toFixed(1)}
        </span>
        <span>Avg: {distances.length > 0 ? (distances.reduce((s, d) => s + d, 0) / distances.length).toFixed(1) : '—'}</span>
      </div>
    </div>
  );
}

function MotorTestResults({ results }) {
  const navigate = useNavigate();
  if (results.length === 0) return null;

  const overallScore = Math.round(
    results.reduce((sum, r) => sum + r.percentage, 0) / results.length
  );

  const getScoreLevel = (pct) => {
    if (pct >= 80) return { label: 'Excellent', color: 'var(--success)', emoji: '🟢' };
    if (pct >= 60) return { label: 'Good', color: 'var(--gold)', emoji: '🟡' };
    if (pct >= 40) return { label: 'Fair', color: 'var(--warning)', emoji: '🟠' };
    return { label: 'Needs Attention', color: 'var(--danger)', emoji: '🔴' };
  };

  const overall = getScoreLevel(overallScore);

  const domainInfo = {
    reaction_time: { name: 'Reaction Time', icon: '⚡', domain: 'Simple Reaction Speed', abbr: 'SRS' },
    finger_tapping: { name: 'Finger Tapping', icon: '👆', domain: 'Motor Speed', abbr: 'MS' },
    tracking: { name: 'Target Tracking', icon: '🎯', domain: 'Hand-Eye Coordination', abbr: 'HEC' },
  };

  // Radar chart
  const radarSize = 200;
  const center = radarSize / 2;
  const maxRadius = 70;
  const angles = results.map((_, i) => (i * 2 * Math.PI) / Math.max(results.length, 3) - Math.PI / 2);
  const radarPoints = results.map((r, i) => {
    const radius = (r.percentage / 100) * maxRadius;
    return `${center + radius * Math.cos(angles[i])},${center + radius * Math.sin(angles[i])}`;
  }).join(' ');

  return (
    <div className="mt-results-dashboard">
      <div className="mt-results-header">
        <div className="mt-results-icon">🏃</div>
        <div>
          <h3 className="mt-results-title">Motor Assessment Summary</h3>
          <p className="mt-results-sub">{results.length} of 3 domains assessed</p>
        </div>
      </div>

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

        {results.length >= 2 && (
          <div className="mt-radar-wrap">
            <svg viewBox={`0 0 ${radarSize} ${radarSize}`} className="mt-radar-svg">
              {[0.25, 0.5, 0.75, 1].map((pct) => (
                <polygon key={pct} points={
                  [0, 1, 2].map(i => {
                    const a = (i * 2 * Math.PI) / 3 - Math.PI / 2;
                    const r = maxRadius * pct;
                    return `${center + r * Math.cos(a)},${center + r * Math.sin(a)}`;
                  }).join(' ')
                } fill="none" stroke="var(--border)" strokeWidth="0.5" />
              ))}
              {[0, 1, 2].map(i => {
                const a = (i * 2 * Math.PI) / 3 - Math.PI / 2;
                return <line key={i} x1={center} y1={center} x2={center + maxRadius * Math.cos(a)} y2={center + maxRadius * Math.sin(a)} stroke="var(--border)" strokeWidth="0.5" />;
              })}
              <polygon points={radarPoints} fill="rgba(224, 122, 95, 0.15)" stroke="var(--accent)" strokeWidth="2" />
              {results.map((r, i) => {
                const radius = (r.percentage / 100) * maxRadius;
                return <circle key={i} cx={center + radius * Math.cos(angles[i])} cy={center + radius * Math.sin(angles[i])} r="4" fill="var(--accent)" />;
              })}
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

      <div className="mt-domain-grid">
        {results.map((r, i) => {
          const level = getScoreLevel(r.percentage);
          const info = domainInfo[r.test] || { name: r.test, icon: '🧠', domain: 'Unknown', abbr: '?' };
          const detail = r.test === 'reaction_time' ? `Avg: ${r.avgMs}ms · Best: ${r.bestMs}ms`
            : r.test === 'finger_tapping' ? `${r.totalTaps} taps · ${r.tapsPerSecond}/sec`
            : `Avg distance: ${r.avgDistance}`;
          return (
            <div key={i} className="mt-domain-card" style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="mt-domain-top">
                <span className="mt-domain-icon">{info.icon}</span>
                <div className="mt-domain-score-pill" style={{ background: `${level.color}15`, color: level.color }}>
                  {r.percentage}%
                </div>
              </div>
              <h4 className="mt-domain-name">{info.name}</h4>
              <p className="mt-domain-domain">{info.domain}</p>
              <div className="mt-domain-bar">
                <div className="mt-domain-fill" style={{ width: `${r.percentage}%`, background: level.color }} />
              </div>
              <div className="mt-domain-detail">
                <span>{detail}</span>
                <span className="mt-domain-level" style={{ color: level.color }}>{level.emoji} {level.label}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Transfer score to HD Analysis */}
      <button
        type="button"
        className="btn-use-score"
        onClick={() => {
          localStorage.setItem('neurosense_motor_score', overallScore);
          window.dispatchEvent(new Event('storage'));
          navigate('/');
          setTimeout(() => {
            document.getElementById('analysis')?.scrollIntoView({ behavior: 'smooth' });
          }, 150);
        }}
      >
        ⚡ Use Motor Score ({overallScore}%) in HD Analysis
      </button>

      <div className="mt-results-disclaimer">
        <div className="mt-disclaimer-icon">⚕️</div>
        <div>
          <strong>Clinical Disclaimer</strong>
          <p>These tests are supplementary motor assessments and are not diagnostic tools. For clinical evaluation, consult a qualified neurologist. Results should be interpreted in conjunction with comprehensive clinical data.</p>
        </div>
      </div>
    </div>
  );
}

function MotorTestSection() {
  const [activeTest, setActiveTest] = useState(null);
  const [completedTests, setCompletedTests] = useState([]);
  const [testResults, setTestResults] = useState([]);
  const { t } = useLanguage();
  const { user } = useAuth();

  const handleTestComplete = async (result) => {
    setTestResults((prev) => [...prev, result]);
    setCompletedTests((prev) => [...prev, result.test]);
    setActiveTest(null);

    // Save to MongoDB under user document
    if (user?._id) {
      try {
        await fetch(`${API_BASE}/users/${user._id}/tests`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            testType: 'motor',
            testName: result.test,
            score: result.score,
            total: result.total,
            percentage: result.percentage,
            details: {
              ...(result.avgMs != null && { avgMs: result.avgMs }),
              ...(result.bestMs != null && { bestMs: result.bestMs }),
              ...(result.trials && { trials: result.trials }),
              ...(result.totalTaps != null && { totalTaps: result.totalTaps }),
              ...(result.tapsPerSecond != null && { tapsPerSecond: result.tapsPerSecond }),
              ...(result.avgDistance != null && { avgDistance: result.avgDistance }),
              ...(result.samples != null && { samples: result.samples }),
            },
          }),
        });
      } catch (e) { /* silent fail — local results still shown */ }
    }
  };

  const resetAll = () => {
    setActiveTest(null);
    setCompletedTests([]);
    setTestResults([]);
  };

  const tests = [
    {
      id: 'reaction_time',
      name: t('motor.reactionTime'),
      icon: '⚡',
      desc: t('motor.reactionDesc'),
      duration: '~1 min',
      domain: t('motor.speed'),
      difficulty: t('motor.easy'),
      diffColor: 'var(--success)',
    },
    {
      id: 'finger_tapping',
      name: t('motor.fingerTapping'),
      icon: '👆',
      desc: t('motor.fingerDesc'),
      duration: '~30 sec',
      domain: t('motor.speed'),
      difficulty: t('motor.medium'),
      diffColor: 'var(--warning)',
    },
    {
      id: 'tracking',
      name: t('motor.tracking'),
      icon: '🎯',
      desc: t('motor.trackingDesc'),
      duration: '~20 sec',
      domain: t('motor.coordination'),
      difficulty: t('motor.hard'),
      diffColor: 'var(--danger)',
    },
  ];

  const progress = completedTests.length;

  return (
    <section className="section memory-test-section" id="motor-test">
      <div className="section-inner">
        <div className="section-label reveal">{t('motor.label')}</div>
        <h2 className="section-title reveal">{t('motor.title')}</h2>
        <p className="section-subtitle reveal">
          {t('motor.subtitle')}
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
            <div className="mt-progress-text">{progress}/3 {t('motor.completed')}</div>
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
                    <span className="mt-tag mt-tag-domain">🏃 {test.domain}</span>
                  </div>
                  <p className="mt-selector-desc">{test.desc}</p>
                  <div className="mt-selector-meta">
                    <span>⏱️ {test.duration}</span>
                    <span>📐 {test.domain}</span>
                  </div>
                  {!isCompleted && (
                    <button className="mt-selector-btn">{t('motor.startTest')}</button>
                  )}
                  {isCompleted && result && (
                    <div className="mt-selector-done">
                      <span className="mt-selector-done-score">{result.percentage}%</span>
                      <span className="mt-selector-done-label">{t('motor.scoreAchieved')}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {activeTest === 'reaction_time' && (
          <ReactionTimeTest onComplete={handleTestComplete} />
        )}
        {activeTest === 'finger_tapping' && (
          <FingerTappingTest onComplete={handleTestComplete} />
        )}
        {activeTest === 'tracking' && (
          <TrackingTest onComplete={handleTestComplete} />
        )}

        <MotorTestResults results={testResults} />

        {completedTests.length > 0 && !activeTest && (
          <div className="mt-reset-wrap">
            <button className="mt-btn-reset" onClick={resetAll}>
              {t('motor.retakeAll')}
            </button>
          </div>
        )}

        {!activeTest && completedTests.length === 0 && (
          <div className="mt-section-disclaimer reveal">
            {t('motor.disclaimer')}
          </div>
        )}
      </div>
    </section>
  );
}

function MotorTestPage() {
  const { t } = useLanguage();
  useScrollReveal();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <ActiveSectionContext.Provider value="motor-test">
      <Particles />
      <Navbar />
      <section className="page-hero" id="motor-test-hero">
        <div className="page-hero-inner">
          <div className="hero-badge">
            {t('motorPage.badge')}
          </div>
          <h1 className="hero-title">
            {t('motorPage.title1')}{' '}
            <span className="highlight">{t('motorPage.titleHighlight')}</span>
          </h1>
          <p className="hero-description">
            {t('motorPage.desc')}
          </p>
          <Link to="/" className="btn btn-secondary">
            {t('motorPage.backHome')}
          </Link>
        </div>
      </section>
      <MotorTestSection />
      <Footer />
    </ActiveSectionContext.Provider>
  );
}

/* ═══════════════════════════════════════════
   AI CHATBOT WIDGET
   ═══════════════════════════════════════════ */

const QUICK_PROMPTS = [
  { icon: '🩺', label: 'HD Symptoms (Mayo)', prompt: 'What are the main movement, cognitive, and psychiatric symptoms of Huntington\'s Disease according to Mayo Clinic guidelines?' },
  { icon: '🧠', label: 'Early Signs & Staging', prompt: 'What are the earliest cognitive, mood, and motor warning signs of Huntington\'s Disease, and how are stages classified?' },
  { icon: '🔬', label: 'Symptom Scoring in AI', prompt: 'How does the NeuroSense platform use patient symptoms in its AI disease staging and SHAP feature attribution?' },
  { icon: '💊', label: 'Austedo Dosing', prompt: 'What is the starting dose, titration schedule, and CYP2D6 limits for Deutetrabenazine (Austedo)?' },
  { icon: '⚡', label: 'Chorea Meds', prompt: 'What are the first-line medication options for chorea in Huntington\'s Disease?' },
  { icon: '🚨', label: 'Dysphagia / ODT', prompt: 'Which HD medications are available in ODT or liquid formulations for patients with swallowing difficulties?' },
  { icon: '🩺', label: 'Baclofen / Rigidity', prompt: 'How is Baclofen titrated for rigidity and dystonia in advanced HD, and what are the precautions?' },
  { icon: '😴', label: 'Sleep & Mood', prompt: 'What are safe medication options for nocturnal agitation, insomnia, and depression in HD?' },
  { icon: '🧬', label: 'Gene Silencing', prompt: 'What is the current status of ASO gene silencing (like Tominersen/WVE-003) and AMT-130 trials?' },
];

function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('neurosense-chat-messages') || '[]');
    } catch { return []; }
  });
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [hasUnread, setHasUnread] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState(null);
  const [chatSessionId, setChatSessionId] = useState(
    () => localStorage.getItem('neurosense-chat-session-id') || ''
  );
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const { t } = useLanguage();

  // Persist messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('neurosense-chat-messages', JSON.stringify(messages));
    } catch { /* storage full */ }
  }, [messages]);

  // Persist session ID to localStorage whenever it changes
  useEffect(() => {
    if (chatSessionId) {
      localStorage.setItem('neurosense-chat-session-id', chatSessionId);
    }
  }, [chatSessionId]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isTyping]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [isOpen]);

  // Show welcome message on first open
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([{
        role: 'assistant',
        content: "Hello! 👋 I'm the **NeuroSense AI Assistant**, specialised in Huntington's Disease clinical symptoms, pharmacology, and platform analysis.\n\nI can help you with:\n- 🩺 **Clinical Symptoms & Staging** — Mayo Clinic criteria: Movement (chorea, dystonia), cognitive, and psychiatric signs\n- 🔬 **Platform AI Analysis** — How symptom selections, CAG repeats, digital tests & MRI predict disease stage\n- 💊 **Medications & Dosing** — Deutetrabenazine, Xenazine, Olanzapine, Baclofen, SSRIs\n- 🚨 **Special Formulations** — Dysphagia-safe ODTs, liquids, crushing rules\n- 🧬 **Genetics & Progression** — CAG repeats, UHDRS, TFC tracking\n- 🔬 **Emerging Therapies** — ASO gene silencing, AMT-130, clinical trials\n\nSelect a topic below or type your question!",
        timestamp: new Date(),
      }]);
    }
  }, [isOpen, messages.length]);

  const toggleChat = () => {
    setIsOpen((prev) => !prev);
    if (!isOpen) setHasUnread(false);
  };

  const copyMessage = (text, idx) => {
    if (!text) return;
    navigator.clipboard?.writeText(text).then(() => {
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx(null), 2000);
    }).catch(() => {});
  };

  // Structured markdown rendering for bold, italics, bullets, numbered lists, code, and notes
  const renderMessage = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    return lines.map((line, i) => {
      let rendered = line;
      // Bold: **text**
      rendered = rendered.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      // Italic: *text* or _text_
      rendered = rendered.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
      // Inline code: `text`
      rendered = rendered.replace(/`(.+?)`/g, '<code>$1</code>');

      const trimmed = line.trim();

      // Educational / disclaimer / warning callout line
      if (trimmed.startsWith('*Educational') || trimmed.startsWith('⚠️') || trimmed.startsWith('⚕️') || trimmed.startsWith('Note:')) {
        return <div key={i} className="chatbot-note" dangerouslySetInnerHTML={{ __html: rendered }} />;
      }

      // Bullet points
      const isBullet = /^\s*[-•*]\s/.test(line);
      if (isBullet) {
        const bulletContent = rendered.replace(/^\s*[-•*]\s/, '');
        return <div key={i} className="chatbot-bullet" dangerouslySetInnerHTML={{ __html: bulletContent }} />;
      }

      // Numbered list
      const isNumbered = /^\s*\d+\.\s/.test(line);
      if (isNumbered) {
        const numContent = rendered.replace(/^\s*\d+\.\s/, '');
        return <div key={i} className="chatbot-numbered" dangerouslySetInnerHTML={{ __html: numContent }} />;
      }

      // Empty line = spacer
      if (trimmed === '') {
        return <div key={i} className="chatbot-spacer" />;
      }

      return <div key={i} className="chatbot-line" dangerouslySetInnerHTML={{ __html: rendered }} />;
    });
  };

  const sendCustomMessage = async (msgText) => {
    const trimmed = (msgText || '').trim();
    if (!trimmed || isTyping) return;

    const userMsg = { role: 'user', content: trimmed, timestamp: new Date() };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput('');
    setIsTyping(true);

    try {
      // Build conversation history for the backend (last 12 messages for conciseness)
      const historyMsgs = updatedMessages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .slice(-12)
        .map((m) => ({ role: m.role, content: m.content }));

      // Call backend /chatbot/chat — which saves to MongoDB
      const res = await fetch(`${API_BASE}/chatbot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmed,
          history: historyMsgs.slice(0, -1), // Exclude the current message (backend adds it)
          session_id: chatSessionId,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `API error (${res.status})`);
      }

      const data = await res.json();
      const reply = data.reply || 'Sorry, I could not generate a response.';

      // Store session_id from backend for conversation continuity
      if (data.session_id) {
        setChatSessionId(data.session_id);
      }

      const assistantMsg = { role: 'assistant', content: reply, timestamp: new Date() };
      setMessages((prev) => [...prev, assistantMsg]);

      // Set unread if chat is closed
      if (!isOpen) setHasUnread(true);
    } catch (err) {
      const errorMsg = {
        role: 'assistant',
        content: `⚠️ Sorry, I couldn't process your request. ${err.message || 'Please try again.'}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const sendMessage = () => {
    sendCustomMessage(input);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    // Start a new conversation — old conversations are preserved in MongoDB
    setMessages([]);
    setChatSessionId('');
    localStorage.removeItem('neurosense-chat-session-id');
    localStorage.removeItem('neurosense-chat-messages');
    // Re-trigger welcome message
    setTimeout(() => {
      setMessages([{
        role: 'assistant',
        content: "Chat cleared! 🔄 How can I help you with Huntington's Disease symptoms, medications, or platform analysis today?",
        timestamp: new Date(),
      }]);
    }, 100);
  };

  const formatTime = (date) => {
    if (!date) return '';
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="chatbot-container">
      {/* Chat Panel */}
      <div className={`chatbot-panel ${isOpen ? 'open' : ''}`}>
        {/* Header */}
        <div className="chatbot-header">
          <div className="chatbot-header-left">
            <div className="chatbot-avatar">
              <span className="chatbot-avatar-icon">🧬</span>
              <span className="chatbot-status-dot" />
            </div>
            <div className="chatbot-header-info">
              <span className="chatbot-header-title">NeuroSense AI</span>
              <span className="chatbot-header-subtitle">
                {isTyping ? 'Consulting clinical knowledge base...' : 'HD Symptoms, Clinical & Medication Assistant'}
              </span>
            </div>
          </div>
          <div className="chatbot-header-actions">
            <button
              className="chatbot-header-btn"
              onClick={clearChat}
              title="Clear conversation"
              aria-label="Clear conversation"
            >
              🗑️
            </button>
            <button
              className="chatbot-header-btn chatbot-close-btn"
              onClick={toggleChat}
              aria-label="Close chat"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="chatbot-messages">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`chatbot-msg ${msg.role === 'user' ? 'chatbot-msg-user' : 'chatbot-msg-assistant'} ${msg.isError ? 'chatbot-msg-error' : ''}`}
            >
              {msg.role === 'assistant' && (
                <div className="chatbot-msg-avatar">🧬</div>
              )}
              <div className="chatbot-msg-bubble">
                <div className="chatbot-msg-content">
                  {renderMessage(msg.content)}
                </div>
                <div className="chatbot-msg-footer">
                  {msg.role === 'assistant' && !msg.isError && (
                    <button
                      className="chatbot-copy-btn"
                      onClick={() => copyMessage(msg.content, i)}
                      title="Copy response"
                    >
                      {copiedIdx === i ? '✓ Copied' : '📋 Copy'}
                    </button>
                  )}
                  <span className="chatbot-msg-time">{formatTime(msg.timestamp)}</span>
                </div>
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {isTyping && (
            <div className="chatbot-msg chatbot-msg-assistant">
              <div className="chatbot-msg-avatar">🧬</div>
              <div className="chatbot-msg-bubble chatbot-typing-bubble">
                <div className="chatbot-typing">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Prompts Bar */}
        <div className="chatbot-quick-prompts">
          <div className="chatbot-quick-prompts-scroll">
            {QUICK_PROMPTS.map((qp, idx) => (
              <button
                key={idx}
                className="chatbot-quick-pill"
                onClick={() => sendCustomMessage(qp.prompt)}
                disabled={isTyping}
                title={qp.prompt}
              >
                <span>{qp.icon}</span> {qp.label}
              </button>
            ))}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="chatbot-disclaimer">
          ⚕️ AI responses are educational & clinical support — not medical advice
        </div>

        {/* Input */}
        <div className="chatbot-input-area">
          <textarea
            ref={inputRef}
            className="chatbot-input"
            placeholder="Ask about HD medications, dosages, or stages..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isTyping}
          />
          <button
            className={`chatbot-send-btn ${input.trim() && !isTyping ? 'active' : ''}`}
            onClick={sendMessage}
            disabled={!input.trim() || isTyping}
            aria-label="Send message"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>

      {/* Floating Action Button */}
      <button
        className={`chatbot-fab ${isOpen ? 'chatbot-fab-active' : ''}`}
        onClick={toggleChat}
        aria-label={isOpen ? 'Close chat' : 'Open AI chat assistant'}
        id="chatbot-fab"
      >
        <span className="chatbot-fab-icon chatbot-fab-icon-chat">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </span>
        <span className="chatbot-fab-icon chatbot-fab-icon-close">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </span>
        {hasUnread && <span className="chatbot-fab-badge" />}
        <span className="chatbot-fab-pulse" />
      </button>
    </div>
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

  const { user } = useAuth();

  const handleSubmit = async (form, mriFile, symptoms = []) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setLastForm(form);

    try {
      const fd = new FormData();
      if (form.cag_repeat) {
        fd.append('cag_repeat', form.cag_repeat);
      }
      fd.append('motor_score', form.motor_score);
      fd.append('memory_score', form.memory_score);
      fd.append('functional_score', '100');
      fd.append('age', form.age);

      // Legacy field fallback for backward compatibility
      if (form.uhdrs_motor != null) fd.append('uhdrs_motor', form.uhdrs_motor);
      if (form.uhdrs_cognitive != null) fd.append('uhdrs_cognitive', form.uhdrs_cognitive);
      if (form.tfc_score != null) fd.append('tfc_score', form.tfc_score);

      // Append symptoms if any were selected
      if (symptoms && symptoms.length > 0) {
        fd.append('symptoms', symptoms.join(','));
      }

      // Attach user ID so prediction is linked to this account
      if (user?._id) {
        fd.append('user_id', user._id);
      }

      // Route to the correct endpoint based on file type
      let endpoint = '/predict';
      if (mriFile) {
        const fname = mriFile.name.toLowerCase();
        if (fname.endsWith('.png') || fname.endsWith('.jpg') || fname.endsWith('.jpeg')) {
          // Image file → use image prediction endpoint
          fd.append('mri_image', mriFile);
          endpoint = '/predict-image';
        } else {
          // NIfTI file → use original endpoint
          fd.append('mri_file', mriFile);
        }
      }

      const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST', body: fd });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.message || err.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      saveAnalysis(form, data, user?._id);
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
   AUTH CONTEXT
   ═══════════════════════════════════════════ */

const AuthContext = createContext({ user: null, setUser: () => {}, logout: () => {} });

function useAuth() {
  return useContext(AuthContext);
}

/* ═══════════════════════════════════════════
   AUTH PAGE (Login / Register)
   ═══════════════════════════════════════════ */

function AuthPage() {
  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const { isDark } = useContext(ThemeContext);
  const navigate = useNavigate();
  const { user, setUser } = useAuth();

  // Redirect if already logged in
  useEffect(() => {
    if (user) navigate('/');
  }, [user, navigate]);

  // Login form state
  const [loginForm, setLoginForm] = useState({ email: '', password: '' });
  // Register form state
  const [registerForm, setRegisterForm] = useState({
    name: '', email: '', password: '', confirmPassword: '',
    age: '', gender: 'male', role: 'patient',
  });

  const switchMode = (newMode) => {
    setMode(newMode);
    setError('');
    setSuccess('');
    setShowPassword(false);
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);

    try {
      const res = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: loginForm.email, password: loginForm.password }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Invalid email or password');
      }

      const data = await res.json();
      localStorage.setItem('neurosense-user', JSON.stringify(data.user));
      setUser(data.user);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    // Validation
    if (registerForm.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }
    if (registerForm.password !== registerForm.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (!registerForm.age || parseFloat(registerForm.age) <= 0) {
      setError('Please enter a valid age');
      return;
    }

    setIsSubmitting(true);

    try {
      const res = await fetch(`${API_BASE}/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: registerForm.name,
          email: registerForm.email,
          password: registerForm.password,
          age: parseFloat(registerForm.age),
          gender: registerForm.gender,
          role: registerForm.role,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Registration failed');
      }

      setSuccess('Account created successfully! You can now log in.');
      setRegisterForm({ name: '', email: '', password: '', confirmPassword: '', age: '', gender: 'male', role: 'patient' });
      setTimeout(() => switchMode('login'), 1500);
    } catch (err) {
      setError(err.message || 'Registration failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  useScrollReveal();

  return (
    <>
      <Particles />
      <div className="auth-page">
        <div className="auth-container reveal visible">
          {/* Left: Branding */}
          <div className="auth-branding">
            <div className="auth-branding-content">
              <div className="auth-branding-logo" onClick={() => navigate('/')}>
                <div className="navbar-logo">🧬</div>
                <span className="navbar-name">NeuroSense</span>
              </div>
              <h1 className="auth-branding-title">
                {mode === 'login' ? 'Welcome Back' : 'Join NeuroSense'}
              </h1>
              <p className="auth-branding-subtitle">
                {mode === 'login'
                  ? 'Sign in to access your HD analysis reports, cognitive assessments, and personalized insights.'
                  : 'Create your account to begin AI-powered Huntington\'s Disease analysis and tracking.'}
              </p>
              <div className="auth-branding-features">
                <div className="auth-feature">
                  <span className="auth-feature-icon">🧠</span>
                  <span>AI-Powered MRI Analysis</span>
                </div>
                <div className="auth-feature">
                  <span className="auth-feature-icon">📊</span>
                  <span>Disease Progression Tracking</span>
                </div>
                <div className="auth-feature">
                  <span className="auth-feature-icon">🔒</span>
                  <span>Secure Clinical Data Storage</span>
                </div>
              </div>
            </div>
            <div className="auth-branding-decoration">
              <div className="auth-deco-circle auth-deco-1" />
              <div className="auth-deco-circle auth-deco-2" />
              <div className="auth-deco-circle auth-deco-3" />
            </div>
          </div>

          {/* Right: Form */}
          <div className="auth-form-panel">
            {/* Tab Switcher */}
            <div className="auth-tabs">
              <button
                className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
                onClick={() => switchMode('login')}
              >
                Sign In
              </button>
              <button
                className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
                onClick={() => switchMode('register')}
              >
                Create Account
              </button>
              <div
                className="auth-tab-indicator"
                style={{ transform: mode === 'register' ? 'translateX(100%)' : 'translateX(0)' }}
              />
            </div>

            {/* Error / Success Messages */}
            {error && (
              <div className="auth-message auth-error">
                <span className="auth-msg-icon">⚠️</span>
                {error}
              </div>
            )}
            {success && (
              <div className="auth-message auth-success">
                <span className="auth-msg-icon">✅</span>
                {success}
              </div>
            )}

            {/* Login Form */}
            {mode === 'login' && (
              <form className="auth-form" onSubmit={handleLogin}>
                <div className="auth-field">
                  <label className="auth-label" htmlFor="login-email">Email Address</label>
                  <div className="auth-input-wrap">
                    <span className="auth-input-icon">✉️</span>
                    <input
                      id="login-email"
                      type="email"
                      className="auth-input"
                      placeholder="you@example.com"
                      value={loginForm.email}
                      onChange={(e) => setLoginForm({ ...loginForm, email: e.target.value })}
                      required
                      autoComplete="email"
                    />
                  </div>
                </div>

                <div className="auth-field">
                  <label className="auth-label" htmlFor="login-password">Password</label>
                  <div className="auth-input-wrap">
                    <span className="auth-input-icon">🔑</span>
                    <input
                      id="login-password"
                      type={showPassword ? 'text' : 'password'}
                      className="auth-input"
                      placeholder="Enter your password"
                      value={loginForm.password}
                      onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                      required
                      autoComplete="current-password"
                    />
                    <button
                      type="button"
                      className="auth-password-toggle"
                      onClick={() => setShowPassword(!showPassword)}
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                    >
                      {showPassword ? '🙈' : '👁️'}
                    </button>
                  </div>
                </div>

                <button
                  type="submit"
                  className="auth-submit"
                  disabled={isSubmitting}
                >
                  {isSubmitting ? (
                    <span className="auth-spinner" />
                  ) : (
                    <>Sign In<span className="auth-submit-arrow">→</span></>
                  )}
                </button>

                <p className="auth-switch-text">
                  Don't have an account?{' '}
                  <button type="button" className="auth-switch-link" onClick={() => switchMode('register')}>
                    Create one
                  </button>
                </p>
              </form>
            )}

            {/* Register Form */}
            {mode === 'register' && (
              <form className="auth-form" onSubmit={handleRegister}>
                <div className="auth-field">
                  <label className="auth-label" htmlFor="reg-name">Full Name</label>
                  <div className="auth-input-wrap">
                    <span className="auth-input-icon">👤</span>
                    <input
                      id="reg-name"
                      type="text"
                      className="auth-input"
                      placeholder="John Doe"
                      value={registerForm.name}
                      onChange={(e) => setRegisterForm({ ...registerForm, name: e.target.value })}
                      required
                      autoComplete="name"
                    />
                  </div>
                </div>

                <div className="auth-field">
                  <label className="auth-label" htmlFor="reg-email">Email Address</label>
                  <div className="auth-input-wrap">
                    <span className="auth-input-icon">✉️</span>
                    <input
                      id="reg-email"
                      type="email"
                      className="auth-input"
                      placeholder="you@example.com"
                      value={registerForm.email}
                      onChange={(e) => setRegisterForm({ ...registerForm, email: e.target.value })}
                      required
                      autoComplete="email"
                    />
                  </div>
                </div>

                <div className="auth-row">
                  <div className="auth-field">
                    <label className="auth-label" htmlFor="reg-age">Age</label>
                    <div className="auth-input-wrap">
                      <span className="auth-input-icon">🎂</span>
                      <input
                        id="reg-age"
                        type="number"
                        className="auth-input"
                        placeholder="25"
                        min="1"
                        max="120"
                        value={registerForm.age}
                        onChange={(e) => setRegisterForm({ ...registerForm, age: e.target.value })}
                        required
                      />
                    </div>
                  </div>
                  <div className="auth-field">
                    <label className="auth-label" htmlFor="reg-gender">Gender</label>
                    <div className="auth-input-wrap auth-select-wrap">
                      <span className="auth-input-icon">⚧️</span>
                      <select
                        id="reg-gender"
                        className="auth-input auth-select"
                        value={registerForm.gender}
                        onChange={(e) => setRegisterForm({ ...registerForm, gender: e.target.value })}
                      >
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="auth-field">
                  <label className="auth-label" htmlFor="reg-password">Password</label>
                  <div className="auth-input-wrap">
                    <span className="auth-input-icon">🔑</span>
                    <input
                      id="reg-password"
                      type={showPassword ? 'text' : 'password'}
                      className="auth-input"
                      placeholder="Min 6 characters"
                      value={registerForm.password}
                      onChange={(e) => setRegisterForm({ ...registerForm, password: e.target.value })}
                      required
                      minLength={6}
                      autoComplete="new-password"
                    />
                    <button
                      type="button"
                      className="auth-password-toggle"
                      onClick={() => setShowPassword(!showPassword)}
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                    >
                      {showPassword ? '🙈' : '👁️'}
                    </button>
                  </div>
                </div>

                <div className="auth-field">
                  <label className="auth-label" htmlFor="reg-confirm">Confirm Password</label>
                  <div className="auth-input-wrap">
                    <span className="auth-input-icon">🔐</span>
                    <input
                      id="reg-confirm"
                      type={showPassword ? 'text' : 'password'}
                      className="auth-input"
                      placeholder="Repeat password"
                      value={registerForm.confirmPassword}
                      onChange={(e) => setRegisterForm({ ...registerForm, confirmPassword: e.target.value })}
                      required
                      minLength={6}
                      autoComplete="new-password"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  className="auth-submit"
                  disabled={isSubmitting}
                >
                  {isSubmitting ? (
                    <span className="auth-spinner" />
                  ) : (
                    <>Create Account<span className="auth-submit-arrow">→</span></>
                  )}
                </button>

                <p className="auth-switch-text">
                  Already have an account?{' '}
                  <button type="button" className="auth-switch-link" onClick={() => switchMode('login')}>
                    Sign in
                  </button>
                </p>
              </form>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════
   APP ROOT (Router + Theme + Auth)
   ═══════════════════════════════════════════ */

export default function App() {
  const theme = useTheme();
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('neurosense-user');
      return saved ? JSON.parse(saved) : null;
    } catch { return null; }
  });

  const logout = useCallback(() => {
    localStorage.removeItem('neurosense-user');
    setUser(null);
  }, []);

  return (
    <LanguageProvider>
      <ThemeContext.Provider value={theme}>
        <AuthContext.Provider value={{ user, setUser, logout }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<AuthPage />} />
            <Route path="/memory-test" element={<MemoryTestPage />} />
            <Route path="/motor-test" element={<MotorTestPage />} />
            <Route path="/progression" element={<DiseaseProgressionPage />} />
            <Route path="/history" element={<AnalysisHistoryPage />} />
            <Route path="/medications" element={<MedicationsPage />} />
          </Routes>
          <ChatBot />
        </AuthContext.Provider>
      </ThemeContext.Provider>
    </LanguageProvider>
  );
}

