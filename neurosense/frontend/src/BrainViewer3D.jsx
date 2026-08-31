/* ═══════════════════════════════════════════════════════════
   Interactive 3D Brain Viewer — NeuroSense
   
   Three.js-powered anatomical brain visualization. Styled as an
   exact replica of a realistic human anatomical specimen, featuring
   organic fleshy/pinkish textures with procedurally generated blood
   vessels, realistic gyri/sulci folds, a detailed cerebellum, and
   an interactive translucency slider to reveal internal HD-affected
   structures (caudate/putamen).
   ═══════════════════════════════════════════════════════════ */

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';

/* ─── 3D Perlin Noise Implementation for Realistic Gyri/Sulci ─── */
const permutation = [
  151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,
  125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,
  105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,
  82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,
  153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,
  84,201,127,150,27,33,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,
  100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,
  16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,
  22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,
  191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,84,201,127,150,27,33,166,77,
  146,158,231,83,111
];

const p = new Array(512);
for (let i = 0; i < 256; i++) {
  p[256 + i] = p[i] = permutation[i];
}

function fade(t) {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(t, a, b) {
  return a + t * (b - a);
}

function grad(hash, x, y, z) {
  const h = hash & 15;
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
  return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}

function noise3D(x, y, z) {
  const X = Math.floor(x) & 255;
  const Y = Math.floor(y) & 255;
  const Z = Math.floor(z) & 255;
  
  x -= Math.floor(x);
  y -= Math.floor(y);
  z -= Math.floor(z);
  
  const u = fade(x);
  const v = fade(y);
  const w = fade(z);
  
  const A = p[X] + Y;
  const AA = p[A] + Z;
  const AB = p[A + 1] + Z;
  const B = p[X + 1] + Y;
  const BB = p[B] + Z;
  const BC = p[B + 1] + Z;
  
  return lerp(w,
    lerp(v,
      lerp(u, grad(p[AA], x, y, z), grad(p[BB], x - 1, y, z)),
      lerp(u, grad(p[AB], x, y - 1, z), grad(p[BC], x - 1, y - 1, z))
    ),
    lerp(v,
      lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BB + 1], x - 1, y, z - 1)),
      lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BC + 1], x - 1, y - 1, z - 1))
    )
  );
}

/* ─── Procedural Vein and Fleshy Tissue Canvas Texture ─── */
function createBrainTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  // Base fleshy pink/beige tone
  ctx.fillStyle = '#e8b5a3';
  ctx.fillRect(0, 0, 1024, 1024);

  // Subtle color noise / blotches for realistic organic variation
  for (let i = 0; i < 200; i++) {
    const x = Math.random() * 1024;
    const y = Math.random() * 1024;
    const r = 80 + Math.random() * 160;
    const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, 'rgba(215, 120, 105, 0.4)');
    grad.addColorStop(1, 'rgba(215, 120, 105, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw organic capillary/vein structures
  ctx.strokeStyle = 'rgba(145, 30, 42, 0.4)';
  ctx.lineCap = 'round';
  
  for (let i = 0; i < 45; i++) {
    let x = Math.random() * 1024;
    let y = Math.random() * 1024;
    ctx.lineWidth = 1.0 + Math.random() * 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y);

    const length = 100 + Math.random() * 200;
    let angle = Math.random() * Math.PI * 2;
    for (let j = 0; j < length; j += 6) {
      angle += (Math.random() - 0.5) * 0.45;
      x += Math.cos(angle) * 6;
      y += Math.sin(angle) * 6;
      ctx.lineTo(x, y);

      if (Math.random() < 0.04) {
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
        let branchAngle = angle + (Math.random() - 0.5) * 1.4;
        let bx = x;
        let by = y;
        ctx.lineWidth = ctx.lineWidth * 0.7;
        for (let k = 0; k < 35; k += 5) {
          branchAngle += (Math.random() - 0.5) * 0.35;
          bx += Math.cos(branchAngle) * 5;
          by += Math.sin(branchAngle) * 5;
          ctx.lineTo(bx, by);
        }
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineWidth = 1.0 + Math.random() * 1.5;
      }
    }
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(1.5, 1.5);
  return texture;
}

/* ─── HD Brain Region Definitions (Internal Disease Areas) ─── */
const BRAIN_REGIONS = [
  {
    name: 'Caudate Nucleus (Right)',
    description: 'Primary target of HD neurodegeneration. CAG repeat expansion causes progressive atrophy, leading to motor control deficits.',
    position: [0.08, 0.12, 0.08],
    size: 0.11,
    severity: 'critical',
    color: '#ff3344',
    glowColor: '#ff0022',
  },
  {
    name: 'Caudate Nucleus (Left)',
    description: 'Bilateral degeneration of caudate nuclei is a hallmark of HD. Atrophy correlates with CAG repeat length.',
    position: [-0.08, 0.12, 0.08],
    size: 0.11,
    severity: 'critical',
    color: '#ff3344',
    glowColor: '#ff0022',
  },
  {
    name: 'Putamen (Right)',
    description: 'Part of the striatum; severely affected in HD. Degeneration causes chorea and dystonia.',
    position: [0.15, 0.06, 0.04],
    size: 0.12,
    severity: 'critical',
    color: '#ff4455',
    glowColor: '#ff2233',
  },
  {
    name: 'Putamen (Left)',
    description: 'Striatal neuronal loss in the putamen is among the earliest changes detectable on MRI in HD.',
    position: [-0.15, 0.06, 0.04],
    size: 0.12,
    severity: 'critical',
    color: '#ff4455',
    glowColor: '#ff2233',
  },
  {
    name: 'Frontal Cortex',
    description: 'Cortical thinning in frontal regions contributes to executive dysfunction and personality changes in HD.',
    position: [0, 0.3, 0.35],
    size: 0.2,
    severity: 'moderate',
    color: '#ff8833',
    glowColor: '#ff6600',
  },
  {
    name: 'Prefrontal Cortex',
    description: 'Involved in cognitive decline, impaired decision-making, and behavioral symptoms in HD patients.',
    position: [0, 0.22, 0.42],
    size: 0.16,
    severity: 'moderate',
    color: '#ffaa33',
    glowColor: '#ff8800',
  },
  {
    name: 'Globus Pallidus (Right)',
    description: 'Basal ganglia structure affected in later stages. Contributes to rigidity and bradykinesia.',
    position: [0.12, 0.02, 0.02],
    size: 0.07,
    severity: 'moderate',
    color: '#ff9944',
    glowColor: '#ff7722',
  },
  {
    name: 'Globus Pallidus (Left)',
    description: 'Bilateral pallidal involvement is associated with advanced HD motor symptoms.',
    position: [-0.12, 0.02, 0.02],
    size: 0.07,
    severity: 'moderate',
    color: '#ff9944',
    glowColor: '#ff7722',
  },
  {
    name: 'Temporal Cortex (Right)',
    description: 'Temporal lobe atrophy contributes to memory impairment and language difficulties in HD.',
    position: [0.35, -0.12, 0.12],
    size: 0.14,
    severity: 'mild',
    color: '#ffcc44',
    glowColor: '#ffaa00',
  },
  {
    name: 'Temporal Cortex (Left)',
    description: 'Bilateral temporal changes visible in moderate-to-advanced HD stages.',
    position: [-0.35, -0.12, 0.12],
    size: 0.14,
    severity: 'mild',
    color: '#ffcc44',
    glowColor: '#ffaa00',
  },
  {
    name: 'Cerebellum',
    description: 'Cerebellar involvement in late-stage HD contributes to gait ataxia and coordination problems.',
    position: [0, -0.32, -0.3],
    size: 0.2,
    severity: 'mild',
    color: '#ffdd55',
    glowColor: '#ffcc00',
  },
  {
    name: 'Thalamus',
    description: 'Thalamic neuronal loss affects sensory relay and contributes to sleep disturbances in HD.',
    position: [0, 0.08, -0.06],
    size: 0.1,
    severity: 'moderate',
    color: '#ff8844',
    glowColor: '#ff6622',
  },
];

const SEVERITY_CONFIG = {
  critical: { label: 'Critical (Striatal)', color: '#ff3344' },
  moderate: { label: 'Moderate (Cortical/BG)', color: '#ff8833' },
  mild: { label: 'Mild (Secondary)', color: '#ffcc44' },
};

/* ─── Deform Cerebrum Hemispheres with organic winding folds ─── */
function deformHemisphere(geometry, isLeft) {
  const positions = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  const sideSign = isLeft ? -1 : 1;

  for (let i = 0; i < positions.count; i++) {
    vertex.fromBufferAttribute(positions, i);

    // Realistic scaling: total brain width ~1.1, height ~0.95, length ~1.2
    vertex.x *= 0.90;
    vertex.y *= 0.95;
    vertex.z *= 1.15;

    // Fissure division: keep a tiny slit, but allow them to meet near 0
    const medialDist = vertex.x * sideSign;
    if (medialDist < 0.005) {
      vertex.x = 0.005 * sideSign;
    }

    // Anatomical taper profiles
    if (vertex.z > 0.3) {
      vertex.x *= (1.0 - (vertex.z - 0.3) * 0.25);
      vertex.y *= (1.0 - (vertex.z - 0.3) * 0.18);
    } else if (vertex.z < -0.4) {
      vertex.x *= (1.0 - (-vertex.z - 0.4) * 0.35);
      vertex.y *= (1.0 - (-vertex.z - 0.4) * 0.28);
    }

    // Generate winding serpentine cortical folds
    const noiseVal = noise3D(vertex.x * 3.5, vertex.y * 3.5, vertex.z * 3.5);
    const ridgePattern = Math.sin(noiseVal * 10.0);
    const gyrus = Math.pow(Math.max(0, ridgePattern), 0.7) * 0.075;
    const sulcus = Math.pow(Math.max(0, -ridgePattern), 0.75) * 0.065;
    const displacement = gyrus - sulcus;

    const boundarySuppressor = Math.min(1.0, Math.abs(vertex.x) * 8.0);
    const normal = vertex.clone().normalize();
    vertex.addScaledVector(normal, displacement * boundarySuppressor);

    positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }

  geometry.computeVertexNormals();
  return geometry;
}

/* ─── Deform Temporal Lobes ─── */
function deformTemporalLobe(geometry, isLeft) {
  const positions = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  
  for (let i = 0; i < positions.count; i++) {
    vertex.fromBufferAttribute(positions, i);

    vertex.x *= 0.35;
    vertex.y *= 0.30;
    vertex.z *= 0.50;

    const noiseVal = noise3D(vertex.x * 4.0, vertex.y * 4.0, vertex.z * 4.0);
    const ridgePattern = Math.sin(noiseVal * 8.0);
    const displacement = Math.pow(Math.max(0, ridgePattern), 0.7) * 0.038 - Math.pow(Math.max(0, -ridgePattern), 0.7) * 0.03;

    const normal = vertex.clone().normalize();
    vertex.addScaledVector(normal, displacement);

    positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

/* ─── Deform Cerebellum ─── */
function deformCerebellum(geometry) {
  const positions = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  
  for (let i = 0; i < positions.count; i++) {
    vertex.fromBufferAttribute(positions, i);

    vertex.x *= 0.80;
    vertex.y *= 0.40;
    vertex.z *= 0.55;

    const wave = Math.sin(vertex.y * 32.0) * 0.016;
    const normal = vertex.clone().normalize();
    vertex.addScaledVector(normal, wave);

    positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

/* ─── Main Component ─── */
export default function BrainViewer3D({ stage, visible = true }) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const brainGroupRef = useRef(null);
  const regionMeshesRef = useRef([]);
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());
  const frameRef = useRef(null);
  
  const isDraggingRef = useRef(false);
  const previousMouseRef = useRef({ x: 0, y: 0 });
  const rotationRef = useRef({ x: 0.2, y: -0.4 });
  const targetRotationRef = useRef({ x: 0.2, y: -0.4 });
  const autoRotateRef = useRef(true);
  const zoomRef = useRef(2.8);
  const targetZoomRef = useRef(2.8);

  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [isAutoRotating, setIsAutoRotating] = useState(true);
  const [transparency, setTransparency] = useState(0.85);

  const getRegionsForStage = useCallback((stageType) => {
    switch (stageType) {
      case 'advanced':
        return BRAIN_REGIONS;
      case 'early':
        return BRAIN_REGIONS.filter((r) => r.severity === 'critical' || r.severity === 'moderate');
      case 'pre_manifest':
        return BRAIN_REGIONS.filter((r) => r.severity === 'critical');
      default:
        return BRAIN_REGIONS;
    }
  }, []);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current || !visible) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = null;
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 0.1, zoomRef.current);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xfff5ea, 1.5);
    mainLight.position.set(2, 4, 3);
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0xadc7f5, 0.55);
    fillLight.position.set(-3, -1, 1);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xfff5ee, 0.9);
    rimLight.position.set(0, 2, -4);
    scene.add(rimLight);

    const brainTexture = createBrainTexture();

    // Brain Group
    const brainGroup = new THREE.Group();
    scene.add(brainGroup);
    brainGroupRef.current = brainGroup;

    // Materials
    const brainspecimenMaterial = new THREE.MeshPhysicalMaterial({
      map: brainTexture,
      bumpMap: brainTexture,
      bumpScale: 0.015,
      color: 0xffe2d5,
      roughness: 0.42,
      metalness: 0.05,
      clearcoat: 0.55,
      clearcoatRoughness: 0.25,
      transparent: true,
      opacity: transparency,
      side: THREE.DoubleSide,
    });

    const cerebellumMaterial = new THREE.MeshPhysicalMaterial({
      map: brainTexture,
      bumpMap: brainTexture,
      bumpScale: 0.012,
      color: 0xca8c7e,
      roughness: 0.52,
      metalness: 0.05,
      clearcoat: 0.35,
      transparent: true,
      opacity: transparency,
      side: THREE.DoubleSide,
    });

    const brainstemMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xe3aba0,
      roughness: 0.5,
      clearcoat: 0.3,
      transparent: true,
      opacity: transparency,
      side: THREE.DoubleSide,
    });

    // 1. CEREBRUM HEMISPHERES (Brought close to 0.015 for realistic midline fissure)
    const rightCerebrumGeo = deformHemisphere(new THREE.SphereGeometry(0.5, 64, 48), false);
    const rightCerebrum = new THREE.Mesh(rightCerebrumGeo, brainspecimenMaterial);
    rightCerebrum.position.x = 0.015;
    rightCerebrum.position.y = 0.12;
    brainGroup.add(rightCerebrum);

    const leftCerebrumGeo = deformHemisphere(new THREE.SphereGeometry(0.5, 64, 48), true);
    const leftCerebrum = new THREE.Mesh(leftCerebrumGeo, brainspecimenMaterial);
    leftCerebrum.position.x = -0.015;
    leftCerebrum.position.y = 0.12;
    brainGroup.add(leftCerebrum);

    // 2. TEMPORAL LOBES (Brought closer to center)
    const rightTemporalGeo = deformTemporalLobe(new THREE.SphereGeometry(0.5, 40, 30), false);
    const rightTemporal = new THREE.Mesh(rightTemporalGeo, brainspecimenMaterial);
    rightTemporal.position.set(0.28, -0.10, 0.18);
    rightTemporal.rotation.set(0.1, 0, -0.25);
    brainGroup.add(rightTemporal);

    const leftTemporalGeo = deformTemporalLobe(new THREE.SphereGeometry(0.5, 40, 30), true);
    const leftTemporal = new THREE.Mesh(leftTemporalGeo, brainspecimenMaterial);
    leftTemporal.position.set(-0.28, -0.10, 0.18);
    leftTemporal.rotation.set(0.1, 0, 0.25);
    brainGroup.add(leftTemporal);

    // 3. CEREBELLUM
    const cerebellumGeo = deformCerebellum(new THREE.SphereGeometry(0.5, 48, 36));
    const cerebellum = new THREE.Mesh(cerebellumGeo, cerebellumMaterial);
    cerebellum.position.set(0, -0.34, -0.35);
    cerebellum.rotation.x = -0.15;
    brainGroup.add(cerebellum);

    // 4. BRAINSTEM + PONS
    const brainstemGeo = new THREE.CylinderGeometry(0.18, 0.12, 0.70, 32);
    const bsPositions = brainstemGeo.attributes.position;
    const bsVertex = new THREE.Vector3();
    for (let i = 0; i < bsPositions.count; i++) {
      bsVertex.fromBufferAttribute(bsPositions, i);
      
      const angle = Math.atan2(bsVertex.z, bsVertex.x);
      const stripe = Math.sin(angle * 12) * 0.005;
      bsVertex.x += Math.cos(angle) * stripe;
      bsVertex.z += Math.sin(angle) * stripe;

      if (bsVertex.y > 0.12 && bsVertex.z > 0) {
        bsVertex.z += 0.055;
        bsVertex.x *= 1.15;
      }

      bsPositions.setXYZ(i, bsVertex.x, bsVertex.y, bsVertex.z);
    }
    brainstemGeo.computeVertexNormals();

    const brainstem = new THREE.Mesh(brainstemGeo, brainstemMaterial);
    brainstem.position.set(0, -0.45, -0.10);
    brainstem.rotation.x = 0.15;
    brainGroup.add(brainstem);

    // 5. INTERNAL DISEASE REGIONS
    const activeRegions = getRegionsForStage(stage);
    const regionMeshes = [];

    activeRegions.forEach((region) => {
      const regionGroup = new THREE.Group();

      // Core sphere
      const coreGeo = new THREE.SphereGeometry(region.size * 0.6, 24, 24);
      const coreMat = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(region.color),
        emissive: new THREE.Color(region.color),
        emissiveIntensity: 0.8,
        transparent: true,
        opacity: 0.9,
        roughness: 0.25,
        metalness: 0.1,
      });
      const coreMesh = new THREE.Mesh(coreGeo, coreMat);
      regionGroup.add(coreMesh);

      // Outer glow
      const glowGeo = new THREE.SphereGeometry(region.size, 20, 20);
      const glowMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(region.glowColor),
        transparent: true,
        opacity: 0.22,
        side: THREE.BackSide,
      });
      const glowMesh = new THREE.Mesh(glowGeo, glowMat);
      regionGroup.add(glowMesh);

      // Pulsing outer ring
      const ringGeo = new THREE.RingGeometry(region.size * 0.72, region.size * 0.88, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(region.color),
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
      });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.lookAt(camera.position);
      regionGroup.add(ringMesh);

      regionGroup.position.set(...region.position);
      regionGroup.userData = { region, coreMat, glowMat, ringMat };

      brainGroup.add(regionGroup);
      regionMeshes.push(regionGroup);
    });

    regionMeshesRef.current = regionMeshes;

    // Animation loop
    let time = 0;
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      time += 0.016;

      if (autoRotateRef.current) {
        targetRotationRef.current.y += 0.0035;
      }

      rotationRef.current.x += (targetRotationRef.current.x - rotationRef.current.x) * 0.08;
      rotationRef.current.y += (targetRotationRef.current.y - rotationRef.current.y) * 0.08;
      zoomRef.current += (targetZoomRef.current - zoomRef.current) * 0.08;

      brainGroup.rotation.x = rotationRef.current.x;
      brainGroup.rotation.y = rotationRef.current.y;
      camera.position.z = zoomRef.current;

      regionMeshes.forEach((group, idx) => {
        const { coreMat, glowMat, ringMat } = group.userData;
        const pulse = Math.sin(time * 2.5 + idx * 0.8) * 0.5 + 0.5;

        coreMat.emissiveIntensity = 0.5 + pulse * 0.5;
        coreMat.opacity = 0.75 + pulse * 0.15;
        glowMat.opacity = 0.15 + pulse * 0.12;

        const ring = group.children[2];
        if (ring) {
          ring.lookAt(camera.position);
          ringMat.opacity = 0.2 + pulse * 0.25;
          ring.scale.setScalar(1.0 + pulse * 0.18);
        }

        group.children[0].scale.setScalar(0.92 + pulse * 0.12);
      });

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameRef.current);
      renderer.dispose();
      rightCerebrumGeo.dispose();
      leftCerebrumGeo.dispose();
      rightTemporalGeo.dispose();
      leftTemporalGeo.dispose();
      cerebellumGeo.dispose();
      brainstemGeo.dispose();
      brainTexture.dispose();
      brainspecimenMaterial.dispose();
      cerebellumMaterial.dispose();
      brainstemMaterial.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [visible, stage, getRegionsForStage, transparency]);

  // Handle transparency slider update
  useEffect(() => {
    if (!brainGroupRef.current) return;
    brainGroupRef.current.children.forEach((mesh) => {
      if (mesh.material && mesh.material.transparent) {
        mesh.material.opacity = transparency;
      }
    });
  }, [transparency]);

  // Mouse interaction handlers
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleMouseDown = (e) => {
      isDraggingRef.current = true;
      autoRotateRef.current = false;
      setIsAutoRotating(false);
      previousMouseRef.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e) => {
      if (isDraggingRef.current) {
        const dx = e.clientX - previousMouseRef.current.x;
        const dy = e.clientY - previousMouseRef.current.y;
        targetRotationRef.current.y += dx * 0.008;
        targetRotationRef.current.x += dy * 0.008;
        targetRotationRef.current.x = Math.max(-1.0, Math.min(1.0, targetRotationRef.current.x));
        previousMouseRef.current = { x: e.clientX, y: e.clientY };
      }

      const rect = container.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      if (cameraRef.current && sceneRef.current) {
        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
        const regionObjects = regionMeshesRef.current.flatMap((g) => g.children);
        const intersects = raycasterRef.current.intersectObjects(regionObjects, false);

        if (intersects.length > 0) {
          const parentGroup = intersects[0].object.parent;
          if (parentGroup?.userData?.region) {
            setHoveredRegion(parentGroup.userData.region);
            container.style.cursor = 'pointer';
          }
        } else {
          setHoveredRegion(null);
          container.style.cursor = isDraggingRef.current ? 'grabbing' : 'grab';
        }
      }
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
      container.style.cursor = 'grab';
    };

    const handleClick = (e) => {
      if (!cameraRef.current || !sceneRef.current) return;
      const rect = container.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      raycasterRef.current.setFromCamera(mouse, cameraRef.current);
      const regionObjects = regionMeshesRef.current.flatMap((g) => g.children);
      const intersects = raycasterRef.current.intersectObjects(regionObjects, false);
      if (intersects.length > 0) {
        const parentGroup = intersects[0].object.parent;
        if (parentGroup?.userData?.region) {
          setSelectedRegion(parentGroup.userData.region);
        }
      } else {
        setSelectedRegion(null);
      }
    };

    const handleWheel = (e) => {
      e.preventDefault();
      targetZoomRef.current += e.deltaY * 0.003;
      targetZoomRef.current = Math.max(1.4, Math.min(4.5, targetZoomRef.current));
    };

    let touchStartDistance = 0;
    const handleTouchStart = (e) => {
      if (e.touches.length === 1) {
        isDraggingRef.current = true;
        autoRotateRef.current = false;
        setIsAutoRotating(false);
        previousMouseRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        touchStartDistance = Math.sqrt(dx * dx + dy * dy);
      }
    };

    const handleTouchMove = (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && isDraggingRef.current) {
        const dx = e.touches[0].clientX - previousMouseRef.current.x;
        const dy = e.touches[0].clientY - previousMouseRef.current.y;
        targetRotationRef.current.y += dx * 0.008;
        targetRotationRef.current.x += dy * 0.008;
        targetRotationRef.current.x = Math.max(-1.0, Math.min(1.0, targetRotationRef.current.x));
        previousMouseRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const delta = touchStartDistance - dist;
        targetZoomRef.current += delta * 0.01;
        targetZoomRef.current = Math.max(1.4, Math.min(4.5, targetZoomRef.current));
        touchStartDistance = dist;
      }
    };

    const handleTouchEnd = () => {
      isDraggingRef.current = false;
    };

    container.addEventListener('mousedown', handleMouseDown);
    container.addEventListener('mousemove', handleMouseMove);
    container.addEventListener('mouseup', handleMouseUp);
    container.addEventListener('mouseleave', handleMouseUp);
    container.addEventListener('click', handleClick);
    container.addEventListener('wheel', handleWheel, { passive: false });
    container.addEventListener('touchstart', handleTouchStart, { passive: false });
    container.addEventListener('touchmove', handleTouchMove, { passive: false });
    container.addEventListener('touchend', handleTouchEnd);

    return () => {
      container.removeEventListener('mousedown', handleMouseDown);
      container.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('mouseup', handleMouseUp);
      container.removeEventListener('mouseleave', handleMouseUp);
      container.removeEventListener('click', handleClick);
      container.removeEventListener('wheel', handleWheel);
      container.removeEventListener('touchstart', handleTouchStart);
      container.removeEventListener('touchmove', handleTouchMove);
      container.removeEventListener('touchend', handleTouchEnd);
    };
  }, [visible]);

  const toggleAutoRotate = () => {
    autoRotateRef.current = !autoRotateRef.current;
    setIsAutoRotating(autoRotateRef.current);
  };

  const resetView = () => {
    targetRotationRef.current = { x: 0.2, y: -0.4 };
    targetZoomRef.current = 2.8;
    autoRotateRef.current = true;
    setIsAutoRotating(true);
    setSelectedRegion(null);
  };

  if (!visible) return null;

  const activeRegions = getRegionsForStage(stage);
  const severities = [...new Set(activeRegions.map((r) => r.severity))];

  return (
    <div className="brain-viewer-container">
      {/* Header */}
      <div className="brain-viewer-header">
        <div className="brain-viewer-title-area">
          <span className="brain-viewer-icon">🧠</span>
          <div>
            <h3 className="brain-viewer-title">Interactive 3D Brain Viewer</h3>
            <p className="brain-viewer-subtitle">
              Huntington's Disease affected regions •{' '}
              <span className="brain-viewer-stage-tag" data-stage={stage}>
                {stage === 'pre_manifest' ? 'Pre-manifest' : stage === 'early' ? 'Early HD' : 'Advanced HD'}
              </span>
            </p>
          </div>
        </div>
        <div className="brain-viewer-controls">
          {/* Transparency slider */}
          <div className="brain-transparency-ctrl">
            <span className="brain-slider-label">Translucency</span>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={transparency}
              onChange={(e) => setTransparency(parseFloat(e.target.value))}
              title="Adjust outer brain translucency to view inner regions"
              className="brain-slider"
            />
          </div>
          <button
            className={`brain-ctrl-btn ${isAutoRotating ? 'active' : ''}`}
            onClick={toggleAutoRotate}
            title={isAutoRotating ? 'Stop auto-rotation' : 'Start auto-rotation'}
          >
            {isAutoRotating ? '⏸' : '▶'}
          </button>
          <button className="brain-ctrl-btn" onClick={resetView} title="Reset view">
            ↺
          </button>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="brain-viewer-canvas-wrap">
        <div ref={containerRef} className="brain-viewer-canvas" style={{ cursor: 'grab' }} />

        {/* Hover tooltip */}
        {hoveredRegion && !selectedRegion && (
          <div className="brain-hover-tooltip">
            <span className="brain-tooltip-severity" data-severity={hoveredRegion.severity} />
            <span>{hoveredRegion.name}</span>
            <span className="brain-tooltip-hint">Click for details</span>
          </div>
        )}

        {/* Interaction hint */}
        <div className="brain-viewer-hint">
          <span>🖱️ Drag to rotate • Scroll to zoom • Click glowing nodes to inspect</span>
        </div>
      </div>

      {/* Selected region detail panel */}
      {selectedRegion && (
        <div className="brain-region-detail">
          <div className="brain-detail-header">
            <span className="brain-detail-severity" data-severity={selectedRegion.severity} />
            <h4 className="brain-detail-name">{selectedRegion.name}</h4>
            <button className="brain-detail-close" onClick={() => setSelectedRegion(null)}>✕</button>
          </div>
          <p className="brain-detail-desc">{selectedRegion.description}</p>
          <div className="brain-detail-meta">
            <span className="brain-detail-tag" data-severity={selectedRegion.severity}>
              {SEVERITY_CONFIG[selectedRegion.severity]?.label}
            </span>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="brain-viewer-legend">
        <div className="brain-legend-title">Affected Regions</div>
        <div className="brain-legend-items">
          {severities.map((sev) => (
            <div className="brain-legend-item" key={sev}>
              <span className="brain-legend-dot" style={{ background: SEVERITY_CONFIG[sev]?.color }} />
              <span className="brain-legend-label">{SEVERITY_CONFIG[sev]?.label}</span>
            </div>
          ))}
        </div>
        <div className="brain-legend-count">
          {activeRegions.length} region{activeRegions.length !== 1 ? 's' : ''} affected
        </div>
      </div>
    </div>
  );
}
