// AgriScout 3D viewer: real Musanze terrain + live drone state over WebSocket.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const WORLD = 800;            // world units across the whole terrain patch
const PITCH = 20;             // distance between field centres
const FIELD = 17;             // rendered size of one field tile
const TILE_OPACITY = 0.35;    // fields blend with the real satellite imagery beneath
const DRONE_ALTITUDE = 19;
const DRONE_SCALE = 2.3;
const STRESS = 0.5;

let scene, camera, renderer, controls, clock;
let heightData = null, heightSize = 0, heightScale = 1, heightMin = 0;
let fieldTiles = [], scanRings = [], basePad, baseGlow;
let drone, props = [], droneBeam, droneMarker, navLights = [];
let effects = [];
let gridN = 8, state = null, prev = null;
let droneTarget = new THREE.Vector3(), droneYaw = 0;

init();

async function init() {
  // Build the scene, terrain, drone and start the render + WebSocket loops.
  scene = new THREE.Scene();
  clock = new THREE.Clock();

  camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 0.1, 4000);
  camera.position.set(110, 112, 148);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 18, 0);
  controls.maxPolarAngle = Math.PI * 0.487;
  controls.minDistance = 45;
  controls.maxDistance = 520;
  controls.enableDamping = true;

  buildSky();
  buildLights();
  await buildTerrain();
  buildGrid(gridN);
  buildDrone();
  connect();
  animate();
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
}

function buildSky() {
  // Vertical gradient sky dome plus matching distance fog.
  const cv = Object.assign(document.createElement("canvas"), { width: 2, height: 256 });
  const ctx = cv.getContext("2d");
  const grad = ctx.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0.0, "#2f6fb5");
  grad.addColorStop(0.45, "#87b6e2");
  grad.addColorStop(0.75, "#cfe2f2");
  grad.addColorStop(1.0, "#e8eef2");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 2, 256);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(1800, 32, 16),
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, depthWrite: false })
  );
  scene.add(sky);
  scene.fog = new THREE.Fog(0xbcd4e8, 620, 1700);
}

function buildLights() {
  // Warm key light with soft shadows, plus sky/ground bounce.
  const sun = new THREE.DirectionalLight(0xfff0d6, 2.5);
  sun.position.set(-230, 330, 170);
  sun.castShadow = true;
  sun.shadow.camera.left = sun.shadow.camera.bottom = -190;
  sun.shadow.camera.right = sun.shadow.camera.top = 190;
  sun.shadow.camera.far = 900;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0008;
  sun.shadow.normalBias = 0.6;
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0xbfdcff, 0x4a5f3a, 1.0));
  scene.add(new THREE.AmbientLight(0xffffff, 0.25));
}

async function buildTerrain() {
  // Displace a plane with real terrarium elevation and drape satellite imagery.
  let texture;
  try {
    const img = await loadImage("terrain/terrain.png");
    const cv = Object.assign(document.createElement("canvas"), { width: img.width, height: img.height });
    const ctx = cv.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const px = ctx.getImageData(0, 0, img.width, img.height).data;
    heightSize = img.width;
    heightData = new Float32Array(heightSize * heightSize);
    let min = Infinity;
    for (let i = 0; i < heightData.length; i++) {
      heightData[i] = px[i * 4] * 256 + px[i * 4 + 1] + px[i * 4 + 2] / 256 - 32768;
      min = Math.min(min, heightData[i]);
    }
    heightMin = min;
    heightScale = (WORLD / (heightSize * 37.5)) * 1.45;
    texture = new THREE.TextureLoader().load("terrain/satellite.jpg");
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  } catch {
    makeFallbackHills();
    texture = makeFallbackTexture();
  }

  const geo = new THREE.PlaneGeometry(WORLD, WORLD, 255, 255);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, sampleHeight(pos.getX(i), pos.getZ(i)));
  geo.computeVertexNormals();
  const terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    map: texture, roughness: 0.96, metalness: 0.0,
  }));
  terrain.receiveShadow = true;
  scene.add(terrain);
}

function sampleHeight(x, z) {
  // Bilinear-sample the elevation grid at a world coordinate.
  if (!heightData) return 0;
  const u = ((x + WORLD / 2) / WORLD) * (heightSize - 1);
  const v = ((z + WORLD / 2) / WORLD) * (heightSize - 1);
  const x0 = Math.floor(u), y0 = Math.floor(v);
  const x1 = Math.min(x0 + 1, heightSize - 1), y1 = Math.min(y0 + 1, heightSize - 1);
  const fx = u - x0, fy = v - y0;
  const h = (xx, yy) => heightData[yy * heightSize + xx] - heightMin;
  const top = h(x0, y0) * (1 - fx) + h(x1, y0) * fx;
  const bot = h(x0, y1) * (1 - fx) + h(x1, y1) * fx;
  return (top * (1 - fy) + bot * fy) * heightScale;
}

function makeFallbackHills() {
  // Procedural hills used only when the cached terrain files are missing.
  heightSize = 256;
  heightMin = 0;
  heightScale = 1;
  heightData = new Float32Array(heightSize * heightSize);
  for (let y = 0; y < heightSize; y++)
    for (let x = 0; x < heightSize; x++)
      heightData[y * heightSize + x] =
        18 * Math.sin(x * 0.045) * Math.cos(y * 0.038) + 9 * Math.sin(x * 0.11 + y * 0.07);
}

function makeFallbackTexture() {
  // Flat green texture for the procedural fallback terrain.
  const cv = Object.assign(document.createElement("canvas"), { width: 64, height: 64 });
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#4e7a3d";
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
}

function makeCropTexture() {
  // Procedural crop rows so each field reads as planted farmland, not a flat tile.
  const S = 128;
  const cv = Object.assign(document.createElement("canvas"), { width: S, height: S });
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, S, S);
  ctx.strokeStyle = "rgba(0,0,0,0.26)";
  ctx.lineWidth = 3;
  for (let i = 6; i < S; i += 11) {
    ctx.beginPath();
    ctx.moveTo(i, 4);
    ctx.lineTo(i, S - 4);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.33)";
  ctx.lineWidth = 2;
  for (let i = 11; i < S; i += 11) {
    ctx.beginPath();
    ctx.moveTo(i, 4);
    ctx.lineTo(i, S - 4);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(0,0,0,0.16)";
  ctx.lineWidth = 5;
  ctx.strokeRect(2, 2, S - 4, S - 4);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function cellPosition(row, col, n) {
  // Convert a grid cell to a world-space position on the terrain.
  const x = (col - (n - 1) / 2) * PITCH;
  const z = (row - (n - 1) / 2) * PITCH;
  return new THREE.Vector3(x, sampleHeight(x, z), z);
}

function buildGrid(n) {
  // Lay the farm's field tiles, scan rings and charging base onto the terrain.
  fieldTiles.forEach(t => scene.remove(t));
  scanRings.forEach(r => scene.remove(r));
  if (basePad) scene.remove(basePad);
  if (baseGlow) scene.remove(baseGlow);
  fieldTiles = []; scanRings = [];

  const crop = makeCropTexture();
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const p = cellPosition(r, c, n);
      const tile = new THREE.Mesh(
        new THREE.BoxGeometry(FIELD, 0.9, FIELD),
        new THREE.MeshStandardMaterial({
          map: crop, color: 0x3fbf6a, roughness: 0.9, metalness: 0.0,
          transparent: true, opacity: TILE_OPACITY,
        })
      );
      tile.position.set(p.x, p.y + 0.7, p.z);
      tile.rotation.y = ((r + c) % 2) * Math.PI / 2;
      tile.receiveShadow = true;
      scene.add(tile);
      fieldTiles.push(tile);

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(FIELD * 0.43, 0.3, 8, 32),
        new THREE.MeshStandardMaterial({ color: 0xf2f6fb, emissive: 0x7f8c9c, emissiveIntensity: 0.22, roughness: 0.6 })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(p.x, p.y + 1.4, p.z);
      ring.visible = false;
      scene.add(ring);
      scanRings.push(ring);
    }
  }

  const bp = cellPosition(0, 0, n);
  basePad = new THREE.Group();
  const pad = new THREE.Mesh(
    new THREE.CylinderGeometry(FIELD * 0.46, FIELD * 0.52, 1.6, 28),
    new THREE.MeshStandardMaterial({ color: 0x2f6fd0, roughness: 0.45, metalness: 0.35 })
  );
  pad.receiveShadow = true;
  basePad.add(pad);
  const mark = new THREE.Mesh(
    new THREE.TorusGeometry(FIELD * 0.26, 0.5, 8, 28),
    new THREE.MeshStandardMaterial({ color: 0xdce9ff, emissive: 0x6fa8ff, emissiveIntensity: 0.8, roughness: 0.4 })
  );
  mark.rotation.x = -Math.PI / 2;
  mark.position.y = 0.9;
  basePad.add(mark);
  basePad.position.set(bp.x, bp.y + 2.0, bp.z);
  scene.add(basePad);

  baseGlow = new THREE.PointLight(0x5fa8ff, 0, 60);
  baseGlow.position.set(bp.x, bp.y + 10, bp.z);
  scene.add(baseGlow);
}

function buildDrone() {
  // Assemble a quadcopter: fuselage, canopy, arms, motors, props, skids, gimbal, LEDs.
  drone = new THREE.Group();
  props = []; navLights = [];

  const shell = new THREE.MeshStandardMaterial({ color: 0xeef1f6, roughness: 0.35, metalness: 0.25 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x232a36, roughness: 0.55, metalness: 0.5 });

  const body = new THREE.Mesh(new THREE.CapsuleGeometry(1.9, 3.0, 6, 16), shell);
  body.rotation.z = Math.PI / 2;
  body.scale.set(1, 1, 0.78);
  body.castShadow = true;
  drone.add(body);

  const canopy = new THREE.Mesh(
    new THREE.SphereGeometry(1.45, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x1b2430, roughness: 0.12, metalness: 0.75 })
  );
  canopy.position.set(0.5, 0.75, 0);
  canopy.scale.set(1.5, 0.85, 1.05);
  drone.add(canopy);

  for (const [dx, dz] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 4.6, 10), dark);
    arm.rotation.z = Math.PI / 2;
    arm.rotation.y = -Math.atan2(dz, dx);
    arm.position.set(dx * 1.7, 0.15, dz * 1.7);
    arm.castShadow = true;
    drone.add(arm);

    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.68, 1.0, 14), dark);
    motor.position.set(dx * 3.4, 0.55, dz * 3.4);
    motor.castShadow = true;
    drone.add(motor);

    const hub = new THREE.Group();
    hub.position.set(dx * 3.4, 1.15, dz * 3.4);
    for (const ang of [0, Math.PI / 2]) {
      const blade = new THREE.Mesh(
        new THREE.BoxGeometry(5.2, 0.1, 0.62),
        new THREE.MeshStandardMaterial({ color: 0x3a4454, roughness: 0.6 })
      );
      blade.rotation.y = ang;
      hub.add(blade);
    }
    const blur = new THREE.Mesh(
      new THREE.CircleGeometry(2.7, 28),
      new THREE.MeshBasicMaterial({ color: 0xaab6c8, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })
    );
    blur.rotation.x = -Math.PI / 2;
    hub.add(blur);
    drone.add(hub);
    props.push(hub);
  }

  for (const dz of [-1.5, 1.5]) {
    const skid = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 5.4, 8), dark);
    skid.rotation.z = Math.PI / 2;
    skid.position.set(0, -2.0, dz);
    drone.add(skid);
    for (const dx of [-1.6, 1.6]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 1.9, 8), dark);
      leg.position.set(dx, -1.2, dz);
      drone.add(leg);
    }
  }

  const gimbal = new THREE.Mesh(new THREE.SphereGeometry(0.78, 14, 12), dark);
  gimbal.position.set(1.5, -1.25, 0);
  drone.add(gimbal);
  const lens = new THREE.Mesh(
    new THREE.CylinderGeometry(0.42, 0.5, 0.5, 14),
    new THREE.MeshStandardMaterial({ color: 0x0b1f33, roughness: 0.08, metalness: 0.9, emissive: 0x16456b, emissiveIntensity: 0.5 })
  );
  lens.rotation.z = Math.PI / 2;
  lens.position.set(2.1, -1.4, 0);
  drone.add(lens);

  for (const [dx, color] of [[3.9, 0x30ff6a], [-3.9, 0xff3b30]]) {
    const led = new THREE.Mesh(
      new THREE.SphereGeometry(0.3, 10, 8),
      new THREE.MeshBasicMaterial({ color })
    );
    led.position.set(dx, -0.1, 0);
    drone.add(led);
    navLights.push(led);
  }

  droneBeam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.2, 1.7, DRONE_ALTITUDE / DRONE_SCALE, 18, 1, true),
    new THREE.MeshBasicMaterial({
      color: 0x8fd4ff, transparent: true, opacity: 0.17,
      side: THREE.DoubleSide, depthWrite: false,
    })
  );
  droneBeam.position.y = -DRONE_ALTITUDE / (2 * DRONE_SCALE) - 1.8;
  drone.add(droneBeam);

  drone.scale.setScalar(DRONE_SCALE);
  const start = cellPosition(0, 0, gridN);
  drone.position.set(start.x, start.y + DRONE_ALTITUDE, start.z);
  droneTarget.copy(drone.position);
  scene.add(drone);

  droneMarker = new THREE.Mesh(
    new THREE.RingGeometry(FIELD * 0.29, FIELD * 0.45, 40),
    new THREE.MeshBasicMaterial({ color: 0x3ba7ff, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false })
  );
  droneMarker.rotation.x = -Math.PI / 2;
  scene.add(droneMarker);
}

function ndviColor(v) {
  // Map NDVI to a red (stressed) / green (healthy) ramp split at the stress threshold.
  const hue = v < STRESS
    ? 0.045 * Math.max(0, (v - 0.15) / 0.35)
    : 0.245 + 0.085 * Math.min(1, (v - STRESS) / 0.4);
  return new THREE.Color().setHSL(hue, v < STRESS ? 0.72 : 0.6, v < STRESS ? 0.42 : 0.42);
}

function spawnScanPulse(p) {
  // Expanding ring marking a multispectral capture.
  const mesh = new THREE.Mesh(
    new THREE.RingGeometry(FIELD * 0.2, FIELD * 0.3, 36),
    new THREE.MeshBasicMaterial({ color: 0xbfe9ff, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false })
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(p.x, p.y + 3.2, p.z);
  scene.add(mesh);
  effects.push({ mesh, age: 0, life: 1.1, kind: "scan" });
}

function spawnTreatBurst(p) {
  // Spray cloud and flash when a stressed field is treated.
  const group = new THREE.Group();
  for (let i = 0; i < 14; i++) {
    const bit = new THREE.Mesh(
      new THREE.SphereGeometry(0.55, 7, 6),
      new THREE.MeshBasicMaterial({ color: 0xdff6ff, transparent: true, opacity: 0.85, depthWrite: false })
    );
    const a = (i / 14) * Math.PI * 2;
    bit.position.set(Math.cos(a) * 1.5, 1.5, Math.sin(a) * 1.5);
    bit.userData.dir = new THREE.Vector3(Math.cos(a), 0.55, Math.sin(a));
    group.add(bit);
  }
  group.position.set(p.x, p.y + 2.0, p.z);
  scene.add(group);
  effects.push({ mesh: group, age: 0, life: 1.3, kind: "treat" });

  const flash = new THREE.PointLight(0x9ff0c0, 7, 55);
  flash.position.set(p.x, p.y + 9, p.z);
  scene.add(flash);
  effects.push({ mesh: flash, age: 0, life: 0.7, kind: "flash" });
}

function updateEffects(dt) {
  // Age every transient effect and remove the expired ones.
  for (let i = effects.length - 1; i >= 0; i--) {
    const fx = effects[i];
    fx.age += dt;
    const t = fx.age / fx.life;
    if (t >= 1) {
      scene.remove(fx.mesh);
      effects.splice(i, 1);
      continue;
    }
    if (fx.kind === "scan") {
      fx.mesh.scale.setScalar(1 + t * 2.4);
      fx.mesh.material.opacity = 0.85 * (1 - t);
    } else if (fx.kind === "treat") {
      fx.mesh.children.forEach(bit => {
        bit.position.addScaledVector(bit.userData.dir, dt * 7);
        bit.material.opacity = 0.85 * (1 - t);
        bit.scale.setScalar(1 + t * 0.8);
      });
    } else if (fx.kind === "flash") {
      fx.mesh.intensity = 7 * (1 - t);
    }
  }
}

function applyState(s) {
  // Refresh tiles, rings, drone target and HUD, firing effects for what changed.
  if (s.grid_size !== gridN) { gridN = s.grid_size; buildGrid(gridN); buildDrone(); }

  for (let r = 0; r < gridN; r++) {
    for (let c = 0; c < gridN; c++) {
      const i = r * gridN + c;
      const ndvi = s.ndvi[r][c];
      fieldTiles[i].material.color.copy(ndviColor(ndvi));
      fieldTiles[i].material.opacity = ndvi < STRESS ? 0.82 : TILE_OPACITY;
      scanRings[i].visible = s.known[r][c] >= 0;
    }
  }

  if (prev && prev.grid_size === s.grid_size) {
    let revealed = 0, cured = null;
    for (let r = 0; r < gridN; r++) {
      for (let c = 0; c < gridN; c++) {
        if (prev.known[r][c] < 0 && s.known[r][c] >= 0) revealed++;
        if (prev.ndvi[r][c] < STRESS && s.ndvi[r][c] >= STRESS) cured = cellPosition(r, c, gridN);
      }
    }
    if (revealed > 0) spawnScanPulse(cellPosition(s.pos[0], s.pos[1], gridN));
    if (cured) spawnTreatBurst(cured);
  }

  const p = cellPosition(s.pos[0], s.pos[1], gridN);
  droneTarget.set(p.x, p.y + DRONE_ALTITUDE, p.z);
  if (prev && (prev.pos[0] !== s.pos[0] || prev.pos[1] !== s.pos[1])) {
    droneYaw = Math.atan2(s.pos[1] - prev.pos[1], -(s.pos[0] - prev.pos[0]));
  }
  const atBase = s.pos[0] === s.base[0] && s.pos[1] === s.base[1];
  baseGlow.intensity = atBase ? 2.6 : 0.0;

  document.getElementById("battery").textContent = `${s.battery}%`;
  const bar = document.getElementById("batteryBar");
  bar.style.width = `${Math.max(0, s.battery)}%`;
  bar.style.background = s.battery > 40 ? "#41d97d" : s.battery > 15 ? "#f5b93e" : "#e2574c";
  document.getElementById("stressed").textContent = s.stressed_left;
  document.getElementById("wind").textContent = `${Math.round(s.wind * 30)} km/h`;
  document.getElementById("steps").textContent = s.steps;

  prev = s;
  state = s;
}

function connect() {
  // Keep a WebSocket to the Python simulation, reconnecting on drops.
  const status = document.getElementById("status");
  const ws = new WebSocket("ws://localhost:8765");
  ws.onopen = () => (status.textContent = "live — AgriScout simulation connected");
  ws.onmessage = e => applyState(JSON.parse(e.data));
  ws.onclose = () => {
    status.textContent = "simulation offline — retrying…";
    setTimeout(connect, 1500);
  };
}

function animate() {
  // Per-frame updates: flight motion, rotor spin, effects, stressed-field pulse.
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.getElapsedTime();

  props.forEach((hub, i) => (hub.rotation.y += dt * (48 + i * 3)));
  navLights[0].visible = Math.sin(t * 5) > -0.4;
  navLights[1].visible = Math.sin(t * 5 + Math.PI) > -0.4;

  const before = drone.position.clone();
  drone.position.lerp(droneTarget, 1 - Math.pow(0.0015, dt));
  drone.position.y += Math.sin(t * 2.1) * 0.012;
  const vel = drone.position.clone().sub(before).multiplyScalar(1 / Math.max(dt, 1e-4));
  drone.rotation.z = THREE.MathUtils.clamp(-vel.x * 0.012, -0.3, 0.3);
  drone.rotation.x = THREE.MathUtils.clamp(vel.z * 0.012, -0.3, 0.3);
  drone.rotation.y += (droneYaw - drone.rotation.y) * Math.min(1, dt * 4);

  droneMarker.position.set(drone.position.x, sampleHeight(drone.position.x, drone.position.z) + 3.6, drone.position.z);
  droneMarker.material.opacity = 0.5 + 0.35 * Math.sin(t * 4);
  droneBeam.material.opacity = 0.13 + 0.06 * Math.sin(t * 3);

  if (state) {
    for (let i = 0; i < fieldTiles.length; i++) {
      const ndvi = state.ndvi[Math.floor(i / gridN)][i % gridN];
      const pulse = ndvi < STRESS ? 0.5 + 0.5 * Math.sin(t * 4.5) : 0;
      fieldTiles[i].material.emissive.setRGB(pulse * 0.42, 0, 0);
    }
  }

  updateEffects(dt);
  controls.update();
  renderer.render(scene, camera);
}

function loadImage(src) {
  // Promise wrapper around image loading so missing files can be caught.
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
