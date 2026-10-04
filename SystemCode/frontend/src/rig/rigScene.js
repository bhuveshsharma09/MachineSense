// MachineSense — illustrative 3D rotor test rig (three.js), ported from design/prototype_v2/rig-scene.js. Geometry is approximate; not CAD.
import * as THREE from 'three';

const COL = { orange: 0xff6428, amber: 0xf2ae2e, lime: 0xc6e03a, violet: 0x5a5cf5 };
const PICK = new Set(['motor', 'coupling', 'in_brg', 'rotor', 'shaft', 'out_brg', 's_in', 's_out', 's_tach', 's_mic']);
const FOCUS = { motor: 0.9, coupling: 0.55, in_brg: 0.6, rotor: 0.78, shaft: 0.8, out_brg: 0.6, s_in: 0.5, s_out: 0.5, s_tach: 0.5, s_mic: 0.55 };

function radialTex(stops, size = 256) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'); const gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  stops.forEach(s => gr.addColorStop(s[0], s[1])); g.fillStyle = gr; g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function envScene() {
  const s = new THREE.Scene();
  const room = new THREE.Mesh(new THREE.BoxGeometry(10, 6, 10), new THREE.MeshBasicMaterial({ color: 0x8e8b86, side: THREE.BackSide }));
  room.position.y = 2; s.add(room);
  const panel = (w, h, p, r, k, col = 0xffffff) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(col).multiplyScalar(k), side: THREE.DoubleSide }));
    m.position.set(...p); m.rotation.set(...r); s.add(m);
  };
  panel(6, 3, [0, 4.9, 0], [Math.PI / 2, 0, 0], 3.4, 0xfffaf2);
  panel(1.4, 4, [-4.9, 2, -0.5], [0, Math.PI / 2, 0], 2.4);
  panel(1.2, 4, [4.9, 2, 0.6], [0, -Math.PI / 2, 0], 1.7, 0xf2f4ff);
  panel(5, 1.2, [0, 1.4, 4.9], [0, Math.PI, 0], 1.3);
  panel(10, 10, [0, -0.95, 0], [-Math.PI / 2, 0, 0], 0.9, 0xd8d3cb);
  return s;
}
function rboxGeo(w, h, d, r) {
  r = Math.min(r, w / 2 - 1e-4, h / 2 - 1e-4, d / 2 - 1e-4);
  const iw = w - 2 * r, ih = h - 2 * r, cr = Math.min(r, iw / 2, ih / 2) * 0.9, x0 = -iw / 2, y0 = -ih / 2;
  const s = new THREE.Shape();
  s.moveTo(x0 + cr, y0); s.lineTo(x0 + iw - cr, y0); s.quadraticCurveTo(x0 + iw, y0, x0 + iw, y0 + cr);
  s.lineTo(x0 + iw, y0 + ih - cr); s.quadraticCurveTo(x0 + iw, y0 + ih, x0 + iw - cr, y0 + ih);
  s.lineTo(x0 + cr, y0 + ih); s.quadraticCurveTo(x0, y0 + ih, x0, y0 + ih - cr);
  s.lineTo(x0, y0 + cr); s.quadraticCurveTo(x0, y0, x0 + cr, y0);
  const g = new THREE.ExtrudeGeometry(s, { depth: d - 2 * r, bevelEnabled: true, bevelThickness: r, bevelSize: r, bevelSegments: 4, curveSegments: 8 });
  g.center(); return g;
}
const cylX = (r, len, seg = 48) => new THREE.CylinderGeometry(r, r, len, seg).rotateZ(Math.PI / 2);
const cylY = (r, len, seg = 32) => new THREE.CylinderGeometry(r, r, len, seg);
const cylZ = (r, len, seg = 24) => new THREE.CylinderGeometry(r, r, len, seg).rotateX(Math.PI / 2);

export function createRig(container, opts = {}) {
  const reduce = !!opts.reducedMotion;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  if (!renderer.getContext()) throw new Error('WebGL unavailable');
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.setClearColor(0x000000, 0);
  const cv = renderer.domElement;
  cv.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;outline:none';
  container.appendChild(cv);

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(envScene(), 0.035).texture;
  scene.environmentIntensity = 0.95;
  const camera = new THREE.PerspectiveCamera(26, 1, 0.05, 20);

  scene.add(new THREE.HemisphereLight(0xffffff, 0xd6cfc4, 0.55));
  const key = new THREE.DirectionalLight(0xfff5ea, 1.9);
  key.position.set(0.7, 2.1, 1.3); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -0.8, right: 0.8, top: 0.8, bottom: -0.8, near: 0.5, far: 5 });
  key.shadow.bias = -0.0003; key.shadow.normalBias = 0.012; key.shadow.radius = 5;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xe6ebff, 0.7); rim.position.set(-1.3, 0.9, -1.1); scene.add(rim);
  const hot = new THREE.PointLight(COL.orange, 0, 0.45, 2); scene.add(hot);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: 0.13 }));
  ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
  const blob = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.0), new THREE.MeshBasicMaterial({ map: radialTex([[0, 'rgba(70,64,58,0.30)'], [0.55, 'rgba(70,64,58,0.12)'], [1, 'rgba(70,64,58,0)']]), transparent: true, depthWrite: false }));
  blob.rotation.x = -Math.PI / 2; blob.position.y = 0.0006; scene.add(blob);

  const M = {
    alu: new THREE.MeshPhysicalMaterial({ name: 'brushed_aluminium', color: 0xd6d9dd, metalness: 1, roughness: 0.34, anisotropy: 0.6 }),
    cast: new THREE.MeshStandardMaterial({ name: 'cast_aluminium', color: 0xc4c8cd, metalness: 0.85, roughness: 0.42 }),
    paint: new THREE.MeshStandardMaterial({ name: 'motor_paint', color: 0xa7b1bb, metalness: 0.45, roughness: 0.42 }),
    steel: new THREE.MeshStandardMaterial({ name: 'polished_steel', color: 0xeef0f2, metalness: 1, roughness: 0.12 }),
    dark: new THREE.MeshStandardMaterial({ name: 'black_polymer', color: 0x2b2d31, metalness: 0.1, roughness: 0.55 }),
    anod: new THREE.MeshStandardMaterial({ name: 'black_anodised', color: 0x3a3d43, metalness: 0.75, roughness: 0.34 }),
    panel: new THREE.MeshStandardMaterial({ name: 'drive_panel', color: 0xe4e6e9, metalness: 0.05, roughness: 0.5 }),
    screen: new THREE.MeshStandardMaterial({ name: 'display', color: 0x1a1e22, metalness: 0.2, roughness: 0.18, emissive: 0x30424c, emissiveIntensity: 0.5 }),
    tape: new THREE.MeshStandardMaterial({ name: 'reflective_tape', color: 0xf4f4f2, metalness: 0, roughness: 0.3 })
  };
  const glowTex = radialTex([[0, 'rgba(255,255,255,0.95)'], [0.22, 'rgba(255,255,255,0.55)'], [0.55, 'rgba(255,255,255,0.14)'], [1, 'rgba(255,255,255,0)']]);

  const root = new THREE.Group(); root.name = 'rotor_test_rig'; scene.add(root);
  const G = {}, anchors = {}, animated = [], glows = [];
  const V = (a) => new THREE.Vector3(...a);
  function mk(geo, mat, name, parent, p, opt) {
    const m = new THREE.Mesh(geo, mat); m.name = name; m.castShadow = true; m.receiveShadow = true;
    if (p) m.position.set(...p);
    if (opt && opt.noOutline) m.userData.noOutline = true;
    parent.add(m); return m;
  }
  function grp(id, name, parent, p = [0, 0, 0]) {
    const g = new THREE.Group(); g.name = name; g.position.set(...p); parent.add(g);
    if (id) { g.userData.id = id; G[id] = g; }
    return g;
  }
  function movable(o) { o.userData.home = o.position.clone(); o.userData.off = new THREE.Vector3(); o.userData.offT = new THREE.Vector3(); animated.push(o); return o; }
  function anchor(id, parent, p) { const o = new THREE.Object3D(); o.position.set(...p); parent.add(o); anchors[id] = o; }
  function glow(parent, p, scale) {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: COL.orange, transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
    s.position.set(...p); s.scale.setScalar(scale); s.raycast = () => {}; s.userData.op = 0; s.userData.opT = 0; parent.add(s); glows.push(s); return s;
  }

  // Base frame
  const frame = grp(null, 'base frame', root);
  mk(rboxGeo(1.0, 0.02, 0.5, 0.005), M.alu, 'base plate', frame, [0, 0.025, 0]);
  [[-0.44, -0.2], [0.44, -0.2], [-0.44, 0.2], [0.44, 0.2]].forEach(q => mk(cylY(0.022, 0.015), M.dark, 'foot', frame, [q[0], 0.0075, q[1]]));
  for (let i = 0; i < 12; i++) mk(cylY(0.0055, 0.004, 6), M.steel, 'bolt', frame, [-0.46 + i * (0.92 / 11), 0.037, 0.232]);
  [-0.1, 0.3].forEach(x => mk(rboxGeo(0.05, 0.057, 0.14, 0.003), M.cast, 'rotor base support', frame, [x, 0.0635, 0]));
  mk(rboxGeo(0.5, 0.012, 0.16, 0.003), M.alu, 'rotor base plate', frame, [0.1, 0.098, 0]);
  // Drive unit (illustrative)
  const drive = grp(null, 'drive unit', frame, [-0.4, 0, 0.145]);
  mk(rboxGeo(0.06, 0.025, 0.04, 0.003), M.cast, 'drive bracket', drive, [0, 0.0475, 0]);
  mk(rboxGeo(0.1, 0.16, 0.05, 0.005), M.panel, 'drive housing', drive, [0, 0.14, 0]);
  mk(rboxGeo(0.101, 0.03, 0.051, 0.004), M.anod, 'drive band', drive, [0, 0.085, 0]);
  mk(new THREE.BoxGeometry(0.05, 0.018, 0.002), M.screen, 'drive display', drive, [0, 0.19, 0.0255]);
  for (let i = 0; i < 6; i++) mk(cylZ(0.0042, 0.003, 20), M.anod, 'drive key', drive, [-0.018 + (i % 3) * 0.018, 0.158 - Math.floor(i / 3) * 0.016, 0.026]);

  // Motor
  const motor = movable(grp('motor', 'Motor', root, [-0.34, 0, 0]));
  mk(rboxGeo(0.15, 0.09, 0.15, 0.005), M.cast, 'motor stand', motor, [0, 0.08, 0]);
  mk(rboxGeo(0.13, 0.012, 0.11, 0.003), M.paint, 'motor feet', motor, [0, 0.131, 0]);
  mk(cylX(0.07, 0.17, 72), M.paint, 'motor body', motor, [0, 0.2, 0]);
  for (let i = 0; i < 9; i++) mk(cylX(0.0742, 0.0045, 72), M.paint, 'cooling fin', motor, [-0.068 + i * 0.017, 0.2, 0]);
  mk(cylX(0.066, 0.024, 72), M.cast, 'rear end cap', motor, [-0.097, 0.2, 0]);
  mk(cylX(0.05, 0.01, 48), M.cast, 'fan cover', motor, [-0.113, 0.2, 0]);
  mk(cylX(0.078, 0.012, 72), M.cast, 'front flange', motor, [0.091, 0.2, 0]);
  mk(rboxGeo(0.055, 0.034, 0.062, 0.003), M.paint, 'terminal box', motor, [0, 0.287, 0]);
  for (let i = 0; i < 4; i++) { const a = Math.PI / 4 + i * Math.PI / 2; mk(cylX(0.004, 0.016, 6), M.steel, 'flange bolt', motor, [0.093, 0.2 + Math.cos(a) * 0.066, Math.sin(a) * 0.066], { noOutline: true }); }
  anchor('motor', motor, [0, 0.27, 0.04]);

  // Shaft
  const shaft = grp('shaft', 'Shaft', root);
  mk(cylX(0.008, 0.66, 40), M.steel, 'shaft', shaft, [0.085, 0.2, 0]);
  mk(cylX(0.0087, 0.008, 40), M.tape, 'tachometer tape', shaft, [-0.165, 0.2, 0], { noOutline: true });
  anchor('shaft', shaft, [0.24, 0.2, 0.01]);

  // Coupling
  const coupling = grp('coupling', 'Coupling', root, [-0.215, 0, 0]);
  const cA = movable(grp(null, 'coupling hub A', coupling)), cB = movable(grp(null, 'coupling hub B', coupling));
  mk(cylX(0.019, 0.018, 48), M.alu, 'hub', cA, [-0.0165, 0.2, 0]);
  mk(cylX(0.019, 0.018, 48), M.alu, 'hub', cB, [0.0165, 0.2, 0]);
  const spiderMat = new THREE.MeshStandardMaterial({ name: 'coupling_spider', color: 0x2f3236, metalness: 0.1, roughness: 0.6, emissive: COL.amber, emissiveIntensity: 0 });
  mk(cylX(0.0165, 0.013, 48), spiderMat, 'spider', coupling, [0, 0.2, 0]);
  mk(cylY(0.003, 0.005, 10), M.dark, 'set screw', cA, [-0.0165, 0.2205, 0], { noOutline: true });
  mk(cylY(0.003, 0.005, 10), M.dark, 'set screw', cB, [0.0165, 0.2205, 0], { noOutline: true });
  coupling.userData.mat = spiderMat;
  coupling.userData.glow = glow(coupling, [0, 0.2, 0], 0.14);
  anchor('coupling', coupling, [0, 0.2, 0.025]);

  // Tachometer (illustrative position)
  const tach = grp('s_tach', 'Tachometer', root, [-0.165, 0, 0.07]);
  const tachMat = new THREE.MeshStandardMaterial({ name: 'sensor_s_tach', color: 0x2a2c30, metalness: 0.3, roughness: 0.45 });
  mk(cylY(0.012, 0.004, 24), M.anod, 'tach base', tach, [0, 0.037, 0]);
  mk(cylY(0.0035, 0.155, 12), M.steel, 'tach post', tach, [0, 0.1125, 0]);
  mk(rboxGeo(0.018, 0.018, 0.03, 0.003), tachMat, 'tach head', tach, [0, 0.2, -0.012]);
  tach.userData.sm = tachMat;
  anchor('s_tach', tach, [0, 0.212, -0.012]);

  // Bearings
  function makeBearing(id, name, x, sid) {
    const g = grp(id, name, root, [x, 0, 0]);
    const shell = movable(grp(null, name + ' housing', g));
    const hm = new THREE.MeshStandardMaterial({ name: 'housing_' + id, color: 0xcbd0d5, metalness: 0.82, roughness: 0.36, transparent: true, opacity: 1 });
    mk(rboxGeo(0.052, 0.016, 0.13, 0.003), hm, 'housing foot', shell, [0, 0.112, 0]);
    mk(rboxGeo(0.046, 0.08, 0.072, 0.004), hm, 'housing body', shell, [0, 0.16, 0]);
    mk(cylX(0.036, 0.046, 72), hm, 'housing cap', shell, [0, 0.2, 0]);
    [-0.05, 0.05].forEach(z => mk(cylY(0.0055, 0.008, 6), M.steel, 'housing bolt', shell, [0, 0.124, z], { noOutline: true }));
    const bm = new THREE.MeshStandardMaterial({ name: 'bearing_steel_' + id, color: 0xdfe3e8, metalness: 1, roughness: 0.16, emissive: COL.orange, emissiveIntensity: 0 });
    const inner = grp(null, name + ' internals', g);
    mk(new THREE.TorusGeometry(0.0235, 0.0036, 20, 72).rotateY(Math.PI / 2), bm, 'outer race', inner, [0, 0.2, 0], { noOutline: true });
    mk(new THREE.TorusGeometry(0.0132, 0.0028, 16, 64).rotateY(Math.PI / 2), bm, 'inner race', inner, [0, 0.2, 0], { noOutline: true });
    mk(new THREE.TorusGeometry(0.0185, 0.0011, 8, 64).rotateY(Math.PI / 2), M.dark, 'cage', inner, [0.0045, 0.2, 0], { noOutline: true });
    for (let i = 0; i < 10; i++) { const a = i / 10 * Math.PI * 2; mk(new THREE.SphereGeometry(0.0052, 20, 14), bm, 'ball', inner, [0, 0.2 + Math.cos(a) * 0.0185, Math.sin(a) * 0.0185], { noOutline: true }); }
    g.userData.hm = hm; g.userData.bm = bm; g.userData.shell = shell; g.userData.glow = glow(inner, [0, 0.2, 0], 0.2);
    const s = grp(sid, name + ' accelerometer', shell);
    const sm = new THREE.MeshStandardMaterial({ name: 'sensor_' + sid, color: 0x2a2c30, metalness: 0.3, roughness: 0.45, emissive: 0x000000 });
    mk(cylY(0.011, 0.004, 6), M.steel, 'sensor mount', s, [0, 0.238, 0]);
    mk(cylY(0.0088, 0.022, 32), sm, 'sensor body', s, [0, 0.251, 0]);
    mk(cylY(0.0092, 0.002, 32), M.steel, 'sensor band', s, [0, 0.258, 0], { noOutline: true });
    mk(cylZ(0.0035, 0.012, 16), M.steel, 'sensor connector', s, [0, 0.248, -0.014], { noOutline: true });
    const curve = new THREE.CatmullRomCurve3([[0, 0.248, -0.02], [0, 0.252, -0.05], [0.01, 0.2, -0.12], [0.02, 0.12, -0.19], [0.03, 0.037, -0.215]].map(V));
    mk(new THREE.TubeGeometry(curve, 64, 0.0022, 10), M.dark, 'sensor cable', s, null, { noOutline: true });
    s.userData.sm = sm;
    anchor(sid, s, [0, 0.264, 0]);
    anchor(id, g, [0, 0.2, 0.04]);
    return g;
  }
  makeBearing('in_brg', 'Inner bearing', -0.1, 's_in');
  makeBearing('out_brg', 'Outer bearing', 0.3, 's_out');

  // Rotor discs
  const rotor = grp('rotor', 'Rotor discs', root);
  const prof = [[0.0085, -0.016], [0.022, -0.016], [0.023, -0.007], [0.071, -0.0062], [0.075, -0.0035], [0.075, 0.0035], [0.071, 0.0062], [0.023, 0.007], [0.022, 0.016], [0.0085, 0.016]].map(p => new THREE.Vector2(p[0], p[1]));
  const discGeo = new THREE.LatheGeometry(prof, 120).rotateZ(-Math.PI / 2);
  const holeGeo = cylX(0.0042, 0.0128, 16);
  rotor.userData.discs = [];
  [0.03, 0.17].forEach((x, k) => {
    const dm = new THREE.MeshStandardMaterial({ name: 'brass_disc_' + k, color: 0xd6ab61, metalness: 1, roughness: 0.24, emissive: COL.orange, emissiveIntensity: 0 });
    const d = movable(grp(null, 'rotor disc ' + (k ? 'B' : 'A'), rotor, [x, 0.2, 0]));
    mk(discGeo, dm, 'disc', d, [0, 0, 0]);
    for (let i = 0; i < 12; i++) { const a = i / 12 * Math.PI * 2; mk(holeGeo, M.dark, 'balance hole', d, [0, Math.cos(a) * 0.058, Math.sin(a) * 0.058], { noOutline: true }); }
    if (k === 0) { const a = 2 / 12 * Math.PI * 2; mk(cylX(0.0038, 0.022, 12), M.steel, 'test mass screw', d, [0, Math.cos(a) * 0.058, Math.sin(a) * 0.058], { noOutline: true }); }
    rotor.userData.discs.push({ d, dm, glow: glow(d, [0, 0, 0], 0.26) });
  });
  anchor('rotor', rotor, [0.1, 0.2, 0]);

  // Belt drive (not instrumented)
  const belt = grp(null, 'belt drive', frame);
  mk(cylX(0.042, 0.022, 64), M.anod, 'drive pulley', belt, [0.395, 0.2, 0]);
  [0.3845, 0.4055].forEach(x => mk(cylX(0.0455, 0.003, 64), M.anod, 'pulley rim', belt, [x, 0.2, 0]));
  mk(cylX(0.03, 0.02, 48), M.anod, 'driven pulley', belt, [0.395, 0.075, 0]);
  mk(cylX(0.006, 0.05, 16), M.steel, 'driven shaft', belt, [0.42, 0.075, 0]);
  mk(rboxGeo(0.04, 0.05, 0.06, 0.004), M.cast, 'driven support', belt, [0.45, 0.06, 0]);
  const beltMat = M.dark;
  mk(new THREE.TorusGeometry(0.043, 0.0028, 8, 48, Math.PI).rotateY(Math.PI / 2), beltMat, 'belt', belt, [0.395, 0.2, 0]);
  mk(new THREE.TorusGeometry(0.031, 0.0028, 8, 48, Math.PI).rotateZ(Math.PI).rotateY(Math.PI / 2), beltMat, 'belt', belt, [0.395, 0.075, 0]);
  const L = Math.hypot(0.012, 0.125), th = Math.atan2(0.012, 0.125);
  const bs1 = mk(new THREE.BoxGeometry(0.0056, L, 0.0056), beltMat, 'belt', belt, [0.395, 0.1375, 0.037]); bs1.rotation.x = th;
  const bs2 = mk(new THREE.BoxGeometry(0.0056, L, 0.0056), beltMat, 'belt', belt, [0.395, 0.1375, -0.037]); bs2.rotation.x = -th;

  // Microphone (illustrative position)
  const mic = grp('s_mic', 'Microphone', root, [0.1, 0, 0.17]);
  const micMat = new THREE.MeshStandardMaterial({ name: 'sensor_s_mic', color: 0x2a2c30, metalness: 0.3, roughness: 0.45 });
  mk(cylY(0.018, 0.005, 32), M.anod, 'mic base', mic, [0, 0.0375, 0]);
  mk(cylY(0.0022, 0.13, 12), M.steel, 'mic rod', mic, [0, 0.1025, 0]);
  mk(cylZ(0.0068, 0.045, 24), micMat, 'mic capsule', mic, [0, 0.168, -0.012]);
  mk(cylZ(0.0072, 0.008, 24), M.steel, 'mic grille', mic, [0, 0.168, -0.038], { noOutline: true });
  mic.userData.sm = micMat;
  anchor('s_mic', mic, [0, 0.18, -0.02]);

  // Outlines (inverted hull, per-axis scaled for even width)
  const olSel = new THREE.MeshBasicMaterial({ color: COL.violet, side: THREE.BackSide, transparent: true, opacity: 0.95, depthWrite: false, toneMapped: false });
  const olHov = new THREE.MeshBasicMaterial({ color: COL.violet, side: THREE.BackSide, transparent: true, opacity: 0.35, depthWrite: false, toneMapped: false });
  const tmpS = new THREE.Vector3(), tmpC = new THREE.Vector3();
  Object.keys(G).forEach(id => {
    const g = G[id], list = [];
    const walk = o => o.children.forEach(c => { if (c.userData.id && c !== g) return; if (c.isMesh && !c.userData.noOutline) list.push(c); walk(c); });
    walk(g);
    g.userData.outlines = list.map(m => {
      m.geometry.computeBoundingBox(); m.geometry.boundingBox.getSize(tmpS); m.geometry.boundingBox.getCenter(tmpC);
      const w = 0.0017, o = new THREE.Mesh(m.geometry, olSel); o.userData.src = m;
      o.raycast = () => {}; o.userData.noOutline = true; o.renderOrder = 2; o.castShadow = false;
      o.position.copy(m.position); o.rotation.copy(m.rotation);
      o.scale.set((tmpS.x + 2 * w) / Math.max(tmpS.x, 1e-4), (tmpS.y + 2 * w) / Math.max(tmpS.y, 1e-4), (tmpS.z + 2 * w) / Math.max(tmpS.z, 1e-4));
      o.position.x += tmpC.x * (1 - o.scale.x); o.position.y += tmpC.y * (1 - o.scale.y); o.position.z += tmpC.z * (1 - o.scale.z);
      o.visible = false; m.parent.add(o); return o;
    });
  });

  // Camera rig
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const DEF = { target: new THREE.Vector3(0.02, 0.13, 0), theta: 0.58, phi: 1.12, radius: 1.75 };
  const view = { target: DEF.target.clone(), theta: DEF.theta, phi: DEF.phi, radius: DEF.radius };
  const cur = { target: DEF.target.clone(), theta: DEF.theta, phi: DEF.phi + 0.06, radius: DEF.radius * 1.12 };
  let st = { sel: null, hover: null, fault: [], cand: [], unusual: [], unusable: [], exploded: false, zoomOut: 1, insets: { top: 0, right: 0, bottom: 0, left: 0 } };
  let W = 1, H = 1, hoverPick = null, zoomK = 1;
  
  function resize() {
    const r = container.getBoundingClientRect(); W = Math.max(1, r.width); H = Math.max(1, r.height);
    renderer.setSize(W, H, false); camera.aspect = W / H;
    const ins = st.insets;
    camera.setViewOffset(W, H, (ins.right - ins.left) / 2, (ins.bottom - ins.top) / 2, W, H);
    camera.updateProjectionMatrix();
    const tf = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const fw = Math.max(0.3, (W - ins.left - ins.right) / W), fh = Math.max(0.3, (H - ins.top - ins.bottom) / H);
    const fit = clamp(Math.max(1.22 / (tf * camera.aspect * fw), 0.6 / (tf * fh)) * st.zoomOut, 1.2, 4.5);
    const was = DEF.radius; DEF.radius = fit; zoomK = fit / 1.75;
    if (!st.sel && Math.abs(view.radius - was) < 0.02) view.radius = fit;
  }
  const ro = new ResizeObserver(resize); ro.observe(container); resize();

  const wp = new THREE.Vector3();
  function focus(id) {
    const a = anchors[id]; if (!a) return;
    a.getWorldPosition(wp);
    view.target.set(wp.x, wp.y - 0.02, wp.z * 0.4);
    view.radius = DEF.radius * (FOCUS[id] || 0.7) / 1.75 * 1.3;
    view.phi = clamp(view.phi, 0.95, 1.18);
    if (reduce) Object.assign(cur, { theta: view.theta, phi: view.phi, radius: view.radius }), cur.target.copy(view.target);
  }
  function resetView() { view.target.copy(DEF.target); view.theta = DEF.theta; view.phi = DEF.phi; view.radius = DEF.radius; }

  function apply() {
    const s = st;
    ['in_brg', 'out_brg'].forEach(id => {
      const g = G[id], f = s.fault.includes(id), c = s.cand.includes(id);
      const reveal = (f || c || s.sel === id) && !s.exploded;
      g.userData.hm.userData.opT = reveal ? 0.16 : 1;
      g.userData.bm.emissive.setHex(c ? COL.amber : COL.orange);
      g.userData.bm.userData.eT = f ? 1.6 : c ? 0.8 : 0;
      g.userData.glow.material.color.setHex(c ? COL.amber : COL.orange);
      g.userData.glow.userData.opT = f ? 0.9 : c ? 0.55 : 0;
      g.userData.shell.userData.offT.set(0, s.exploded ? 0.12 : 0, 0);
    });
    Object.keys(G).forEach(id => {
      const g = G[id], mat = id === s.sel ? olSel : (id === s.hover || id === hoverPick) ? olHov : null;
      g.userData.outlines.forEach(o => { const sm = o.userData.src.material, faded = sm.transparent && (sm.userData.opT ?? 1) < 0.99; o.visible = !!mat && !faded; if (mat) o.material = mat; });
    });
    const rf = s.fault.includes('rotor'), rc = s.cand.includes('rotor');
    G.rotor.userData.discs.forEach((q, k) => {
      q.dm.emissive.setHex(rc ? COL.amber : COL.orange); q.dm.userData.eT = rf ? 0.55 : rc ? 0.3 : 0;
      q.glow.material.color.setHex(rc ? COL.amber : COL.orange); q.glow.userData.opT = rf ? 0.75 : rc ? 0.45 : 0;
      q.d.userData.offT.set(s.exploded ? (k ? 0.035 : -0.035) : 0, 0, 0);
    });
    const cc = s.cand.includes('coupling') || s.fault.includes('coupling');
    G.coupling.userData.mat.userData.eT = cc ? 0.9 : 0;
    G.coupling.userData.glow.material.color.setHex(s.fault.includes('coupling') ? COL.orange : COL.amber);
    G.coupling.userData.glow.userData.opT = cc ? 0.6 : 0;
    G.coupling.children.forEach(c => { if (c.userData.home && c.name.indexOf('hub A') > -1) c.userData.offT.set(s.exploded ? -0.02 : 0, 0, 0); if (c.userData.home && c.name.indexOf('hub B') > -1) c.userData.offT.set(s.exploded ? 0.02 : 0, 0, 0); });
    G.motor.userData.offT.set(s.exploded ? -0.06 : 0, 0, 0);
    ['s_in', 's_out', 's_tach', 's_mic'].forEach(id => {
      const sm = G[id].userData.sm, u = s.unusual.includes(id), b = s.unusable.includes(id);
      sm.color.setHex(u ? COL.lime : b ? COL.amber : 0x2a2c30);
      sm.emissive.setHex(u ? COL.lime : b ? COL.amber : 0x000000);
      sm.emissiveIntensity = u || b ? 0.35 : 0;
    });
    const hotId = s.fault[0];
    if (hotId && anchors[hotId]) { anchors[hotId].getWorldPosition(wp); hot.position.set(wp.x, 0.2, 0.0); hot.userData.iT = 0.05; } else hot.userData.iT = 0;
  }

  // Input
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  function pickAt(cx, cy) {
    const r = cv.getBoundingClientRect(); ndc.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects(root.children, true);
    for (const h of hits) { let o = h.object; if (o.material && o.material.transparent && o.material.opacity < 0.5) continue; while (o && !(o.userData.id && PICK.has(o.userData.id))) o = o.parent; return o ? o.userData.id : null; }
    return null;
  }
  let down = null, dragging = false;
  const onDown = e => { down = { x: e.clientX, y: e.clientY }; dragging = false; cv.setPointerCapture(e.pointerId); };
  const onMove = e => {
    if (down && (e.buttons & 1)) {
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (!dragging && Math.hypot(dx, dy) > 4) dragging = true;
      if (dragging) {
        view.theta = clamp(view.theta - e.movementX * 0.004, -0.15, 1.1);
        view.phi = clamp(view.phi - e.movementY * 0.0035, 0.88, 1.28);
        cv.style.cursor = 'grabbing';
      }
      return;
    }
    const id = pickAt(e.clientX, e.clientY);
    if (id !== hoverPick) { hoverPick = id; cv.style.cursor = id ? 'pointer' : 'grab'; apply(); if (opts.onHover) opts.onHover(id); }
  };
  const onUp = e => {
    if (down && !dragging && opts.onPick) opts.onPick(pickAt(e.clientX, e.clientY));
    down = null; dragging = false; cv.style.cursor = hoverPick ? 'pointer' : 'grab';
  };
  const onLeave = () => { if (hoverPick) { hoverPick = null; apply(); if (opts.onHover) opts.onHover(null); } };
  const onWheel = e => { e.preventDefault(); view.radius = clamp(view.radius * Math.exp(e.deltaY * 0.0011), DEF.radius * 0.65, DEF.radius * 1.65); };
  cv.addEventListener('pointerdown', onDown); cv.addEventListener('pointermove', onMove); cv.addEventListener('pointerup', onUp);
  cv.addEventListener('pointerleave', onLeave); cv.addEventListener('wheel', onWheel, { passive: false });
  cv.style.cursor = 'grab';

  // Loop
  const cam = new THREE.Vector3(), pv = new THREE.Vector3();
  let last = performance.now();
  function tickFrame() {
    const now = performance.now(), dt = Math.min(0.05, (now - last) / 1000); last = now;
    const k = reduce ? 1 : 1 - Math.exp(-dt * 5.5);
    cur.theta += (view.theta - cur.theta) * k; cur.phi += (view.phi - cur.phi) * k; cur.radius += (view.radius - cur.radius) * k; cur.target.lerp(view.target, k);
    cam.set(Math.sin(cur.phi) * Math.sin(cur.theta), Math.cos(cur.phi), Math.sin(cur.phi) * Math.cos(cur.theta)).multiplyScalar(cur.radius).add(cur.target);
    camera.position.copy(cam); camera.lookAt(cur.target);
    animated.forEach(o => { o.userData.off.lerp(o.userData.offT, k); o.position.copy(o.userData.home).add(o.userData.off); });
    const breath = reduce ? 1 : 0.84 + 0.16 * Math.sin(now * 0.0024);
    glows.forEach(g => { g.userData.op += (g.userData.opT - g.userData.op) * k; g.material.opacity = g.userData.op * breath; g.visible = g.userData.op > 0.01; });
    ['in_brg', 'out_brg'].forEach(id => {
      const hm = G[id].userData.hm, bm = G[id].userData.bm;
      const t = hm.userData.opT ?? 1; hm.opacity += (t - hm.opacity) * k; hm.depthWrite = hm.opacity > 0.98;
      bm.emissiveIntensity += ((bm.userData.eT || 0) * breath - bm.emissiveIntensity) * k;
    });
    G.rotor.userData.discs.forEach(q => { q.dm.emissiveIntensity += ((q.dm.userData.eT || 0) * breath - q.dm.emissiveIntensity) * k; });
    const sp = G.coupling.userData.mat; sp.emissiveIntensity += ((sp.userData.eT || 0) * breath - sp.emissiveIntensity) * k;
    hot.intensity += ((hot.userData.iT || 0) * breath - hot.intensity) * k;
    renderer.render(scene, camera);
    overlay();
  }
  function overlay() {
    const ov = opts.overlay; if (!ov) return;
    const ins = st.insets;
    let sx = null, sy = null;
    ov.querySelectorAll('[data-anchor]').forEach(el => {
      const id = el.getAttribute('data-anchor'), a = anchors[id]; if (!a) return;
      a.getWorldPosition(pv); pv.project(camera);
      const x = (pv.x + 1) / 2 * W, y = (1 - pv.y) / 2 * H;
      el.style.transform = 'translate3d(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px,0)';
      el.style.visibility = pv.z < 1 ? 'visible' : 'hidden';
      if (id === st.sel) { sx = x; sy = y; }
    });
    avoidCollisions(ov, ins);
    const card = ov.querySelector('[data-follow]');
    if (card && sx !== null) {
      const cw = card.offsetWidth, ch = card.offsetHeight;
      let x = sx + 48; if (x + cw > W - ins.right - 16) x = sx - 48 - cw;
      x = clamp(x, ins.left + 16, W - ins.right - cw - 16);
      const y = clamp(sy - ch * 0.4, ins.top + 12, Math.max(ins.top + 12, H - ins.bottom - ch - 12));
      card.style.transform = 'translate3d(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px,0)';
      card.style.opacity = '1';
    }
  }
  // Callout collision avoidance (Stage App-2a.2): place labels top to bottom; flip a label to the left of its marker when
  // it would run under the right-hand panel; push it down past labels already placed and past the legend.
  function avoidCollisions(ov, ins) {
    const placed = [], items = [];
    const cr = container.getBoundingClientRect();
    const lg = document.querySelector('.legend');
    if (lg) { const r = lg.getBoundingClientRect(); placed.push({ x: r.left - cr.left, y: r.top - cr.top, w: r.width, h: r.height }); }
    ov.querySelectorAll('[data-anchor]').forEach(el => {
      const lab = el.querySelector('.ms-label');
      if (!lab || el.style.visibility === 'hidden') return;
      const m = /translate3d\(([-\d.]+)px, ?([-\d.]+)px/.exec(el.style.transform);
      if (m) items.push({ lab, x: +m[1], y: +m[2], w: lab.offsetWidth, h: lab.offsetHeight });
    });
    items.sort((a, b) => a.y - b.y);
    items.forEach(it => {
      const flip = it.x + 18 + it.w > W - ins.right - 8;
      const bx = flip ? it.x - 18 - it.w : it.x + 18, by = it.y - 14;
      let dy = 0;
      for (let k = 0; k < 10; k++) {
        const hit = placed.find(p => bx < p.x + p.w && bx + it.w > p.x && by + dy < p.y + p.h && by + dy + it.h > p.y);
        if (!hit) break;
        dy = hit.y + hit.h + 4 - by;
      }
      placed.push({ x: bx, y: by + dy, w: it.w, h: it.h });
      it.lab.style.left = (flip ? -18 - it.w : 18) + 'px';
      it.lab.style.top = (-14 + dy) + 'px';
    });
  }

  renderer.setAnimationLoop(tickFrame);
  apply();

  return {
    update(s) {
      const prevSel = st.sel, prevIns = JSON.stringify(st.insets), prevZoom = st.zoomOut;
      st = Object.assign({}, st, s);
      if (JSON.stringify(st.insets) !== prevIns || st.zoomOut !== prevZoom) resize();
      if (st.sel !== prevSel) { if (st.sel) focus(st.sel); else resetView(); }
      apply();
    },
    resetView() { resetView(); },
    dispose() {
      renderer.setAnimationLoop(null); ro.disconnect();
      cv.removeEventListener('pointerdown', onDown); cv.removeEventListener('pointermove', onMove); cv.removeEventListener('pointerup', onUp);
      cv.removeEventListener('pointerleave', onLeave); cv.removeEventListener('wheel', onWheel);
      renderer.dispose(); pmrem.dispose(); if (cv.parentNode) cv.parentNode.removeChild(cv);
    }
  };
}
