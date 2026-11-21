// "Try it" demo: recorded outputs of the trained readers (one example episode, seed 0).
// The particle flow is a schematic illustration; no model weights are loaded.
(() => {
  const S = {
    A: {rows: [["1", "Fonin Telu", "Sunte", 1]], kv: ["Sunte", 1], ours: ["Sunte", 1],
        r2: false, del1: false, flow: [[[0, 1, "good"]], [[0, 1, "good"]]]},
    B: {rows: [["1", "Fonin Telu", "Felaga", 1]], kv: ["Felaga", 1], ours: ["Felaga", 1],
        r2: false, del1: false, flow: [[[0, 1, "good"]], [[0, 1, "good"]]]},
    C: {rows: [["1", "Fonin Telu", "Sunte", 0]], kv: ["I don’t know", 1], ours: ["I don’t know", 1],
        r2: false, del1: true, flow: [[[2, 1, "good"]], [[2, 1, "good"]]]},
    E: {rows: [["1", "Fonin Telu (country 1)", "Sunte", 0], ["2", "Fonin Telu (country 2)", "Vake", 1]],
        kv: ["Vake ✗ (answered with the namesake)", 0], ours: ["Which Fonin Telu do you mean?", 1],
        r2: true, del1: true, flow: [[[1, 1, "bad"]], [[2, 1, "amber"]]]},
    F: {rows: [["1", "Fonin Telu (country)", "Sunte", 1], ["2", "Fonin Telu (a person)", "—", 1]], kv: ["Sunte", 1], ours: ["Sunte", 1],
        r2: true, del1: false, flow: [[[0, 1, "good"]], [[0, 1, "good"]]]},
  };
  const COL = {good: [0.50, 0.85, 0.71], bad: [1.0, 0.56, 0.50], amber: [1.0, 0.83, 0.54], blue: [0.54, 0.71, 1.0], dim: [0.36, 0.42, 0.53]};
  const LANES = [0.25, 0.75], SRC_X = 0.12, DST_X = 0.86, OFF = [-0.12, 0, 0.12];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FR = {
    "I don’t know": "Je ne sais pas",
    "Vake ✗ (answered with the namesake)": "Vake ✗ (a répondu avec l’homonyme)",
    "Which Fonin Telu do you mean?": "De quel Fonin Telu parlez-vous ?",
    "Fonin Telu (country 1)": "Fonin Telu (pays 1)", "Fonin Telu (country 2)": "Fonin Telu (pays 2)",
    "Fonin Telu (country)": "Fonin Telu (pays)", "Fonin Telu (a person)": "Fonin Telu (une personne)",
    " (deleted)": " (supprimée)", "row 1": "ligne 1", "row 1 ∅": "ligne 1 ∅",
  };
  const t = (s) => (document.documentElement.lang === "fr" && FR[s]) || s;

  const cases = document.querySelector(".cases"), btns = [...cases.querySelectorAll("button")];
  const rows = document.getElementById("rows"), kv = document.getElementById("kv"), ab = document.getElementById("ours");
  const canvas = document.getElementById("flow"), engine = document.getElementById("engine");
  let state = S.E;

  // ---------- particules (simulation CPU partagée par les deux rendus) ----------
  const N = 160, P = [];
  const pick = (lane) => {
    const fl = state.flow[lane]; let r = Math.random() * fl.reduce((a, f) => a + f[1], 0);
    for (const f of fl) { if ((r -= f[1]) <= 0) return f; } return fl[fl.length - 1];
  };
  for (let l = 0; l < 2; l++) for (let i = 0; i < N; i++) {
    const f = pick(l);
    P.push({lane: l, dst: f[0], col: f[2], t: Math.random(), v: 0.0035 + Math.random() * 0.004, j: (Math.random() - .5) * 0.05});
  }
  function step() {
    for (const p of P) {
      p.t += reduce ? 0 : p.v;
      if (p.t >= 1) { const f = pick(p.lane); p.t = 0; p.dst = f[0]; p.col = f[2]; p.j = (Math.random() - .5) * 0.05; }
    }
  }
  function particlePos(p) {
    const ly = LANES[p.lane], dy = ly + OFF[p.dst], t = p.t, u = 1 - t;
    const cx = 0.5, cy = ly + p.j * 2;
    const x = u * u * SRC_X + 2 * u * t * cx + t * t * DST_X;
    const y = u * u * ly + 2 * u * t * cy + t * t * dy;
    return [x, y, Math.sin(Math.PI * t)];
  }
  // instances : x, y, taille(px), r, g, b, a
  function instances(dpr) {
    const out = [];
    for (const p of P) {
      const [x, y, a] = particlePos(p), c = COL[p.col];
      out.push(x, y, 2.6 * dpr, c[0], c[1], c[2], 0.85 * a);
    }
    for (let l = 0; l < 2; l++) {                       // nœuds : source et trois destinations
      out.push(SRC_X, LANES[l], 7 * dpr, ...COL.blue, 0.9);
      const vis = [state.del1 ? COL.dim : COL.blue, state.r2 ? COL.blue : null, COL.dim];
      vis.forEach((c, k) => { if (c) out.push(DST_X, LANES[l] + OFF[k], 6.5 * dpr, ...c, k === 2 ? 0.55 : 0.9); });
    }
    return new Float32Array(out);
  }

  // ---------- rendu WebGPU ----------
  const WGSL = `
struct U { res: vec2f };
@group(0) @binding(0) var<uniform> u: U;
struct VO { @builtin(position) p: vec4f, @location(0) c: vec4f, @location(1) q: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) pos: vec2f, @location(1) size: f32, @location(2) col: vec4f) -> VO {
  var k = array<vec2f, 6>(vec2f(-1.,-1.), vec2f(1.,-1.), vec2f(-1.,1.), vec2f(-1.,1.), vec2f(1.,-1.), vec2f(1.,1.));
  let q = k[vi];
  let px = pos * u.res + q * size;
  var o: VO;
  o.p = vec4f(px.x / u.res.x * 2. - 1., 1. - px.y / u.res.y * 2., 0., 1.);
  o.c = col; o.q = q; return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let d = length(i.q);
  let a = (1. - smoothstep(0.35, 1.0, d)) * i.c.a;
  return vec4f(i.c.rgb * a, a);
}`;
  async function startWebGPU() {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter(); if (!adapter) return false;
    const device = await adapter.requestDevice();
    const ctx = canvas.getContext("webgpu"); if (!ctx) return false;
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({device, format, alphaMode: "premultiplied"});
    const mod = device.createShaderModule({code: WGSL});
    const pipe = device.createRenderPipeline({
      layout: "auto",
      vertex: {module: mod, entryPoint: "vs", buffers: [{arrayStride: 28, stepMode: "instance", attributes: [
        {shaderLocation: 0, offset: 0, format: "float32x2"}, {shaderLocation: 1, offset: 8, format: "float32"},
        {shaderLocation: 2, offset: 12, format: "float32x4"}]}]},
      fragment: {module: mod, entryPoint: "fs", targets: [{format, blend: {
        color: {srcFactor: "one", dstFactor: "one", operation: "add"}, alpha: {srcFactor: "one", dstFactor: "one", operation: "add"}}}]},
      primitive: {topology: "triangle-list"},
    });
    const ubuf = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    const bind = device.createBindGroup({layout: pipe.getBindGroupLayout(0), entries: [{binding: 0, resource: {buffer: ubuf}}]});
    let vbuf = null, cap = 0;
    function frame() {
      const dpr = Math.min(devicePixelRatio || 1, 2), w = canvas.clientWidth * dpr | 0, h = canvas.clientHeight * dpr | 0;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      step();
      const data = instances(dpr);
      if (data.byteLength > cap) { vbuf?.destroy(); cap = data.byteLength * 2; vbuf = device.createBuffer({size: cap, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST}); }
      device.queue.writeBuffer(vbuf, 0, data);
      device.queue.writeBuffer(ubuf, 0, new Float32Array([w, h, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({colorAttachments: [{view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store",
        clearValue: {r: 0.047, g: 0.082, b: 0.161, a: 1}}]});
      pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.setVertexBuffer(0, vbuf); pass.draw(6, data.length / 7);
      pass.end(); device.queue.submit([enc.finish()]);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    return true;
  }

  // ---------- repli Canvas 2D ----------
  function start2D() {
    const g = canvas.getContext("2d");
    function frame() {
      const dpr = Math.min(devicePixelRatio || 1, 2), w = canvas.clientWidth * dpr | 0, h = canvas.clientHeight * dpr | 0;
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      step();
      g.globalCompositeOperation = "source-over"; g.fillStyle = "#0c1529"; g.fillRect(0, 0, w, h);
      g.globalCompositeOperation = "lighter";
      const d = instances(dpr);
      for (let i = 0; i < d.length; i += 7) {
        g.fillStyle = `rgba(${d[i+3]*255|0},${d[i+4]*255|0},${d[i+5]*255|0},${d[i+6]})`;
        g.beginPath(); g.arc(d[i] * w, d[i+1] * h, d[i+2] * 0.7, 0, 6.283); g.fill();
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // ---------- interface ----------
  function show(k) {
    state = S[k];
    rows.innerHTML = "";
    for (const [n, e, v, p] of state.rows) {
      const tr = document.createElement("tr"); if (!p) tr.className = "gone";
      tr.innerHTML = "<td></td><td></td><td></td>";
      tr.children[0].textContent = n; tr.children[1].textContent = t(e); tr.children[2].textContent = p ? v : v + t(" (deleted)");
      rows.appendChild(tr);
    }
    kv.textContent = t(state.kv[0]); kv.className = state.kv[1] ? "ok" : "ko";
    ab.textContent = t(state.ours[0]); ab.className = state.ours[1] ? "ok" : "ko";
    document.querySelectorAll(".n-r2").forEach(n => n.style.opacity = state.r2 ? 1 : 0);
    document.querySelectorAll(".n-r1").forEach(n => n.textContent = t(state.del1 ? "row 1 ∅" : "row 1"));
    for (const b of btns) b.setAttribute("aria-pressed", String(b.dataset.case === k));
  }
  cases.hidden = false;
  btns.forEach(b => b.addEventListener("click", () => show(b.dataset.case)));
  show("E");
  startWebGPU().then(ok => { engine.textContent = ok ? "WebGPU" : "Canvas 2D"; if (!ok) start2D(); })
               .catch(() => { engine.textContent = "Canvas 2D"; start2D(); });
})();
