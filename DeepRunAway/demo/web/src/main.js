import * as ort from "onnxruntime-web/webgpu";

let forwardSession = null;
let residualSession = null;
let running = false;
let pending = false;

const NX = 100;
const NY = 100;

let ENERGY_MIN = 1.0e4;
let ENERGY_MAX = 5.0e6;
const MEC_SQ_EV = 511.0e3;

const XI_MIN = -1.0;
const XI_MAX = 1.0;

let P_MIN = momentumFromEnergy(ENERGY_MIN);
let P_MAX = momentumFromEnergy(ENERGY_MAX);

let EVERT_MIN = 1.0;
let EVERT_MAX = 10.0;
let ZEFF_MIN = 1.0;
let ZEFF_MAX = 10.0;
let ALPHA_MIN = 0.0;
let ALPHA_MAX = 0.2;
let RESIDUAL_ONNX_BATCH = 4096;

const FIG_WIDTH = 1700;
const FIG_HEIGHT = 1120;

const PLOT_MARGIN = {
  left: 220,
  right: 155,
  top: 132,
  bottom: 195,
};

const COLORBAR_CSS_WIDTH = 48;
const COLORBAR_TEXT_FONT = "18px sans-serif";

const colorbarSpecs = {};

function $(id) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing element #${id}`);
  }
  return el;
}

function maybe$(id) {
  return document.getElementById(id);
}

function momentumFromEnergy(energyEv) {
  const gamma = 1.0 + energyEv / MEC_SQ_EV;
  return Math.sqrt(gamma * gamma - 1.0);
}

function updateMomentumRange() {
  P_MIN = momentumFromEnergy(ENERGY_MIN);
  P_MAX = momentumFromEnergy(ENERGY_MAX);
}

function superscriptInt(n) {
  const map = {
    "-": "⁻",
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
  };

  return String(n)
    .split("")
    .map((ch) => map[ch] ?? ch)
    .join("");
}

function ensureInteractiveSliders() {
  hideSingleControlOnly("nx");
  hideSingleControlOnly("ny");
  hideSingleControlOnly("backendMode");

  hideSingleElementOnly("status");
  hideSingleElementOnly("stats");

  const existingE = maybe$("evert");
  const existingZ = maybe$("zeff");
  const existingA = maybe$("alpha");

  if (existingE && existingZ && existingA) {
    forceVisible(existingE);
    forceVisible(existingZ);
    forceVisible(existingA);

    forceVisible(maybe$("evertValue"));
    forceVisible(maybe$("zeffValue"));
    forceVisible(maybe$("alphaValue"));

    return;
  }

  if (maybe$("dynamicControls")) {
    return;
  }

  const root = document.querySelector("#app") || document.body;

  const panel = document.createElement("section");
  panel.id = "dynamicControls";
  panel.style.margin = "18px 0 22px 0";
  panel.style.padding = "18px 20px";
  panel.style.border = "1px solid #334155";
  panel.style.borderRadius = "12px";
  panel.style.background = "#151b24";
  panel.style.maxWidth = "1500px";

  panel.innerHTML = `
    <div style="display:grid; grid-template-columns: 140px 1fr 90px; gap: 14px; align-items:center; margin-bottom: 14px;">
      <label for="evert" style="font-weight:700;">|Eφ|</label>
      <input id="evert" type="range" min="1" max="10" step="0.01" value="10">
      <span id="evertValue">10.00</span>

      <label for="zeff" style="font-weight:700;">Zeff</label>
      <input id="zeff" type="range" min="1" max="10" step="0.01" value="1">
      <span id="zeffValue">1.00</span>

      <label for="alpha" style="font-weight:700;">α</label>
      <input id="alpha" type="range" min="0" max="0.2" step="0.001" value="0">
      <span id="alphaValue">0.000</span>
    </div>
  `;

  root.prepend(panel);
}

function forceVisible(el) {
  if (!el) {
    return;
  }

  el.style.display = "";
  el.style.visibility = "visible";
  el.style.opacity = "1";

  let p = el.parentElement;
  for (let i = 0; i < 4 && p && p !== document.body; i += 1) {
    p.style.display = "";
    p.style.visibility = "visible";
    p.style.opacity = "1";
    p = p.parentElement;
  }
}

function hideSingleElementOnly(id) {
  const el = maybe$(id);
  if (el) {
    el.style.display = "none";
  }
}

function hideSingleControlOnly(id) {
  const el = maybe$(id);
  if (!el) {
    return;
  }

  el.style.display = "none";

  const valueEl = maybe$(`${id}Value`);
  if (valueEl) {
    valueEl.style.display = "none";
  }

  const labelEl = document.querySelector(`label[for="${id}"]`);
  if (labelEl) {
    labelEl.style.display = "none";
  }
}

function cleanPlotCards() {
  const captionsToHide = [
    "x-axis: log energy",
    "Residual color scale",
  ];

  for (const canvasId of ["probCanvas", "resCanvas"]) {
    const canvas = maybe$(canvasId);
    if (!canvas) {
      continue;
    }

    let card = canvas.parentElement;
    for (let i = 0; i < 12 && card && card !== document.body; i += 1) {
      const hasCanvas = card.querySelector(`#${canvasId}`);
      const hasLegacyHeading = card.querySelector("h2, h3, h4");
      const hasLegacyCaption = Array.from(card.querySelectorAll("p, div, span")).some((el) => {
        const text = el.textContent || "";
        return captionsToHide.some((txt) => text.includes(txt));
      });

      if (hasCanvas && (hasLegacyHeading || hasLegacyCaption)) {
        break;
      }

      card = card.parentElement;
    }

    if (!card || card === document.body) {
      continue;
    }

    const headings = card.querySelectorAll("h2, h3, h4");
    for (const h of headings) {
      h.style.display = "none";
    }

    const allTextEls = card.querySelectorAll("p, div, span");
    for (const el of allTextEls) {
      const text = el.textContent || "";
      if (captionsToHide.some((txt) => text.includes(txt))) {
        el.style.display = "none";
      }
    }

    card.style.background = "transparent";
    card.style.border = "none";
    card.style.boxShadow = "none";
    card.style.padding = "0";
    card.style.margin = "0";
    card.style.overflow = "visible";

    let p = card.parentElement;
    for (let i = 0; i < 3 && p && p !== document.body; i += 1) {
      if (p.querySelector(`#${canvasId}`)) {
        p.style.background = "transparent";
        p.style.border = "none";
        p.style.boxShadow = "none";
        p.style.overflow = "visible";
      }
      p = p.parentElement;
    }
  }

  const probCanvas = maybe$("probCanvas");
  const resCanvas = maybe$("resCanvas");

  if (probCanvas && resCanvas) {
    const probCard =
      probCanvas.closest(".card") ||
      probCanvas.closest(".plot-card") ||
      probCanvas.parentElement;

    const resCard =
      resCanvas.closest(".card") ||
      resCanvas.closest(".plot-card") ||
      resCanvas.parentElement;

    if (probCard && resCard && probCard.parentElement === resCard.parentElement) {
      const parent = probCard.parentElement;
      parent.style.display = "grid";
      parent.style.gridTemplateColumns = "minmax(0, 1fr) minmax(0, 1fr)";
      parent.style.columnGap = "220px";
      parent.style.rowGap = "42px";
      parent.style.alignItems = "start";
      parent.style.overflow = "visible";
      parent.style.maxWidth = "none";
    }

    if (probCard) {
      probCard.style.marginRight = "0";
      probCard.style.paddingRight = "80px";
      probCard.style.overflow = "visible";
      probCard.style.maxWidth = "none";
    }

    if (resCard) {
      resCard.style.marginLeft = "0";
      resCard.style.paddingLeft = "80px";
      resCard.style.overflow = "visible";
      resCard.style.maxWidth = "none";
    }
  }

  syncColorbarHeight("probCanvas", "probColorbar");
  syncColorbarHeight("resCanvas", "resColorbar");
}

function syncColorbarHeight(plotCanvasId, colorbarCanvasId) {
  const plotCanvas = maybe$(plotCanvasId);
  const colorbarCanvas = maybe$(colorbarCanvasId);

  if (!plotCanvas || !colorbarCanvas) {
    return;
  }

  const plotRect = plotCanvas.getBoundingClientRect();
  if (!Number.isFinite(plotRect.width) || plotRect.width <= 0) {
    return;
  }

  const scale = plotRect.width / FIG_WIDTH;
  const yAxisTop = PLOT_MARGIN.top * scale;
  const yAxisHeight =
    (FIG_HEIGHT - PLOT_MARGIN.top - PLOT_MARGIN.bottom) * scale;

  colorbarCanvas.style.marginTop = `${yAxisTop}px`;
  colorbarCanvas.style.marginBottom = "0";
  colorbarCanvas.style.alignSelf = "flex-start";

  const spec = colorbarSpecs[colorbarCanvasId];
  if (spec) {
    renderColorbarToCanvas(
      colorbarCanvas,
      yAxisHeight,
      spec.min,
      spec.max,
      spec.cmap,
      spec.options,
    );
  }
}

function configureSlidersFromMetadata() {
  const evert = $("evert");
  evert.min = String(EVERT_MIN);
  evert.max = String(EVERT_MAX);
  evert.step = "0.01";
  evert.value = String(clamp(Number(evert.value), EVERT_MIN, EVERT_MAX));

  const zeff = $("zeff");
  zeff.min = String(ZEFF_MIN);
  zeff.max = String(ZEFF_MAX);
  zeff.step = "0.01";
  zeff.value = String(clamp(Number(zeff.value), ZEFF_MIN, ZEFF_MAX));

  const alpha = $("alpha");
  alpha.min = String(ALPHA_MIN);
  alpha.max = String(ALPHA_MAX);
  alpha.step = "0.001";
  alpha.value = String(clamp(Number(alpha.value), ALPHA_MIN, ALPHA_MAX));

  updateSliderLabels();
}

function updateSliderLabels() {
  const evert = Number($("evert").value);
  const zeff = Number($("zeff").value);
  const alpha = Number($("alpha").value);

  const evertValue = maybe$("evertValue");
  if (evertValue) {
    evertValue.textContent = evert.toFixed(2);
  }

  const zeffValue = maybe$("zeffValue");
  if (zeffValue) {
    zeffValue.textContent = zeff.toFixed(2);
  }

  const alphaValue = maybe$("alphaValue");
  if (alphaValue) {
    alphaValue.textContent = alpha.toFixed(3);
  }
}

async function createWasmSession(path, name) {
  console.log(`Loading ${name} with WASM:`, path);
  return await ort.InferenceSession.create(path, {
    executionProviders: ["wasm"],
  });
}

async function loadMetadata() {
  const base = import.meta.env.BASE_URL;
  const version = Date.now();
  const metadataPath = `${base}models/metadata.json?v=${version}`;

  try {
    const response = await fetch(metadataPath);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const metadata = await response.json();
    const ranges = metadata.parameter_ranges ?? {};

    if (Array.isArray(ranges.Evert)) {
      [EVERT_MIN, EVERT_MAX] = ranges.Evert;
    }

    if (Array.isArray(ranges.Zeff)) {
      [ZEFF_MIN, ZEFF_MAX] = ranges.Zeff;
    }

    if (Array.isArray(ranges.alpha)) {
      [ALPHA_MIN, ALPHA_MAX] = ranges.alpha;
    }

    if (Array.isArray(ranges.energy_eV)) {
      [ENERGY_MIN, ENERGY_MAX] = ranges.energy_eV;
    }

    updateMomentumRange();

    const residualBatch = metadata.onnx?.residual_batch_size;
    if (Number.isFinite(residualBatch) && residualBatch > 0) {
      RESIDUAL_ONNX_BATCH = Math.trunc(residualBatch);
    }
  } catch (err) {
    console.warn("Could not load metadata.json; using built-in defaults.", err);
    updateMomentumRange();
  }

  ensureInteractiveSliders();
  configureSlidersFromMetadata();
}

async function loadModels() {
  await loadMetadata();

  const base = import.meta.env.BASE_URL;
  const version = Date.now();

  const forwardPath = `${base}models/rpf_forward.onnx?v=${version}`;
  const residualPath = `${base}models/rpf_residual.onnx?v=${version}`;

  forwardSession = await createWasmSession(forwardPath, "forward model");
  residualSession = await createWasmSession(residualPath, "residual model");

  console.log("Forward input/output:", forwardSession.inputNames, forwardSession.outputNames);
  console.log("Residual input/output:", residualSession.inputNames, residualSession.outputNames);
}

function norm(x, lo, hi) {
  return (x - lo) / (hi - lo);
}

function energyAtIndex(i, nx) {
  if (nx <= 1) {
    return ENERGY_MIN;
  }

  const t = i / (nx - 1);
  const logMin = Math.log10(ENERGY_MIN);
  const logMax = Math.log10(ENERGY_MAX);

  return Math.pow(10.0, logMin + t * (logMax - logMin));
}

function xiAtIndex(j, ny) {
  if (ny <= 1) {
    return XI_MIN;
  }

  return XI_MIN + (j / (ny - 1)) * (XI_MAX - XI_MIN);
}

function makeInput(nx, ny, evert, zeff, alpha) {
  const data = new Float32Array(nx * ny * 5);

  const eNorm = norm(evert, EVERT_MIN, EVERT_MAX);
  const zNorm = norm(zeff, ZEFF_MIN, ZEFF_MAX);
  const aNorm = norm(alpha, ALPHA_MIN, ALPHA_MAX);

  let k = 0;

  for (let j = 0; j < ny; j += 1) {
    const xi = xiAtIndex(j, ny);
    const xiNorm = norm(xi, XI_MIN, XI_MAX);

    for (let i = 0; i < nx; i += 1) {
      const energy = energyAtIndex(i, nx);
      const p = momentumFromEnergy(energy);
      const pNorm = norm(p, P_MIN, P_MAX);

      data[5 * k + 0] = pNorm;
      data[5 * k + 1] = xiNorm;
      data[5 * k + 2] = eNorm;
      data[5 * k + 3] = zNorm;
      data[5 * k + 4] = aNorm;

      k += 1;
    }
  }

  return data;
}

async function runResidualInChunks(inputData, totalRows) {
  const chunkRows = RESIDUAL_ONNX_BATCH;
  const output = new Float32Array(totalRows);

  for (let offset = 0; offset < totalRows; offset += chunkRows) {
    const rows = Math.min(chunkRows, totalRows - offset);
    const chunkData = new Float32Array(chunkRows * 5);

    const src0 = offset * 5;
    const src1 = (offset + rows) * 5;
    chunkData.set(inputData.subarray(src0, src1), 0);

    if (rows < chunkRows && rows > 0) {
      const lastRowStart = (rows - 1) * 5;
      const lastRow = chunkData.subarray(lastRowStart, lastRowStart + 5);

      for (let r = rows; r < chunkRows; r += 1) {
        chunkData.set(lastRow, r * 5);
      }
    }

    const tensor = new ort.Tensor("float32", chunkData, [chunkRows, 5]);
    const result = await residualSession.run({
      [residualSession.inputNames[0]]: tensor,
    });

    const values = result[residualSession.outputNames[0]].data;
    output.set(values.subarray(0, rows), offset);
  }

  return output;
}

async function runInference() {
  if (!forwardSession || !residualSession || running) {
    return;
  }

  running = true;

  try {
    updateSliderLabels();
    cleanPlotCards();

    const evert = Number($("evert").value);
    const zeff = Number($("zeff").value);
    const alpha = Number($("alpha").value);

    const inputData = makeInput(NX, NY, evert, zeff, alpha);
    const inputTensor = new ort.Tensor("float32", inputData, [NX * NY, 5]);

    const fwd = await forwardSession.run({
      [forwardSession.inputNames[0]]: inputTensor,
    });

    const r = await runResidualInChunks(inputData, NX * NY);
    const p = fwd[forwardSession.outputNames[0]].data;

    const rAbs = new Float32Array(r.length);
    for (let i = 0; i < r.length; i += 1) {
      rAbs[i] = Math.abs(r[i]);
    }

    drawHeatmap("probCanvas", p, NX, NY, {
      min: 0.0,
      max: 1.0,
      cmap: turboColormap,
      title: "Runaway probability",
    });

    drawColorbar("probColorbar", 0.0, 1.0, turboColormap, {
      showMax: true,
      maxText: "1",
      topTextColor: "white",
    });

    const vmax = Math.max(percentile(rAbs, 0.99), 1.0e-16);

    drawHeatmap("resCanvas", rAbs, NX, NY, {
      min: 0.0,
      max: vmax,
      cmap: infernoColormap,
      title: "PDE residual magnitude",
    });

    drawColorbar("resColorbar", 0.0, vmax, infernoColormap, {
      showMax: true,
      maxText: vmax.toExponential(1),
      topTextColor: "black",
    });

    cleanPlotCards();
  } catch (err) {
    console.error(err);

    const status = maybe$("status");
    if (status) {
      status.style.display = "";
      status.textContent = `Error: ${err.message}`;
    }
  } finally {
    running = false;
  }
}

function drawHeatmap(canvasId, values, nx, ny, options) {
  const canvas = $(canvasId);
  const ctx = canvas.getContext("2d");

  canvas.width = FIG_WIDTH;
  canvas.height = FIG_HEIGHT;

  canvas.style.width = "100%";
  canvas.style.height = "auto";

  const margin = PLOT_MARGIN;

  const plotX = margin.left;
  const plotY = margin.top;
  const plotW = FIG_WIDTH - margin.left - margin.right;
  const plotH = FIG_HEIGHT - margin.top - margin.bottom;

  ctx.clearRect(0, 0, FIG_WIDTH, FIG_HEIGHT);
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, FIG_WIDTH, FIG_HEIGHT);

  const offscreen = document.createElement("canvas");
  offscreen.width = nx;
  offscreen.height = ny;

  const offCtx = offscreen.getContext("2d");
  const image = offCtx.createImageData(nx, ny);

  const min = options.min;
  const max = options.max;
  const scale = max > min ? 1.0 / (max - min) : 1.0;

  for (let j = 0; j < ny; j += 1) {
    const srcJ = ny - 1 - j;

    for (let i = 0; i < nx; i += 1) {
      const src = srcJ * nx + i;
      const dst = j * nx + i;

      const v = clamp01((values[src] - min) * scale);
      const [r, g, b] = options.cmap(v);

      image.data[4 * dst + 0] = r;
      image.data[4 * dst + 1] = g;
      image.data[4 * dst + 2] = b;
      image.data[4 * dst + 3] = 255;
    }
  }

  offCtx.putImageData(image, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, plotX, plotY, plotW, plotH);

  drawAxes(ctx, plotX, plotY, plotW, plotH, options.title);
}

function drawAxes(ctx, plotX, plotY, plotW, plotH, title) {
  ctx.save();

  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 3.0;
  ctx.strokeRect(plotX, plotY, plotW, plotH);

  if (title) {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "white";
    ctx.font = "54px sans-serif";
    ctx.fillText(title, plotX + plotW / 2, plotY - 62);
  }

  const xTicks = logEnergyTicks(ENERGY_MIN, ENERGY_MAX);

  for (const energy of xTicks) {
    const x = plotX + energyToX(energy, plotW);

    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.lineWidth = 3.0;
    ctx.beginPath();
    ctx.moveTo(x, plotY + plotH);
    ctx.lineTo(x, plotY + plotH + 22);
    ctx.stroke();

    drawEnergyTickLabel(ctx, x, plotY + plotH + 78, energy, plotX, plotW);
  }

  const yTicks = [-1.0, -0.5, 0.0, 0.5, 1.0];

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "white";
  ctx.font = "42px sans-serif";

  for (const xi of yTicks) {
    const y = plotY + (1.0 - norm(xi, XI_MIN, XI_MAX)) * plotH;

    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.lineWidth = 3.0;
    ctx.beginPath();
    ctx.moveTo(plotX - 22, y);
    ctx.lineTo(plotX, y);
    ctx.stroke();

    ctx.fillText(xi.toFixed(1), plotX - 36, y);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "white";
  ctx.font = "46px sans-serif";
  ctx.fillText("Energy [eV]", plotX + plotW / 2, plotY + plotH + 140);

  ctx.save();
  ctx.translate(70, plotY + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("ξ", 0, 0);
  ctx.restore();

  ctx.restore();
}

function drawEnergyTickLabel(ctx, x, y, energy, plotX, plotW) {
  const exponent = Math.floor(Math.log10(energy));
  const mantissa = energy / Math.pow(10, exponent);
  const mantissaIsOne = Math.abs(mantissa - 1.0) < 1.0e-6;

  const isRightEndpoint = Math.abs(energy - ENERGY_MAX) / ENERGY_MAX < 1.0e-10;
  const isLeftEndpoint = Math.abs(energy - ENERGY_MIN) / ENERGY_MIN < 1.0e-10;

  ctx.save();
  ctx.textBaseline = "middle";
  ctx.fillStyle = "white";
  ctx.font = "42px sans-serif";

  let label;

  if (mantissaIsOne) {
    label = `10${superscriptInt(exponent)}`;
  } else {
    label = `${mantissa.toFixed(1)}×10${superscriptInt(exponent)}`;
  }

  if (isRightEndpoint) {
    ctx.textAlign = "right";
    ctx.fillText(label, plotX + plotW - 6, y);
  } else if (isLeftEndpoint) {
    ctx.textAlign = "left";
    ctx.fillText(label, plotX + 6, y);
  } else {
    ctx.textAlign = "center";
    ctx.fillText(label, x, y);
  }

  ctx.restore();
}

function energyToX(energy, plotW) {
  const logMin = Math.log10(ENERGY_MIN);
  const logMax = Math.log10(ENERGY_MAX);
  const t = (Math.log10(energy) - logMin) / (logMax - logMin);
  return t * plotW;
}

function logEnergyTicks(lo, hi) {
  const ticks = [];
  const p0 = Math.ceil(Math.log10(lo));
  const p1 = Math.floor(Math.log10(hi));

  for (let p = p0; p <= p1; p += 1) {
    const e = Math.pow(10, p);
    if (e >= lo && e <= hi) {
      ticks.push(e);
    }
  }

  if (ticks.length === 0 || ticks[0] > lo * 1.01) {
    ticks.unshift(lo);
  }

  if (ticks[ticks.length - 1] < hi / 1.01) {
    ticks.push(hi);
  }

  return ticks;
}

function drawColorbar(canvasId, min, max, cmap, options = {}) {
  colorbarSpecs[canvasId] = {
    min,
    max,
    cmap,
    options,
  };

  const plotCanvasId = canvasId === "probColorbar" ? "probCanvas" : "resCanvas";
  const plotCanvas = maybe$(plotCanvasId);
  const colorbarCanvas = $(canvasId);

  let displayHeight = 360;

  if (plotCanvas) {
    const plotRect = plotCanvas.getBoundingClientRect();
    if (Number.isFinite(plotRect.width) && plotRect.width > 0) {
      const scale = plotRect.width / FIG_WIDTH;
      displayHeight =
        (FIG_HEIGHT - PLOT_MARGIN.top - PLOT_MARGIN.bottom) * scale;
    }
  }

  renderColorbarToCanvas(colorbarCanvas, displayHeight, min, max, cmap, options);
}

function renderColorbarToCanvas(canvas, displayHeight, min, max, cmap, options = {}) {
  const cssW = COLORBAR_CSS_WIDTH;
  const cssH = Math.max(120, Math.round(displayHeight));
  const dpr = window.devicePixelRatio || 1.0;

  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);

  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const image = ctx.createImageData(canvas.width, canvas.height);

  for (let j = 0; j < image.height; j += 1) {
    const v = 1.0 - j / Math.max(image.height - 1, 1);
    const [r, g, b] = cmap(v);

    for (let i = 0; i < image.width; i += 1) {
      const k = 4 * (j * image.width + i);
      image.data[k + 0] = r;
      image.data[k + 1] = g;
      image.data[k + 2] = b;
      image.data[k + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const showMax = options.showMax ?? true;
  const maxText = options.maxText ?? max.toExponential(1);
  const topTextColor = options.topTextColor ?? "white";

  ctx.font = COLORBAR_TEXT_FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";

  if (showMax) {
    ctx.fillStyle = topTextColor;
    ctx.fillText(maxText, 6, 8);
  }

  ctx.fillStyle = "white";
  ctx.textBaseline = "bottom";
  ctx.fillText(min.toFixed(0), 6, cssH - 8);
}

function scheduleInference() {
  if (pending) {
    return;
  }

  pending = true;

  requestAnimationFrame(async () => {
    pending = false;
    await runInference();
  });
}

function percentile(values, q) {
  const arr = Array.from(values).sort((a, b) => a - b);
  const idx = Math.min(
    arr.length - 1,
    Math.max(0, Math.floor(q * (arr.length - 1))),
  );

  return arr[idx];
}

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) {
    return lo;
  }

  return Math.max(lo, Math.min(hi, x));
}

function clamp01(x) {
  return Math.max(0.0, Math.min(1.0, x));
}

function turboColormap(v) {
  const x = clamp01(v);

  const r =
    34.61 +
    x * (1172.33 + x * (-10793.56 + x * (33300.12 + x * (-38394.49 + x * 14825.05))));

  const g =
    23.31 +
    x * (557.33 + x * (1225.33 + x * (-3574.96 + x * (1073.77 + x * 707.56))));

  const b =
    27.2 +
    x * (3211.1 + x * (-15327.97 + x * (27814.0 + x * (-22569.18 + x * 6838.66))));

  return [
    Math.round(clamp255(r)),
    Math.round(clamp255(g)),
    Math.round(clamp255(b)),
  ];
}

function infernoColormap(v) {
  const anchors = [
    [0.000, 0, 0, 4],
    [0.100, 22, 11, 57],
    [0.200, 66, 10, 104],
    [0.300, 106, 23, 110],
    [0.400, 147, 38, 103],
    [0.500, 188, 55, 84],
    [0.600, 221, 81, 58],
    [0.700, 243, 120, 25],
    [0.800, 252, 165, 10],
    [0.900, 246, 215, 70],
    [1.000, 252, 255, 164],
  ];

  const x = clamp01(v);

  for (let i = 0; i < anchors.length - 1; i += 1) {
    const a = anchors[i];
    const b = anchors[i + 1];

    if (x >= a[0] && x <= b[0]) {
      const t = (x - a[0]) / (b[0] - a[0]);

      return [
        Math.round(a[1] + t * (b[1] - a[1])),
        Math.round(a[2] + t * (b[2] - a[2])),
        Math.round(a[3] + t * (b[3] - a[3])),
      ];
    }
  }

  return [252, 255, 164];
}

function clamp255(x) {
  return Math.max(0, Math.min(255, x));
}

async function main() {
  ensureInteractiveSliders();
  cleanPlotCards();

  await loadModels();
  await runInference();

  for (const id of ["evert", "zeff", "alpha"]) {
    $(id).addEventListener("input", scheduleInference);
    $(id).addEventListener("change", scheduleInference);
  }

  window.addEventListener("resize", () => {
    syncColorbarHeight("probCanvas", "probColorbar");
    syncColorbarHeight("resCanvas", "resColorbar");
  });
}

main().catch((err) => {
  console.error(err);

  const status = maybe$("status");
  if (status) {
    status.style.display = "";
    status.textContent = `Error: ${err.message}`;
  }
});
