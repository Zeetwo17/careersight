/* CareerSight dashboard — single-page client.
 * Sections: tabs · live demo · portfolio · PRI · architecture.
 */

const API = ""; // same-origin

// Keep Chart.js animations off — long render loops can lock headless renderers.
if (typeof Chart !== "undefined") {
  Chart.defaults.animation = false;
  Chart.defaults.animations = { colors: false, x: false };
  Chart.defaults.transitions = { active: { animation: { duration: 0 } } };
}

// ----------- helpers -----------
const $  = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

const fmtPct  = (x) => `${Math.round(x * 100)}%`;
const fmtRs   = (x) => `₹${Number(x).toFixed(1)}L`;
const fmtRs2  = (x) => `₹${Number(x).toFixed(2)}L`;
const initials = (name) =>
  (name || "?")
    .split(/\s+/).filter(Boolean).slice(0, 2)
    .map(w => w[0].toUpperCase()).join("") || "?";

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status}: ${txt}`);
  }
  return res.json();
}


// ----------- tabs -----------
// Centralised tab switcher — used by the top nav AND the in-result rail nav.
// Keeps tab-active class + lazy-loads heavy tabs on first visit.
function switchToTab(target) {
  if (!target) return;
  $$(".nav-tab").forEach(t => t.classList.toggle("active", t.dataset.tab === target));
  $$(".tab").forEach(s => {
    s.classList.toggle("tab-active", s.id === `tab-${target}`);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (target === "portfolio" && !portfolioLoaded) loadPortfolio();
  if (target === "pri"       && !priLoaded)       loadPri();
  if (target === "arch"      && !archLoaded)      renderArchitecture();
}

function setupTabs() {
  $$(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => switchToTab(tab.dataset.tab));
  });

  // Rail-nav links inside the result panel — clicking jumps to the
  // corresponding top-level tab. This makes "Portfolio View / PRI Index /
  // Architecture" actionable instead of being decorative dead labels.
  document.addEventListener("click", (e) => {
    const li = e.target.closest("[data-go-tab]");
    if (!li) return;
    const target = li.dataset.goTab;
    if (target && target !== "demo") switchToTab(target);
  });
}


// ----------- health / status -----------
async function pingHealth() {
  const pill = $("#nav-status");
  try {
    const h = await api("/api/health");
    pill.classList.add("ok");
    const ece = h.calibration?.placed_6m?.ece_post;
    const eceStr = (ece != null) ? ` · ECE ${ece.toFixed(3)}` : "";
    pill.querySelector("span:last-child").textContent =
      `Model live · n=${h.n_train.toLocaleString()} · AUC ${h.model_aucs.placed_6m.toFixed(2)}${eceStr}`;
    return h;
  } catch (e) {
    pill.classList.add("err");
    pill.querySelector("span:last-child").textContent = "API unreachable";
    return null;
  }
}

async function pingLLM(probe = false) {
  const pill = $("#llm-status");
  if (!pill) return;
  if (probe) pill.classList.add("probing");
  pill.classList.remove("ok", "err", "warn", "ready");
  try {
    const url = probe ? "/api/llm_health?probe=true" : "/api/llm_health";
    const h = await api(url);
    pill.classList.remove("probing");
    pill.querySelector("span:last-child").textContent = h.label || "LLM";

    if (h.mode === "heuristic_only") {
      pill.classList.add("warn");
      _updateDropzoneSub(false, null);
    } else if (h.ok) {
      pill.classList.add("ok");
      _updateDropzoneSub(true, h.provider);
    } else if (h.status === "rate_limited") {
      pill.classList.add("warn");
      _updateDropzoneSub(false, null);
    } else if (h.status === "auth_failed") {
      pill.classList.add("err");
      _updateDropzoneSub(false, null);
    } else if (h.status === "no_calls_yet" || h.status === "ok") {
      // Configured, untested — show a subtle ready state (not full green yet)
      pill.classList.add("ready");
      _updateDropzoneSub(true, h.provider);
    } else {
      pill.classList.add("warn");
    }

    const providerLabel = h.provider === "openrouter" ? "OpenRouter LLM" : "Gemini Flash";
    pill.title = `Click to probe ${providerLabel}.\n${h.label || ""}\nCalls: ${h.calls_ok}/${h.calls_total}` +
      (h.last_error ? `\nLast error: ${h.last_error}` : "");
    return h;
  } catch (e) {
    pill.classList.remove("probing");
    pill.classList.add("err");
    pill.querySelector("span:last-child").textContent = "LLM · unreachable";
    return null;
  }
}

// Update the dropzone sub-label to reflect whether LLM extraction is active.
function _updateDropzoneSub(llmActive, provider) {
  const sub = $("#dropzone .dropzone-sub");
  if (!sub) return;
  if (llmActive) {
    const name = provider === "openrouter" ? "OpenRouter LLM" : "Gemini Flash";
    sub.innerHTML = `<span class="dropzone-llm-badge">✦ ${name} active</span> · structured JSON extraction`;
  } else {
    sub.textContent = "Parsed via Gemini Flash if API key set, else rule-based fallback.";
  }
}


// ----------- demo flow -----------
function setupDemo() {
  // Persona buttons
  $$(".persona-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      $$(".persona-btn").forEach(b => b.classList.remove("busy"));
      btn.classList.add("busy");
      try {
        const result = await api(`/api/demo_profile/${btn.dataset.persona}`);
        renderResult(result);
      } catch (e) {
        alert("Failed to load persona: " + e.message);
      } finally {
        btn.classList.remove("busy");
      }
    });
  });

  // File input — selecting a file STAGES it, doesn't analyze yet.
  const dropzone = $("#dropzone");
  const input    = $("#resume-input");
  input.addEventListener("change", e => {
    if (e.target.files[0]) stageResumeFile(e.target.files[0]);
  });

  // Drag & drop — also stages, doesn't auto-analyze.
  ["dragover", "dragenter"].forEach(evt =>
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.add("drag-active");
    }));
  ["dragleave", "dragend", "drop"].forEach(evt =>
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.remove("drag-active");
    }));
  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) stageResumeFile(file);
  });

  // Explicit Analyze + Remove buttons — user has to click Analyze to score.
  $("#analyze-btn").addEventListener("click", () => {
    if (window.__stagedFile) handleResumeUpload(window.__stagedFile);
  });
  $("#remove-staged-btn").addEventListener("click", clearStagedFile);
}

// Stage a selected/dropped PDF file. Shows the confirmation card with the
// Analyze button — no API call is made until the user clicks Analyze.
function stageResumeFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    alert("Please upload a PDF resume.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert("File too large (>10MB).");
    return;
  }
  window.__stagedFile = file;
  $("#staged-file-name").textContent = file.name;
  const sizeKb = (file.size / 1024).toFixed(1);
  $("#staged-file-meta").textContent = `${sizeKb} KB · PDF · ready to analyze`;
  $("#staged-file").classList.remove("hidden");
  $("#dropzone").classList.add("has-staged");
  $("#dropzone .dropzone-text").textContent = `Selected ${file.name}`;
  // Hide any prior result while a fresh file is staged
  $("#discovery-panel")?.classList.add("hidden");
  $("#result-panel")?.classList.add("hidden");
  $("#summary-panel")?.classList.add("hidden");
  $("#empty-state")?.classList.remove("hidden");
}

function clearStagedFile() {
  window.__stagedFile = null;
  $("#staged-file").classList.add("hidden");
  $("#dropzone").classList.remove("has-staged");
  $("#dropzone .dropzone-text").textContent = "Drop a PDF resume here or click to upload";
  // Reset the file input so re-selecting the same file fires the change event
  $("#resume-input").value = "";
}

async function handleResumeUpload(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    alert("Please upload a PDF resume.");
    return;
  }
  const fd = new FormData();
  fd.append("file", file);

  const btn = $("#analyze-btn");
  const btnLabel = $(".analyze-btn-label");
  if (btn) {
    btn.disabled = true;
    btnLabel.textContent = "Analyzing";
  }
  $("#dropzone .dropzone-text").textContent = `Examining ${file.name}…`;
  resetDiscoveryPanel();

  // Auto-scroll: bring the live discovery panel into view as soon as analysis
  // starts so the user sees the streaming events without scrolling manually.
  scrollToAnalysis("#discovery-panel");

  try {
    // Try the SSE forensic flow first; if browser/proxy doesn't tolerate it,
    // fall back to the one-shot endpoint.
    const ok = await streamResumeUpload(file);
    if (!ok) {
      const result = await api("/api/score_from_resume", { method: "POST", body: fd });
      if (result.rejected) {
        showRejectBanner(result.reason);
      } else {
        renderResult(result);
      }
    }
    $("#dropzone .dropzone-text").textContent = `Done · ${file.name}`;
    pingLLM(false);
  } catch (e) {
    alert("Upload failed: " + e.message);
    $("#dropzone .dropzone-text").textContent = "Drop a PDF resume here or click to upload";
  } finally {
    if (btn) {
      btn.disabled = false;
      btnLabel.textContent = "Re-analyze";
    }
  }
}

// ----------- SSE forensic flow -----------
const STAGE_ORDER = [
  ["validity", "Validity gate"],
  ["parse",    "Parsing resume"],
  ["ipr",      "Institute lookup"],
  ["score",    "Scoring + anchor"],
];

function resetDiscoveryPanel() {
  const panel = $("#discovery-panel");
  panel.classList.remove("hidden");
  $("#discovery-feed").innerHTML = "";
  $("#ipr-card").classList.add("hidden");
  $("#elite-banner").classList.add("hidden");
  $("#anchor-card").classList.add("hidden");
  $("#reject-banner").classList.add("hidden");
  $("#discovery-title").textContent = "Examining your file…";
  $("#discovery-meta").textContent = "—";
  $("#empty-state").classList.add("hidden");
  $("#result-panel").classList.add("hidden");
  $("#summary-panel").classList.add("hidden");
  const _gapsCard = $("#profile-gaps");
  if (_gapsCard) _gapsCard.classList.add("hidden");

  // Render stage chips fresh
  const stagesEl = $("#discovery-stages");
  stagesEl.innerHTML = "";
  STAGE_ORDER.forEach(([key, label]) => {
    const div = document.createElement("div");
    div.className = "discovery-stage"; div.dataset.stage = key;
    div.textContent = label;
    stagesEl.appendChild(div);
  });
}

function setStage(name) {
  $$(".discovery-stage").forEach(div => {
    if (div.dataset.stage === name) {
      div.classList.add("active");
      div.classList.remove("done");
    } else if (div.classList.contains("active")) {
      div.classList.remove("active");
      div.classList.add("done");
    }
  });
}

function pushFeedItem(html, classes = "") {
  const li = document.createElement("li");
  if (classes) li.className = classes;
  li.innerHTML = html;
  $("#discovery-feed").appendChild(li);
  // Keep latest 14 visible (don't grow unbounded mid-demo)
  const feed = $("#discovery-feed");
  while (feed.children.length > 14) feed.removeChild(feed.firstChild);
}

function fmtConf(c) {
  return `${Math.round(c * 100)}%`;
}

function showIprCard(ipr) {
  const card = $("#ipr-card");
  const fb = ipr.fallback_level;
  const fbClass = fb >= 5 ? "l5" : fb >= 4 ? "l4" : "";
  const sp = ipr.salary_percentiles_lpa;
  const pl = ipr.placement_rate;
  const headline = ipr.canonical_name || ipr.level_label;
  card.classList.remove("hidden");
  card.innerHTML = `
    <div class="ipr-card-eyebrow">INSTITUTE PLACEMENT REGISTRY · LIVE LOOKUP</div>
    <div class="ipr-card-title">${escapeHtml(headline)}
      <span class="ipr-card-fb ${fbClass}">L${fb} · ${ipr.data_quality}</span>
    </div>
    <div class="ipr-card-row">
      <span>Median salary</span><span>₹${sp.p50.toFixed(1)}L</span>
      <span>p25 – p75</span><span>₹${sp.p25.toFixed(1)}L – ₹${sp.p75.toFixed(1)}L</span>
      <span>p90 (top decile)</span><span>₹${sp.p90.toFixed(1)}L</span>
      <span>Placement @ 6m</span><span>${Math.round(pl.month_6 * 100)}%</span>
      <span>Placement @ 12m</span><span>${Math.round(pl.month_12 * 100)}%</span>
      <span>Cohort size</span><span>n = ${ipr.sample_size.toLocaleString()}</span>
    </div>
    <div class="ipr-card-source">Source: ${escapeHtml(ipr.source || "—")}${ipr.year_bin ? " · " + ipr.year_bin : ""}</div>
  `;
}

function showEliteBanner(reasons) {
  if (!reasons || reasons.length === 0) return;
  const el = $("#elite-banner");
  el.classList.remove("hidden");
  el.innerHTML = `
    <div class="elite-banner-title">⚡ Elite outlier detected — bypassing median anchor</div>
    <div class="elite-banner-reasons">${reasons.map(escapeHtml).join(" · ")}</div>
  `;
}

function showAnchorCard(salaryCard) {
  const c = $("#anchor-card");
  c.classList.remove("hidden");
  const w = Math.round((salaryCard.anchor_weight || 0) * 100);
  c.innerHTML = `
    <div class="anchor-card-title">SALARY ANCHOR · how much the institute drives the prediction</div>
    <div class="anchor-card-bar">
      <div class="anchor-card-bar-fill" style="width: ${w}%"></div>
    </div>
    <div class="anchor-card-row">
      <span>IPR weight: <b>${w}%</b></span>
      <span>Model weight: <b>${100 - w}%</b></span>
      <span>IPR p50: <b>₹${(salaryCard.ipr_p50 || 0).toFixed(1)}L</b></span>
    </div>
  `;
}

function showRejectBanner(reason) {
  resetDiscoveryPanel();
  $("#reject-banner").classList.remove("hidden");
  $("#reject-reason").textContent = reason || "No resume signals found.";
  $("#discovery-title").textContent = "Document rejected";
  $("#discovery-meta").textContent = "No salary or risk score generated";
}

function escapeHtml(s) {
  return String(s).replace(/[<>&"']/g, c =>
    ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function streamResumeUpload(file) {
  // POST to SSE endpoint, parse text/event-stream from the response stream
  const fd = new FormData();
  fd.append("file", file);

  let res;
  try {
    res = await fetch("/api/score_from_resume_stream", { method: "POST", body: fd });
  } catch (e) {
    return false;
  }
  if (!res.ok || !res.body) return false;
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("text/event-stream")) return false;

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const evtBlock = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      handleSSEBlock(evtBlock);
    }
  }
  return true;
}

function handleSSEBlock(block) {
  const lines = block.split("\n");
  let evt = "message", data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) evt = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  let payload;
  try { payload = JSON.parse(data); } catch { payload = data; }
  dispatchSSE(evt, payload);
}

function dispatchSSE(evt, p) {
  switch (evt) {
    case "upload":
      $("#discovery-meta").textContent = `${p.size_kb} KB · ${p.filename}`;
      break;
    case "stage":
      $("#discovery-title").textContent = p.label;
      setStage(p.name);
      break;
    case "validity_signal":
      pushFeedItem(escapeHtml(p.detail || p.name), p.weight > 0 ? "signal-pos" : "signal-neg");
      break;
    case "validity_result":
      pushFeedItem(`File verified — looks like a resume (confidence ${fmtConf(p.p_resume)})`,
                   p.band === "accept" ? "signal-pos" : "signal-neg");
      break;
    case "rejected":
      showRejectBanner(p.reason);
      break;
    case "parsed_field": {
      const cls = "field" + (p.imputed ? " imputed" : "");
      const value = Array.isArray(p.value) ? p.value.slice(0, 5).join(", ") + (p.count > 5 ? ` (+${p.count - 5} more)` : "") : (p.value ?? "—");
      const conf = p.imputed
        ? `<span class="field-conf">imputed</span>`
        : `<span class="field-conf">${fmtConf(p.confidence)}</span>`;
      pushFeedItem(
        `<span class="field-label">${escapeHtml(p.label)}:</span> <span class="field-value">${escapeHtml(value)}</span> ${conf}`,
        cls,
      );
      break;
    }
    case "parse_result":
      if (p.layout === "multicolumn") {
        pushFeedItem(`Multi-column layout detected — extracted columns separately`, "signal-pos");
      }
      if (p.duplicate_pages > 0) {
        pushFeedItem(`Removed ${p.duplicate_pages} duplicate page(s)`, "signal-pos");
      }
      if (p.is_elite_outlier) {
        showEliteBanner(p.elite_outlier_reasons);
      }
      break;
    case "ipr_card":
      showIprCard(p);
      break;
    case "salary_card":
      showAnchorCard(p);
      if (p.is_elite_outlier && p.elite_reasons?.length) {
        showEliteBanner(p.elite_reasons);
      }
      break;
    case "placement_card":
      // handled by `result` payload — placement_card is mostly a paced ping
      break;
    case "drivers_card":
    case "copilot_card":
      break;
    case "result":
      renderResult(p);
      break;
    case "done":
      $("#discovery-meta").textContent = `${(p.latency_ms / 1000).toFixed(1)}s · parser ${p.parser_used || "—"}`;
      // Mark all stages done
      $$(".discovery-stage.active").forEach(s => { s.classList.remove("active"); s.classList.add("done"); });
      break;
    case "error":
      pushFeedItem(`<span style="color:#ef4444">Error in ${p.stage}: ${escapeHtml(p.message)}</span>`);
      break;
  }
}


// ----------- result rendering -----------
let survivalChart = null;

function renderResult(r) {
  $("#empty-state").classList.add("hidden");
  $("#result-panel").classList.remove("hidden");

  const p = r.profile, risk = r.risk, pp = r.placement_probabilities, sb = r.salary_band_lpa;

  // Rail
  $("#result-avatar").textContent = initials(p.name);
  $("#result-name").textContent = p.name || "Anonymous";
  let metaText = `${p.course_type} · Tier-${p.institute_tier} · CGPA ${Number(p.cgpa).toFixed(1)}`;
  if (r.nirf_match && r.nirf_match.matched_to) {
    metaText += ` · NIRF #${Math.round(r.nirf_match.nirf_rank)} ${r.nirf_match.matched_to}`;
  }
  $("#result-meta").textContent = metaText;

  // Risk score circle
  const num = $("#risk-num");
  num.textContent = "0";
  // animate from 0 to risk.score
  const target = risk.score;
  const start = performance.now();
  const dur = 900;
  const tick = (t) => {
    const k = Math.min(1, (t - start) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    num.textContent = Math.round(target * eased);
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  const bar = $("#risk-bar");
  const circumference = 326; // 2 * pi * 52
  bar.style.strokeDashoffset = circumference - (circumference * target / 100);
  bar.style.stroke = risk.tier_color;

  const tierEl = $("#result-tier");
  tierEl.textContent = `${risk.tier} RISK`;
  tierEl.className = `result-tier-pill t-${risk.tier.toLowerCase()}`;
  $(".result-tierbar").className = `result-tierbar t-${risk.tier.toLowerCase()}`;

  // Show the BCC-derived CRC threshold instead of a hard-coded constant
  const crcEl = $(".result-crc");
  if (crcEl && risk.crc_threshold != null) {
    const fnrPct = Math.round((risk.crc_fnr_target || 0.10) * 100);
    const confPct = Math.round((risk.crc_confidence || 0.90) * 100);
    crcEl.innerHTML = `
      <div class="result-crc-label" title="${risk.crc_note || ''}">CRC (Bates 2024) · FNR ≤ ${fnrPct}%</div>
      <div class="result-crc-val">@ score ≥ ${risk.crc_threshold}</div>
      <div class="result-crc-sub">w.p. ≥ ${confPct}%</div>
    `;
  }

  // Probabilities + salary
  $("#p-3m").textContent = fmtPct(pp.p_3m);
  $("#p-6m").textContent = fmtPct(pp.p_6m);
  $("#p-12m").textContent = fmtPct(pp.p_12m);
  $("#salary-range").textContent = `${fmtRs(sb.low)} – ${fmtRs(sb.high)}`;

  // Rail "Decision snapshot" — populate with derived numbers so the rail
  // shows live, profile-specific values instead of static placeholders.
  const railBand = $("#rail-risk-band");
  if (railBand) {
    railBand.textContent = risk.tier || "—";
    railBand.className = `rail-stat-val t-${(risk.tier || "").toLowerCase()}`;
  }
  const railP6 = $("#rail-p6m");
  if (railP6) railP6.textContent = fmtPct(pp.p_6m);
  const railPct = $("#rail-cohort-pct");
  if (railPct) {
    const peer = r.peer_benchmark || {};
    const speedPct = peer.speed_percentile;
    const salPct   = peer.salary_percentile;
    const best = (speedPct != null) ? speedPct : salPct;
    railPct.textContent = (best != null) ? `${Math.round(best)}th` : "—";
  }

  // Drivers
  const driversRow = $("#drivers-row");
  driversRow.innerHTML = "";
  const maxAbs = Math.max(...r.top_drivers.map(d => Math.abs(d.risk_contribution)), 0.0001);
  r.top_drivers.forEach(d => {
    const helping = d.risk_contribution < 0;
    const w = Math.abs(d.risk_contribution) / maxAbs * 100;
    const chip = document.createElement("span");
    chip.className = `driver-chip${helping ? " helping" : ""}`;
    if (d.tooltip) chip.title = d.tooltip;            // native hover tooltip
    chip.innerHTML = `
      <span>${d.label}</span>
      <span class="driver-chip-bar"><span style="width:${w}%"></span></span>
    `;
    driversRow.appendChild(chip);
  });

  // Explanation
  $("#explanation").textContent = `"${r.explanation}"`;

  // Curve — eyebrow honestly reflects which method produced the curve.
  if (r.survival_method === "deephit") {
    $("#curve-eyebrow").textContent = "DEEPHIT SURVIVAL · LEE ET AL. 2018";
  } else {
    $("#curve-eyebrow").textContent = "INTERPOLATED SURVIVAL · 3/6/12-MO ANCHORS";
  }
  $("#curve-meta").textContent =
    `EWS lead: ${r.early_warning_days}d · CRC FNR ≤ ${(risk.crc_fnr_target * 100).toFixed(0)}%`;
  drawSurvivalCurve(r.survival_curve, risk, r.ipr_survival_curve);

  // Co-Pilot actions (Thompson Sampling — Russo & Van Roy 2018)
  const al = $("#actions-list");
  al.innerHTML = "";
  r.copilot_actions.forEach(a => {
    const node = document.createElement("div");
    node.className = "action-item";
    // Counterfactual sentence is the new hero line: "rises from 83% to 89%"
    // sells the action's value far better than "+6pp" did.
    const counter = a.counterfactual ||
      `${a.detail} (+${a.expected_uplift_pp}pp expected uplift)`;
    const baseline = a.baseline_p_6m_pct;
    const projected = a.projected_p_6m_pct;
    const arrow = (baseline != null && projected != null)
      ? `<span class="action-arrow"><b>${baseline.toFixed(0)}%</b> → <b>${projected.toFixed(0)}%</b></span>`
      : "";
    node.innerHTML = `
      <div class="action-item-text">
        <div class="action-item-title">${a.title} ${arrow}</div>
        <div class="action-item-detail">${counter}</div>
      </div>
      <div class="action-impact ${a.impact}">${a.impact}</div>
    `;
    al.appendChild(node);
  });

  // Live feed
  const feed = $("#feed-list");
  feed.innerHTML = "";
  const survivalTag = r.survival_method === "deephit"
    ? ["DeepHit survival curve", "Computed", "green"]
    : ["Interpolated survival curve", "Computed", "amber"];
  const nirfMatch = r.nirf_match && r.nirf_match.matched_to;
  const nirfTag = nirfMatch
    ? [`NIRF match: ${r.nirf_match.matched_to}`, `rank ${Math.round(r.nirf_match.nirf_rank)}`, "green"]
    : [`NIRF match: tier-${r.profile.institute_tier} fallback`, "Done", "amber"];
  const items = [
    [`Resume parsed via ${r.parser_used}`, `${(r.latency_ms / 1000).toFixed(1)}s`, "green"],
    [`Feature vector: 60 features (incl. NIRF)`, "Done", "green"],
    nirfTag,
    [`Beta-calibrated probabilities (Kull 2017)`, "Done", "green"],
    survivalTag,
    [`SHAP top-${r.top_drivers.length} explained`, "Ready", "green"],
    [`Thompson Sampling Co-Pilot`, "Sampled", "green"],
    [`CRC threshold (Bates 2024)`, `≥${r.risk?.crc_threshold ?? "—"}`, "orange"],
    [`Lender alert dispatched`, `${r.early_warning_days || 0}d early`, r.early_warning_days >= 90 ? "orange" : "amber"],
  ];
  for (const [k, v, color] of items) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${k}</span><span class="feed-tag ${color}">${v}</span>`;
    feed.appendChild(li);
  }

  // Render the new Decision Intelligence Summary cards (readiness / peers /
  // what-if simulator). Renders BEFORE we scroll so the auto-scroll target
  // is the populated summary panel.
  renderSummary(r);
  maybeShowProfileGaps(r);

  // Auto-scroll: if the gaps card is showing (CGPA / institute missing),
  // land on it so the warning is immediately visible. Otherwise prefer
  // the result panel (placement risk score is the headline metric and
  // now renders first in the DOM); summary panel sits below as a follow-up.
  const gapsCard    = $("#profile-gaps");
  const resultPanel = $("#result-panel");
  if (gapsCard && !gapsCard.classList.contains("hidden")) {
    scrollToAnalysis("#profile-gaps");
  } else if (resultPanel && !resultPanel.classList.contains("hidden")) {
    scrollToAnalysis("#result-panel");
  } else {
    scrollToAnalysis("#summary-panel");
  }
}

// Smooth-scroll a target element into view + briefly flash a border so the
// user's eye lands on it. Used during the upload flow so the user never has
// to manually scroll to find the analysis.
function scrollToAnalysis(selector) {
  const el = document.querySelector(selector);
  if (!el) return;
  // Account for the sticky nav (~60px) so the panel header isn't hidden.
  const navOffset = 72;
  const top = el.getBoundingClientRect().top + window.scrollY - navOffset;
  window.scrollTo({ top, behavior: "smooth" });
  // Brief visual flash to confirm focus
  el.classList.remove("scroll-target-flash");
  // Force reflow so re-adding the class restarts the animation
  void el.offsetWidth;
  el.classList.add("scroll-target-flash");
  setTimeout(() => el.classList.remove("scroll-target-flash"), 1500);
}

// ============================================================
// DECISION INTELLIGENCE SUMMARY (readiness · peers · what-if)
// ============================================================

// Last-rendered baseline result + profile — what-if uses these to know the
// "before" numbers and the StudentProfile fields to send to /api/whatif.
let __summaryBaseline = null;
let __whatifInFlight = null; // AbortController of pending request, if any

function renderSummary(r) {
  __summaryBaseline = r;
  const panel = $("#summary-panel");
  if (!panel) return;

  const readiness = r.readiness || {};
  const peer = r.peer_benchmark || {};

  // Show the panel
  panel.classList.remove("hidden");

  // Meta line — show profile name + cohort context
  const meta = $("#summary-meta");
  if (meta) {
    const name = (r.profile?.name && r.profile.name !== "Anonymous") ? r.profile.name : "";
    const cohort = peer.cohort_size ? `${peer.cohort_size} peers` : "";
    const lvl = peer.level ? peer.level : "";
    const parts = [name, cohort && lvl ? `${cohort} · ${lvl}` : (cohort || lvl)].filter(Boolean);
    meta.textContent = parts.join(" · ") || "—";
  }

  renderReadiness(readiness);
  renderPeer(peer);
  renderWhatIfInitial(r);
}

function renderReadiness(readiness) {
  if (!readiness || readiness.available === false) {
    $("#readiness-num").textContent = "—";
    $("#readiness-narrative").textContent = "Readiness unavailable.";
    return;
  }
  const target = Number(readiness.score || 0);
  // Animate count-up
  animateNumber($("#readiness-num"), 0, target, 1100, v => Math.round(v));

  // Stroke gradient color and arc length
  const bar = $("#readiness-bar");
  const C = 2 * Math.PI * 60;   // r=60
  bar.style.strokeDasharray = String(C);
  // Reset to 0 first (forces a fresh transition)
  bar.style.strokeDashoffset = String(C);
  // Pick the gradient color based on band
  const color = readinessColor(readiness.band);
  bar.style.stroke = color;
  // Run the animation on next frame
  requestAnimationFrame(() => {
    bar.style.strokeDashoffset = String(C - (C * target / 100));
  });

  // Band pill
  const bandEl = $("#readiness-band");
  const tierClass = readiness.tier === "LOW"
    ? ""
    : readiness.tier === "MEDIUM" ? "t-medium" : "t-high";
  bandEl.className = `readiness-band ${tierClass}`;
  bandEl.textContent = `${readiness.band || "—"} · ${readiness.tier || "—"} RISK`;

  // Narrative
  $("#readiness-narrative").textContent = readiness.narrative || "—";

  // Sub-score bars
  const bars = $("#readiness-bars");
  bars.innerHTML = "";
  const order = ["academic", "exposure", "skill", "activity", "institute"];
  order.forEach((key, idx) => {
    const c = (readiness.components || {})[key];
    if (!c) return;
    const pct = Math.round((c.score || 0) * 100);
    const labelClass = (c.label || "").toLowerCase();
    const row = document.createElement("div");
    row.className = "readiness-bar-row";
    row.innerHTML = `
      <span class="readiness-bar-label" title="${escapeHtml(c.detail || '')}">${key}</span>
      <span class="readiness-bar-track">
        <span class="readiness-bar-fill ${labelClass}" style="width:0%"></span>
      </span>
      <span class="readiness-bar-val">${pct}</span>
    `;
    bars.appendChild(row);
    // Stagger the fill animation
    setTimeout(() => {
      const fill = row.querySelector(".readiness-bar-fill");
      if (fill) fill.style.width = `${pct}%`;
    }, 200 + idx * 90);
  });
}

function readinessColor(band) {
  if (band === "STRONG") return "#22c55e";
  if (band === "MEDIUM") return "#f59e0b";
  return "#ef4444";
}

function renderPeer(peer) {
  if (!peer || !peer.available) {
    $("#peer-narrative").textContent = "Peer cohort unavailable.";
    $("#peer-cohort-pill").textContent = "—";
    $("#peer-rows").innerHTML = "";
    return;
  }
  const cohortPill = $("#peer-cohort-pill");
  if (cohortPill) {
    const n = peer.cohort_size ? ` · n=${peer.cohort_size}` : "";
    cohortPill.textContent = `${peer.definition || "—"}${n}`;
  }
  $("#peer-narrative").textContent = peer.comparison?.narrative || "—";

  // Three side-by-side comparison bars: months, salary, P(6m)
  const rows = $("#peer-rows");
  rows.innerHTML = "";
  const items = [
    {
      label: "Months to placement",
      student: peer.student.expected_months_to_placement,
      peer:    peer.peer.median_months_to_placement,
      // Faster (smaller) is better — invert the bar fill so a quicker
      // student fills less of the bar AND we color it positive.
      max: Math.max(peer.student.expected_months_to_placement, peer.peer.median_months_to_placement, 12),
      suffix: " mo",
      lowerIsBetter: true,
    },
    {
      label: "Median salary (LPA)",
      student: peer.student.median_salary_lpa,
      peer:    peer.peer.salary_p50_lpa,
      max: Math.max(peer.student.median_salary_lpa, peer.peer.salary_p50_lpa) * 1.15 || 25,
      suffix: " L",
      lowerIsBetter: false,
    },
    {
      label: "P(placed @ 6m)",
      student: peer.student.p_6m * 100,
      peer:    peer.peer.p_6m * 100,
      max: 100,
      suffix: "%",
      lowerIsBetter: false,
    },
  ];
  items.forEach((it, idx) => {
    const row = document.createElement("div");
    row.className = "peer-row";
    const studentPct = Math.min(100, (it.student / it.max) * 100);
    const peerPct = Math.min(100, (it.peer / it.max) * 100);
    row.innerHTML = `
      <div class="peer-row-head">
        <span class="peer-row-label">${it.label}</span>
        <span class="peer-row-val">
          <span style="color:#f15a29">${formatBarVal(it.student, it.suffix)}</span>
          <span style="color:var(--ink-3)">vs ${formatBarVal(it.peer, it.suffix)}</span>
        </span>
      </div>
      <div class="peer-bar">
        <div class="peer-bar-label"><span>You</span><span>${formatBarVal(it.student, it.suffix)}</span></div>
        <div class="peer-bar-track"><div class="peer-bar-fill student" style="width:0%"></div></div>
      </div>
      <div class="peer-bar">
        <div class="peer-bar-label"><span>Peer median</span><span>${formatBarVal(it.peer, it.suffix)}</span></div>
        <div class="peer-bar-track"><div class="peer-bar-fill peer" style="width:0%"></div></div>
      </div>
    `;
    rows.appendChild(row);
    // Fill in next frame so transition fires
    setTimeout(() => {
      const fills = row.querySelectorAll(".peer-bar-fill");
      if (fills[0]) fills[0].style.width = `${studentPct}%`;
      if (fills[1]) fills[1].style.width = `${peerPct}%`;
    }, 200 + idx * 110);
  });

  // Percentile sliders (markers and fills)
  const salaryPct = Math.round((peer.comparison.salary_percentile || 0) * 100);
  const speedPct  = Math.round((peer.comparison.speed_percentile  || 0) * 100);
  setTimeout(() => {
    $("#peer-salary-pct-fill").style.width = `${salaryPct}%`;
    $("#peer-salary-pct-marker").style.left = `${salaryPct}%`;
    $("#peer-speed-pct-fill").style.width = `${speedPct}%`;
    $("#peer-speed-pct-marker").style.left = `${speedPct}%`;
  }, 350);
  $("#peer-salary-pct-val").textContent = `top ${100 - salaryPct}%`;
  $("#peer-speed-pct-val").textContent  = `top ${100 - speedPct}%`;
}

function formatBarVal(v, suffix) {
  if (suffix === "%")  return `${Number(v).toFixed(0)}%`;
  if (suffix === " L") return `₹${Number(v).toFixed(1)}L`;
  if (suffix === " mo") return `${Number(v).toFixed(1)} mo`;
  return `${v}${suffix}`;
}

// What-If: build the toggle pills, wire change handlers, render the initial
// "before" state from the baseline.
function renderWhatIfInitial(r) {
  const togglesEl = $("#whatif-toggles");
  togglesEl.innerHTML = "";

  // Initial pre-baked catalog (kept in sync with backend INTERVENTION_CATALOG)
  const catalog = [
    { id: "add_top_tier_internship", label: "FAANG-tier internship",   icon: "★" },
    { id: "add_internship",          label: "Industry internship",     icon: "+" },
    { id: "boost_cgpa",              label: "+0.5 CGPA",                icon: "↑" },
    { id: "add_skill",               label: "+1 in-demand skill",       icon: "✦" },
    { id: "add_certification",       label: "Cloud / data cert",        icon: "✓" },
    { id: "add_coding_problems",     label: "+200 coding problems",     icon: "{ }" },
    { id: "add_hackathon_win",       label: "Hackathon win",            icon: "⚡" },
    { id: "boost_activity",          label: "Engage portal +interview", icon: "↗" },
    { id: "remove_backlog",          label: "Clear a backlog",          icon: "✕" },
  ];
  catalog.forEach(item => {
    const btn = document.createElement("button");
    btn.className = "whatif-toggle";
    btn.dataset.id = item.id;
    btn.type = "button";
    btn.innerHTML = `<span class="whatif-toggle-icon">${item.icon}</span>${escapeHtml(item.label)}`;
    btn.addEventListener("click", () => {
      btn.classList.toggle("active");
      runWhatIfFromToggles();
    });
    togglesEl.appendChild(btn);
  });

  // Initial "before" = baseline values; "after" = same (no interventions yet)
  const pp = r.placement_probabilities || {};
  const sb = r.salary_band_lpa || {};
  const beforePct = Math.round((pp.p_6m || 0) * 100);
  const beforeSal = (sb.median || 0).toFixed(1);

  $("#whatif-before-pct").textContent = `${beforePct}%`;
  $("#whatif-after-pct").textContent  = `${beforePct}%`;
  $("#whatif-before-fill").style.width = `${beforePct}%`;
  $("#whatif-after-fill").style.width  = `${beforePct}%`;
  $("#whatif-delta").textContent = "+0.0pp";
  $("#whatif-delta").className = "whatif-delta-pill zero";
  $("#whatif-narrative").textContent = "Pick at least one intervention to simulate.";
  $("#whatif-salary-before").textContent = `₹${beforeSal}L`;
  $("#whatif-salary-after").textContent  = `₹${beforeSal}L`;
  $("#whatif-salary-delta").textContent  = "+0.0L";
  $("#whatif-salary-delta").className = "whatif-salary-delta";
}

async function runWhatIfFromToggles() {
  const ids = $$(".whatif-toggle.active").map(b => b.dataset.id);
  if (!__summaryBaseline) return;

  // No interventions = reset to baseline
  if (ids.length === 0) {
    renderWhatIfInitial(__summaryBaseline);
    return;
  }

  // Cancel any in-flight call so rapid toggling doesn't race
  if (__whatifInFlight) __whatifInFlight.abort();
  const ctrl = new AbortController();
  __whatifInFlight = ctrl;

  $("#whatif-loading").classList.remove("hidden");
  $("#whatif-narrative").textContent = "Re-running model with new profile…";

  try {
    // Send the profile back to the server. We trim the profile to fields the
    // ProfileBody pydantic model expects so the backend doesn't reject it.
    const p = __summaryBaseline.profile || {};
    const profileBody = {
      name: p.name || "Anonymous",
      course_type: p.course_type || "BTech-CS",
      institute_name: p.institute_name || "Tier-2 Institute",
      institute_tier: p.institute_tier ?? 2,
      region: p.region || "Tier2",
      graduation_year: p.graduation_year || 2024,
      cgpa: Number(p.cgpa ?? 7.0),
      backlogs_count: p.backlogs_count ?? 0,
      internships: p.internships || [],
      certifications: p.certifications || [],
      skills: p.skills || [],
      projects_count: p.projects_count ?? 0,
      github_projects: p.github_projects ?? 0,
      coding_problem_count: p.coding_problem_count ?? 0,
      hackathon_wins: p.hackathon_wins ?? 0,
      leadership_roles_count: p.leadership_roles_count ?? 0,
      paper_publications: p.paper_publications ?? 0,
      extracurriculars_count: p.extracurriculars_count ?? 0,
      languages_known: p.languages_known ?? 1,
      portal_activity_30d: p.portal_activity_30d ?? 5,
      interview_invites_count: p.interview_invites_count ?? 0,
      salary_expectation_lpa: Number(p.salary_expectation_lpa ?? 6.0),
      // Preserve elite-outlier flag so salary anchor is consistent on round-trip
      is_elite_outlier: p.is_elite_outlier || false,
      elite_outlier_reasons: p.elite_outlier_reasons || [],
    };

    const res = await fetch("/api/whatif", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: profileBody, interventions: ids }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    renderWhatIfDelta(data.result);
  } catch (e) {
    if (e.name === "AbortError") return;       // newer call superseded this one
    console.error("whatif failed:", e);
    $("#whatif-narrative").textContent = "Simulation failed: " + e.message;
  } finally {
    if (__whatifInFlight === ctrl) {
      __whatifInFlight = null;
      $("#whatif-loading").classList.add("hidden");
    }
  }
}

function renderWhatIfDelta(result) {
  if (!result) return;
  const after  = result.after  || {};
  const delta  = result.delta  || {};

  // Always use the ORIGINAL baseline for "before" — the API re-computes it
  // and tiny floating-point differences (from profile round-trip) cause
  // confusing inconsistencies between the initial display and what-if view.
  const baselinePP = __summaryBaseline?.placement_probabilities || {};
  const baselineSB = __summaryBaseline?.salary_band_lpa || {};
  const beforePct = Math.round((baselinePP.p_6m || 0) * 100);
  const beforeSal = Number(baselineSB.median || 0);

  const afterPct  = Math.round((after.p_6m || 0) * 100);
  const afterSal  = Number(after.salary_median || 0);

  $("#whatif-before-pct").textContent = `${beforePct}%`;
  $("#whatif-after-pct").textContent  = `${afterPct}%`;

  // Animate the bars
  $("#whatif-before-fill").style.width = `${beforePct}%`;
  requestAnimationFrame(() => {
    $("#whatif-after-fill").style.width = `${afterPct}%`;
  });

  // Recompute delta against true baseline (not API's re-computed before)
  const dpp = afterPct - beforePct;
  const sign = dpp > 0 ? "+" : "";
  const deltaEl = $("#whatif-delta");
  deltaEl.textContent = `${sign}${dpp.toFixed(1)}pp`;
  deltaEl.className = "whatif-delta-pill " + (dpp > 0 ? "" : dpp < 0 ? "neg" : "zero");

  $("#whatif-narrative").textContent = result.narrative || "";

  const sDelta = afterSal - beforeSal;
  $("#whatif-salary-before").textContent = `₹${beforeSal.toFixed(1)}L`;
  $("#whatif-salary-after").textContent  = `₹${afterSal.toFixed(1)}L`;
  const sd = $("#whatif-salary-delta");
  sd.textContent = (sDelta >= 0 ? "+" : "") + sDelta.toFixed(2) + "L";
  sd.className = "whatif-salary-delta" + (sDelta < 0 ? " neg" : "");
}

// ============================================================
// PROFILE GAPS — amber warning card when critical fields were imputed
// ============================================================

// Set to true for exactly one renderResult call after a correction re-run
// so the gaps card isn't re-opened when the corrected profile is re-scored
// (the corrected /api/predict call has no field_confidence, so backend
// would report all fields as imputed again).
let __suppressGapsOnce = false;

const _COURSE_TYPE_OPTIONS = [
  "BTech-CS", "BTech-ECE", "BTech-ME", "BTech-CE", "BTech-EE",
  "MBA-General", "MBA-Finance", "MCA", "MSc-CS", "BCA", "BSc-CS",
];

function maybeShowProfileGaps(r) {
  const card = $("#profile-gaps");
  if (!card) return;
  // After a correction re-run, skip for this one call
  if (__suppressGapsOnce) { __suppressGapsOnce = false; card.classList.add("hidden"); return; }
  // Only show for real resume parses — not demo personas or direct /api/predict calls
  const parserUsed = r.parser_used || "";
  const isResumeResult = parserUsed !== "demo_profile" && parserUsed !== "corrected"
                      && parserUsed !== "" && parserUsed !== "manual";
  const imputed = (isResumeResult ? r.imputed_fields : null) || [];
  if (imputed.length === 0) {
    card.classList.add("hidden");
    return;
  }
  buildGapsForm(imputed);
  card.classList.remove("hidden");

  // Wire dismiss — one-shot onclick so it doesn't stack
  const dismiss = $("#gaps-dismiss-btn");
  if (dismiss) dismiss.onclick = () => card.classList.add("hidden");

  // Wire rerun
  const rerun = $("#gaps-rerun-btn");
  if (rerun) rerun.onclick = () => rerunWithCorrections();
}

function buildGapsForm(imputed) {
  const container = $("#gaps-fields");
  if (!container) return;
  container.innerHTML = "";
  for (const item of imputed) {
    const div = document.createElement("div");
    div.className = "gaps-field";
    let inputHtml;
    if (item.field === "course_type") {
      const opts = _COURSE_TYPE_OPTIONS.map(o =>
        `<option value="${o}"${o === item.imputed ? " selected" : ""}>${o}</option>`
      ).join("");
      inputHtml = `<select data-field="${escapeHtml(item.field)}">${opts}</select>`;
    } else if (item.field === "cgpa") {
      // item.imputed may be "7.0 (population mean)" — extract just the number
      const cgpaNum = parseFloat(item.imputed);
      const cgpaVal = (!isNaN(cgpaNum) && cgpaNum >= 1.0) ? cgpaNum : "";
      inputHtml = `<input type="number" data-field="${escapeHtml(item.field)}"
        min="1.0" max="10.0" step="0.1" value="${cgpaVal}" placeholder="e.g. 7.5">`;
    } else if (item.field === "backlogs_count") {
      const blNum = parseInt(item.imputed, 10);
      const blVal = !isNaN(blNum) ? blNum : 0;
      inputHtml = `<input type="number" data-field="${escapeHtml(item.field)}"
        min="0" max="20" step="1" value="${blVal}" placeholder="0">`;
    } else {
      // institute_name and any other text fields
      inputHtml = `<input type="text" data-field="${escapeHtml(item.field)}"
        value="${escapeHtml(String(item.imputed))}" placeholder="Enter ${escapeHtml(item.label)}">`;
    }
    div.innerHTML = `
      <span class="gaps-field-label">${escapeHtml(item.label)}</span>
      <span class="gaps-field-note">estimated: ${escapeHtml(String(item.imputed))}</span>
      ${inputHtml}
    `;
    container.appendChild(div);
  }
}

async function rerunWithCorrections() {
  if (!__summaryBaseline) return;
  const btn = $("#gaps-rerun-btn");
  if (btn) { btn.disabled = true; btn.textContent = "Re-running…"; }

  // Collect corrected values from the form
  const corrections = {};
  $$("#gaps-fields [data-field]").forEach(input => {
    const field = input.dataset.field;
    const raw = (input.value || "").trim();
    if (!raw) return;
    if (field === "cgpa") {
      const v = parseFloat(raw);
      if (!isNaN(v) && v >= 1.0 && v <= 10.0) corrections[field] = v;
    } else if (field === "backlogs_count") {
      const v = parseInt(raw, 10);
      if (!isNaN(v) && v >= 0) corrections[field] = v;
    } else {
      corrections[field] = raw;
    }
  });

  // Build the full profile body: baseline + corrections merged in
  const p = __summaryBaseline.profile || {};
  const profileBody = {
    name:                     p.name || "Anonymous",
    course_type:              corrections.course_type      ?? p.course_type      ?? "BTech-CS",
    institute_name:           corrections.institute_name   ?? p.institute_name   ?? "Tier-2 Institute",
    institute_tier:           p.institute_tier ?? 2,
    region:                   p.region || "Tier2",
    graduation_year:          p.graduation_year || 2024,
    cgpa:                     corrections.cgpa             ?? Number(p.cgpa ?? 7.0),
    backlogs_count:           corrections.backlogs_count   ?? (p.backlogs_count ?? 0),
    internships:              p.internships || [],
    certifications:           p.certifications || [],
    skills:                   p.skills || [],
    projects_count:           p.projects_count ?? 0,
    github_projects:          p.github_projects ?? 0,
    coding_problem_count:     p.coding_problem_count ?? 0,
    hackathon_wins:           p.hackathon_wins ?? 0,
    leadership_roles_count:   p.leadership_roles_count ?? 0,
    paper_publications:       p.paper_publications ?? 0,
    extracurriculars_count:   p.extracurriculars_count ?? 0,
    languages_known:          p.languages_known ?? 1,
    portal_activity_30d:      p.portal_activity_30d ?? 5,
    interview_invites_count:  p.interview_invites_count ?? 0,
    salary_expectation_lpa:   Number(p.salary_expectation_lpa ?? 6.0),
    is_elite_outlier:         p.is_elite_outlier || false,
    elite_outlier_reasons:    p.elite_outlier_reasons || [],
  };

  try {
    const data = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profileBody),
    });
    // Mark parser as "corrected" so the feed shows a meaningful label
    data.parser_used = data.parser_used || "corrected";
    data.latency_ms  = data.latency_ms  || 0;
    // Suppress the gaps card for this one renderResult call — the corrections
    // the user just entered should not trigger another warning cycle.
    __suppressGapsOnce = true;
    renderResult(data);
  } catch (e) {
    alert("Re-analysis failed: " + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Re-analyse with corrections →"; }
  }
}


// Generic count-up animation. el: DOM element; from -> to over `dur` ms;
// `formatter(value)` returns the string to set as textContent.
function animateNumber(el, from, to, dur = 900, formatter = v => Math.round(v)) {
  if (!el) return;
  const start = performance.now();
  const tick = (t) => {
    const k = Math.min(1, (t - start) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    const v = from + (to - from) * eased;
    el.textContent = formatter(v);
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}


function drawSurvivalCurve(curve, risk, iprCurve) {
  const canvas = $("#survival-canvas");
  if (survivalChart) survivalChart.destroy();
  const ctx = canvas.getContext("2d");
  // Build gradient under line
  const grad = ctx.createLinearGradient(0, 0, 0, 160);
  grad.addColorStop(0, "rgba(241, 90, 41, 0.35)");
  grad.addColorStop(1, "rgba(241, 90, 41, 0.0)");

  // Primary curve: this student's predicted survival
  const datasets = [{
    label: "This student",
    data: curve.map(c => c.p_unplaced),
    fill: true,
    backgroundColor: grad,
    borderColor: "#f15a29",
    borderWidth: 2.5,
    tension: 0.35,
    pointRadius: 0,
    pointHoverRadius: 4,
    pointHoverBackgroundColor: "#f15a29",
    order: 1,
  }];

  // Reference curve: institute average (if IPR data is available)
  if (iprCurve && iprCurve.length) {
    datasets.push({
      label: "Institute average",
      data: iprCurve.map(c => c.p_unplaced),
      fill: false,
      borderColor: "rgba(255,255,255,0.55)",
      borderWidth: 1.5,
      borderDash: [5, 4],
      tension: 0.30,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: "#fff",
      order: 0,
    });
  }

  survivalChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: curve.map(c => c.month),
      datasets: datasets,
    },
    options: {
      animation: false,
      plugins: {
        legend: {
          display: !!(iprCurve && iprCurve.length),
          position: "top",
          align: "end",
          labels: { color: "rgba(255,255,255,0.75)", font: { size: 11 }, boxWidth: 18, boxHeight: 2 },
        },
        tooltip: {
          backgroundColor: "#0a1f44",
          callbacks: {
            title: items => `Month ${items[0].label}`,
            label: ctx => `${ctx.dataset.label}: P(unplaced) = ${(ctx.parsed.y * 100).toFixed(1)}%`,
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "rgba(255,255,255,0.5)", font: { size: 11 } },
          title: { display: true, text: "Months since graduation", color: "rgba(255,255,255,0.45)", font: { size: 11 } },
        },
        y: {
          min: 0, max: 1,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: {
            color: "rgba(255,255,255,0.5)", font: { size: 11 },
            callback: v => `${(v * 100).toFixed(0)}%`,
          },
        },
      },
    },
  });
}


// ----------- portfolio tab -----------
let portfolioLoaded = false;
let portfolioCache = null;
let riskDistChart  = null;

async function loadPortfolio() {
  try {
    portfolioCache = await api("/api/portfolio?limit=200");
    portfolioLoaded = true;
    renderPortfolio("all");
  } catch (e) {
    console.error(e);
  }
}

function renderPortfolio(filter) {
  const d = portfolioCache;
  const s = d.summary;
  $("#m-students").textContent = s.n_students;
  $("#m-loan").textContent = `₹${s.total_loan_lakhs.toLocaleString()}L`;
  $("#m-risk").textContent = `₹${s.at_risk_loan_lakhs.toLocaleString()}L`;
  $("#m-high").textContent = s.high_risk_count;
  $("#m-med").textContent = s.medium_risk_count;
  $("#m-low").textContent = s.low_risk_count;

  // PRI overall metric
  const mPri = $("#m-pri");
  if (mPri) mPri.textContent = s.pri_overall.toFixed(1);

  // Distribution chart
  const distCtx = $("#risk-dist-canvas").getContext("2d");
  if (riskDistChart) riskDistChart.destroy();
  riskDistChart = new Chart(distCtx, {
    type: "doughnut",
    data: {
      labels: ["High", "Medium", "Low"],
      datasets: [{
        data: [s.high_risk_count, s.medium_risk_count, s.low_risk_count],
        backgroundColor: ["#ef4444", "#f59e0b", "#22c55e"],
        borderWidth: 0,
      }],
    },
    options: {
      cutout: "65%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, font: { size: 12 }, color: "#364259" },
        },
      },
    },
  });

  // Sector bars (PRI by sector)
  const sb = $("#sector-bars");
  sb.innerHTML = "";
  const entries = Object.entries(d.pri_by_sector).sort((a, b) => b[1].value - a[1].value);
  const maxV = Math.max(...entries.map(([, v]) => v.value), 0.001);
  for (const [name, v] of entries) {
    const row = document.createElement("div");
    row.className = "sector-bar";
    const riskCls = v.value >= 50 ? "sector-high" : v.value >= 30 ? "sector-mid" : "sector-low";
    row.innerHTML = `
      <div class="sector-bar-name">${name}<span class="sector-n">n=${v.n}</span></div>
      <div class="sector-bar-track"><div class="sector-bar-fill ${riskCls}" style="width:${(v.value/maxV)*100}%"></div></div>
      <div class="sector-bar-val ${riskCls}">${v.value.toFixed(1)}</div>
    `;
    sb.appendChild(row);
  }

  // Table
  const tbody = $("#portfolio-tbody");
  tbody.innerHTML = "";
  const rows = filter === "all"
    ? d.students
    : d.students.filter(s => s.risk_tier.toLowerCase() === filter);

  for (const s of rows.slice(0, 80)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><div class="borrower-cell"><strong>${s.name}</strong><span class="id">${s.id}</span></div></td>
      <td>${s.course}</td>
      <td><span title="${s.region}">${s.institute}</span></td>
      <td>${s.cgpa.toFixed(2)}</td>
      <td>₹${s.loan_lakhs}L</td>
      <td>${(s.p_6m*100).toFixed(0)}%</td>
      <td>₹${s.salary_lpa_med.toFixed(1)}L</td>
      <td>
        <span class="tier-tag ${s.risk_tier}">${s.risk_tier} · ${s.risk_score}</span>
        <div class="risk-bar ${s.risk_tier}"><span style="width:${s.risk_score}%"></span></div>
      </td>
    `;
    tbody.appendChild(tr);
  }

  // Filters
  $$(".filter-btn").forEach(b => b.classList.toggle("active", b.dataset.filter === filter));
}

function setupPortfolioFilters() {
  $$(".filter-btn").forEach(b => {
    b.addEventListener("click", () => {
      if (!portfolioCache) return;
      renderPortfolio(b.dataset.filter);
    });
  });
}


// ----------- PRI tab -----------
let priLoaded = false;
let priChart = null;

async function loadPri() {
  try {
    const d = await api("/api/pri");
    priLoaded = true;

    $("#pri-now").textContent = d.overall.toFixed(1);

    // Delta vs nearest prior month that actually differs (PRI falling = improving = green)
    const hist = d.history;
    const deltaEl = $("#pri-delta");
    if (deltaEl && hist.length >= 2) {
      // Walk back to find the last month with a meaningfully different PRI
      let prevIdx = hist.length - 2;
      while (prevIdx > 0 && Math.abs(hist[prevIdx].pri - d.overall) < 0.05) prevIdx--;
      const prev = hist[prevIdx].pri;
      const delta = +(d.overall - prev).toFixed(1);
      const sign = delta > 0 ? "+" : "";
      const isGood = delta < 0;   // PRI down = fewer at-risk = good
      const prevLabel = hist[prevIdx].month;
      deltaEl.textContent = `${sign}${delta} vs ${prevLabel}`;
      deltaEl.className = `pri-delta ${isGood ? "pri-delta-good" : delta > 0 ? "pri-delta-bad" : "pri-delta-flat"}`;
    }

    const ctx = $("#pri-canvas").getContext("2d");
    if (priChart) priChart.destroy();
    const grad = ctx.createLinearGradient(0, 0, 0, 200);
    grad.addColorStop(0, "rgba(241,90,41,0.30)");
    grad.addColorStop(1, "rgba(241,90,41,0.0)");

    priChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: d.history.map(h => h.month || `W${h.week}`),
        datasets: [
          {
            label: "Monthly PRI",
            data: d.history.map(h => h.pri),
            fill: true,
            backgroundColor: grad,
            borderColor: "#f15a29",
            borderWidth: 2.5,
            tension: 0.4,
            pointRadius: 3,
            pointBackgroundColor: "#f15a29",
            yAxisID: "y",
          },
          {
            label: "JobSpeak Index",
            data: d.history.map(h => h.jobspeak ?? null),
            borderColor: "rgba(107,118,145,0.6)",
            borderDash: [6, 4],
            borderWidth: 1.6,
            pointRadius: 0,
            fill: false,
            yAxisID: "y2",
            spanGaps: true,
          },
        ],
      },
      options: {
        animation: false,
        plugins: {
          legend: {
            display: true,
            position: "top",
            align: "end",
            labels: { color: "#6b7691", font: { size: 11 }, boxWidth: 22, boxHeight: 2, padding: 14 },
          },
          tooltip: {
            backgroundColor: "#0a1f44",
            callbacks: {
              label: ctx => ctx.dataset.yAxisID === "y2"
                ? `JobSpeak: ${ctx.parsed.y.toLocaleString()}`
                : `PRI: ${ctx.parsed.y.toFixed(1)}`,
            },
          },
        },
        scales: {
          x: { grid: { color: "rgba(0,0,0,0.04)" }, ticks: { color: "#6b7691", font: { size: 11 } } },
          y: {
            grid: { color: "rgba(0,0,0,0.04)" },
            ticks: { color: "#6b7691", font: { size: 11 } },
            title: { display: true, text: "PRI", color: "#f15a29", font: { size: 11, weight: 600 } },
          },
          y2: {
            position: "right",
            grid: { display: false },
            ticks: { color: "#6b7691", font: { size: 11 }, callback: v => v.toLocaleString() },
            title: { display: true, text: "JobSpeak", color: "#6b7691", font: { size: 11, weight: 600 } },
          },
        },
      },
    });

    const list = $("#pri-sector-list");
    list.innerHTML = "";
    const entries = Object.entries(d.by_sector).sort((a, b) => b[1].value - a[1].value);
    const maxV = Math.max(...entries.map(([, v]) => v.value), 0.001);
    for (const [name, v] of entries) {
      const row = document.createElement("div");
      row.className = "pri-sector";
      const riskCls = v.value >= 50 ? "sector-high" : v.value >= 30 ? "sector-mid" : "sector-low";
      row.innerHTML = `
        <div class="pri-sector-name">${name}</div>
        <div class="pri-sector-bar"><span class="${riskCls}" style="width:${(v.value/maxV)*100}%"></span></div>
        <div class="pri-sector-val ${riskCls}">${v.value.toFixed(1)}<span class="sector-n"> n=${v.n}</span></div>
      `;
      list.appendChild(row);
    }
  } catch (e) {
    console.error(e);
  }
}


// ----------- architecture tab -----------
let archLoaded = false;

const INNOVATIONS = [
  { n: "01", title: "DeepHit Survival",           sub: "Discrete-time deep survival on {3,6,12}-month grid (Lee 2018).",        tag: "LIVE",      cls: "live" },
  { n: "02", title: "Drift Detection",            sub: "PSI + Bonferroni-corrected KS over the feature store.",                tag: "LIVE",      cls: "live" },
  { n: "03", title: "Federated SHAP + DP",        sub: "Gaussian-mechanism DP (ε=1.0) across lender shards. ρ=0.94.",            tag: "LIVE",      cls: "live" },
  { n: "04", title: "Conformal Risk Control",     sub: "MAPIE Learn-Then-Test (Bates 2024). FNR ≤ 10% w.p. ≥ 90%.",            tag: "LIVE",      cls: "live" },
  { n: "05", title: "Thompson Sampling Bandit",   sub: "Beta-Bernoulli posteriors per action × segment + /api/record_outcome.", tag: "LIVE",      cls: "live" },
  { n: "06", title: "PC Algorithm Causal DAG",    sub: "Auto-discovered CPDAG + Markov blanket (causal-learn).",               tag: "LIVE",      cls: "live" },
  { n: "07", title: "Born-Again Edge Tree",       sub: "Distilled to 10.7KB int16 JSON tree (Vidal & Schiffer 2020).",          tag: "LIVE",      cls: "live" },
  { n: "08", title: "Counterfactual ATE",         sub: "EconML LinearDRLearner per action + DoWhy placebo refuter.",            tag: "LIVE",      cls: "live" },
  { n: "09", title: "PlacementRisk Index (PRI)",  sub: "Naukri-JobSpeak-anchored macro signal, 13-month history.",              tag: "LIVE",      cls: "live" },
  { n: "10", title: "Live Resume-to-Risk",        sub: "Upload PDF → 30-second risk score + SHAP + Co-Pilot.",                  tag: "LIVE",      cls: "live" },
  { n: "11", title: "Beta Calibration",           sub: "Kull 2017 over 15% holdout. ECE 0.027 < 0.05 target.",                  tag: "LIVE",      cls: "live" },
  { n: "12", title: "NIRF 2024 Priors",           sub: "Per-institute placement + salary signal via fuzzy NIRF lookup.",        tag: "LIVE",      cls: "live" },
  { n: "13", title: "Fairlearn Audit",            sub: "DPDP §10 + EEOC four-fifths rule on tier × is_metro × course.",          tag: "LIVE",      cls: "live" },
  { n: "14", title: "Kaggle OOD Validation",      sub: "Real-label transfer test, with covariate-shift fallback.",              tag: "LIVE",      cls: "live" },
];

async function renderArchitecture() {
  archLoaded = true;

  const list = $("#innov-list");
  list.innerHTML = "";
  for (const i of INNOVATIONS) {
    const node = document.createElement("div");
    node.className = "innov";
    node.innerHTML = `
      <div class="innov-num">${i.n}</div>
      <div class="innov-body">
        <div class="innov-title">${i.title}</div>
        <div class="innov-sub">${i.sub}</div>
      </div>
      <div class="innov-tag ${i.cls || ""}">${i.tag}</div>
    `;
    list.appendChild(node);
  }

  try {
    const h = await api("/api/health");
    const grid = $("#model-meta-grid");
    const cal6m = h.calibration?.placed_6m;
    const eceStr = cal6m ? cal6m.ece_post.toFixed(3) : "—";
    grid.innerHTML = `
      <div class="mm-cell"><div class="mm-label">Train rows</div><div class="mm-val">${h.n_train.toLocaleString()}</div></div>
      <div class="mm-cell"><div class="mm-label">AUC · 3m</div><div class="mm-val">${h.model_aucs.placed_3m.toFixed(3)}</div></div>
      <div class="mm-cell"><div class="mm-label">AUC · 6m</div><div class="mm-val">${h.model_aucs.placed_6m.toFixed(3)}</div></div>
      <div class="mm-cell"><div class="mm-label">AUC · 12m</div><div class="mm-val">${h.model_aucs.placed_12m.toFixed(3)}</div></div>
      <div class="mm-cell"><div class="mm-label">ECE · 6m</div><div class="mm-val" style="color:${cal6m && cal6m.ece_post < 0.05 ? '#22c55e' : '#f59e0b'}">${eceStr}</div></div>
      <div class="mm-cell"><div class="mm-label">Salary MAE</div><div class="mm-val">₹${h.salary_mae_lpa.toFixed(2)}L</div></div>
      <div class="mm-cell"><div class="mm-label">Inference</div><div class="mm-val">&lt;100ms</div></div>
      <div class="mm-cell"><div class="mm-label">Status</div><div class="mm-val" style="color:#22c55e">LIVE</div></div>
    `;
    if (cal6m) renderReliabilityChart(cal6m);
  } catch (e) {}

  try {
    const ood = await api("/api/ood");
    renderOOD(ood);
  } catch (e) {}

  try {
    const drift = await api("/api/drift");
    renderDrift(drift);
  } catch (e) {}

  try {
    const bandit = await api("/api/bandit_state");
    renderBandit(bandit);
  } catch (e) {}

  try {
    const fair = await api("/api/fairness");
    renderFairness(fair);
  } catch (e) {}

  try {
    const c = await api("/api/causal");
    renderCausal(c);
  } catch (e) {}

  try {
    const fed = await api("/api/federated_shap");
    renderFederated(fed);
  } catch (e) {}

  try {
    const cf = await api("/api/counterfactual");
    renderCounterfactual(cf);
  } catch (e) {}

  try {
    const edge = await api("/api/edge_model");
    renderEdge(edge);
  } catch (e) {}
}

function renderEdge(e) {
  const meta = $("#edge-meta");
  const sum = $("#edge-summary");
  if (!sum) return;
  if (!e || e.size_bytes == null) {
    sum.innerHTML = `<div class="es"><div class="es-label">Status</div><div class="es-val warn">no artefact</div></div>`;
    return;
  }
  const sizeKb = (e.size_bytes / 1024).toFixed(1);
  const sizeOk = e.size_bytes <= 47 * 1024;
  const gapOk = e.auc_gap_pp < 5;
  const jaccOk = e.feature_jaccard_top5 >= 0.4;
  if (meta) {
    meta.innerHTML = `
      <div class="cm"><div class="cm-label">Leaves</div><div class="cm-val">${e.n_leaves}</div></div>
      <div class="cm"><div class="cm-label">Top-5 Jaccard</div><div class="cm-val ${jaccOk ? 'good' : 'warn'}">${e.feature_jaccard_top5.toFixed(2)}</div></div>
    `;
  }
  sum.innerHTML = `
    <div class="es"><div class="es-label">Payload size</div><div class="es-val ${sizeOk ? 'good' : 'warn'}">${sizeKb} KB</div></div>
    <div class="es"><div class="es-label">Teacher AUC</div><div class="es-val">${e.teacher_auc.toFixed(3)}</div></div>
    <div class="es"><div class="es-label">Student AUC</div><div class="es-val">${e.student_auc.toFixed(3)}</div></div>
    <div class="es"><div class="es-label">AUC gap</div><div class="es-val ${gapOk ? 'good' : 'warn'}">${e.auc_gap_pp.toFixed(1)}pp</div></div>
  `;
}

function renderCounterfactual(cf) {
  const meta = $("#counterfactual-meta");
  const wrap = $("#counterfactual-rows");
  if (!meta || !wrap) return;
  if (!cf || !cf.actions || !cf.actions.length) {
    wrap.innerHTML = `<div class="cf-row"><div class="cf-name">No counterfactual artefact.</div></div>`;
    return;
  }
  const passCount = cf.actions.filter(a => a.placebo_passes).length;
  meta.innerHTML = `
    <div class="cm"><div class="cm-label">Actions</div><div class="cm-val">${cf.actions.length}</div></div>
    <div class="cm"><div class="cm-label">Placebo pass</div><div class="cm-val ${passCount === cf.actions.length ? 'good' : 'warn'}">${passCount}/${cf.actions.length}</div></div>
  `;
  // CI track scale: -2pp ... +18pp
  const minPP = -2, maxPP = 18;
  const pos = (pp) => `${Math.max(0, Math.min(100, ((pp - minPP) / (maxPP - minPP)) * 100))}%`;

  wrap.innerHTML = "";
  for (const a of cf.actions) {
    const ciL = pos(a.dr_ci_low_pp);
    const ciR = pos(a.dr_ci_high_pp);
    const ciW = `calc(${ciR} - ${ciL})`;
    const meanLeft = pos(a.dr_ate_pp);
    const passed = a.placebo_passes;
    const div = document.createElement("div");
    div.className = "cf-row";
    div.innerHTML = `
      <div class="cf-name">${a.action_id}</div>
      <div class="cf-ci-track">
        <div class="cf-ci-bar" style="left:${ciL}; width:${ciW}"></div>
        <div class="cf-mean-tick" style="left:${meanLeft}"></div>
      </div>
      <div class="cf-num">DR ${a.dr_ate_pp >= 0 ? '+' : ''}${a.dr_ate_pp.toFixed(1)}pp</div>
      <div class="cf-num naive">Naïve ${a.naive_diff_pp >= 0 ? '+' : ''}${a.naive_diff_pp.toFixed(1)}pp</div>
      <div class="cf-placebo">placebo ${a.placebo_ate_pp >= 0 ? '+' : ''}${a.placebo_ate_pp.toFixed(2)}pp</div>
      <div class="cf-status ${passed ? 'pass' : 'fail'}">${passed ? 'PASS' : 'WATCH'}</div>
    `;
    wrap.appendChild(div);
  }
}

function renderFederated(fs) {
  const meta = $("#federated-meta");
  const cen = $("#federated-centralized");
  const fed = $("#federated-federated");
  if (!meta || !cen || !fed) return;
  if (!fs || !fs.centralized_top) {
    meta.innerHTML = `<div class="cm"><div class="cm-label">Status</div><div class="cm-val">no artefact</div></div>`;
    return;
  }
  const goodFaith = fs.spearman_rho >= 0.9;
  meta.innerHTML = `
    <div class="cm"><div class="cm-label">Spearman ρ</div><div class="cm-val ${goodFaith ? 'good' : 'warn'}">${fs.spearman_rho.toFixed(3)}</div></div>
    <div class="cm"><div class="cm-label">σ (DP)</div><div class="cm-val">${fs.sigma.toFixed(3)}</div></div>
    <div class="cm"><div class="cm-label">Shards</div><div class="cm-val">${fs.n_shards}</div></div>
    <div class="cm"><div class="cm-label">n / shard</div><div class="cm-val">${fs.n_per_shard.toLocaleString()}</div></div>
  `;
  cen.innerHTML = "";
  for (const [name, val] of fs.centralized_top) {
    cen.innerHTML += `<li><span>${name}</span><span class="fl-num">${val.toFixed(4)}</span></li>`;
  }
  fed.innerHTML = "";
  for (const [name, val] of fs.federated_top) {
    fed.innerHTML += `<li><span>${name}</span><span class="fl-num">${val.toFixed(4)}</span></li>`;
  }
}

function renderCausal(c) {
  const svg = $("#causal-svg");
  const meta = $("#causal-meta");
  if (!svg) return;
  if (!c || !c.feature_subset || c.feature_subset.length === 0) {
    svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" font-family="Inter" font-size="13" fill="#6b7691">No causal artefact (causallearn unavailable).</text>`;
    return;
  }
  const nodes = [...c.feature_subset, "placement_6m"];
  const target = c.target_idx;
  const mb = new Set(c.markov_blanket);

  // Layout: target at centre, MB on inner ring, others on outer ring.
  const W = 900, H = 460, cx = W/2, cy = H/2;
  const others = nodes.map((n, i) => i).filter(i => i !== target && !mb.has(nodes[i]));
  const mbList = nodes.map((n, i) => i).filter(i => i !== target && mb.has(nodes[i]));

  const positions = new Array(nodes.length);
  positions[target] = {x: cx, y: cy};
  mbList.forEach((idx, k) => {
    const a = (k / Math.max(1, mbList.length)) * 2 * Math.PI - Math.PI/2;
    positions[idx] = {x: cx + 130 * Math.cos(a), y: cy + 110 * Math.sin(a)};
  });
  others.forEach((idx, k) => {
    const a = (k / Math.max(1, others.length)) * 2 * Math.PI;
    positions[idx] = {x: cx + 200 * Math.cos(a), y: cy + 180 * Math.sin(a)};
  });

  // Build SVG markup
  const parts = [
    `<defs>
       <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
         <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(10,31,68,0.30)"/>
       </marker>
       <marker id="arrowmb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
         <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(241,90,41,0.7)"/>
       </marker>
     </defs>`
  ];

  for (const [s, t, kind] of c.edges) {
    const ps = positions[s], pt = positions[t];
    if (!ps || !pt) continue;
    const involvesTarget = (s === target || t === target);
    const cls = involvesTarget ? "edge edge-mb" : "edge";
    const marker = (kind === "->" || kind === "<->") ? (involvesTarget ? "url(#arrowmb)" : "url(#arrow)") : "";
    parts.push(`<line class="${cls}" x1="${ps.x}" y1="${ps.y}" x2="${pt.x}" y2="${pt.y}" marker-end="${marker}"/>`);
  }

  for (let i = 0; i < nodes.length; i++) {
    const p = positions[i];
    if (!p) continue;
    const isTarget = i === target;
    const isMB = mb.has(nodes[i]);
    const cls = isTarget ? "node-target" : (isMB ? "node-mb" : "node-other");
    const r = isTarget ? 24 : (isMB ? 10 : 6);
    parts.push(`<circle class="${cls}" cx="${p.x}" cy="${p.y}" r="${r}"/>`);
    const labelDx = isTarget ? 0 : (isMB ? 0 : 0);
    const labelDy = isTarget ? 4 : (p.y > cy ? 18 : -10);
    const txt = isTarget ? "P(placed_6m)" : nodes[i];
    parts.push(`<text class="node-label ${isTarget ? 'target' : ''}" x="${p.x + labelDx}" y="${p.y + labelDy}" text-anchor="middle">${txt}</text>`);
  }
  svg.innerHTML = parts.join("\n");

  if (meta) {
    const fullAuc = c.full_auc != null ? c.full_auc.toFixed(3) : "—";
    const mbAuc = c.mb_auc != null ? c.mb_auc.toFixed(3) : "—";
    const ratio = (c.full_auc && c.mb_auc) ? ((c.mb_auc / c.full_auc) * 100).toFixed(1) + "%" : "—";
    meta.innerHTML = `
      <div class="cm"><div class="cm-label">|MB|</div><div class="cm-val">${c.mb_size}/${c.full_size}</div></div>
      <div class="cm"><div class="cm-label">Full AUC</div><div class="cm-val">${fullAuc}</div></div>
      <div class="cm"><div class="cm-label">MB-only AUC</div><div class="cm-val">${mbAuc}</div></div>
      <div class="cm"><div class="cm-label">Retention</div><div class="cm-val good">${ratio}</div></div>
    `;
  }
}

function renderFairness(f) {
  const summary = $("#fairness-summary");
  const rows = $("#fairness-rows");
  if (!summary || !rows) return;
  const breach = f.audits.filter(a => a.breaches.length > 0).length;
  summary.innerHTML = `
    <div class="ds"><div class="ds-label">Audits</div><div class="ds-val">${f.n_audits}</div></div>
    <div class="ds"><div class="ds-label">Pass</div><div class="ds-val stable">${f.pass_count}</div></div>
    <div class="ds"><div class="ds-label">Breach</div><div class="ds-val drift">${breach}</div></div>
  `;
  rows.innerHTML = "";
  const t = f.thresholds;
  for (const a of f.audits) {
    const passed = a.breaches.length === 0;
    const div = document.createElement("div");
    div.className = `fairness-row ${passed ? "pass" : "breach"}`;
    const cell = (label, val, good) => `
      <div class="fr-cell">
        <div class="fr-cell-label">${label}</div>
        <div class="fr-cell-val ${good ? "good" : "bad"}">${val}</div>
      </div>`;
    div.innerHTML = `
      <div class="fr-name">${a.sensitive}</div>
      ${cell("DI ratio (≥" + t.di_ratio_min + ")", a.di_ratio.toFixed(2), a.di_ratio >= t.di_ratio_min)}
      ${cell("DP diff (≤" + t.dp_diff_max + ")", a.dp_diff.toFixed(2), a.dp_diff <= t.dp_diff_max)}
      ${cell("EO diff (≤" + t.eo_diff_max + ")", a.eo_diff.toFixed(2), a.eo_diff <= t.eo_diff_max)}
      ${cell("EOpp diff (≤" + t.eopp_diff_max + ")", a.eopp_diff.toFixed(2), a.eopp_diff <= t.eopp_diff_max)}
      ${cell("AUC gap", a.auc_gap.toFixed(3), a.auc_gap < 0.05)}
      <div class="fr-status ${passed ? "pass" : "breach"}" title="${a.note}">${passed ? "PASS" : "BREACH"}</div>
    `;
    rows.appendChild(div);
  }
}

function renderBandit(b) {
  const summary = $("#bandit-summary");
  const wrap = $("#bandit-segments");
  if (!summary || !wrap) return;
  const totalTrials = b.segments.reduce((s, seg) => s + seg.actions.reduce((t, a) => t + a.n_trials, 0), 0);
  summary.innerHTML = `
    <div class="ds"><div class="ds-label">Segments</div><div class="ds-val">${b.n_segments}</div></div>
    <div class="ds"><div class="ds-label">Outcomes</div><div class="ds-val">${totalTrials}</div></div>
  `;
  wrap.innerHTML = "";
  if (!b.segments.length) {
    wrap.innerHTML = `<div class="bandit-segment"><div class="bandit-segment-head">No outcomes recorded yet — the dashboard's recommendations are using the warm-start prior.</div></div>`;
    return;
  }
  for (const seg of b.segments.slice(0, 5)) {
    const segTrials = seg.actions.reduce((t, a) => t + a.n_trials, 0);
    const div = document.createElement("div");
    div.className = "bandit-segment";
    let actionsHtml = "";
    for (const a of seg.actions) {
      const ciLeft = (a.ci_low * 100).toFixed(0);
      const ciWidth = ((a.ci_high - a.ci_low) * 100).toFixed(0);
      const meanLeft = (a.posterior_mean * 100).toFixed(0);
      actionsHtml += `
        <div class="bandit-action">
          <div class="ba-name">${a.action_id}</div>
          <div class="ba-bar">
            <div class="ba-bar-ci" style="left:${ciLeft}%; width:${ciWidth}%"></div>
            <div class="ba-bar-mean" style="left:${meanLeft}%"></div>
          </div>
          <div class="ba-mean-num">${(a.posterior_mean*100).toFixed(1)}% [${(a.ci_low*100).toFixed(0)}-${(a.ci_high*100).toFixed(0)}]</div>
          <div class="ba-trials">n=${a.n_trials}</div>
        </div>
      `;
    }
    div.innerHTML = `
      <div class="bandit-segment-head">
        <span>${seg.segment}</span>
        <span class="seg-trials">${segTrials} outcome${segTrials === 1 ? '' : 's'} recorded</span>
      </div>
      ${actionsHtml}
    `;
    wrap.appendChild(div);
  }
}

function renderDrift(d) {
  const summary = $("#drift-summary");
  const rows = $("#drift-rows");
  if (!summary || !rows) return;
  const s = d.summary;
  summary.innerHTML = `
    <div class="ds"><div class="ds-label">Stable</div><div class="ds-val stable">${s.stable}</div></div>
    <div class="ds"><div class="ds-label">Moderate</div><div class="ds-val moderate">${s.moderate}</div></div>
    <div class="ds"><div class="ds-label">Drift</div><div class="ds-val drift">${s.drift}</div></div>
  `;
  rows.innerHTML = "";
  // Cap PSI bar at 0.5 so the spectrum is readable
  const maxPSI = 0.5;
  for (const f of d.features.slice(0, 12)) {
    const div = document.createElement("div");
    div.className = `drift-row ${f.band}`;
    const w = Math.min(100, (f.psi / maxPSI) * 100);
    div.innerHTML = `
      <div class="drift-name">${f.feature}</div>
      <div class="drift-bar"><span style="width:${w}%"></span></div>
      <div class="drift-num">PSI ${f.psi.toFixed(3)}</div>
      <div class="drift-num">p ${f.ks_pvalue < 0.001 ? '&lt;0.001' : f.ks_pvalue.toFixed(3)}</div>
      <div class="drift-band ${f.band}">${f.band}</div>
    `;
    rows.appendChild(div);
  }
}

function renderOOD(ood) {
  const tag = $("#ood-source-tag");
  const rows = $("#ood-rows");
  if (!tag || !rows) return;
  if (!ood.reports.length) {
    tag.textContent = "NO FOLDS";
    tag.className = "ood-source-tag fallback";
    rows.innerHTML = "<div class='ood-row'><div class='ood-name'>No OOD evaluation present.</div></div>";
    return;
  }
  if (ood.kaggle_present) {
    tag.textContent = "KAGGLE · REAL LABELS";
    tag.className = "ood-source-tag kaggle";
  } else {
    tag.textContent = "FALLBACK · COVARIATE-SHIFT FOLD";
    tag.className = "ood-source-tag fallback";
  }
  rows.innerHTML = "";
  for (const r of ood.reports) {
    const auc = r.auc_12m;
    const cls = (val, hi=0.80, mid=0.65) => val >= hi ? "good" : (val >= mid ? "warn" : "bad");
    const eceCls = r.ece_post < 0.05 ? "good" : (r.ece_post < 0.10 ? "warn" : "bad");
    const div = document.createElement("div");
    div.className = "ood-row" + (auc < 0.65 ? " bad" : "");
    div.innerHTML = `
      <div>
        <div class="ood-name">${r.source.replace(/_/g, " ")}</div>
        <div class="ood-note">${r.note}</div>
      </div>
      <div class="ood-cell"><div class="ood-cell-label">N rows</div><div class="ood-cell-val">${r.n_rows.toLocaleString()}</div></div>
      <div class="ood-cell"><div class="ood-cell-label">Pos rate</div><div class="ood-cell-val">${(r.pos_rate*100).toFixed(0)}%</div></div>
      <div class="ood-cell"><div class="ood-cell-label">AUC 12m</div><div class="ood-cell-val ${cls(auc)}">${auc.toFixed(3)}</div></div>
      <div class="ood-cell"><div class="ood-cell-label">ECE post</div><div class="ood-cell-val ${eceCls}">${r.ece_post.toFixed(3)}</div></div>
    `;
    rows.appendChild(div);
  }
}

let reliabilityChart = null;
function renderReliabilityChart(cal) {
  const canvas = $("#reliability-canvas");
  if (!canvas) return;
  if (reliabilityChart) reliabilityChart.destroy();
  const points = (cal.reliability_curve || []).map(p => ({ x: p.p_pred, y: p.p_true }));
  const meta = $("#calibration-meta");
  if (meta) {
    const ecePre = cal.ece_pre.toFixed(3);
    const ecePost = cal.ece_post.toFixed(3);
    const brier = cal.brier_post.toFixed(4);
    const goodPost = cal.ece_post < 0.05;
    meta.innerHTML = `
      <div class="cm"><div class="cm-label">ECE pre</div><div class="cm-val">${ecePre}</div></div>
      <div class="cm"><div class="cm-label">ECE post</div><div class="cm-val ${goodPost ? 'good' : 'warn'}">${ecePost}</div></div>
      <div class="cm"><div class="cm-label">Brier</div><div class="cm-val">${brier}</div></div>
      <div class="cm"><div class="cm-label">Calibration n</div><div class="cm-val">${cal.n_calibration.toLocaleString()}</div></div>
    `;
  }

  const ctx = canvas.getContext("2d");
  reliabilityChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Reliability bins",
          data: points,
          backgroundColor: "#f15a29",
          borderColor: "#f15a29",
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: true,
          borderWidth: 2,
          tension: 0.0,
        },
        {
          label: "Perfect calibration",
          data: [{x: 0, y: 0}, {x: 1, y: 1}],
          borderColor: "rgba(10,31,68,0.35)",
          borderDash: [6, 6],
          borderWidth: 1.4,
          pointRadius: 0,
          showLine: true,
          fill: false,
        },
      ],
    },
    options: {
      animation: false,
      plugins: {
        legend: { display: true, labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          backgroundColor: "#0a1f44",
          callbacks: {
            label: ctx => ctx.dataset.label === "Perfect calibration"
              ? `(${ctx.parsed.x.toFixed(2)}, ${ctx.parsed.y.toFixed(2)})`
              : `Predicted ${(ctx.parsed.x*100).toFixed(0)}% → Empirical ${(ctx.parsed.y*100).toFixed(0)}%`,
          },
        },
      },
      scales: {
        x: {
          min: 0, max: 1,
          grid: { color: "rgba(0,0,0,0.04)" },
          ticks: { color: "#6b7691", font: { size: 11 }, callback: v => `${(v*100).toFixed(0)}%` },
          title: { display: true, text: "Predicted P(placed @ 6m)", color: "#6b7691", font: { size: 11 } },
        },
        y: {
          min: 0, max: 1,
          grid: { color: "rgba(0,0,0,0.04)" },
          ticks: { color: "#6b7691", font: { size: 11 }, callback: v => `${(v*100).toFixed(0)}%` },
          title: { display: true, text: "Empirical placement rate", color: "#6b7691", font: { size: 11 } },
        },
      },
    },
  });
}


// Scroll smoothly past the hero to the first tab section.
function scrollToDemo() {
  const target = document.querySelector(".container");
  if (!target) return;
  const navH = document.querySelector(".nav")?.offsetHeight || 64;
  const top = target.getBoundingClientRect().top + window.scrollY - navH;
  window.scrollTo({ top, behavior: "smooth" });
}


// ----------- boot -----------
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupDemo();
  setupPortfolioFilters();
  pingHealth();
  pingLLM(false);

  // "Live Demo ↓" button in the compact hero
  const heroBtn = $("#hero-scroll-btn");
  if (heroBtn) heroBtn.addEventListener("click", scrollToDemo);

  // Click the LLM pill to fire a real probe. Consumes one free-tier call.
  const llm = $("#llm-status");
  if (llm) llm.addEventListener("click", () => pingLLM(true));
});
