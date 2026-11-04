/* SVLA-Agent showcase: renders the data bundle produced by scripts/build_site.py. */
(() => {
  "use strict";

  // Fixed categorical hue per failure type (dark-surface steps of a CVD-validated palette).
  const TYPE_COLORS = {
    PCIE_TIMEOUT: "#3987e5", DDR_ECC: "#d95926", PMU_DROOP: "#199e70",
    THERMAL_TRIP: "#c98500", FW_EXCEPTION: "#d55181", NOC_TIMEOUT: "#5cb85c", UNKNOWN_FAILURE: "#9085e9",
  };
  const TYPE_ORDER = Object.keys(TYPE_COLORS);
  const color = (t) => TYPE_COLORS[t] || TYPE_COLORS.UNKNOWN_FAILURE;
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
  const pct = (n) => (n == null ? "—" : Math.round(n * 100) + "%");

  const STAGES = [
    ["Ingest", "stream · gzip", "<b>ingest/reader.py</b> discovers <code>run_&lt;id&gt;.&lt;source&gt;.log[.gz]</code> files, groups them per run and streams lines lazily, so a multi-GB run never sits in memory."],
    ["Parse", "regex · Drain3", "<b>parse/parser.py</b> recovers timestamp, subsystem, level, message and <code>key=value</code> pairs; indented register dumps and stack traces attach to their event; Drain3 mines a stable template per message for dedup, clustering and retrieval."],
    ["Redact", "PII · secrets", "<b>privacy/redact.py</b> replaces e-mails, IPs, MACs, home-directory names, serial/fuse ids and credential-looking pairs with typed placeholders before anything is indexed, prompted or reported."],
    ["Classify", "rules · evidence", "<b>classify/classifier.py</b> turns ERROR/FATAL events into prioritised failures with the distinct WARN context from any subsystem in the window before the first error — where root-cause tell-tales live. A fingerprint counts recurrences across the corpus."],
    ["Cluster", "HDBSCAN", "<b>cluster/clusterer.py</b> embeds signatures (sentence-transformers, hashing fallback offline) and runs HDBSCAN within each failure type, so families are explainable sub-groups — typically root causes."],
    ["Retrieve", "RAG", "<b>rag/</b> indexes historical failures with their known root causes in a pluggable vector store (Chroma persistent or in-memory) and returns the closest past incidents for each new failure."],
    ["Hypothesize", "team · agent · rules", "<b>hypothesize/</b> names a specific root cause. Rule-based: catalogue markers, retrieval votes, prior. Direct: one strict Claude call. Agentic: one agent with tools in a bounded loop. <b>Multi-agent</b>: a LangGraph team — a triage agent dispatches subsystem specialists in parallel, each scoped to its own lines and catalogue slice, and a reviewer cross-checks the findings into a causal chain. All modes return the same Hypothesis shape and keep a trace."],
    ["Report", "MD · JSON · API", "<b>report/</b> renders Markdown and JSON per run; the FastAPI service, the CLI and the Streamlit dashboard all read the same objects."],
  ];

  // ---------- tooltip ----------
  const tip = $("#tooltip");
  function showTip(html, x, y) { tip.innerHTML = html; tip.hidden = false; moveTip(x, y); }
  function moveTip(x, y) { const pad = 14; const w = tip.offsetWidth, h = tip.offsetHeight; tip.style.left = Math.min(x + pad, window.innerWidth - w - 8) + "px"; tip.style.top = (y + pad + h > window.innerHeight ? y - h - pad : y + pad) + "px"; }
  function hideTip() { tip.hidden = true; }
  document.addEventListener("mousemove", (e) => { if (!tip.hidden) moveTip(e.clientX, e.clientY); });

  // ---------- charts (inline SVG) ----------
  function hbar(container, rows, { valueFmt = (v) => String(v), max = null, unit = "" } = {}) {
    // rows: [{label, value, color, tip}]
    const W = 560, rowH = 30, left = 150, right = 60, top = 8;
    const H = top + rows.length * rowH + 20;
    const vmax = max ?? Math.max(1, ...rows.map((r) => r.value));
    const scale = (v) => left + (v / vmax) * (W - left - right);
    const ticks = 4;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="bar chart">`;
    for (let i = 0; i <= ticks; i++) { const v = (vmax / ticks) * i; const x = scale(v); s += `<line class="grid" x1="${x}" y1="${top}" x2="${x}" y2="${H - 20}"/><text x="${x}" y="${H - 4}" text-anchor="middle">${valueFmt(v)}${unit}</text>`; }
    rows.forEach((r, i) => {
      const y = top + i * rowH + 6, h = 14, x1 = scale(r.value);
      s += `<text x="${left - 10}" y="${y + 11}" text-anchor="end">${esc(r.label)}</text>`;
      s += `<rect class="bar" x="${left}" y="${y}" width="${Math.max(2, x1 - left)}" height="${h}" fill="${r.color}"></rect>`;
      s += `<text class="val" x="${x1 + 8}" y="${y + 11}">${valueFmt(r.value)}${unit}</text>`;
      s += `<rect class="hit" x="0" y="${y - 6}" width="${W}" height="${rowH}" data-tip="${esc(r.tip || r.label)}"></rect>`;
    });
    s += "</svg>";
    container.innerHTML = s;
    container.querySelectorAll(".hit").forEach((el) => { el.addEventListener("mouseenter", (e) => showTip(el.dataset.tip, e.clientX, e.clientY)); el.addEventListener("mouseleave", hideTip); });
  }

  function timeline(container, failures, nEvents) {
    const W = 640, rowH = 34, left = 130, right = 20, top = 10;
    const H = top + failures.length * rowH + 28;
    const tmax = Math.max(...failures.map((f) => f.last_ts_ms)) * 1.04;
    const tmin = Math.max(0, Math.min(...failures.map((f) => f.first_ts_ms)) - tmax * 0.06);
    const x = (t) => left + ((t - tmin) / (tmax - tmin)) * (W - left - right);
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="failure windows">`;
    for (let i = 0; i <= 4; i++) { const t = tmin + ((tmax - tmin) / 4) * i; s += `<line class="grid" x1="${x(t)}" y1="${top}" x2="${x(t)}" y2="${H - 24}"/><text x="${x(t)}" y="${H - 8}" text-anchor="middle">${(t / 1000).toFixed(1)} s</text>`; }
    failures.forEach((f, i) => {
      const y = top + i * rowH + 12, x1 = x(f.first_ts_ms), x2 = Math.max(x(f.last_ts_ms), x1 + 6);
      s += `<text x="${left - 10}" y="${y + 4}" text-anchor="end">${esc(f.failure_type)}</text>`;
      s += `<line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}" stroke="${color(f.failure_type)}" stroke-width="6" stroke-linecap="round"/>`;
      s += `<circle cx="${x1}" cy="${y}" r="5" fill="${color(f.failure_type)}" stroke="#0b0f14" stroke-width="2"/><circle cx="${x2}" cy="${y}" r="5" fill="${color(f.failure_type)}" stroke="#0b0f14" stroke-width="2"/>`;
      s += `<rect class="hit" x="0" y="${y - 16}" width="${W}" height="${rowH}" data-tip="<b>${esc(f.failure_type)}</b><br>${(f.first_ts_ms / 1000).toFixed(2)} s → ${(f.last_ts_ms / 1000).toFixed(2)} s (${f.last_ts_ms - f.first_ts_ms} ms)<br>${esc(f.summary.slice(0, 110))}"></rect>`;
    });
    s += `</svg><p class="muted" style="margin:4px 0 0;font-size:.8rem">${nEvents.toLocaleString()} events in this run; windows show first to last error of each failure.</p>`;
    container.innerHTML = s;
    container.querySelectorAll(".hit").forEach((el) => { el.addEventListener("mouseenter", (e) => showTip(el.dataset.tip, e.clientX, e.clientY)); el.addEventListener("mouseleave", hideTip); });
  }

  // ---------- sections ----------
  function renderHero(b) {
    const m = b.metrics, rules = m.rules_sentence_transformers, claude = m.claude, tp = b.throughput;
    const rc = rules?.root_cause, rcc = claude?.root_cause, rt = rules?.retrieval, ho = rules?.holdout;
    $("#hero-stats").innerHTML = `
      <div class="stat"><div class="v">${pct(rc?.accuracy)}</div><div class="l">root-cause top-1, rule-based</div><div class="s">${rc?.n ?? "—"} held-out failures</div></div>
      <div class="stat"><div class="v">${pct(rcc?.accuracy)}</div><div class="l">root-cause top-1, Claude</div><div class="s">confidence when wrong ${fmt(rcc?.mean_confidence_wrong)}</div></div>
      <div class="stat"><div class="v">${fmt(rt?.recall_at_k)}</div><div class="l">retrieval recall@${rt?.k ?? 5}</div><div class="s">MRR ${fmt(rt?.mrr)} · hold-out ${ho?.train_runs}/${ho?.test_runs} runs</div></div>
      <div class="stat"><div class="v">${(tp.events / 1e6).toFixed(2)}&thinsp;M</div><div class="l">events parsed in ${tp.seconds.toFixed(0)} s</div><div class="s">${tp.size_mb} MB · ${tp.files} files · ${Math.round(tp.events / tp.seconds / 1000)} k events/s</div></div>`;
    $("#footer-version").textContent = `SVLA-Agent v${b.version} · data generated ${b.generated_at.slice(0, 10)}`;
    $("#nav-live").href = b.live_app; $("#nav-repo").href = b.repo;
  }

  function renderPipeline() {
    const ol = $("#pipeline-stages");
    ol.innerHTML = STAGES.map(([n, sub]) => `<li><b>${n}</b><span>${sub}</span></li>`).join("");
    const items = [...ol.children];
    const show = (i) => { items.forEach((li, j) => li.classList.toggle("active", i === j)); $("#stage-detail").innerHTML = STAGES[i][2]; };
    items.forEach((li, i) => li.addEventListener("click", () => show(i)));
    show(0);
  }

  function renderResults(b) {
    const m = b.metrics;
    const cols = [["claude", "Claude agent"], ["rules_sentence_transformers", "Rule-based · transformers"], ["rules_hashing", "Rule-based · hashing (offline)"]].filter(([k]) => m[k]);
    const get = (k, path) => path.split(".").reduce((o, p) => (o == null ? undefined : o[p]), m[k]);
    const rows = [
      ["Anomaly detection F1", (k) => fmt(get(k, "detection.f1"))],
      ["Failure typing F1 (micro)", (k) => fmt(get(k, "failure_typing.micro.f1"))],
      ["Cluster purity vs. failure type", (k) => fmt(get(k, "clustering.purity"))],
      ["Cluster homogeneity vs. root cause", (k) => fmt(get(k, "clustering.homogeneity_vs_root_cause"))],
      ["Retrieval recall@5 (same type) / MRR", (k) => `${fmt(get(k, "retrieval.recall_at_k"))} / ${fmt(get(k, "retrieval.mrr"))}`],
      ["Retrieval recall@5 (same root cause)", (k) => fmt(get(k, "retrieval.root_cause_recall_at_k"))],
      ["Root-cause top-1 accuracy", (k) => `${fmt(get(k, "root_cause.accuracy"))}`, true],
      ["Mean confidence when right / wrong", (k) => `${fmt(get(k, "root_cause.mean_confidence_correct"))} / ${fmt(get(k, "root_cause.mean_confidence_wrong"))}`],
    ];
    const any = m[cols[0][0]];
    $("#results-lede").textContent = `${any.n_runs} synthetic runs, ${any.n_failures} failures, ${any.holdout?.test_failures ?? "—"} held out. Retrieval and root-cause rows are scored on the held-out runs only.`;
    $("#metrics-table").innerHTML = `<thead><tr><th>Metric</th>${cols.map(([, n]) => `<th>${n}</th>`).join("")}</tr></thead><tbody>` +
      rows.map(([label, f, hl]) => `<tr class="${hl ? "hl" : ""}"><td>${label}</td>${cols.map(([k]) => `<td>${f(k)}</td>`).join("")}</tr>`).join("") + "</tbody>";
    const per = (m.claude || any).root_cause?.per_type || {};
    const bars = Object.entries(per).map(([t, v]) => ({ label: t, value: v.accuracy * 100, color: color(t), tip: `<b>${t}</b><br>${v.correct} of ${v.total} held-out failures correct` }));
    hbar($("#rc-chart"), bars, { valueFmt: (v) => Math.round(v), unit: "%", max: 100 });
  }

  function renderOOD(b) {
    const m = b.metrics;
    const cols = [["bgl_claude", "Claude agent"], ["bgl_sentence_transformers", "rules · transformers"], ["bgl_hashing", "rules · hashing"]].filter(([k]) => m[k]);
    if (!cols.length) { $("#ood-lede").textContent = "No out-of-distribution metrics in this build."; return; }
    const any = m[cols[0][0]], ho = any.holdout || {};
    const get = (k, path) => path.split(".").reduce((o, p) => (o == null ? undefined : o[p]), m[k]);
    $("#ood-lede").textContent = `Loghub BGL (BlueGene/L): 4.75 M real log lines with operator-assigned alert labels, cut into ${any.n_runs.toLocaleString()} thirty-minute windows and run through the unchanged pipeline; ${ho.test_runs?.toLocaleString() ?? "—"} windows held out.`;
    const rows = [
      ["Window anomaly detection P / R / F1", (k) => `${fmt(get(k, "detection.precision"))} / ${fmt(get(k, "detection.recall"))} / ${fmt(get(k, "detection.f1"))}`],
      ["Retrieval recall@5 (same component) / MRR", (k) => `${fmt(get(k, "retrieval.recall_at_k"))} / ${fmt(get(k, "retrieval.mrr"))}`],
      ["Retrieval recall@5 (same alert tag)", (k) => fmt(get(k, "retrieval.root_cause_recall_at_k"))],
      ["Alert tag from retrieved incidents, top-1", (k) => `${fmt(get(k, "root_cause.accuracy"))} (n=${get(k, "root_cause.n") ?? "—"})`, true],
      ["Answered (did not say OTHER)", (k) => pct(get(k, "root_cause.predicted_rate"))],
      ["Mean confidence when right / wrong", (k) => `${fmt(get(k, "root_cause.mean_confidence_correct"))} / ${fmt(get(k, "root_cause.mean_confidence_wrong"))}`],
      ["Cluster purity vs. alert tag", (k) => fmt(get(k, "clustering.purity_vs_root_cause"))],
    ];
    $("#ood-table").innerHTML = `<thead><tr><th>Metric</th>${cols.map(([, n]) => `<th>${n}</th>`).join("")}</tr></thead><tbody>` +
      rows.map(([label, f, hl]) => `<tr class="${hl ? "hl" : ""}"><td>${label}</td>${cols.map(([k]) => `<td>${f(k)}</td>`).join("")}</tr>`).join("") + "</tbody>";
    const per = any.root_cause?.per_type || {};
    const bars = Object.entries(per).sort((a, b) => b[1].total - a[1].total).map(([t, v]) => ({ label: t.replace("_FAILURE", ""), value: v.accuracy * 100, color: "#3987e5", tip: `<b>${t}</b><br>${v.correct} of ${v.total} held-out windows named correctly` }));
    hbar($("#ood-chart"), bars, { valueFmt: (v) => Math.round(v), unit: "%", max: 100 });
  }

  function renderCatalogue(b) {
    $("#catalogue-grid").innerHTML = Object.entries(b.knowledge).filter(([t]) => t !== "UNKNOWN_FAILURE").map(([t, k]) => `
      <article class="cat" style="--c:${color(t)}"><h3>${t}</h3><p class="prior">${esc(k.prior.root_cause)}</p>
        <ul>${k.root_causes.map((rc) => `<li><b>${rc.rc_id}</b>${esc(rc.description)}<span>tell-tales: ${rc.markers.map(esc).join(" · ")}</span></li>`).join("")}</ul></article>`).join("");
  }

  // ---------- explorer ----------
  let reports = [], current = null;
  function runMatches(r) {
    const onlyAnom = $("#only-anomalous").checked, t = $("#type-filter").value;
    if (onlyAnom && !r.anomalous) return false;
    if (t && !r.failures.some((f) => f.failure_type === t)) return false;
    return true;
  }
  function renderRunList() {
    const ul = $("#runs"); const vis = reports.filter(runMatches);
    ul.innerHTML = vis.map((r) => `<li class="${r.anomalous ? "anom" : ""} ${current === r.run_id ? "active" : ""}" data-id="${r.run_id}" role="option" aria-selected="${current === r.run_id}">
      <span class="dot"></span><span class="id">run ${r.run_id}</span><span class="types">${r.failures.map((f) => f.failure_type).join(", ") || "pass"}</span></li>`).join("") || `<li class="empty">no runs match</li>`;
    ul.querySelectorAll("li[data-id]").forEach((li) => li.addEventListener("click", () => selectRun(li.dataset.id)));
    if (!vis.some((r) => r.run_id === current) && vis.length) selectRun(vis[0].run_id);
  }
  function highlightEvidence(lines, hyp, knowledge, ftype) {
    const markers = (knowledge[ftype]?.root_causes.find((rc) => rc.rc_id === hyp?.root_cause_id)?.markers) || [];
    return lines.map((l) => {
      let h = esc(l);
      if (/: (ERROR|FATAL):/.test(l)) h = `<span class="err">${h}</span>`;
      markers.forEach((mk) => { const re = new RegExp(mk.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"); if (re.test(l)) h = h.replace(new RegExp("(" + mk.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "i"), '<span class="hl">$1</span>'); });
      return h;
    }).join("\n");
  }
  function selectRun(id) {
    current = id; const r = reports.find((x) => x.run_id === id); if (!r) return;
    document.querySelectorAll("#runs li").forEach((li) => { const on = li.dataset.id === id; li.classList.toggle("active", on); li.setAttribute("aria-selected", on); });
    const m = r.meta, hyps = Object.fromEntries(r.hypotheses.map((h) => [h.failure_id, h]));
    let html = `<div class="run-head"><h3>Run ${r.run_id}</h3><span class="pill ${r.anomalous ? "bad" : "good"}">${r.anomalous ? "ANOMALOUS" : "PASS"}</span>${r.failures.map((f) => `<span class="pill type" style="border-color:${color(f.failure_type)}">${f.failure_type}</span>`).join("")}</div>
      <div class="meta"><div>SoC<b>${esc(m.soc)} ${esc(m.stepping)}</b></div><div>Board<b>${esc(m.board)}</b></div><div>Firmware<b>${esc(m.fw_version)}</b></div><div>Test<b>${esc(m.test_suite)}/${esc(m.test_case)}</b></div><div>Workload<b>${esc(m.workload)}</b></div><div>Conditions<b>${m.voltage_mv} mV · ${m.temp_c} °C · ${m.clk_mhz} MHz</b></div><div>Events<b>${r.n_events.toLocaleString()}</b></div></div>`;
    if (!r.failures.length) { html += `<p class="empty">No failures detected. ${r.n_events.toLocaleString()} events parsed, nothing above WARN.</p>`; $("#run-detail").innerHTML = html; return; }
    html += `<div id="tl"></div>`;
    r.failures.forEach((f) => {
      const h = hyps[f.failure_id];
      html += `<section class="failure"><header><span class="sev ${f.severity}">${f.severity}</span><strong>${f.failure_type}</strong><span class="muted">${esc(f.subsys)}</span><span class="seen">${f.occurrences > 1 ? `seen ${f.occurrences}× in corpus` : "first occurrence"} · ${f.last_ts_ms - f.first_ts_ms} ms</span></header><div class="body">
        <h4>Evidence (${f.evidence.length} lines, redacted)</h4><pre class="evidence">${highlightEvidence(f.evidence, h, window.__bundle.knowledge, f.failure_type)}</pre>`;
      if (h) {
        html += `<h4>Root-cause hypothesis</h4><div class="hyp"><div class="rc">${esc(h.root_cause)}${h.root_cause_id ? `<span class="rcid">${h.root_cause_id}</span>` : ""}</div>
          <div class="conf"><div class="bar"><i style="width:${Math.round(h.confidence * 100)}%"></i></div><span>${pct(h.confidence)} · <span class="src">${esc(h.source)}</span></span></div>
          <p class="why">${esc(h.rationale)}</p><ol>${h.recommended_actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ol>`;
        if (h.trace?.length) {
          const agents = new Set(h.trace.map((t) => t.agent).filter(Boolean));
          html += `<h4 style="margin-top:12px">${agents.size ? `Team investigation · ${agents.size} agents · ${h.trace.length} steps` : `Agent investigation · ${h.trace.length} tool calls`} <span class="muted">(${agents.size ? "triage → specialists → reviewer" : "what Claude looked at before answering"})</span></h4><ol class="trace">` +
            h.trace.map((t) => `<li>${t.note ? `<p class="note">${esc(t.note)}</p>` : ""}${t.agent ? `<span class="agent">${esc(t.agent)}</span> ` : ""}<code class="call">${esc(t.tool)}(${esc(Object.entries(t.input || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", "))})</code><pre class="evidence small">${esc((t.output || "").slice(0, 700))}${(t.output || "").length > 700 ? "\n…" : ""}</pre></li>`).join("") + `</ol>`;
        }
        if (h.similar_incidents?.length) html += `<h4 style="margin-top:12px">Similar historical incidents (RAG)</h4><table class="sim"><thead><tr><th>run</th><th>type</th><th>root cause</th><th>similarity</th></tr></thead><tbody>${h.similar_incidents.map((s) => `<tr><td class="mono">${esc(s.run_id)}</td><td>${esc(s.failure_type)}</td><td class="mono">${esc(s.root_cause_id || "—")}</td><td>${fmt(1 - s.distance, 3)}</td></tr>`).join("")}</tbody></table>`;
        html += `</div>`;
      }
      html += `</div></section>`;
    });
    $("#run-detail").innerHTML = html;
    timeline($("#tl"), r.failures, r.n_events);
  }
  function renderOverview() {
    const counts = {}; const clusters = {};
    reports.forEach((r) => { r.failures.forEach((f) => { counts[f.failure_type] = (counts[f.failure_type] || 0) + 1; }); r.clusters.forEach((c) => { clusters[c.cluster_id] = c; }); });
    const order = TYPE_ORDER.filter((t) => counts[t]).concat(Object.keys(counts).filter((t) => !TYPE_ORDER.includes(t)));
    hbar($("#type-chart"), order.map((t) => ({ label: t, value: counts[t], color: color(t), tip: `<b>${t}</b><br>${counts[t]} failure(s) across ${reports.length} runs` })), { valueFmt: (v) => Math.round(v) });
    const cl = Object.values(clusters).sort((a, b) => b.size - a.size);
    $("#cluster-table").innerHTML = `<table class="clusters"><thead><tr><th>family</th><th>size</th><th>representative signature</th></tr></thead><tbody>${cl.map((c) => `<tr><td><span class="swatch" style="background:${color(c.label)}"></span>${c.label} <span class="muted">#${c.cluster_id}</span></td><td>${c.size}</td><td class="muted" style="font-family:var(--mono);font-size:.75rem">${esc(c.representative_signature.slice(0, 90))}…</td></tr>`).join("")}</tbody></table>`;
    const sel = $("#type-filter"); order.forEach((t) => { const o = document.createElement("option"); o.value = t; o.textContent = t; sel.appendChild(o); });
  }

  // ---------- boot ----------
  fetch("data/bundle.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then((b) => {
    window.__bundle = b; reports = b.reports;
    renderHero(b); renderPipeline(); renderResults(b); renderOOD(b); renderCatalogue(b); renderOverview(); renderRunList();
    $("#only-anomalous").addEventListener("change", renderRunList); $("#type-filter").addEventListener("change", renderRunList);
    const first = reports.find((r) => r.anomalous) || reports[0]; if (first) selectRun(first.run_id);
  }).catch((err) => { console.error("SVLA site failed to render:", err); $("#run-detail").innerHTML = `<p class="empty">Could not load data bundle (${esc(err.message)}). Run <code>python scripts/build_site.py</code>.</p>`; renderPipeline(); });
})();
