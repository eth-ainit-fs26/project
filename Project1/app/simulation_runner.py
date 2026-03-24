"""
SimulationDashboard — HTML/Chart.js dashboard rendered directly in the notebook.

No backend required. The dashboard is injected as HTML, and each simulation
step pushes data into it via a JavaScript call.

Usage:
    from app.simulation_runner import SimulationDashboard
    dash = SimulationDashboard(w, use_opro2_mode)
    dash.display()
    dash.run()
"""
import json
import traceback

import pandas as pd
from IPython.display import display, HTML, Javascript

SEG_COLORS = {"H": "#2196F3", "N": "#4CAF50", "R": "#FF9800", "L": "#9C27B0", "A": "#F44336"}


def _seg_color(seg):
    return SEG_COLORS.get(seg, "#607D8B")


_DASHBOARD_HTML = """
<style>
  #sim-dash * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  #sim-dash {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f5f7fa; color: #2d3748; padding: 16px; border-radius: 12px;
  }}
  #sim-dash header {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; }}
  #sim-dash h1 {{ font-size:1.3rem; font-weight:700; }}
  #sim-dash .meta {{ font-size:0.8rem; color:#718096; margin-top:3px; }}
  #sim-dash .meta span {{ margin-right:12px; }}
  #sim-progress-wrap {{
    display:flex; align-items:center; gap:12px;
    background:white; border-radius:8px; padding:10px 14px;
    margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,.08);
  }}
  #sim-day-label {{ font-size:0.82rem; min-width:95px; font-variant-numeric:tabular-nums; }}
  #sim-bar-wrap {{ flex:1; background:#e2e8f0; border-radius:6px; height:9px; overflow:hidden; }}
  #sim-bar {{ height:100%; width:0%; background:#4f46e5; border-radius:6px; transition:width .3s; }}
  #sim-vr-label {{ font-size:0.82rem; min-width:160px; text-align:right; }}
  #sim-cards {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }}
  .sim-card {{
    background:white; border-radius:8px; padding:10px 14px; flex:1; min-width:120px;
    box-shadow:0 1px 3px rgba(0,0,0,.08); border-top:4px solid #ccc;
  }}
  .sim-card-name  {{ font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:#718096; }}
  .sim-card-acc   {{ font-size:1.7rem; font-weight:800; line-height:1.1; }}
  .sim-card-count {{ font-size:0.72rem; color:#a0aec0; margin-top:2px; }}
  .sim-card-hook  {{ font-size:0.68rem; color:#718096; margin-top:5px; font-style:italic; line-height:1.3; }}
  #sim-charts {{
    display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;
  }}
  .sim-chart-card {{
    background:white; border-radius:8px; padding:12px 14px;
    box-shadow:0 1px 3px rgba(0,0,0,.08);
  }}
  .sim-chart-card h3 {{
    font-size:0.72rem; font-weight:600; color:#718096; margin-bottom:8px;
    text-transform:uppercase; letter-spacing:.05em;
  }}
  .sim-chart-card canvas {{ max-height:170px; }}
  #sim-hooks-section {{
    background:white; border-radius:8px; padding:12px 14px;
    box-shadow:0 1px 3px rgba(0,0,0,.08);
  }}
  #sim-hooks-section h3 {{
    font-size:0.72rem; font-weight:600; color:#718096; margin-bottom:8px;
    text-transform:uppercase; letter-spacing:.05em;
  }}
  #sim-hooks-table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
  #sim-hooks-table th {{
    text-align:left; padding:5px 8px; background:#f7fafc;
    color:#4a5568; font-weight:600; border-bottom:2px solid #e2e8f0;
  }}
  #sim-hooks-table td {{ padding:5px 8px; border-bottom:1px solid #edf2f7; }}
  #sim-hooks-table tr:last-child td {{ border-bottom:none; }}
  .sim-pill {{
    display:inline-block; padding:1px 7px; border-radius:10px;
    font-weight:700; font-size:0.75rem; background:#ebf8ff; color:#2b6cb0;
  }}
  #sim-status {{ margin-top:10px; font-size:0.75rem; color:#718096; text-align:center; }}
</style>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<div id="sim-dash">
  <header>
    <div>
      <h1>AI Marketing Simulation</h1>
      <div class="meta">
        <span>Mode: <b>{mode}</b></span>
        <span>Duration: <b>{total_days} days</b></span>
        <span>OPRO every <b>{opro_cycle} days</b></span>
      </div>
    </div>
  </header>

  <div id="sim-progress-wrap">
    <span id="sim-day-label">Day 0 / {total_days}</span>
    <div id="sim-bar-wrap"><div id="sim-bar"></div></div>
    <span id="sim-vr-label">Overall visit rate: —</span>
  </div>

  <div id="sim-cards"></div>

  <div id="sim-charts">
    <div class="sim-chart-card"><h3>Accuracy per segment</h3><canvas id="sim-c-acc"></canvas></div>
    <div class="sim-chart-card"><h3>Visit rate per segment</h3><canvas id="sim-c-vr"></canvas></div>
    <div class="sim-chart-card"><h3>Overall visit rate</h3><canvas id="sim-c-ovr"></canvas></div>
    <div class="sim-chart-card"><h3>Final accuracy (bar)</h3><canvas id="sim-c-bar"></canvas></div>
  </div>

  <div id="sim-hooks-section">
    <h3>Best hook per segment</h3>
    <table id="sim-hooks-table">
      <thead><tr><th>Segment</th><th>Best hook</th><th>Score</th></tr></thead>
      <tbody id="sim-hooks-body"><tr><td colspan="3" style="color:#a0aec0">No data yet</td></tr></tbody>
    </table>
  </div>
  <div id="sim-status">Initialising…</div>
</div>

<script>
(function() {{
  const COLORS = {seg_colors};
  const TOTAL  = {total_days};
  function c(s) {{ return COLORS[s] || "#607D8B"; }}

  let days=[], segs=[], accD={{}}, vrD={{}}, ovrD=[], bestHooks={{}};

  // ── charts ──────────────────────────────────────────────────────────
  function lineChart(id) {{
    return new Chart(document.getElementById(id), {{
      type:"line", data:{{labels:[], datasets:[]}},
      options:{{
        animation:false, responsive:true,
        scales:{{
          x:{{title:{{display:true,text:"Day"}}, ticks:{{maxTicksLimit:10}}}},
          y:{{min:0,max:1}}
        }},
        plugins:{{legend:{{position:"bottom", labels:{{boxWidth:10,font:{{size:10}}}}}}}}
      }}
    }});
  }}

  const cAcc = lineChart("sim-c-acc");
  const cVR  = lineChart("sim-c-vr");
  const cOvr = new Chart(document.getElementById("sim-c-ovr"), {{
    type:"line",
    data:{{labels:[], datasets:[{{label:"Overall",data:[],borderColor:"#37474F",
      backgroundColor:"rgba(55,71,79,.1)",fill:true,borderWidth:2,pointRadius:2}}]}},
    options:{{animation:false,responsive:true,
      scales:{{x:{{title:{{display:true,text:"Day"}},ticks:{{maxTicksLimit:10}}}},y:{{min:0,max:1}}}},
      plugins:{{legend:{{display:false}}}}}}
  }});
  const cBar = new Chart(document.getElementById("sim-c-bar"), {{
    type:"bar", data:{{labels:[],datasets:[{{data:[],backgroundColor:[]}}]}},
    options:{{animation:false,responsive:true,
      scales:{{y:{{min:0,max:1}}}},plugins:{{legend:{{display:false}}}}}}
  }});

  // ── init segments ────────────────────────────────────────────────────
  function initSegs(names) {{
    if (segs.length) return;
    segs = [...names].sort();
    segs.forEach(s => {{
      accD[s]=[]; vrD[s]=[];
      const ds = color => ({{label:s,data:[],borderColor:color,backgroundColor:color+"22",
        borderWidth:2,pointRadius:2,tension:.3}});
      cAcc.data.datasets.push(ds(c(s)));
      cVR.data.datasets.push(ds(c(s)));
    }});
    buildCards();
    cAcc.update("none"); cVR.update("none");
  }}

  function buildCards() {{
    const wrap = document.getElementById("sim-cards");
    wrap.innerHTML = "";
    segs.forEach(s => {{
      wrap.innerHTML += `<div class="sim-card" id="card-${{s}}" style="border-top-color:${{c(s)}}">
        <div class="sim-card-name">${{s}}</div>
        <div class="sim-card-acc" id="cacc-${{s}}">—</div>
        <div class="sim-card-count" id="ccnt-${{s}}">— / —</div>
        <div class="sim-card-hook" id="chook-${{s}}">—</div>
      </div>`;
    }});
  }}

  function updateCard(s, acc, vis, tot, hook) {{
    const a=document.getElementById("cacc-"+s);
    if(a) a.textContent=(acc*100).toFixed(1)+"%";
    const cnt=document.getElementById("ccnt-"+s);
    if(cnt) cnt.textContent=vis+" / "+tot+" visited";
    const h=document.getElementById("chook-"+s);
    if(h) h.textContent=hook?(hook.length>58?hook.slice(0,58)+"…":hook):"—";
  }}

  // ── main update ──────────────────────────────────────────────────────
  window.updateSim = function(snap) {{
    const segAcc  = snap.segment_accuracy       || {{}};
    const segVis  = snap.segment_visit_metrics  || {{}};
    const curHooks= snap.current_hooks          || {{}};
    const hist    = snap.historical_hooks_and_scores || [];
    const day     = snap.day_number ?? days.length;
    const overall = (snap.daily_metrics||{{}}).visit_rate ?? NaN;

    const segNames = Object.keys(segAcc).length ? Object.keys(segAcc) : Object.keys(curHooks);
    if (segNames.length && !segs.length) initSegs(segNames);
    if (!segs.length) return;

    days.push(day);
    cAcc.data.labels = days; cVR.data.labels = days; cOvr.data.labels = days;

    segs.forEach((s,i) => {{
      const acc = segAcc[s] ?? NaN;
      const vr  = (segVis[s]||{{}}).visit_rate ?? NaN;
      accD[s].push(acc); vrD[s].push(vr);
      cAcc.data.datasets[i].data = accD[s];
      cVR.data.datasets[i].data  = vrD[s];
      updateCard(s, isNaN(acc)?0:acc,
        (segVis[s]||{{}}).visited_customers??0,
        (segVis[s]||{{}}).total_customers??0,
        curHooks[s]||"");
    }});

    ovrD.push(overall);
    cOvr.data.datasets[0].data = ovrD;
    cBar.data.labels = segs;
    cBar.data.datasets[0].data = segs.map(s => segAcc[s]??0);
    cBar.data.datasets[0].backgroundColor = segs.map(s => c(s));

    cAcc.update("none"); cVR.update("none");
    cOvr.update("none"); cBar.update("none");

    // progress
    const pct = TOTAL ? Math.min(day/TOTAL*100,100) : 0;
    document.getElementById("sim-bar").style.width = pct+"%";
    document.getElementById("sim-day-label").textContent = "Day "+day+" / "+(TOTAL||"?");
    if (!isNaN(overall))
      document.getElementById("sim-vr-label").textContent =
        "Overall visit rate: "+(overall*100).toFixed(1)+"%";

    // best hooks
    hist.forEach(e => {{
      if (!bestHooks[e.segment] || e.accuracy_score > bestHooks[e.segment].score)
        bestHooks[e.segment] = {{hook:e.hook, score:e.accuracy_score}};
    }});
    const tbody = document.getElementById("sim-hooks-body");
    tbody.innerHTML = Object.entries(bestHooks)
      .sort(([a],[b])=>a.localeCompare(b))
      .map(([s,{{hook,score}}])=>`<tr>
        <td><b style="color:${{c(s)}}">${{s}}</b></td>
        <td>${{hook}}</td>
        <td><span class="sim-pill">${{score.toFixed(3)}}</span></td>
      </tr>`).join("");
  }};

  window.simDone = function() {{
    document.getElementById("sim-bar").style.width = "100%";
    document.getElementById("sim-bar").style.background = "#48bb78";
    document.getElementById("sim-day-label").textContent = "Done ✓";
    document.getElementById("sim-status").textContent = "Simulation complete.";
  }};

  window.simError = function(msg) {{
    document.getElementById("sim-status").textContent = "Error: "+msg;
  }};

  document.getElementById("sim-status").textContent = "Running…";
}})();
</script>
"""


class SimulationDashboard:
    def __init__(self, world_simulator, use_opro2_mode: bool = True):
        self.w = world_simulator
        self.use_opro2_mode = use_opro2_mode

    def display(self):
        mode       = "OPRO2" if self.use_opro2_mode else "OPRO"
        total_days = getattr(self.w, "simulation_duration", "?")
        opro_cycle = getattr(self.w, "opro_cycle_days", "?")
        html = _DASHBOARD_HTML.format(
            mode=mode,
            total_days=total_days,
            opro_cycle=opro_cycle,
            seg_colors=json.dumps(SEG_COLORS),
        )
        display(HTML(html))

    def run(self):
        try:
            self.w.runSimulation(self._on_progress)
            display(Javascript("if(window.simDone) simDone();"))
        except NotImplementedError as e:
            display(Javascript(f"if(window.simError) simError({json.dumps(str(e))});"))
            print(f"NotImplementedError: {e}")
        except Exception as e:
            display(Javascript(f"if(window.simError) simError({json.dumps(str(e))});"))
            traceback.print_exc()

    def _on_progress(self, snapshot: dict):
        if not isinstance(snapshot, dict):
            return
        snapshot.pop("type", None)
        js = f"if(window.updateSim) updateSim({json.dumps(snapshot, default=str)});"
        display(Javascript(js))


# ---------------------------------------------------------------------------
# Standalone notebook helpers
# ---------------------------------------------------------------------------

def display_customers(world_simulator):
    """Print a summary table of all customers currently in the registry."""
    all_customers = world_simulator.customer_registry.get_all_customers()
    rows = [
        {
            'CID': c.cid, 'Name': c.name, 'Age': c.age, 'Gender': c.gender,
            'Job': c.job_type, 'Income': f"${c.income:,}", 'Segment': c.segment,
            'Satisfaction': round(c.satisfaction, 3),
            'Visit_Prob': round(c.visit_probability, 3),
        }
        for c in all_customers
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal customers loaded: {len(all_customers)}")


def cleanup():
    """Close the database connection and print a completion message."""
    try:
        from models.database import close_database
        close_database()
        print("Database connection closed.")
    except Exception as e:
        print(f"Warning: could not close database: {e}")
    print("\nSimulation complete. Check the dashboard above for results.")
