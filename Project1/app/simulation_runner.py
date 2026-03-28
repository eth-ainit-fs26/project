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
#sim-progress-wrap {{
    display:flex; align-items:center; gap:12px;
    background:white; border-radius:8px; padding:10px 14px;
    margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,.08);
  }}
  #sim-day-label {{ font-size:0.82rem; min-width:95px; font-variant-numeric:tabular-nums; }}
  #sim-bar-wrap {{ flex:1; background:#e2e8f0; border-radius:6px; height:9px; overflow:hidden; }}
  #sim-bar {{ height:100%; width:0%; background:#4f46e5; border-radius:6px; transition:width .3s; }}
#sim-charts {{
    display:grid; grid-template-columns:1fr; gap:10px; margin-bottom:12px;
  }}
  .sim-chart-card {{
    background:white; border-radius:8px; padding:12px 14px;
    box-shadow:0 1px 3px rgba(0,0,0,.08); margin-top:16px;
  }}
  .sim-chart-card h3 {{
    font-size:0.72rem; font-weight:600; color:#718096; margin-top:10px; margin-bottom:8px;
    text-transform:uppercase; letter-spacing:.05em; text-align:center;
  }}
  .sim-chart-card canvas {{ max-height:340px; }}
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
  <div id="sim-progress-wrap">
    <span id="sim-day-label">Day 0 / {total_days}</span>
    <div id="sim-bar-wrap"><div id="sim-bar"></div></div>
  </div>

  <div id="sim-charts">
    <div class="sim-chart-card"><canvas id="sim-c-acc"></canvas></div>
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

  let days=[], segs=[], bestAccD={{}};

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
        plugins:{{legend:{{display:false}}}}
      }}
    }});
  }}

  const cAcc = lineChart("sim-c-acc");

  // ── init segments ────────────────────────────────────────────────────
  function initSegs(names) {{
    if (segs.length) return;
    segs = [...names].sort();
    segs.forEach(s => {{
      bestAccD[s]=[];
      const dsBest = color => ({{label:s,data:[],borderColor:color,backgroundColor:"transparent",
        borderWidth:2,pointRadius:0,stepped:true,fill:false}});
      cAcc.data.datasets.push(dsBest(c(s)));
    }});
    cAcc.update("none");
  }}


  // ── main update ──────────────────────────────────────────────────────
  window.updateSim = function(snap) {{
    const segAcc      = snap.segment_accuracy      || {{}};
    const bestAcc     = snap.best_segment_accuracy || {{}};
    const curHooks    = snap.current_hooks         || {{}};
    const day         = snap.day_number ?? days.length;

    const segNames = Object.keys(segAcc);
    if (segNames.length && !segs.length) initSegs(segNames);
    if (!segs.length) return;

    days.push(day);
    cAcc.data.labels = days;

    segs.forEach((s,i) => {{
      const best = bestAcc[s] ?? (bestAccD[s].length ? bestAccD[s][bestAccD[s].length-1] : NaN);
      bestAccD[s].push(best);
      cAcc.data.datasets[i].data = bestAccD[s];
    }});

    cAcc.update("none");

    // progress
    const pct = TOTAL ? Math.min(day/TOTAL*100,100) : 0;
    document.getElementById("sim-bar").style.width = pct+"%";
    document.getElementById("sim-day-label").textContent = "Day "+day+" / "+(TOTAL||"?");

    // current hooks — update every cycle so the table always reflects what's being tried
    const tbody = document.getElementById("sim-hooks-body");
    const hookEntries = Object.entries(curHooks).filter(([,h]) => h);
    if (hookEntries.length) {{
      tbody.innerHTML = hookEntries
        .sort(([a],[b]) => a.localeCompare(b))
        .map(([s, hook]) => `<tr>
          <td><b style="color:${{c(s)}}">${{s}}</b></td>
          <td>${{hook}}</td>
          <td><span class="sim-pill">${{(bestAcc[s]||0).toFixed(3)}}</span></td>
        </tr>`).join("");
    }}
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
        total_days = getattr(self.w, "simulation_duration", "?")
        html = _DASHBOARD_HTML.format(
            total_days=total_days,
            seg_colors=json.dumps(SEG_COLORS),
        )
        display(HTML(html))

    def run(self):
        try:
            self.w.reset()
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
