"""
SimulationDashboard — live ipywidgets dashboard for the Project 1 simulation.
display_customers() — print a summary table of all loaded customers.
cleanup()          — close the database connection and print a completion message.

Usage (notebook):
    from app.simulation_runner import SimulationDashboard
    dash = SimulationDashboard(w, use_opro2_mode)
    dash.display()
    dash.run()
"""
import math
import traceback
from collections import deque

import ipywidgets as W
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output


class SimulationDashboard:
    """Encapsulates the live dashboard and simulation runner for WorldSimulatorP1."""

    def __init__(self, world_simulator, use_opro2_mode: bool = True):
        self.w = world_simulator
        self.use_opro2_mode = use_opro2_mode

        # --- state ---
        self.x_days = []
        self.segment_order = []
        self.ys_by_segment = {}
        self.last_snapshot = None
        self.rolling_snapshots = deque(maxlen=200)

        # --- widgets ---
        self.out_plot  = W.Output(layout=W.Layout(border='1px solid #ddd', height='360px'))
        self.out_tbl_1 = W.Output(layout=W.Layout(border='1px solid #ddd', height='260px', overflow='auto', width='50%'))
        self.out_tbl_2 = W.Output(layout=W.Layout(border='1px solid #ddd', height='260px', overflow='auto', width='50%'))

        total_days = self._safe_int(getattr(world_simulator, "simulation_duration", None))
        self.day_prog = W.IntProgress(
            value=0, min=0,
            max=(total_days if total_days else 1),
            layout=W.Layout(width='260px'),
        )
        self.day_text = W.HTML("<b>Day 0 / ?</b>")
        self._total_days_hint = total_days

        day_row = W.HBox(
            [W.Label("Day:"), self.day_prog, self.day_text],
            layout=W.Layout(align_items='center', gap='8px'),
        )
        self._dashboard_widget = W.VBox(
            [
                W.HTML("<h3 style='margin:6px 0'>AI Marketing Simulation — Live</h3>"),
                W.HTML("<p style='margin:0 0 8px 0'>Accuracy lines (one per segment) — final plot renders after the run • Per-segment table • Historical hooks (top per segment)</p>"),
                day_row,
                self.out_plot,
                W.HBox([self.out_tbl_1, self.out_tbl_2], layout=W.Layout(gap='10px')),
            ],
            layout=W.Layout(gap='10px'),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self):
        """Render the dashboard widget and show placeholder text."""
        display(self._dashboard_widget)
        with self.out_plot:
            clear_output(wait=True)
            print("The accuracy chart will render here once the simulation completes...")
        self._refresh_tables()

    def run(self):
        """Run the simulation and render the final plot."""
        mode_label = "OPRO2 (Direct Hook Optimization)" if self.use_opro2_mode else "Original OPRO (Prompt Optimization)"
        all_customers = self.w.customer_registry.get_all_customers()
        print(f"Starting simulation — {len(all_customers)} customers, {self.w.simulation_duration} days, mode: {mode_label}")
        print("=" * 60)

        try:
            self.w.runSimulation(self._on_progress)
            print("\nSimulation completed successfully!")

            # Snap progress bar to done
            total = self._safe_int(getattr(self.w, "simulation_duration", None))
            observed = len(self.x_days)
            end = total if total else observed
            self.day_prog.max = max(end, 1)
            self.day_prog.value = self.day_prog.max
            self.day_text.value = f"<b>Day {self.day_prog.max} / {self.day_prog.max}</b>"

            self._render_final_plot()

        except NotImplementedError as nie:
            print(f"\nImplementation Error: {nie}")
            print("You need to implement the required functions before the simulation can run:")
            print("  • HookEvaluator.cosine_similarity")
            print("  • HookEvaluator.evaluate_hook_quality_with_embedding")
            print("  • MarketingSession.run_session")
            print("  • MarketingSession.run_step")
            print("  • WorldSimulatorP1.calculate_segment_hook_accuracy")
            print("  • Opro.optimize_prompt")

        except Exception as e:
            print(f"\nSimulation Error: {e}")
            traceback.print_exc()

        print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_int(val):
        try:
            return int(val)
        except Exception:
            return None

    def _init_segments(self, snapshot: dict):
        if self.segment_order:
            return
        seg_names = (
            list((snapshot.get("segment_accuracy") or {}).keys())
            or list((snapshot.get("current_hooks") or {}).keys())
            or list((snapshot.get("segment_visit_metrics") or {}).keys())
            or list((snapshot.get("data_summary", {}).get("segments_tracked") or []))
        )
        if seg_names:
            self.segment_order[:] = sorted(seg_names)[:5]
            for seg in self.segment_order:
                self.ys_by_segment.setdefault(seg, [])

    @staticmethod
    def _top_hooks_per_segment(historical_entries, topk=5) -> pd.DataFrame:
        if not historical_entries:
            return pd.DataFrame(columns=["segment", "hook", "best_accuracy", "count"])
        df = pd.DataFrame(historical_entries)
        for col in ["segment", "hook", "accuracy_score"]:
            if col not in df.columns:
                return pd.DataFrame(columns=["segment", "hook", "best_accuracy", "count"])
        df["accuracy_score"] = pd.to_numeric(df["accuracy_score"], errors="coerce")
        agg = (
            df.groupby(["segment", "hook"], dropna=False)
            .agg(best_accuracy=("accuracy_score", "max"), count=("hook", "size"))
            .reset_index()
        )
        agg = agg.sort_values(["segment", "best_accuracy", "count"], ascending=[True, False, False])
        agg["rank_in_segment"] = agg.groupby("segment").cumcount() + 1
        top = (
            agg[agg["rank_in_segment"] <= topk]
            .sort_values(["segment", "best_accuracy", "count"], ascending=[True, False, False])
            .reset_index(drop=True)
        )
        return top[["segment", "hook", "best_accuracy", "count"]]

    def _refresh_tables(self):
        with self.out_tbl_1:
            clear_output(wait=True)
            if not self.last_snapshot:
                display(pd.DataFrame({"status": ["waiting for data"]}))
            else:
                seg_acc   = self.last_snapshot.get("segment_accuracy", {}) or {}
                cur_hooks = self.last_snapshot.get("current_hooks", {}) or {}
                segments  = self.segment_order or sorted(set(seg_acc) | set(cur_hooks))
                rows = [{"segment": s, "accuracy": seg_acc.get(s), "current_hook": cur_hooks.get(s)} for s in segments]
                display(pd.DataFrame(rows, columns=["segment", "accuracy", "current_hook"])
                        .style.set_caption("Per-segment (latest): accuracy & current hook"))

        with self.out_tbl_2:
            clear_output(wait=True)
            if not self.last_snapshot:
                display(pd.DataFrame({"status": ["waiting for data"]}))
            else:
                hist   = self.last_snapshot.get("historical_hooks_and_scores", []) or []
                top_df = self._top_hooks_per_segment(hist, topk=5)
                if top_df.empty:
                    display(pd.DataFrame({"status": ["no historical hooks yet"]})
                            .style.set_caption("Historical hooks — top per segment"))
                else:
                    display(top_df.head(25).style.set_caption("Historical hooks — top per segment (by best accuracy)"))

    def _on_progress(self, snapshot: dict):
        if not isinstance(snapshot, dict):
            return
        self._init_segments(snapshot)
        if not self.segment_order:
            self.last_snapshot = snapshot
            self._refresh_tables()
            return

        day_raw = snapshot.get("day_number", snapshot.get("day", len(self.x_days)))
        day = self._safe_int(day_raw) or len(self.x_days)
        self.x_days.append(day)

        disp_day = max(day, 1)
        if self.day_prog.max <= 1 and self._total_days_hint:
            self.day_prog.max = max(self._total_days_hint, 1)
        if self.day_prog.max <= 1:
            self.day_prog.max = max(disp_day, 1)
        self.day_prog.value = min(disp_day, self.day_prog.max)
        total_label = self.day_prog.max if (self._total_days_hint or self.day_prog.max > 1) else "…"
        self.day_text.value = f"<b>Day {disp_day} / {total_label}</b>"

        seg_acc = snapshot.get("segment_accuracy", {}) or {}
        for seg in self.segment_order:
            try:
                y = float(seg_acc.get(seg, float("nan")))
            except Exception:
                y = float("nan")
            self.ys_by_segment[seg].append(y)

        self.last_snapshot = snapshot
        self.rolling_snapshots.append(snapshot)
        self._refresh_tables()

    def _render_final_plot(self):
        with self.out_plot:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7, 3))
            if not self.x_days or not self.segment_order:
                ax.set_title("No data to plot")
            else:
                n = min(len(self.x_days), min(len(self.ys_by_segment.get(s, [])) for s in self.segment_order))
                for seg in self.segment_order:
                    ax.plot(self.x_days[:n], self.ys_by_segment[seg][:n], lw=2, label=seg)
                ax.set_xlabel("Day")
                ax.set_ylabel("Accuracy")
                ax.set_title("Accuracy per segment (final)")
                ax.legend(loc='best')
                plt.tight_layout()
            display(fig)


# ---------------------------------------------------------------------------
# Standalone notebook helpers
# ---------------------------------------------------------------------------

def display_customers(world_simulator):
    """Print a summary table of all customers currently in the registry."""
    import pandas as pd
    all_customers = world_simulator.customer_registry.get_all_customers()
    rows = [
        {
            'CID': c.cid,
            'Name': c.name,
            'Age': c.age,
            'Gender': c.gender,
            'Job': c.job_type,
            'Income': f"${c.income:,}",
            'Segment': c.segment,
            'Satisfaction': round(c.satisfaction, 3),
            'Visit_Prob': round(c.visit_probability, 3),
        }
        for c in all_customers
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nTotal customers loaded: {len(all_customers)}")


def cleanup():
    """Close the database connection and print a completion message."""
    try:
        from models.database import close_database
        close_database()
        print("Database connection closed.")
    except Exception as e:
        print(f"Warning: could not close database: {e}")
    print("\nSimulation complete. Check the plots above for results.")
