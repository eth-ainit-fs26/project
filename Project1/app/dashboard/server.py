"""
Flask-based simulation dashboard with real-time Server-Sent Events.

Usage (notebook):
    from app.dashboard.server import SimulationServer
    server = SimulationServer(w, use_opro2_mode)
    server.launch()   # starts server + shows iframe in notebook
"""
import json
import queue
import threading
import time

from flask import Flask, Response, render_template, request


class SimulationServer:
    def __init__(self, world_simulator, use_opro2_mode: bool = True, port: int = 7860):
        self.w = world_simulator
        self.use_opro2_mode = use_opro2_mode
        self.port = port

        self._queue: queue.Queue = queue.Queue()
        self._sim_thread: threading.Thread | None = None
        self._running = False

        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Flask app
    # ------------------------------------------------------------------

    def _build_app(self):
        import os
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        static_dir   = os.path.join(os.path.dirname(__file__), "static")
        app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
        app.config["SECRET_KEY"] = "sim-dashboard"

        @app.get("/")
        def index():
            return render_template(
                "index.html",
                total_days=getattr(self.w, "simulation_duration", "?"),
                opro_cycle=getattr(self.w, "opro_cycle_days", "?"),
                mode="OPRO2" if self.use_opro2_mode else "OPRO",
            )

        @app.post("/start")
        def start():
            if self._running:
                return {"status": "already running"}, 409
            self._sim_thread = threading.Thread(target=self._run_simulation, daemon=True)
            self._sim_thread.start()
            return {"status": "started"}, 200

        @app.get("/stream")
        def stream():
            def event_generator():
                while True:
                    try:
                        data = self._queue.get(timeout=30)
                        yield f"data: {json.dumps(data)}\n\n"
                        if data.get("type") == "done" or data.get("type") == "error":
                            break
                    except queue.Empty:
                        yield "data: {\"type\": \"ping\"}\n\n"
            return Response(event_generator(), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        return app

    # ------------------------------------------------------------------
    # Simulation thread
    # ------------------------------------------------------------------

    def _run_simulation(self):
        self._running = True
        try:
            self.w.runSimulation(self._on_progress)
            self._queue.put({"type": "done"})
        except NotImplementedError as e:
            self._queue.put({"type": "error", "message": str(e)})
        except Exception as e:
            self._queue.put({"type": "error", "message": str(e)})
        finally:
            self._running = False

    def _on_progress(self, snapshot: dict):
        if not isinstance(snapshot, dict):
            return
        snapshot["type"] = "snapshot"
        self._queue.put(snapshot)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self):
        """Start the Flask server in a background thread and display an iframe."""
        server_thread = threading.Thread(
            target=lambda: self._app.run(port=self.port, threaded=True, use_reloader=False),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1.5)  # wait for server to be ready

        url = f"http://localhost:{self.port}"
        try:
            # Colab environment
            from google.colab.output import eval_js
            url = eval_js(f"google.colab.kernel.proxyPort({self.port})")
        except Exception:
            pass

        try:
            from IPython.display import IFrame, display, HTML
            display(HTML(f'<a href="{url}" target="_blank">Open dashboard in new tab →</a>'))
            display(IFrame(src=url, width="100%", height="750px"))
        except Exception:
            print(f"Dashboard running at: {url}")
