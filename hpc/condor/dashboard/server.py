#!/usr/bin/env python3
"""Dashboard for the correlated Bose-Hubbard sweeps on the QuantumGhent HTCondor pool.

Runs on the laptop and talks to the access point (the access point) over one persistent ssh connection:

    python3 hpc/condor/dashboard/server.py            # asks for the the access point password once, opens the browser

It shows every point (t, μ) of a sweep in a grid (not submitted / queued / running / held / failed / unconverged /
done, plus whether the observables exist) and runs the condor scripts of hpc/condor/ from buttons: submit or continue
the sweep, observables, plot, remove held jobs, and gather the data (rsync to this repo's data/ and plots/).

Options: --host (<access point>), --port (<port>), --user (<login>), --remote-dir (DisorderKit.jl, relative to the
remote home), --julia (julia command on the access point, default juliaup with the Manifest's version), --http-port
(8765), --no-browser, --local DIR (no ssh: run the "remote" commands in DIR, for testing), --gather-dest (default: this
repo), --selftest (check the state logic and exit). Python 3 standard library only.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = "BoseHubbardCorrelated_NormalizedITE"
JOB_ATTRS = ("ClusterId,ProcId,JobStatus,BHC_stage,BHC_tag,BHC_L,BHC_t,BHC_mu,RemoteHost,LastRemoteHost,"
             "HoldReason,EnteredCurrentStatus,QDate,CompletionDate,ExitCode,RemoveReason")

RE_TAG = re.compile(r"^[A-Za-z0-9._+-]+$")
RE_POINT = re.compile(r"^-?\d+\.\d{4}$")
# Environment variables the actions may pass to the condor scripts, with the allowed values
PARAMS = {
    "TS": r"^[0-9.eE+\- ]+$", "MUS": r"^[0-9.eE+\- ]+$",
    "U": r"^[0-9.eE+-]+$", "R": r"^[0-9.eE+-]+$", "DELTA": r"^[0-9.eE+-]+$", "OMEGA": r"^[0-9.eE+-]+$",
    "NMAX": r"^\d+$", "DTAU": r"^[0-9.eE+-]+$", "MAXD": r"^\d+$", "MAXITER": r"^\d+$", "CONVTOL": r"^[0-9.eE+-]+$",
    "CROSS_WIDTH": r"^\d+$", "PPN": r"^\d+$", "JULIA_THREADS": r"^\d+$", "BLAS_THREADS": r"^\d+$",
    "MEM": r"^\d+(\.\d+)? ?[KMGT]?B?$", "DISK": r"^\d+(\.\d+)? ?[KMGT]?B?$",
    "REQUIREMENTS": r"^[A-Za-z0-9_ .<>=!&|()\"']*$", "NICE": r"^(TRUE|FALSE)$",
    "N_SAMPLES": r"^\d+$", "N_EDGE": r"^\d+$", "KAPPA_TOL": r"^[0-9.eE+-]+$", "CV_TOL": r"^[0-9.eE+-]+$",
    "FORCE": r"^[01]$", "DRYRUN": r"^[01]$", "CONDOR": r"^[01]$",
}
# Which parameters every action passes on
ACTION_PARAMS = {
    "submit": ["TS", "MUS", "U", "R", "DELTA", "OMEGA", "NMAX", "DTAU", "MAXD", "MAXITER", "CONVTOL", "CROSS_WIDTH",
               "PPN", "JULIA_THREADS", "BLAS_THREADS", "MEM", "DISK", "REQUIREMENTS", "NICE", "FORCE"],
    "observables": ["U", "R", "DELTA", "OMEGA", "NMAX", "DTAU", "MAXD", "N_SAMPLES", "N_EDGE", "PPN", "MEM",
                    "REQUIREMENTS", "NICE", "FORCE"],
    "plot": ["U", "R", "DELTA", "OMEGA", "NMAX", "DTAU", "MAXD", "KAPPA_TOL", "CV_TOL", "CONDOR"],
}
ACTION_SCRIPT = {"submit": "hpc/condor/sweep_BHCorr.sh", "observables": "hpc/condor/obs_BHCorr.sh",
                 "plot": "hpc/condor/plot_BHCorr.sh"}


class Remote:
    """Runs shell commands in the repo on the access point (over the ssh master connection) or locally (--local)."""

    def __init__(self, args):
        self.local = args.local
        self.host, self.port, self.user, self.dir = args.host, str(args.port), args.user, args.remote_dir
        self.sock = os.path.expanduser(f"~/.ssh/condor-dashboard-{args.user}@{args.host}-{args.port}.sock")
        self.julia = args.julia
        self.lost = False

    def _argv(self, cmd):
        if self.local:
            return ["bash", "-c", f"cd {shlex.quote(self.local)} && {cmd}"]
        return ["ssh", "-S", self.sock, "-o", "BatchMode=yes", "-p", self.port, f"{self.user}@{self.host}",
                f"cd {shlex.quote(self.dir)} && {cmd}"]

    def connect(self):
        if self.local:
            return True
        check = ["ssh", "-S", self.sock, "-O", "check", "-p", self.port, f"{self.user}@{self.host}"]
        if subprocess.run(check, capture_output=True).returncode == 0:
            return True
        print(f"Connecting to {self.user}@{self.host}:{self.port} (one password prompt; the connection is kept open) ...")
        rc = subprocess.run(["ssh", "-M", "-S", self.sock, "-o", "ControlPersist=12h", "-o", "ServerAliveInterval=60",
                             "-p", self.port, "-fN", f"{self.user}@{self.host}"]).returncode
        self.lost = rc != 0
        return rc == 0

    def run(self, cmd, timeout=300):
        try:
            p = subprocess.run(self._argv(cmd), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {timeout} s"
        if p.returncode == 255 and not self.local:
            self.lost = subprocess.run(["ssh", "-S", self.sock, "-O", "check", "-p", self.port,
                                        f"{self.user}@{self.host}"], capture_output=True).returncode != 0
        return p.returncode, p.stdout, p.stderr

    def stream(self, cmd, emit):
        p = subprocess.Popen(self._argv(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            emit(line.rstrip("\n"))
        return p.wait()

    def rsync_source(self, path):
        if self.local:
            return os.path.join(self.local, path)
        return f"{self.user}@{self.host}:{self.dir}/{path}"

    def rsync_ssh(self):
        return [] if self.local else ["-e", f"ssh -S {self.sock} -o BatchMode=yes -p {self.port}"]


# ---- state of the points

QUEUE_STATE = {1: "idle", 2: "running", 4: "running", 5: "held", 6: "running", 7: "idle"}


def job_info(j):
    return {"id": f"{j.get('ClusterId')}.{j.get('ProcId')}", "status": j.get("JobStatus"),
            "host": j.get("RemoteHost") or j.get("LastRemoteHost"), "since": j.get("EnteredCurrentStatus"),
            "hold_reason": j.get("HoldReason"), "exit_code": j.get("ExitCode"), "remove_reason": j.get("RemoveReason")}


def merge(status, queue, history):
    """Combine the stored files (sweep_status.jl), the queue and the history into one record per point."""
    points = {}
    for p in status.get("points", []):
        points[(p["t"], p["mu"])] = {"t": p["t"], "mu": p["mu"], "file": p}
    for t, mu in status.get("planned", []):
        points.setdefault((t, mu), {"t": t, "mu": mu, "file": None})
    active = {"ite": {}, "obs": {}}
    for j in queue:
        stage, key = j.get("BHC_stage"), (j.get("BHC_t"), j.get("BHC_mu"))
        if stage not in active or None in key or j.get("JobStatus") == 3:
            continue
        points.setdefault(key, {"t": key[0], "mu": key[1], "file": None})
        old = active[stage].get(key)
        if old is None or (j.get("ClusterId", 0), j.get("ProcId", 0)) > (old.get("ClusterId", 0), old.get("ProcId", 0)):
            active[stage][key] = j
    last = {"ite": {}, "obs": {}}
    for j in history:
        stage, key = j.get("BHC_stage"), (j.get("BHC_t"), j.get("BHC_mu"))
        if stage not in last or None in key:
            continue
        points.setdefault(key, {"t": key[0], "mu": key[1], "file": None})
        rank = (j.get("CompletionDate") or j.get("EnteredCurrentStatus") or 0, j.get("ClusterId", 0))
        old = last[stage].get(key)
        if old is None or rank > old[0]:
            last[stage][key] = (rank, j)

    def failure(j):
        if j.get("JobStatus") == 3:
            return "removed" + (f": {j['RemoveReason']}" if j.get("RemoveReason") else "")
        if j.get("ExitCode") not in (None, 0):
            return f"exit code {j['ExitCode']}"
        return "finished without output file"

    out = []
    for key, p in points.items():
        f = p["file"] or {}
        rec = {"t": p["t"], "mu": p["mu"], "eps": f.get("eps"), "nsteps": f.get("nsteps"), "maxiter": f.get("maxiter"),
               "convtol": f.get("convtol"), "mtime": f.get("mtime"), "n_edge": f.get("n_edge"), "reason": None,
               "job": None, "obs_job": None}
        j = active["ite"].get(key)
        if j is not None:
            rec["state"], rec["job"] = QUEUE_STATE.get(j.get("JobStatus"), "idle"), job_info(j)
            if rec["state"] == "held":
                rec["reason"] = j.get("HoldReason")
        elif f.get("state"):
            if f.get("error"):
                rec["state"], rec["reason"] = "failed", "unreadable state: " + f["error"]
            else:
                rec["state"] = "done" if f.get("converged") else "unconverged"
        elif key in last["ite"]:
            j = last["ite"][key][1]
            rec["state"], rec["reason"], rec["job"] = "failed", failure(j), job_info(j)
        else:
            rec["state"] = "none"
        o = active["obs"].get(key)
        if o is not None:
            rec["obs"], rec["obs_job"] = QUEUE_STATE.get(o.get("JobStatus"), "idle"), job_info(o)
        elif f.get("obs"):
            rec["obs"] = "done"
        elif key in last["obs"] and f.get("state"):
            rec["obs"], rec["obs_job"] = "failed", job_info(last["obs"][key][1])
        else:
            rec["obs"] = "none"
        out.append(rec)
    out.sort(key=lambda r: (float(r["t"]), float(r["mu"])))
    return out


def parse_json_section(text, default):
    text = text.strip()
    if not text:
        return default
    for line in reversed(text.splitlines()):  # sweep_status.jl prints one line; ignore warnings before it
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def sections(out):
    parts, name = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            name = line[2:].strip()
            parts[name] = []
        elif name:
            parts[name].append(line)
    return {k: "\n".join(v) for k, v in parts.items()}


class Dashboard:
    def __init__(self, args):
        self.args = args
        self.remote = Remote(args)
        self.lock = threading.Lock()
        self.cache = {}
        self.tasks = {}
        self.current = None

    def julia(self):
        return self.remote.julia

    def condor_constraint(self, tag, L):
        return f'BHC_tag =?= "{tag}" && BHC_L =?= {int(L)}'

    def status(self, tag, L, fresh=False):
        if not RE_TAG.match(tag):
            raise ValueError("bad tag")
        L = int(L)
        key = (tag, L)
        with self.lock:
            hit = self.cache.get(key)
            if hit and not fresh and time.time() - hit[0] < 10:
                return hit[1]
        c = shlex.quote(self.condor_constraint(tag, L))
        cmd = (f"echo @@STATUS; {self.julia()} --project=. --startup-file=no hpc/condor/sweep_status.jl {tag} {L} 2>/dev/null; "
               f"echo @@QUEUE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint {c} -json -attributes {JOB_ATTRS} 2>/dev/null; "
               f"echo @@HISTORY; command -v condor_history >/dev/null && condor_history \"$USER\" -constraint {c} -json -attributes {JOB_ATTRS} -limit 3000 2>/dev/null; "
               f"echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=180)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        st = parse_json_section(s.get("STATUS", ""), {"points": [], "planned": []})
        queue = parse_json_section(s.get("QUEUE", ""), [])
        history = parse_json_section(s.get("HISTORY", ""), [])
        points = merge(st, queue, history)
        counts = {}
        for p in points:
            counts[p["state"]] = counts.get(p["state"], 0) + 1
        obs_counts = {}
        for p in points:
            obs_counts[p["obs"]] = obs_counts.get(p["obs"], 0) + 1
        clusters = sorted({j.get("ClusterId") for j in queue if j.get("ClusterId") is not None})
        res = {"tag": tag, "L": L, "points": points, "counts": counts, "obs_counts": obs_counts, "clusters": clusters,
               "queue_size": len(queue), "time": time.time(),
               "plot": os.path.isfile(os.path.join(self.args.gather_dest, "plots", f"{MODEL}_{tag}_L{L}.png"))}
        with self.lock:
            self.cache[key] = (time.time(), res)
        return res

    def sweeps(self):
        cmd = (f"echo @@LIST; {self.julia()} --project=. --startup-file=no hpc/condor/sweep_status.jl --list 2>/dev/null; "
               f"echo @@QUEUE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint 'BHC_tag =!= undefined' "
               f"-af BHC_tag BHC_L 2>/dev/null; echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=120)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        found = {(x["tag"], x["L"]): x for x in parse_json_section(s.get("LIST", ""), [])}
        for line in s.get("QUEUE", "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                found.setdefault((parts[0], int(parts[1])), {"tag": parts[0], "L": int(parts[1]), "n_states": 0,
                                                              "n_obs": 0, "mtime": time.time()})
        for (tag, L), x in found.items():
            x["queued"] = sum(1 for line in s.get("QUEUE", "").splitlines() if line.split() == [tag, str(L)])
        return sorted(found.values(), key=lambda x: -x.get("mtime", 0))

    def log(self, tag, L, t, mu, stage):
        if not (RE_TAG.match(tag) and RE_POINT.match(t) and RE_POINT.match(mu)):
            raise ValueError("bad point")
        L = int(L)
        d = f"logs/{MODEL}/{tag}/L{L}/condor" + ("/obs" if stage == "obs" else "")
        prefix = f"BHC_obs_L{L}" if stage == "obs" else f"BHC_L{L}"
        cmd = (f"f=$(ls -t {d}/{prefix}_t{t}_mu{mu}_*.err 2>/dev/null | head -1); "
               f"if [ -z \"$f\" ]; then echo 'No log files for this point yet.'; exit 0; fi; "
               f"o=\"${{f%.err}}.out\"; echo \"== $o\"; tail -n 60 \"$o\" 2>/dev/null; echo; echo \"== $f\"; tail -n 60 \"$f\"")
        rc, out, err = self.remote.run(cmd, timeout=60)
        return out + err

    # ---- actions (one at a time, output streamed to the page)

    def start(self, action, params):
        with self.lock:
            if self.current and not self.tasks[self.current]["done"]:
                raise RuntimeError("another action is still running")
            tid = uuid.uuid4().hex[:8]
            task = {"id": tid, "action": action, "lines": [], "done": False, "rc": None, "needs_confirm": False,
                    "started": time.time()}
            self.tasks[tid] = task
            self.current = tid
        threading.Thread(target=self._run, args=(task, action, params), daemon=True).start()
        return task

    def _run(self, task, action, params):
        emit = task["lines"].append
        try:
            rc = self._action(task, action, params, emit)
        except Exception as e:  # noqa: BLE001 - shown in the page
            emit(f"ERROR: {e}")
            rc = 1
        task["rc"], task["done"] = rc, True
        with self.lock:
            self.cache.clear()

    def _env(self, action, params):
        env = []
        for k in ACTION_PARAMS[action]:
            v = str(params.get(k, "")).strip()
            if v == "":
                continue
            if not re.match(PARAMS[k], v):
                raise ValueError(f"invalid value for {k}: {v!r}")
            env.append(f"{k}={shlex.quote(v)}")
        if params.get("DRYRUN") == "1":
            env.append("DRYRUN=1")
        env.append(f'JULIA="{self.julia()}"')  # double quotes: $HOME expands on the access point
        return " ".join(env)

    def _sync(self, params, emit, task):
        emit("$ git status (laptop)")
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.rstrip()
        ahead = subprocess.run(["git", "-C", REPO, "rev-list", "--count", "@{u}..HEAD"],
                               capture_output=True, text=True).stdout.strip()
        problems = []
        if dirty:
            problems.append("uncommitted changes on the laptop:\n" + dirty)
        if ahead not in ("", "0"):
            problems.append(f"{ahead} commit(s) on the laptop not pushed")
        if problems and params.get("confirm") != "1":
            for p in problems:
                emit("WARNING: " + p)
            emit("These changes will NOT be used on the cluster (it runs the code on GitHub).")
            emit("Commit and push them first, or press 'Continue anyway'.")
            task["needs_confirm"] = True
            return 2
        if problems:
            emit("Continuing with the code on GitHub (laptop changes ignored).")
        else:
            emit("laptop: everything committed and pushed")
        if self.args.local:
            emit("local mode: skipping git pull on the access point")
            return 0
        emit("$ git pull --ff-only (the access point)")
        rc = self.remote.stream("git pull --ff-only 2>&1", emit)
        if rc != 0:
            emit("git pull failed: resolve it on the access point (ssh, cd DisorderKit.jl, git status).")
            return rc
        emit("$ Pkg.instantiate() (the access point)")
        return self.remote.stream(f"{self.julia()} --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()' 2>&1", emit)

    def _action(self, task, action, params, emit):
        tag, L = params.get("tag", ""), params.get("L", "")
        if action in ("submit", "observables", "plot", "remove_held", "remove_all", "gather"):
            if not (RE_TAG.match(tag) and str(L).isdigit()):
                raise ValueError("no sweep selected")
            L = int(L)
        if action == "sync":
            return self._sync(params, emit, task)
        if action in ACTION_SCRIPT:
            env = self._env(action, params)  # validates the parameters before anything runs
            if action == "submit" and params.get("DRYRUN") != "1":
                rc = self._sync(params, emit, task)
                if rc != 0:
                    return rc
            cmd = f"env {env} bash {ACTION_SCRIPT[action]} {L} 2>&1"
            emit(f"$ {cmd}")
            return self.remote.stream(cmd, emit)
        if action in ("remove_held", "remove_all"):
            c = self.condor_constraint(tag, L) + (" && JobStatus == 5" if action == "remove_held" else "")
            cmd = f"condor_rm \"$USER\" -constraint {shlex.quote(c)} 2>&1"
            emit(f"$ {cmd}")
            return self.remote.stream(cmd, emit)
        if action == "gather":
            rel = f"data/{MODEL}/{tag}/L{L}/"
            dest = os.path.join(self.args.gather_dest, rel)
            os.makedirs(dest, exist_ok=True)
            os.makedirs(os.path.join(self.args.gather_dest, "plots"), exist_ok=True)
            cmd = ["rsync", "-av", *self.remote.rsync_ssh(), self.remote.rsync_source(rel), dest]
            emit("$ " + " ".join(shlex.quote(c) for c in cmd))
            rc = self._local_stream(cmd, emit)
            if rc != 0:
                return rc
            plot = f"plots/{MODEL}_{tag}_L{L}.png"
            if self.remote.run(f"test -f {shlex.quote(plot)}", timeout=60)[0] == 0:
                cmd = ["rsync", "-av", *self.remote.rsync_ssh(), self.remote.rsync_source(plot),
                       os.path.join(self.args.gather_dest, "plots") + "/"]
                emit("$ " + " ".join(shlex.quote(c) for c in cmd))
                rc = self._local_stream(cmd, emit)
            else:
                emit(f"No plot {plot} on the access point yet (press Plot first).")
            emit(f"Data in {dest}")
            return rc
        raise ValueError(f"unknown action {action}")

    @staticmethod
    def _local_stream(cmd, emit):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            emit(line.rstrip("\n"))
        return p.wait()

    def info(self):
        rc, out, _ = self.remote.run(f"test -f Project.toml && echo repo; {self.julia()} --startup-file=no -e 'print(VERSION)' 2>/dev/null; echo; "
                                     "command -v condor_q >/dev/null && echo condor", timeout=60)
        lines = out.split()
        return {"host": "local: " + self.args.local if self.args.local else f"{self.args.user}@{self.args.host}",
                "remote_dir": self.args.local or self.args.remote_dir, "connected": rc != 255 and not self.remote.lost,
                "repo": "repo" in lines, "condor": "condor" in lines,
                "julia": next((x for x in lines if re.match(r"^\d+\.\d+\.\d+", x)), None),
                "gather_dest": self.args.gather_dest}


def make_handler(dash):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    with open(os.path.join(HERE, "index.html"), "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                if u.path == "/api/info":
                    return self._send(200, dash.info())
                if u.path == "/api/sweeps":
                    return self._send(200, dash.sweeps())
                if u.path == "/api/status":
                    return self._send(200, dash.status(q["tag"], q["L"], q.get("fresh") == "1"))
                if u.path == "/api/log":
                    return self._send(200, {"text": dash.log(q["tag"], q["L"], q["t"], q["mu"], q.get("stage", "ite"))})
                if u.path == "/api/task":
                    t = dash.tasks.get(q.get("id")) or (dash.tasks.get(dash.current) if dash.current else None)
                    if t is None:
                        return self._send(200, None)
                    since = int(q.get("since", 0))
                    return self._send(200, {**{k: v for k, v in t.items() if k != "lines"}, "lines": t["lines"][since:],
                                            "n": len(t["lines"])})
                return self._send(404, {"error": "not found"})
            except ConnectionError as e:
                return self._send(503, {"error": f"connection to the access point lost ({e}); restart the dashboard"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/action":
                    t = dash.start(body.get("action"), body.get("params", {}))
                    return self._send(200, {"id": t["id"]})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": str(e)})

    return Handler


def selftest():
    status = {"points": [
        {"t": "0.1000", "mu": "0.2000", "state": True, "converged": True, "eps": 1e-7, "obs": True, "error": None},
        {"t": "0.1000", "mu": "0.4000", "state": True, "converged": False, "eps": 1e-3, "obs": False, "error": None},
        {"t": "0.2000", "mu": "0.2000", "state": True, "converged": False, "eps": 1e-3, "obs": False, "error": None}],
        "planned": [["0.1000", "0.2000"], ["0.3000", "0.2000"], ["0.3000", "0.4000"], ["0.2000", "0.4000"]]}
    queue = [
        {"ClusterId": 7, "ProcId": 0, "JobStatus": 2, "BHC_stage": "ite", "BHC_t": "0.2000", "BHC_mu": "0.2000"},
        {"ClusterId": 7, "ProcId": 1, "JobStatus": 5, "BHC_stage": "ite", "BHC_t": "0.3000", "BHC_mu": "0.2000",
         "HoldReason": "Error from slot: failed to transfer output"},
        {"ClusterId": 8, "ProcId": 0, "JobStatus": 1, "BHC_stage": "obs", "BHC_t": "0.1000", "BHC_mu": "0.4000"},
        {"ClusterId": 9, "ProcId": 0, "JobStatus": 1, "BHC_stage": "ite", "BHC_t": "0.4000", "BHC_mu": "0.2000"}]
    history = [{"ClusterId": 5, "ProcId": 3, "JobStatus": 4, "ExitCode": 1, "CompletionDate": 100, "BHC_stage": "ite",
                "BHC_t": "0.3000", "BHC_mu": "0.4000"}]
    got = {(p["t"], p["mu"]): (p["state"], p["obs"]) for p in merge(status, queue, history)}
    want = {("0.1000", "0.2000"): ("done", "done"), ("0.1000", "0.4000"): ("unconverged", "idle"),
            ("0.2000", "0.2000"): ("running", "none"), ("0.3000", "0.2000"): ("held", "none"),
            ("0.3000", "0.4000"): ("failed", "none"), ("0.2000", "0.4000"): ("none", "none"),
            ("0.4000", "0.2000"): ("idle", "none")}
    assert got == want, f"\n got  {got}\n want {want}"
    assert sections("@@A\n1\n2\n@@B\n@@END\n") == {"A": "1\n2", "B": "", "END": ""}
    assert parse_json_section("warning\n[1,2]\n", None) == [1, 2]
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="<access point>")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--user", default="<login>")
    ap.add_argument("--remote-dir", default="DisorderKit.jl")
    ap.add_argument("--julia", default=None)
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--local", default=None)
    ap.add_argument("--gather-dest", default=REPO)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.local:
        args.local = os.path.abspath(args.local)
    if args.julia is None:
        with open(os.path.join(REPO, "Manifest.toml")) as f:
            version = re.search(r'^julia_version = "(.*)"', f.read(), re.M).group(1)
        args.julia = "julia" if args.local else f"$HOME/.juliaup/bin/julia +{version}"
    args.gather_dest = os.path.abspath(args.gather_dest)
    dash = Dashboard(args)
    if not dash.remote.connect():
        sys.exit("Could not connect to the access point (on the UGent network or VPN? right password?)")
    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(dash))
    url = f"http://localhost:{args.http_port}"
    print(f"Dashboard on {url}  (Ctrl-C to stop; jobs keep running on the cluster)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
