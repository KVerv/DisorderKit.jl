#!/usr/bin/env python3
"""Dashboard for the correlated Bose-Hubbard sweeps on an HTCondor pool.

Runs on the laptop and talks to the condor access point over one persistent ssh connection:

    python3 hpc/condor/dashboard/server.py      # asks for the ssh password once, opens the browser

The connection settings (host, port, login, repo folder on the access point) are read from
hpc/condor/condor.local.env (gitignored; asked for and written on the first start; template:
hpc/condor/condor.local.env.example). Command-line flags override them.

Workflow: every condor job runs the ITE and the observables of one point (hpc/condor/sweep_BHCorr.sh). *Gather*
copies finished results to this repo's data/ folder and then deletes them (and their logs) on the access point, so
nothing piles up there. The laptop is the archive: the grid shows the laptop's and the access point's results together,
*Submit* decides which points still need a run from both, *Recompute observables* uploads only the needed states and
runs observables-only jobs (a changed observables script is detected through its git hash), and *Plot* runs the
phase-diagram script locally on the laptop's data.

Options: --host, --port, --user, --remote-dir (override condor.local.env), --julia (Julia command on the access point,
default juliaup with the Manifest's version), --local-julia (Julia on the laptop, default "julia"), --http-port (8765),
--no-browser, --gather-dest (the laptop's data root, default this repo), --local DIR (testing: no ssh, DIR plays the
access point), --selftest. Python 3 standard library only.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CONFIG = os.path.join(REPO, "hpc", "condor", "condor.local.env")
MODEL = "BoseHubbardCorrelated_NormalizedITE"
OBS_SCRIPT = "hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl"
PLOT_SCRIPT = "hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl"
JOB_ATTRS = ("ClusterId,ProcId,JobStatus,BHC_stage,BHC_tag,BHC_L,BHC_t,BHC_mu,RemoteHost,LastRemoteHost,"
             "HoldReason,EnteredCurrentStatus,QDate,CompletionDate,ExitCode,RemoveReason")
GATHER_MIN_AGE = 2  # minutes: younger result files may still be written by condor

RE_TAG = re.compile(r"^[A-Za-z0-9._+-]+$")
RE_POINT = re.compile(r"^-?\d+\.\d{4}$")
NUM = r"^[0-9.eE+-]+$"
# Environment variables the actions may pass to the condor scripts, with the allowed values
PARAMS = {
    "U": NUM, "R": NUM, "DELTA": NUM, "OMEGA": NUM, "NMAX": r"^\d+$", "DTAU": NUM, "MAXD": r"^\d+$",
    "MAXITER": r"^\d+$", "CONVTOL": NUM, "CROSS_WIDTH": r"^\d+$", "PPN": r"^\d+$", "OBS_THREADS": r"^\d+$",
    "MEM": r"^\d+(\.\d+)? ?[KMGT]?B?$", "DISK": r"^\d+(\.\d+)? ?[KMGT]?B?$",
    "REQUIREMENTS": r"^[A-Za-z0-9_ .<>=!&|()\"']*$", "NICE": r"^(TRUE|FALSE)$",
    "N_SAMPLES": r"^\d+$", "N_EDGE": r"^\d+$", "KAPPA_TOL": NUM, "CV_TOL": NUM,
    "POINTS": r"^[0-9.\- ,]*$", "DRYRUN": r"^[01]$",
}
MODEL_PARAMS = ["U", "R", "DELTA", "OMEGA", "NMAX", "DTAU", "MAXD"]
SCRIPT_PARAMS = {
    "submit": MODEL_PARAMS + ["MAXITER", "CONVTOL", "CROSS_WIDTH", "N_SAMPLES", "N_EDGE", "PPN", "OBS_THREADS", "MEM",
                              "DISK", "REQUIREMENTS", "NICE", "POINTS", "DRYRUN"],
    "recompute_obs": MODEL_PARAMS + ["N_SAMPLES", "N_EDGE", "PPN", "MEM", "REQUIREMENTS", "NICE", "POINTS"],
}
SCRIPT = {"submit": "hpc/condor/sweep_BHCorr.sh", "recompute_obs": "hpc/condor/obs_BHCorr.sh"}


# ---- configuration (kept out of the repo)

def read_config():
    cfg = {}
    if os.path.isfile(CONFIG):
        with open(CONFIG) as f:
            for line in f:
                m = re.match(r"^\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
                if m and not line.lstrip().startswith("#"):
                    cfg[m.group(1)] = m.group(2)
    return cfg


def resolve_connection(args):
    cfg = read_config()
    keys = [("host", "CONDOR_HOST", "Access point host name"), ("port", "CONDOR_PORT", "ssh port"),
            ("user", "CONDOR_USER", "Your login on the access point"),
            ("remote_dir", "CONDOR_REMOTE_DIR", "Repo folder on the access point, relative to your home")]
    changed = False
    for attr, key, prompt in keys:
        if getattr(args, attr) is None:
            value = cfg.get(key, "")
            if not value or value.startswith("<"):
                default = "DisorderKit.jl" if key == "CONDOR_REMOTE_DIR" else ""
                value = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip() or default
                cfg[key] = value
                changed = True
            setattr(args, attr, value)
    if changed:
        with open(CONFIG, "w") as f:
            f.write("# Connection to the condor access point for hpc/condor/dashboard/server.py (gitignored, do not commit)\n")
            for _, key, _ in keys:
                f.write(f"{key}={cfg[key]}\n")
        print(f"Saved the connection settings in {CONFIG} (gitignored).")
    args.port = str(args.port)


class Remote:
    """Runs shell commands in the repo on the access point (over the ssh master connection) or locally (--local)."""

    def __init__(self, args):
        self.local = args.local
        self.host, self.port, self.user, self.dir = args.host, args.port, args.user, args.remote_dir
        self.target = f"{self.user}@{self.host}" if not self.local else ""
        self.sock = os.path.expanduser(f"~/.ssh/condor-dashboard-{self.user}@{self.host}-{self.port}.sock")
        self.julia = args.julia
        self.lost = False

    def _argv(self, cmd):
        if self.local:
            return ["bash", "-c", f"cd {shlex.quote(self.local)} && {cmd}"]
        return ["ssh", "-S", self.sock, "-o", "BatchMode=yes", "-p", self.port, self.target,
                f"cd {shlex.quote(self.dir)} && {cmd}"]

    def connect(self):
        if self.local:
            return True
        check = ["ssh", "-S", self.sock, "-O", "check", "-p", self.port, self.target]
        if subprocess.run(check, capture_output=True).returncode == 0:
            return True
        print(f"Connecting to {self.target} (one password prompt; the connection is kept open for 12 h) ...")
        rc = subprocess.run(["ssh", "-M", "-S", self.sock, "-o", "ControlPersist=12h", "-o", "ServerAliveInterval=60",
                             "-p", self.port, "-fN", self.target]).returncode
        self.lost = rc != 0
        return rc == 0

    def run(self, cmd, timeout=300):
        try:
            p = subprocess.run(self._argv(cmd), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {timeout} s"
        if p.returncode == 255 and not self.local:
            self.lost = subprocess.run(["ssh", "-S", self.sock, "-O", "check", "-p", self.port, self.target],
                                       capture_output=True).returncode != 0
        return p.returncode, p.stdout, p.stderr

    def stream(self, cmd, emit):
        return stream(self._argv(cmd), emit)

    def path(self, rel=""):
        """rsync address of a path relative to the repo on the access point."""
        if self.local:
            return os.path.join(self.local, rel)
        return f"{self.target}:{self.dir}/{rel}"

    def rsync_ssh(self):
        return [] if self.local else ["-e", f"ssh -S {self.sock} -o BatchMode=yes -p {self.port}"]


def stream(argv, emit, **kw):
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    for line in p.stdout:
        emit(line.rstrip("\n"))
    return p.wait()


# ---- state of the points

QUEUE_STATE = {1: "idle", 2: "running", 4: "running", 5: "held", 6: "running", 7: "idle"}


def job_info(j):
    return {"id": f"{j.get('ClusterId')}.{j.get('ProcId')}", "status": j.get("JobStatus"),
            "host": j.get("RemoteHost") or j.get("LastRemoteHost"), "since": j.get("EnteredCurrentStatus"),
            "hold_reason": j.get("HoldReason"), "exit_code": j.get("ExitCode"), "remove_reason": j.get("RemoveReason")}


def merge(remote, local, queue, history, obs_version="", n_edge=None):
    """One record per point from the access point's files (remote), the laptop's files (local), the queue and the
    history. A state on the access point is newer than the laptop's copy (it is gathered later)."""
    points = {}

    def entry(key):
        return points.setdefault(key, {"t": key[0], "mu": key[1], "remote": None, "local": None})

    for where, st in (("remote", remote), ("local", local)):
        for p in st.get("points", []):
            entry((p["t"], p["mu"]))[where] = p
    for t, mu in remote.get("planned", []):
        entry((t, mu))
    active = {"ite": {}, "obs": {}}
    for j in queue:
        stage, key = j.get("BHC_stage"), (j.get("BHC_t"), j.get("BHC_mu"))
        if stage not in active or None in key or j.get("JobStatus") == 3:
            continue
        entry(key)
        old = active[stage].get(key)
        if old is None or (j.get("ClusterId", 0), j.get("ProcId", 0)) > (old.get("ClusterId", 0), old.get("ProcId", 0)):
            active[stage][key] = j
    last = {"ite": {}, "obs": {}}
    for j in history:
        stage, key = j.get("BHC_stage"), (j.get("BHC_t"), j.get("BHC_mu"))
        if stage not in last or None in key:
            continue
        entry(key)
        rank = (j.get("CompletionDate") or j.get("EnteredCurrentStatus") or 0, j.get("ClusterId", 0))
        if key not in last[stage] or rank > last[stage][key][0]:
            last[stage][key] = (rank, j)

    def failure(j):
        if j.get("JobStatus") == 3:
            return "removed" + (f": {j['RemoveReason']}" if j.get("RemoveReason") else "")
        if j.get("ExitCode") not in (None, 0):
            return f"exit code {j['ExitCode']}"
        return "finished without its output files"

    out = []
    for key, p in points.items():
        r, l = p["remote"] or {}, p["local"] or {}
        f = r if r.get("state") else l            # the state that counts: the access point's (newer) or the laptop's
        o = r if r.get("obs") else l               # observables: likewise
        where = [w for w, x in (("laptop", l), ("access point", r)) if x.get("state") or x.get("obs")]
        rec = {"t": p["t"], "mu": p["mu"], "eps": f.get("eps"), "nsteps": f.get("nsteps"), "maxiter": f.get("maxiter"),
               "convtol": f.get("convtol"), "mtime": f.get("mtime"), "has_state": bool(f.get("state")),
               "n_edge": o.get("n_edge"), "obs_version": o.get("obs_version"), "where": where,
               "on_remote": bool(r.get("state") or r.get("obs")), "reason": None, "job": None, "obs_job": None}
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
        oj = active["obs"].get(key)
        if oj is not None:
            rec["obs"], rec["obs_job"] = QUEUE_STATE.get(oj.get("JobStatus"), "idle"), job_info(oj)
        elif o.get("obs"):
            stale = (obs_version and o.get("obs_version") != obs_version) or \
                    (n_edge is not None and o.get("n_edge") != n_edge)
            rec["obs"] = "outdated" if stale else "done"
        elif key in last["obs"] and f.get("state"):
            rec["obs"], rec["obs_job"] = "failed", job_info(last["obs"][key][1])
        else:
            rec["obs"] = "none"
        out.append(rec)
    out.sort(key=lambda x: (float(x["t"]), float(x["mu"])))
    return out


def needs_run(p, maxiter, convtol, force=False):
    """Same rule as `hpc/BoseHubbardCorrelated_NormalizedITE.jl --status`, on the merged record of a point."""
    if p.get("state") in ("idle", "running", "held"):
        return None                                   # in the queue
    if force or not p.get("has_state"):
        return "run"
    eps = p.get("eps")
    eps = float("inf") if eps is None else eps
    if eps <= convtol:
        return None
    if maxiter > (p.get("maxiter") or 0) or convtol < (p.get("convtol") or float("inf")):
        return "rerun"
    return None


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
        self.local_cache = {}
        self.tasks = {}
        self.current = None
        self.local_julia = shlex.split(args.local_julia)

    # ---- queries

    def condor_constraint(self, tag, L):
        return f'BHC_tag =?= "{tag}" && BHC_L =?= {int(L)}'

    def obs_version(self):
        p = subprocess.run(["git", "-C", REPO, "hash-object", OBS_SCRIPT], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else ""

    def local_scan(self, tag, L):
        """sweep_status.jl on the laptop's data (cached until the folder changes)."""
        folder = os.path.join(self.args.gather_dest, "data", MODEL, tag, f"L{L}")
        sig = [0, 0.0]
        for root, _, files in os.walk(folder):
            for name in files:
                sig[0] += 1
                sig[1] = max(sig[1], os.path.getmtime(os.path.join(root, name)))
        sig = tuple(sig)
        hit = self.local_cache.get((tag, L))
        if hit and hit[0] == sig:
            return hit[1]
        if sig[0] == 0:
            res = {"points": [], "planned": []}
        else:
            p = subprocess.run([*self.local_julia, f"--project={REPO}", "--startup-file=no",
                                os.path.join(REPO, "hpc", "condor", "sweep_status.jl"), tag, str(L)],
                               cwd=self.args.gather_dest, capture_output=True, text=True, timeout=600)
            res = parse_json_section(p.stdout, None)
            if res is None:
                raise RuntimeError("reading the laptop's data failed: " + (p.stderr.strip().splitlines() or ["?"])[-1])
        self.local_cache[(tag, L)] = (sig, res)
        return res

    def status(self, tag, L, n_edge=None, fresh=False):
        if not RE_TAG.match(tag):
            raise ValueError("bad tag")
        L = int(L)
        key = (tag, L, n_edge)
        with self.lock:
            hit = self.cache.get(key)
            if hit and not fresh and time.time() - hit[0] < 10:
                return hit[1]
        c = shlex.quote(self.condor_constraint(tag, L))
        cmd = (f"echo @@STATUS; {self.remote.julia} --project=. --startup-file=no hpc/condor/sweep_status.jl {tag} {L} 2>/dev/null; "
               f"echo @@QUEUE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint {c} -json -attributes {JOB_ATTRS} 2>/dev/null; "
               f"echo @@HISTORY; command -v condor_history >/dev/null && condor_history \"$USER\" -constraint {c} -json -attributes {JOB_ATTRS} -limit 3000 2>/dev/null; "
               f"echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=180)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        remote = parse_json_section(s.get("STATUS", ""), {"points": [], "planned": []})
        queue = parse_json_section(s.get("QUEUE", ""), [])
        history = parse_json_section(s.get("HISTORY", ""), [])
        local = self.local_scan(tag, L)
        points = merge(remote, local, queue, history, self.obs_version(), n_edge)
        counts, obs_counts = {}, {}
        for p in points:
            counts[p["state"]] = counts.get(p["state"], 0) + 1
            obs_counts[p["obs"]] = obs_counts.get(p["obs"], 0) + 1
        plot = os.path.join(self.args.gather_dest, "plots", f"{MODEL}_{tag}_L{L}.png")
        res = {"tag": tag, "L": L, "points": points, "counts": counts, "obs_counts": obs_counts,
               "clusters": sorted({j.get("ClusterId") for j in queue if j.get("ClusterId") is not None}),
               "queue_size": len(queue), "time": time.time(),
               "to_gather": sum(1 for p in points if p["on_remote"] and p["state"] not in ("idle", "running", "held")
                                and p["obs"] not in ("idle", "running", "held")),
               "on_laptop": sum(1 for p in points if "laptop" in p["where"]),
               "plot_mtime": os.path.getmtime(plot) if os.path.isfile(plot) else None}
        with self.lock:
            self.cache[key] = (time.time(), res)
        return res

    def sweeps(self):
        cmd = (f"echo @@LIST; {self.remote.julia} --project=. --startup-file=no hpc/condor/sweep_status.jl --list 2>/dev/null; "
               f"echo @@QUEUE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint 'BHC_tag =!= undefined' "
               f"-af BHC_tag BHC_L 2>/dev/null; echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=120)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        found = {}
        for x in parse_json_section(s.get("LIST", ""), []):
            found[(x["tag"], x["L"])] = {**x, "n_states": 0, "on_remote": x["n_states"]}
        # the laptop's sweeps (the archive)
        root = os.path.join(self.args.gather_dest, "data", MODEL)
        for tag in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            for Ldir in os.listdir(os.path.join(root, tag)):
                if not re.match(r"^L\d+$", Ldir):
                    continue
                folder = os.path.join(root, tag, Ldir)
                states = [f for _, _, fs in os.walk(folder) for f in fs if re.match(r"^mu-?[\d.]+\.jld2$", f)]
                x = found.setdefault((tag, int(Ldir[1:])), {"tag": tag, "L": int(Ldir[1:]), "on_remote": 0, "n_obs": 0})
                x["n_states"] = len(states)
                x["mtime"] = max(x.get("mtime", 0), os.path.getmtime(folder))
        queue_lines = [line.split() for line in s.get("QUEUE", "").splitlines()]
        for parts in queue_lines:
            if len(parts) == 2 and parts[1].isdigit():
                found.setdefault((parts[0], int(parts[1])), {"tag": parts[0], "L": int(parts[1]), "n_states": 0,
                                                              "on_remote": 0, "n_obs": 0, "mtime": time.time()})
        for (tag, L), x in found.items():
            x["queued"] = sum(1 for parts in queue_lines if parts == [tag, str(L)])
            x.setdefault("n_states", 0)
        return sorted(found.values(), key=lambda x: -x.get("mtime", 0))

    def log(self, tag, L, t, mu, stage):
        if not (RE_TAG.match(tag) and RE_POINT.match(t) and RE_POINT.match(mu)):
            raise ValueError("bad point")
        L = int(L)
        d = f"logs/{MODEL}/{tag}/L{L}/condor" + ("/obs" if stage == "obs" else "")
        prefix = f"BHC_obs_L{L}" if stage == "obs" else f"BHC_L{L}"
        cmd = (f"f=$(ls -t {d}/{prefix}_t{t}_mu{mu}_*.err 2>/dev/null | head -1); "
               f"if [ -z \"$f\" ]; then echo 'No log files for this point on the access point (gathered points have no logs left).'; exit 0; fi; "
               f"o=\"${{f%.err}}.out\"; echo \"== $o\"; tail -n 60 \"$o\" 2>/dev/null; echo; echo \"== $f\"; tail -n 60 \"$f\"")
        rc, out, err = self.remote.run(cmd, timeout=60)
        return out + err

    def plot_file(self, tag, L):
        if not RE_TAG.match(tag):
            raise ValueError("bad tag")
        return os.path.join(self.args.gather_dest, "plots", f"{MODEL}_{tag}_L{int(L)}.png")

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

    @staticmethod
    def _check(params, keys):
        out = {}
        for k in keys:
            v = str(params.get(k, "")).strip()
            if v == "":
                continue
            if not re.match(PARAMS[k], v):
                raise ValueError(f"invalid value for {k}: {v!r}")
            out[k] = v
        return out

    def _env(self, values):
        env = [f"{k}={shlex.quote(v)}" for k, v in values.items()]
        env.append(f'JULIA="{self.remote.julia}"')  # double quotes: $HOME expands on the access point
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
        emit("Continuing with the code on GitHub (laptop changes ignored)." if problems
             else "laptop: everything committed and pushed")
        if self.args.local:
            emit("local mode: skipping git pull on the access point")
            return 0
        emit("$ git pull --ff-only (access point)")
        rc = self.remote.stream("git pull --ff-only 2>&1", emit)
        if rc != 0:
            emit("git pull failed: resolve it on the access point (ssh, cd to the repo, git status).")
            return rc
        emit("$ Pkg.instantiate() (access point)")
        return self.remote.stream(f"{self.remote.julia} --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()' 2>&1", emit)

    def _grid(self, params):
        ts = [f"{float(x):.4f}" for x in str(params.get("TS", "")).split()]
        ms = [f"{float(x):.4f}" for x in str(params.get("MUS", "")).split()]
        return [(t, m) for t in ts for m in ms]

    def _action(self, task, action, params, emit):
        tag, L = params.get("tag", ""), params.get("L", "")
        if not (RE_TAG.match(tag) and str(L).isdigit()):
            raise ValueError("no sweep selected")
        L = int(L)
        n_edge = int(params["N_EDGE"]) if str(params.get("N_EDGE", "")).isdigit() else None
        if action in ("submit", "preview", "recompute_obs") and n_edge is not None and L < 2 * n_edge + 3:
            raise ValueError(f"N_EDGE = {n_edge} leaves fewer than 2 bulk bonds for L = {L} (need L ≥ 2·N_EDGE + 3)")
        if action in ("submit", "preview"):
            values = self._check(params, SCRIPT_PARAMS["submit"])
            maxiter, convtol = int(values.get("MAXITER", 2000)), float(values.get("CONVTOL", 1e-6))
            st = self.status(tag, L, n_edge, fresh=True)
            known = {(p["t"], p["mu"]): p for p in st["points"]}
            grid = self._grid(params) or list(known)
            todo, nrun, nrerun, nskip, nqueued = [], 0, 0, 0, 0
            for key in grid:
                p = known.get(key, {"state": "none", "has_state": False})
                why = needs_run(p, maxiter, convtol, params.get("FORCE") == "1")
                if why is None:
                    nqueued += p.get("state") in ("idle", "running", "held")
                    nskip += p.get("state") not in ("idle", "running", "held")
                    continue
                nrun += why == "run"
                nrerun += why == "rerun"
                todo.append(key)
            emit(f"{len(todo)} point(s) to submit ({nrun} new, {nrerun} rerun with the new MAXITER/CONVTOL), "
                 f"{nskip} done, {nqueued} in the queue")
            if not todo:
                return 0
            values["POINTS"] = ",".join(f"{t} {m}" for t, m in todo)
            if action == "preview":
                values["DRYRUN"] = "1"
            else:
                rc = self._sync(params, emit, task)
                if rc != 0:
                    return rc
            cmd = f"env {self._env(values)} bash {SCRIPT['submit']} {L} 2>&1"
            emit(f"$ bash {SCRIPT['submit']} {L}  ({len(todo)} points)")
            return self.remote.stream(cmd, emit)
        if action == "recompute_obs":
            values = self._check(params, SCRIPT_PARAMS["recompute_obs"])
            st = self.status(tag, L, n_edge, fresh=True)
            todo = [p for p in st["points"] if p["has_state"] and p["obs"] in ("none", "outdated", "failed")
                    and p["state"] not in ("idle", "running", "held")]
            emit(f"{len(todo)} point(s) need observables (missing, outdated or failed)")
            if not todo:
                return 0
            upload = [p for p in todo if "laptop" in p["where"] and not p["on_remote"]]
            if upload:
                emit(f"Uploading {len(upload)} state(s) from the laptop to the access point ...")
                rel = [f"data/{MODEL}/{tag}/L{L}/t{p['t']}/mu{p['mu']}.jld2" for p in upload]
                rc = self._rsync(rel, self.args.gather_dest + "/", self.remote.path(), emit)
                if rc != 0:
                    return rc
            rc = self._sync(params, emit, task)
            if rc != 0:
                return rc
            values["POINTS"] = ",".join(f"{p['t']} {p['mu']}" for p in todo)
            emit(f"$ bash {SCRIPT['recompute_obs']} {L}  ({len(todo)} points)")
            return self.remote.stream(f"env {self._env(values)} bash {SCRIPT['recompute_obs']} {L} 2>&1", emit)
        if action in ("remove_held", "remove_all"):
            c = self.condor_constraint(tag, L) + (" && JobStatus == 5" if action == "remove_held" else "")
            cmd = f"condor_rm \"$USER\" -constraint {shlex.quote(c)} 2>&1"
            emit(f"$ condor_rm (jobs of this sweep{', held only' if action == 'remove_held' else ''})")
            return self.remote.stream(cmd, emit)
        if action == "gather":
            return self._gather(tag, L, emit)
        if action == "plot":
            values = self._check(params, MODEL_PARAMS + ["KAPPA_TOL", "CV_TOL"])
            env = {**os.environ, **values, "DATA_DIR": "data"}
            cmd = [*self.local_julia, f"--project={REPO}", "--startup-file=no", os.path.join(REPO, PLOT_SCRIPT), str(L)]
            emit(f"$ julia {PLOT_SCRIPT} {L}  (on the laptop, data in {self.args.gather_dest}/data)")
            return stream(cmd, emit, cwd=self.args.gather_dest, env=env)
        raise ValueError(f"unknown action {action}")

    def _rsync(self, rel, src, dst, emit, remove_source=False):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(rel) + "\n")
            listfile = f.name
        try:
            cmd = ["rsync", "-a", "--itemize-changes", f"--files-from={listfile}", *self.remote.rsync_ssh()]
            if remove_source:
                cmd.append("--remove-source-files")
            return stream([*cmd, src, dst], emit)
        finally:
            os.unlink(listfile)

    def _gather(self, tag, L, emit):
        """Copy the finished results of this sweep to the laptop, then delete them and their logs on the access point."""
        data = f"data/{MODEL}/{tag}/L{L}"
        logs = f"logs/{MODEL}/{tag}/L{L}/condor"
        c = shlex.quote(self.condor_constraint(tag, L))
        # Result files older than GATHER_MIN_AGE min of points without a job in the queue (any stage)
        cmd = (f"echo @@FILES; [ -d {data} ] && find {data} -type f \\( -name 'mu*.jld2' -o -name 'mu*_obs.csv' \\) "
               f"-mmin +{GATHER_MIN_AGE}; "
               f"echo @@ACTIVE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint {c} -af BHC_t BHC_mu 2>/dev/null; "
               f"echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=120)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        active = {tuple(line.split()) for line in s.get("ACTIVE", "").splitlines() if len(line.split()) == 2}
        files, points = [], set()
        for path in s.get("FILES", "").splitlines():
            m = re.match(rf"^{re.escape(data)}/t(-?\d+\.\d{{4}})/mu(-?\d+\.\d{{4}})(_obs\.csv|\.jld2)$", path.strip())
            if m and (m.group(1), m.group(2)) not in active:
                files.append(path.strip())
                points.add((m.group(1), m.group(2)))
        if not files:
            emit("Nothing to gather: no finished results on the access point (results younger than "
                 f"{GATHER_MIN_AGE} min or of points still in the queue wait for the next Gather).")
            return 0
        emit(f"Copying {len(files)} file(s) of {len(points)} point(s) to the laptop; each file is deleted on the access "
             "point only after it was copied completely:")
        rc = self._rsync(files, self.remote.path(), self.args.gather_dest + "/", emit, remove_source=True)
        if rc != 0:
            emit(f"rsync failed (exit {rc}): nothing more is deleted. Press Gather again to retry.")
            return rc
        missing = [f for f in files if not os.path.isfile(os.path.join(self.args.gather_dest, f))]
        if missing:
            emit(f"ERROR: {len(missing)} file(s) not found on the laptop after the copy, e.g. {missing[0]}")
            return 1
        emit(f"Copied to {os.path.join(self.args.gather_dest, data)}")
        # Logs of the gathered points, and (if no job of this sweep is left) the cluster logs, points and submit files
        rm = [f"rm -f {logs}/BHC_L{L}_t{t}_mu{m}_*.out {logs}/BHC_L{L}_t{t}_mu{m}_*.err "
              f"{logs}/obs/BHC_obs_L{L}_t{t}_mu{m}_*.out {logs}/obs/BHC_obs_L{L}_t{t}_mu{m}_*.err" for t, m in sorted(points)]
        if not active:
            rm.append(f"rm -f {logs}/*.log {logs}/points_*.txt {logs}/*.submit {logs}/obs/*.log {logs}/obs/points_*.txt "
                      f"{logs}/obs/*.submit")
        rm.append(f"for d in {data} logs/{MODEL}/{tag}; do [ -d \"$d\" ] && find \"$d\" -type d -empty -delete; done; true")
        rc, _, err = self.remote.run("; ".join(rm), timeout=120)
        emit(f"Deleted the logs of {len(points)} gathered point(s) on the access point"
             + ("" if active else ", and the cluster logs and submit files (no job of this sweep left)")
             + ("" if rc == 0 else f" (with errors: {err.strip()})"))
        self.local_cache.pop((tag, L), None)
        return 0

    def info(self):
        rc, out, _ = self.remote.run(f"test -f Project.toml && echo repo; {self.remote.julia} --startup-file=no -e 'print(VERSION)' 2>/dev/null; echo; "
                                     "command -v condor_q >/dev/null && echo condor", timeout=60)
        lines = out.split()
        return {"host": "local test: " + self.args.local if self.args.local else self.remote.target,
                "connected": rc != 255 and not self.remote.lost,
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
                    n_edge = int(q["n_edge"]) if q.get("n_edge", "").isdigit() else None
                    return self._send(200, dash.status(q["tag"], q["L"], n_edge, q.get("fresh") == "1"))
                if u.path == "/api/log":
                    return self._send(200, {"text": dash.log(q["tag"], q["L"], q["t"], q["mu"], q.get("stage", "ite"))})
                if u.path == "/api/plot":
                    path = dash.plot_file(q["tag"], q["L"])
                    if not os.path.isfile(path):
                        return self._send(404, {"error": "no plot yet"})
                    with open(path, "rb") as f:
                        return self._send(200, f.read(), "image/png")
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
    remote = {"points": [
        {"t": "0.1000", "mu": "0.2000", "state": True, "converged": True, "eps": 1e-7, "obs": True, "n_edge": 2,
         "obs_version": "v2", "error": None},
        {"t": "0.2000", "mu": "0.2000", "state": True, "converged": False, "eps": 1e-3, "obs": False, "error": None}],
        "planned": [["0.1000", "0.2000"], ["0.3000", "0.2000"], ["0.3000", "0.4000"], ["0.2000", "0.4000"]]}
    local = {"points": [
        {"t": "0.1000", "mu": "0.2000", "state": True, "converged": False, "eps": 1e-3, "obs": True, "n_edge": 2,
         "obs_version": "v1", "error": None},                                     # older copy: remote wins
        {"t": "0.1000", "mu": "0.4000", "state": True, "converged": False, "eps": 1e-3, "maxiter": 2000, "convtol": 1e-6,
         "obs": True, "n_edge": 2, "obs_version": "v1", "error": None},           # gathered, obs from an old script
        {"t": "0.5000", "mu": "0.2000", "state": True, "converged": True, "eps": 1e-8, "maxiter": 2000, "convtol": 1e-6,
         "obs": True, "n_edge": 1, "obs_version": "v2", "error": None}]}          # gathered, other N_EDGE
    queue = [
        {"ClusterId": 7, "ProcId": 0, "JobStatus": 2, "BHC_stage": "ite", "BHC_t": "0.2000", "BHC_mu": "0.2000"},
        {"ClusterId": 7, "ProcId": 1, "JobStatus": 5, "BHC_stage": "ite", "BHC_t": "0.3000", "BHC_mu": "0.2000",
         "HoldReason": "Error from slot: failed to transfer output"},
        {"ClusterId": 9, "ProcId": 0, "JobStatus": 1, "BHC_stage": "ite", "BHC_t": "0.4000", "BHC_mu": "0.2000"}]
    history = [{"ClusterId": 5, "ProcId": 3, "JobStatus": 4, "ExitCode": 1, "CompletionDate": 100, "BHC_stage": "ite",
                "BHC_t": "0.3000", "BHC_mu": "0.4000"}]
    recs = {(p["t"], p["mu"]): p for p in merge(remote, local, queue, history, "v2", 2)}
    got = {k: (p["state"], p["obs"]) for k, p in recs.items()}
    want = {("0.1000", "0.2000"): ("done", "done"), ("0.1000", "0.4000"): ("unconverged", "outdated"),
            ("0.2000", "0.2000"): ("running", "none"), ("0.3000", "0.2000"): ("held", "none"),
            ("0.3000", "0.4000"): ("failed", "none"), ("0.2000", "0.4000"): ("none", "none"),
            ("0.4000", "0.2000"): ("idle", "none"), ("0.5000", "0.2000"): ("done", "outdated")}
    assert got == want, f"\n got  {got}\n want {want}"
    assert recs[("0.1000", "0.2000")]["where"] == ["laptop", "access point"]
    assert recs[("0.1000", "0.4000")]["where"] == ["laptop"] and not recs[("0.1000", "0.4000")]["on_remote"]
    # submit decisions (MAXITER 2000 / 4000, CONVTOL 1e-6)
    assert needs_run(recs[("0.2000", "0.4000")], 2000, 1e-6) == "run"
    assert needs_run(recs[("0.5000", "0.2000")], 2000, 1e-6) is None              # converged
    assert needs_run(recs[("0.1000", "0.4000")], 2000, 1e-6) is None              # unconverged, same settings
    assert needs_run(recs[("0.1000", "0.4000")], 4000, 1e-6) == "rerun"           # more steps
    assert needs_run(recs[("0.2000", "0.2000")], 4000, 1e-6) is None              # in the queue
    assert needs_run(recs[("0.5000", "0.2000")], 2000, 1e-6, force=True) == "run"
    assert sections("@@A\n1\n2\n@@B\n@@END\n") == {"A": "1\n2", "B": "", "END": ""}
    assert parse_json_section("warning\n[1,2]\n", None) == [1, 2]
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--port")
    ap.add_argument("--user")
    ap.add_argument("--remote-dir")
    ap.add_argument("--julia", default=None)
    ap.add_argument("--local-julia", default="julia")
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
        args.host, args.port, args.user, args.remote_dir = "local", "0", "local", args.local
    else:
        resolve_connection(args)
    if args.julia is None:
        with open(os.path.join(REPO, "Manifest.toml")) as f:
            version = re.search(r'^julia_version = "(.*)"', f.read(), re.M).group(1)
        args.julia = "julia" if args.local else f"$HOME/.juliaup/bin/julia +{version}"
    args.gather_dest = os.path.abspath(args.gather_dest)
    dash = Dashboard(args)
    if not dash.remote.connect():
        sys.exit("Could not connect to the access point (on the university network or VPN? right password?)")
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
