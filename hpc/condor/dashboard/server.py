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
JOB_ATTRS = ("ClusterId,ProcId,JobStatus,BHC_stage,BHC_tag,BHC_L,BHC_t,BHC_mu,BHC_points,RemoteHost,LastRemoteHost,"
             "HoldReason,EnteredCurrentStatus,QDate,CompletionDate,ExitCode,RemoveReason,"
             "RemoteUserCpu,RemoteSysCpu,JobCurrentStartDate,RequestCpus,RequestMemory,ResidentSetSize,ResidentSetSize_RAW,"
             "RemoteWallClockTime,NumJobStarts")
STALL_AFTER = 30 * 60   # s: a running job that used less than STALL_CPU cores on average after this long is flagged
STALL_CPU = 0.05
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
    "N_SAMPLES": r"^\d+$", "N_EDGE": r"^\d+$", "KAPPA_TOL": NUM, "SBSTD_TOL": NUM, "A2_TOL": NUM,
    "POINTS": r"^[0-9.\- ,]*$", "DRYRUN": r"^[01]$", "RESUME": r"^[01]$", "CHUNK": r"^[1-9]\d*$",
}
MODEL_PARAMS = ["U", "R", "DELTA", "OMEGA", "NMAX", "DTAU", "MAXD"]
SCRIPT_PARAMS = {
    "submit": MODEL_PARAMS + ["MAXITER", "CONVTOL", "CROSS_WIDTH", "N_SAMPLES", "N_EDGE", "PPN", "OBS_THREADS", "MEM",
                              "DISK", "REQUIREMENTS", "NICE", "POINTS", "DRYRUN", "RESUME"],
    "recompute_obs": MODEL_PARAMS + ["N_SAMPLES", "N_EDGE", "PPN", "MEM", "REQUIREMENTS", "NICE", "POINTS", "CHUNK"],
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
        """argv plus the script to feed on stdin. The script goes over stdin, not as an argument, because a long one
        (e.g. the rm of the logs of many gathered points) exceeds the kernel's argument length limit. It is wrapped in
        { ...; } </dev/null so bash reads it completely before running it and the commands in it do not eat it."""
        script = f"{{ cd {shlex.quote(self.local or self.dir)} && {cmd}\n}} </dev/null\n"
        if self.local:
            return ["bash", "-s"], script
        return ["ssh", "-S", self.sock, "-o", "BatchMode=yes", "-p", self.port, self.target, "bash -s"], script

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
            argv, script = self._argv(cmd)
            p = subprocess.run(argv, input=script, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {timeout} s"
        if p.returncode == 255 and not self.local:
            self.lost = subprocess.run(["ssh", "-S", self.sock, "-O", "check", "-p", self.port, self.target],
                                       capture_output=True).returncode != 0
        return p.returncode, p.stdout, p.stderr

    def stream(self, cmd, emit):
        argv, script = self._argv(cmd)
        return stream(argv, emit, input=script)

    def path(self, rel=""):
        """rsync address of a path relative to the repo on the access point."""
        if self.local:
            return os.path.join(self.local, rel)
        return f"{self.target}:{self.dir}/{rel}"

    def rsync_ssh(self):
        return [] if self.local else ["-e", f"ssh -S {self.sock} -o BatchMode=yes -p {self.port}"]


def stream(argv, emit, input=None, **kw):
    p = subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else None, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    if input is not None:
        p.stdin.write(input)
        p.stdin.close()
    for line in p.stdout:
        emit(line.rstrip("\n"))
    return p.wait()


# ---- state of the points

QUEUE_STATE = {1: "idle", 2: "running", 4: "running", 5: "held", 6: "running", 7: "idle"}


def job_info(j, now=None):
    info = {"id": f"{j.get('ClusterId')}.{j.get('ProcId')}", "status": j.get("JobStatus"),
            "host": j.get("RemoteHost") or j.get("LastRemoteHost"), "since": j.get("EnteredCurrentStatus"),
            "hold_reason": j.get("HoldReason"), "exit_code": j.get("ExitCode"), "remove_reason": j.get("RemoveReason")}
    start = j.get("JobCurrentStartDate")
    if j.get("JobStatus") == 2 and start:
        # CPU use of the current run: average number of busy cores (condor updates RemoteUserCpu every few minutes)
        wall = max((now or time.time()) - start, 1)
        cpu = (j.get("RemoteUserCpu") or 0) + (j.get("RemoteSysCpu") or 0)
        info.update(start=start, wall=wall, cpus=j.get("RequestCpus"), cores_used=cpu / wall,
                    stalled=wall > STALL_AFTER and cpu / wall < STALL_CPU)
    if j.get("JobStatus") == 4:
        # finished job: total wall time over all its runs (RemoteWallClockTime), else that of the last run
        rt = j.get("RemoteWallClockTime")
        if not isinstance(rt, (int, float)) and j.get("CompletionDate") and start:
            rt = j["CompletionDate"] - start
        info.update(runtime=rt if isinstance(rt, (int, float)) else None, starts=j.get("NumJobStarts"))
    req = j.get("RequestMemory")
    if isinstance(req, (int, float)) and req > 0:
        # ResidentSetSize: peak of the job (KB, rounded up by condor), ResidentSetSize_RAW: current (KB). condor's
        # MemoryUsage is the expression (ResidentSetSize + 1023) / 1024, which condor_q -json does not evaluate
        peak = j.get("ResidentSetSize")
        info.update(mem_req=req, mem_peak=(peak + 1023) // 1024 if isinstance(peak, (int, float)) else None)
        if info["mem_peak"] is not None:
            info["mem_frac"] = info["mem_peak"] / req
        rss = j.get("ResidentSetSize_RAW")
        if j.get("JobStatus") == 2 and isinstance(rss, (int, float)):
            info["mem_now"] = rss / 1024
    return info


# ---- progress of running jobs, from their live stdout/stderr (condor_tail)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RE_MARK = re.compile(r"^== (\d\d:\d\d:\d\d) (?:setup: (.*)|julia (\S+).*)$")    # run_julia.sh's phase markers
RE_INFO = re.compile(r"^\[ Info: (?:(\d\d:\d\d:\d\d) )?(.*?)\s*$")               # ITE log lines (time stamp optional)
RE_ITER = re.compile(r"^Iteration (\d+), cap = (\d+)")
RE_DE = re.compile(r"^Convergence error: \(E, dE\) = \(([^,]+), ([^)]+)\)")
ITE_STEPS = ("Partition function", "Normalization filter", "Compress", "Convergence error", "Finalize")


def parse_progress(out, err):
    """Phase of a job from its stdout (run_julia.sh markers) and its ITE step from its stderr (log lines).
    Jobs started before the markers existed: "Julia x (juliaup)" without a later julia marker means Pkg.instantiate."""
    out, err = ANSI.sub("", out or ""), ANSI.sub("", err or "")
    res = {"phase": "starting", "phase_since": None, "script": None, "iteration": None, "cap": None, "step": None,
           "step_since": None, "E": None, "dE": None, "error": None}
    for line in out.splitlines():
        m = RE_MARK.match(line)
        if m:
            res["phase_since"] = m[1]
            if m[2] is not None:
                res["phase"], res["script"] = "setup: " + m[2], None
            else:
                res["script"] = m[3]
                res["phase"] = "observables" if "observables" in m[3] else "ITE"
        elif line.startswith("Julia ") and "(juliaup)" in line:
            res["phase"], res["phase_since"] = "setup: Pkg.instantiate", None
        elif res["phase"] == "starting" and "juliaup" in line.lower():
            res["phase"] = "setup: julia install"
    for line in err.splitlines():
        if line.startswith("ERROR:"):
            res["error"] = line
        m = RE_INFO.match(line)
        if not m:
            continue
        stamp, msg = m[1], m[2]
        it = RE_ITER.match(msg)
        if it:
            res.update(iteration=int(it[1]), cap=int(it[2]), step="Iteration start", step_since=stamp)
            continue
        name = next((x for x in ITE_STEPS if msg.startswith(x)), None)
        if name is None:
            continue
        res["step"], res["step_since"] = name, stamp
        de = RE_DE.match(msg)
        if de:
            try:
                res["E"], res["dE"] = float(de[1]), float(de[2])
            except ValueError:
                pass
    if res["phase"] != "ITE" and res["phase"] != "observables":
        res.update(iteration=None, cap=None, step=None, step_since=None)
    tail = [x for x in err.splitlines() if x.strip()][-12:]
    res["tail"] = "\n".join(tail)
    return res


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
        stage = j.get("BHC_stage")
        if stage not in active or j.get("JobStatus") == 3:
            continue
        for key in job_points(j):
            entry(key)
            old = active[stage].get(key)
            if old is None or (j.get("ClusterId", 0), j.get("ProcId", 0)) > (old.get("ClusterId", 0), old.get("ProcId", 0)):
                active[stage][key] = j
    last = {"ite": {}, "obs": {}}
    for j in history:
        stage = j.get("BHC_stage")
        if stage not in last:
            continue
        rank = (j.get("CompletionDate") or j.get("EnteredCurrentStatus") or 0, j.get("ClusterId", 0))
        for key in job_points(j):
            entry(key)
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
        rec = {"t": p["t"], "mu": p["mu"], "eps": f.get("eps"), "nsteps": f.get("nsteps"), "maxiter": f.get("maxiter"), "resumed_from": f.get("resumed_from"),
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
                if key in last["ite"] and last["ite"][key][1].get("JobStatus") == 4:
                    rec["job"] = job_info(last["ite"][key][1])   # the finished job: runtime, peak RAM
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
            if key in last["obs"] and last["obs"][key][1].get("JobStatus") == 4:
                rec["obs_job"] = job_info(last["obs"][key][1])
        elif key in last["obs"] and f.get("state"):
            rec["obs"], rec["obs_job"] = "failed", job_info(last["obs"][key][1])
        else:
            rec["obs"] = "none"
        out.append(rec)
    out.sort(key=lambda x: (float(x["t"]), float(x["mu"])))
    return out


def job_points(j):
    """The points (t, mu) of a job: BHC_points "t1_mu1,t2_mu2,..." (observables jobs of several points, see
    hpc/condor/obs_BHCorr.sh), else BHC_t and BHC_mu (ITE jobs, older observables jobs)."""
    pts = j.get("BHC_points")
    if isinstance(pts, str) and pts:
        return [tuple(x.split("_", 1)) for x in pts.split(",") if "_" in x]
    key = (j.get("BHC_t"), j.get("BHC_mu"))
    return [] if None in key else [key]


def point_constraint(t, mu):
    """ClassAd constraint matching the jobs of the point (t, mu), single-point or with BHC_points."""
    return f'((BHC_t =?= "{t}" && BHC_mu =?= "{mu}") || stringListMember("{t}_{mu}", BHC_points) =?= true)'


def parse_progress_dump(text):
    """Output of Dashboard.progress's remote loop: @@JOB <id> <stage> <t> <mu>, @@OUT <stdout>, @@ERR <stderr>."""
    jobs, cur, part = [], None, None
    for line in text.splitlines():
        if line.startswith("@@JOB ") or line == "@@END":
            if cur:
                jobs.append(cur)
            cur, part = None, None
            if line != "@@END":
                parts = line.split()
                cur = {"id": parts[1], "stage": parts[2] if len(parts) > 2 else None,
                       "t": parts[3] if len(parts) > 3 else None, "mu": parts[4] if len(parts) > 4 else None,
                       "out": [], "err": []}
        elif cur is not None and line in ("@@OUT", "@@ERR"):
            part = cur["out"] if line == "@@OUT" else cur["err"]
        elif part is not None:
            part.append(line)
    if cur:
        jobs.append(cur)
    res = []
    for j in jobs:
        out, err = "\n".join(j.pop("out")), "\n".join(j.pop("err"))
        res.append({**j, **parse_progress(out, err)})
    return res


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
    if maxiter > (p.get("maxiter") or 0):
        return "resume"                               # continues the ITE from the stored state
    if convtol < (p.get("convtol") or float("inf")):
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
               "stalled": sum(1 for p in points if p["state"] == "running" and (p["job"] or {}).get("stalled")),
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
        # A running job's files on the access point stay empty until it ends: show its live output instead
        pc = f'{self.condor_constraint(tag, L)} && BHC_stage =?= "{"obs" if stage == "obs" else "ite"}" && {point_constraint(t, mu)}'
        c = shlex.quote(pc + " && JobStatus == 2")
        cmd = (f"id=$(condor_q \"$USER\" -constraint {c} -format '%d.' ClusterId -format '%d\\n' ProcId 2>/dev/null | head -1); "
               f"if [ -n \"$id\" ]; then echo \"== job $id is running: live stdout (condor_tail)\"; "
               f"timeout 30 condor_tail -maxbytes 8000 $id 2>&1 | tail -n 60; echo; echo \"== live stderr\"; "
               f"timeout 30 condor_tail -stderr -maxbytes 12000 $id 2>&1 | tail -n 60; exit 0; fi; "
               # observables jobs of several points: the log of the newest finished job of this point (by job id)
               + (f"id=$(condor_history \"$USER\" -constraint {shlex.quote(pc)} -limit 1 -format '%d.' ClusterId "
                  f"-format '%d\\n' ProcId 2>/dev/null | head -1); "
                  f"f=; [ -n \"$id\" ] && f=$(ls {d}/{prefix}_$id.err 2>/dev/null); "
                  # its setup and the section of this point (from its \"== … point k of n: t = …, μ = …\" marker)
                  f"if [ -n \"$f\" ]; then o=\"${{f%.err}}.out\"; echo \"== $o (job $id, setup and this point)\"; "
                  f"awk -v m=\"t = {t}, μ = {mu}\" '/^== .* point [0-9]+ of [0-9]+: / {{ s = index($0, m) > 0; p = 1 }} !p || s' "
                  f"\"$o\" | tail -n 80; echo; echo \"== $f\"; tail -n 60 \"$f\"; exit 0; fi; " if stage == "obs" else "") +
               f"f=$(ls -t {d}/{prefix}_t{t}_mu{mu}_*.err 2>/dev/null | head -1); "
               f"if [ -z \"$f\" ]; then echo 'No log files for this point on the access point (gathered points have no logs left).'; exit 0; fi; "
               f"o=\"${{f%.err}}.out\"; echo \"== $o\"; tail -n 60 \"$o\" 2>/dev/null; echo; echo \"== $f\"; tail -n 60 \"$f\"")
        rc, out, err = self.remote.run(cmd, timeout=60)
        return out + err

    def progress(self, tag, L, t=None, mu=None):
        """Phase and ITE step of the running jobs of a sweep (or of one point), from their live output."""
        if not RE_TAG.match(tag) or (t is not None and not (RE_POINT.match(t) and RE_POINT.match(mu))):
            raise ValueError("bad point")
        L = int(L)
        c = self.condor_constraint(tag, L) + " && JobStatus == 2"
        if t is not None:
            c += " && " + point_constraint(t, mu)
        cmd = (f"condor_q \"$USER\" -constraint {shlex.quote(c)} -af ClusterId ProcId BHC_stage BHC_t BHC_mu 2>/dev/null | "
               "while read c p st t mu; do echo \"@@JOB $c.$p $st $t $mu\"; echo @@OUT; "
               "timeout 30 condor_tail -maxbytes 20000 $c.$p 2>&1 < /dev/null; echo; echo @@ERR; "
               "timeout 30 condor_tail -stderr -maxbytes 20000 $c.$p 2>&1 < /dev/null; echo; done; echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=300)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        jobs = parse_progress_dump(out)
        if t is not None:
            for j in jobs:   # observables jobs of several points have no BHC_t, BHC_mu
                if j.get("t") in (None, "undefined"):
                    j["t"], j["mu"] = t, mu
        return {"time": time.time(), "jobs": jobs}

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
            force = params.get("FORCE") == "1"
            todo, nrun, nrerun, nresume, nskip, nqueued = [], 0, 0, 0, 0, 0
            upload = []   # states of resumed points that are only on the laptop
            for key in grid:
                p = known.get(key, {"state": "none", "has_state": False})
                why = needs_run(p, maxiter, convtol, force)
                if why is None:
                    nqueued += p.get("state") in ("idle", "running", "held")
                    nskip += p.get("state") not in ("idle", "running", "held")
                    continue
                nrun += why == "run"
                nrerun += why == "rerun"
                nresume += why == "resume"
                if why == "resume" and not p.get("on_remote"):
                    upload.append(key)
                todo.append(key)
            emit(f"{len(todo)} point(s) to submit ({nrun} new, {nresume} resumed from their state up to MAXITER, "
                 f"{nrerun} rerun from scratch for the new CONVTOL), {nskip} done, {nqueued} in the queue")
            if not todo:
                return 0
            values["POINTS"] = ",".join(f"{t} {m}" for t, m in todo)
            values["RESUME"] = "0" if force else "1"
            if action == "preview":
                values["DRYRUN"] = "1"
                if upload:
                    emit(f"(submit would first upload {len(upload)} state(s) from the laptop to resume them)")
            else:
                if upload:
                    emit(f"Uploading {len(upload)} state(s) from the laptop to the access point, to resume them ...")
                    rel = [f"data/{MODEL}/{tag}/L{L}/t{t}/mu{m}.jld2" for t, m in upload]
                    rc = self._rsync(rel, self.args.gather_dest + "/", self.remote.path(), emit)
                    if rc != 0:
                        return rc
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
            values = self._check(params, MODEL_PARAMS + ["KAPPA_TOL", "SBSTD_TOL", "A2_TOL"])
            env = {**os.environ, **values, "DATA_DIR": "data"}
            cmd = [*self.local_julia, f"--project={os.path.join(REPO, 'examples')}", "--startup-file=no", os.path.join(REPO, PLOT_SCRIPT), str(L)]
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
               f"echo @@ACTIVE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint {c} -af BHC_t BHC_mu BHC_points 2>/dev/null; "
               f"echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=120)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        active = set()
        for line in s.get("ACTIVE", "").splitlines():
            parts = line.split()
            if len(parts) == 3:
                job = {k: v for k, v in zip(("BHC_t", "BHC_mu", "BHC_points"), parts) if v != "undefined"}
                active.update(job_points(job))
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
                if u.path == "/api/progress":
                    return self._send(200, dash.progress(q["tag"], q["L"], q.get("t"), q.get("mu")))
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
                "BHC_t": "0.3000", "BHC_mu": "0.4000"},
               {"ClusterId": 4, "ProcId": 0, "JobStatus": 4, "ExitCode": 0, "CompletionDate": 90, "BHC_stage": "ite",
                "BHC_t": "0.1000", "BHC_mu": "0.2000", "RemoteWallClockTime": 5400.0, "NumJobStarts": 2,
                "RequestMemory": 6144, "ResidentSetSize": 3145728}]
    recs = {(p["t"], p["mu"]): p for p in merge(remote, local, queue, history, "v2", 2)}
    got = {k: (p["state"], p["obs"]) for k, p in recs.items()}
    want = {("0.1000", "0.2000"): ("done", "done"), ("0.1000", "0.4000"): ("unconverged", "outdated"),
            ("0.2000", "0.2000"): ("running", "none"), ("0.3000", "0.2000"): ("held", "none"),
            ("0.3000", "0.4000"): ("failed", "none"), ("0.2000", "0.4000"): ("none", "none"),
            ("0.4000", "0.2000"): ("idle", "none"), ("0.5000", "0.2000"): ("done", "outdated")}
    assert got == want, f"\n got  {got}\n want {want}"
    assert recs[("0.1000", "0.2000")]["where"] == ["laptop", "access point"]
    j = recs[("0.1000", "0.2000")]["job"]   # finished job attached to a done point
    assert j["id"] == "4.0" and j["runtime"] == 5400.0 and j["starts"] == 2 and j["mem_peak"] == 3072, j
    assert recs[("0.5000", "0.2000")]["job"] is None   # no history job: no info
    assert recs[("0.1000", "0.4000")]["where"] == ["laptop"] and not recs[("0.1000", "0.4000")]["on_remote"]
    # submit decisions (MAXITER 2000 / 4000, CONVTOL 1e-6)
    assert needs_run(recs[("0.2000", "0.4000")], 2000, 1e-6) == "run"
    assert needs_run(recs[("0.5000", "0.2000")], 2000, 1e-6) is None              # converged
    assert needs_run(recs[("0.1000", "0.4000")], 2000, 1e-6) is None              # unconverged, same settings
    assert needs_run(recs[("0.1000", "0.4000")], 4000, 1e-6) == "resume"          # more steps: continue the state
    assert needs_run(recs[("0.1000", "0.4000")], 2000, 1e-8) == "rerun"           # only a smaller CONVTOL
    assert needs_run(recs[("0.2000", "0.2000")], 4000, 1e-6) is None              # in the queue
    assert needs_run(recs[("0.5000", "0.2000")], 2000, 1e-6, force=True) == "run"
    assert sections("@@A\n1\n2\n@@B\n@@END\n") == {"A": "1\n2", "B": "", "END": ""}
    assert parse_json_section("warning\n[1,2]\n", None) == [1, 2]
    # observables jobs of several points (BHC_points): a running chunk applies to each of its points, a finished one
    # is attached to them; an old single-point obs job is still matched
    assert job_points({"BHC_points": "0.1000_0.2000,0.2000_0.2000"}) == [("0.1000", "0.2000"), ("0.2000", "0.2000")]
    assert job_points({"BHC_t": "0.1000", "BHC_mu": "0.2000"}) == [("0.1000", "0.2000")] and job_points({}) == []
    q2 = queue + [{"ClusterId": 11, "ProcId": 0, "JobStatus": 2, "BHC_stage": "obs",
                   "BHC_points": "0.2000_0.2000,0.1000_0.4000"}]
    h2 = history + [{"ClusterId": 10, "ProcId": 1, "JobStatus": 4, "ExitCode": 1, "CompletionDate": 95, "BHC_stage": "obs",
                     "BHC_points": "0.1000_0.2000,0.3000_0.4000", "RemoteWallClockTime": 900.0},
                    {"ClusterId": 3, "ProcId": 0, "JobStatus": 4, "ExitCode": 0, "CompletionDate": 80, "BHC_stage": "obs",
                     "BHC_t": "0.5000", "BHC_mu": "0.2000"}]
    r2 = {(p["t"], p["mu"]): p for p in merge(remote, local, q2, h2, "v2", 2)}
    assert r2[("0.2000", "0.2000")]["obs"] == "running" and r2[("0.1000", "0.4000")]["obs"] == "running"
    assert r2[("0.2000", "0.2000")]["obs_job"]["id"] == "11.0"
    assert r2[("0.1000", "0.2000")]["obs"] == "done" and r2[("0.1000", "0.2000")]["obs_job"]["id"] == "10.1"   # its point ok, another failed
    assert r2[("0.3000", "0.4000")]["obs"] == "none"                     # no state: not "failed"
    assert r2[("0.5000", "0.2000")]["obs"] == "outdated" and r2[("0.5000", "0.2000")]["obs_job"]["id"] == "3.0"
    assert point_constraint("0.1", "0.2") == '((BHC_t =?= "0.1" && BHC_mu =?= "0.2") || stringListMember("0.1_0.2", BHC_points) =?= true)'
    # progress: a job of the old run_julia.sh (no markers) hanging in Pkg.instantiate
    old_out = "Host: x\nWelcome to Julia!\nNow installing Juliaup\nJulia was successfully installed\nJulia 1.12.6 (juliaup), HOME = /tmp/k"
    pr = parse_progress(old_out, "info: downloading installer\n   Configure Default Julia version set to '1.12.6'.")
    assert pr["phase"] == "setup: Pkg.instantiate" and pr["phase_since"] is None and pr["iteration"] is None, pr
    # old format in the ITE (no time stamps), colored log lines
    pr = parse_progress(old_out + "\n== 09:32:05 julia hpc/BoseHubbardCorrelated_NormalizedITE.jl 10 0.0600 1.3000\n",
                        "[ Info: Iteration 3, cap = 16\n[ Info: \x1b[36mPartition function\x1b[39m\n"
                        "[ Info: \x1b[35mCompress\x1b[39m\n")
    assert (pr["phase"], pr["phase_since"], pr["iteration"], pr["cap"], pr["step"], pr["step_since"]) == \
        ("ITE", "09:32:05", 3, 16, "Compress", None), pr
    # new format: markers and time stamps
    new_out = ("== 10:00:00 setup: waiting for the setup lock\n== 10:00:00 setup: julia install\nJulia 1.12.6 (juliaup)\n"
               "== 10:00:05 setup: Pkg.instantiate (attempt 1 of 3)\n")
    pr = parse_progress(new_out, "  Installed X\n")
    assert (pr["phase"], pr["phase_since"]) == ("setup: Pkg.instantiate (attempt 1 of 3)", "10:00:05"), pr
    ite_err = ("[ Info: 10:09:00 Iteration 1, cap = 4\n[ Info: 10:09:01 Partition function\n"
               "[ Info: 10:09:40 Convergence error: (E, dE) = (-0.28, 0.25)\n[ Info: 10:09:40 Finalize\n"
               "[ Info: 10:09:41 Iteration 2, cap = 4\n[ Info: 10:09:41 Normalization filter\n")
    pr = parse_progress(new_out + "== 10:09:00 julia hpc/BoseHubbardCorrelated_NormalizedITE.jl 6 0.1 0.5\n", ite_err)
    assert (pr["phase"], pr["iteration"], pr["step"], pr["step_since"], pr["dE"]) == \
        ("ITE", 2, "Normalization filter", "10:09:41", 0.25), pr
    pr = parse_progress(new_out + "== 10:09:00 julia hpc/BoseHubbardCorrelated_NormalizedITE.jl 6 0.1 0.5\n"
                        "== 11:00:00 julia hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl 6 0.1 0.5\n", ite_err)
    assert (pr["phase"], pr["phase_since"], pr["iteration"]) == ("observables", "11:00:00", 2), pr
    dump = ("@@JOB 7.1 ite 0.1000 0.5000\n@@OUT\n" + old_out + "\n\n@@ERR\n\n"
            "@@JOB 7.2 ite 0.2000 0.5000\n@@OUT\n" + new_out + "\n@@ERR\n" + ite_err + "\n@@END\n")
    jobs = parse_progress_dump(dump)
    assert [(j["id"], j["t"], j["phase"]) for j in jobs] == [("7.1", "0.1000", "setup: Pkg.instantiate"),
                                                              ("7.2", "0.2000", "setup: Pkg.instantiate (attempt 1 of 3)")]
    # stall flag: 27 s CPU in 19 h
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 2, "JobCurrentStartDate": 1000, "RemoteUserCpu": 20,
                   "RemoteSysCpu": 7, "RequestCpus": 8}, now=1000 + 19 * 3600)
    assert ji["stalled"] and ji["cpus"] == 8
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 2, "JobCurrentStartDate": 1000, "RemoteUserCpu": 3600 * 7,
                   "RequestCpus": 8}, now=1000 + 3600)
    assert not ji["stalled"] and abs(ji["cores_used"] - 7) < 1e-9
    # memory: an out-of-memory hold (peak above the request) and a running job
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 5, "RequestMemory": 6144, "ResidentSetSize": 7500000,
                   "ResidentSetSize_RAW": 6227456})
    assert ji["mem_frac"] > 1 and "mem_now" not in ji
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 2, "JobCurrentStartDate": 1000, "RequestMemory": 8192,
                   "ResidentSetSize": 2500000, "ResidentSetSize_RAW": 2171020}, now=2000)
    assert ji["mem_peak"] == 2442 and abs(ji["mem_frac"] - 2442 / 8192) < 1e-12 and abs(ji["mem_now"] - 2171020 / 1024) < 1e-9
    assert "mem_req" not in job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 1})
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
