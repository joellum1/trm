"""Background checkpoint copies to HuggingFace and/or Google Drive (one worker
thread; failures are logged and skipped, the local file stays primary).

  HuggingFace  hf_repo="<user>/<repo>" (private, created if missing) -> <run_name>/<file>.
               Needs HF_TOKEN or `hf auth login`; retries each upload until a token exists.
  Drive        drive_remote="<rclone remote>:<folder>" -> <remote>/<project>/<run>/<file>.
               Needs rclone on PATH with a configured remote.
"""

import os
import queue
import shutil
import subprocess
import threading
from typing import List, Optional


class CheckpointSync:
    def __init__(self, hf_repo: Optional[str], drive_remote: Optional[str],
                 project_name: str, run_name: str):
        self.hf_repo = hf_repo
        self.drive_remote = drive_remote.rstrip("/") if drive_remote else None
        self.project_name = project_name
        self.run_name = run_name
        self._api = None
        self._hf_ready = False

        if self.hf_repo:
            from huggingface_hub import HfApi
            self._api = HfApi()
            self._ensure_hf_repo()
        if self.drive_remote:
            if shutil.which("rclone") is None:
                print("[sync] Drive disabled: rclone not on PATH")
                self.drive_remote = None
            else:
                print(f"[sync] Drive: {self.drive_remote}/{project_name}/{run_name}/")

        self._q: "queue.Queue[Optional[List[str]]]" = queue.Queue()
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return bool(self.hf_repo or self.drive_remote)

    def push(self, *paths: str):
        """Queue files for upload (non-blocking). Missing paths are skipped."""
        if self.enabled:
            self._q.put([p for p in paths if p and os.path.exists(p)])

    def close(self, timeout: float = 1800):
        """Wait for queued uploads to finish (bounded by timeout)."""
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            print("[sync] timed out waiting for uploads; local copies are intact")

    def _ensure_hf_repo(self) -> bool:
        """Create the repo once a token is available. Returns readiness."""
        if not self._hf_ready:
            try:
                self._api.create_repo(self.hf_repo, private=True, exist_ok=True)
                self._hf_ready = True
                print(f"[sync] HF: {self.hf_repo}/{self.run_name}/")
            except Exception as e:
                print(f"[sync] HF not ready, will retry on next checkpoint ({type(e).__name__}: {str(e)[:200]})")
        return self._hf_ready

    def _worker(self):
        while True:
            paths = self._q.get()
            if paths is None:
                return
            for path in paths:
                name = os.path.basename(path)
                if self.hf_repo and self._ensure_hf_repo():
                    try:
                        self._api.upload_file(path_or_fileobj=path, repo_id=self.hf_repo,
                                              path_in_repo=f"{self.run_name}/{name}",
                                              commit_message=f"{self.run_name}: {name}")
                    except Exception as e:
                        print(f"[sync] HF upload of {name} failed: {type(e).__name__}: {e}")
                if self.drive_remote:
                    dest = f"{self.drive_remote}/{self.project_name}/{self.run_name}/{name}"
                    r = subprocess.run(["rclone", "copyto", path, dest], capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"[sync] Drive upload of {name} failed: {r.stderr.strip()[-300:]}")
