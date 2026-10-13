"""Run contract tests and existing checkpoint diagnostics without extra services."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
for command in [[sys.executable,'-m','unittest','discover','-s','tests','-v'],
                ['node','--test','tests/audio-worklet.test.cjs','tests/flow-trace.test.cjs','tests/media-controls.test.cjs'],
                [sys.executable,'scripts/snapshot_evidence.py','--check'],
                [sys.executable,'neural/verify_liquid_core.py']]:
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode:sys.exit(result.returncode)
