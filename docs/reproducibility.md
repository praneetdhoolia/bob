# Dependencies and reproduction

This is an experimental Windows checkout, not a verified fresh-install release.
Python 3.12.15 runs from `.runtime/omni-python/Scripts/python.exe`. That environment
also uses the shared Torch installation in `.runtime/python`; retain both folders.
Node is required for npm commands and audio-worklet tests.

Observed package versions are in `config/runtime-requirements.txt`. They describe
the tested local environment. Installing that list on another machine has not been
verified. The CUDA Torch wheel requires the PyTorch CUDA 12.8 wheel index.

Check the installation without loading neural weights or private live state:

```powershell
& .runtime/omni-python/Scripts/python.exe scripts/preflight.py --manifest artifacts/runtime-preflight.json
npm test
```

Add `--hash-weights` to identify the large local weight files by SHA-256; the
default hashes configs, tokenizer assets and model Python code and records sizes
for large files. A successful preflight proves presence/version agreement, not
CUDA compatibility, provenance, model quality or a cold installation.

The frozen sensory loader needs local MiniCPM-o 4.5 files in
`.runtime/minicpm-omni`, including native configs/processors, tokenizer, trusted
model code, the safetensors index and shards containing vision, audio, projection
and text-embedding tensors. The [upstream model card](https://huggingface.co/openbmb/MiniCPM-o-4_5)
lists Apache 2.0. A subsequent upstream identity audit matches all 15 mandatory
local files to revision `503e754207c94da6bb26850b4469f367c9ea3582`, including
both required weight shards. `config/sensory-assets.json` preserves their sizes
and SHA-256 digests. The audit compares LFS content hashes or Git blob hashes,
including the Git blob header, rather than confusing the two formats. The
default preflight checks small files; `--hash-weights` also checks shards.
Its pretrained language trunk
is unmaterialized and forbidden from execution. Upstream full-model benchmark
scores must not be attributed to this custom core.

`artifacts/liquid-core-training.json` selects the existing trained custom
checkpoint. No trained weights are included in Git. Standard core training also
requires its historical sensory seed and the native cached corpus. These are
local experimental assets, not a published release. Current recordings cover
two video scenes and synthesized speech. Unknown original-media licensing and
missing published asset bundles remain reproducibility gaps; do not redistribute
the recordings as licensed training data without resolving their provenance.

Camera/local-media operation does not require the built-in demonstration images
or speech file. Missing images disable Test recording with a clear error; missing
or unreadable demo audio is omitted. Mandatory sensory weights and the custom
checkpoint still must be present.

Self-supervised experiments run separately with the shared research lock:

```powershell
& .runtime/omni-python/Scripts/python.exe neural/probe_continual_context.py --checkpoint artifacts/liquid-core-1791819735481555400/latest.pt --passes 4 --coupling .02 --device cpu
```

Add `--residual --learning-rate 1e-6` for the persistence-anchored forecast that first beat persistence on held recordings. This learns only the existing shared context pathway from native packet targets.
It writes an isolated candidate and prequential receipt. Answer labels are used
only in the separate retention evaluation. The live framework does not yet run
this learner. A useful forecast must beat persistence and retain prior behaviors
before deployment can be considered.
