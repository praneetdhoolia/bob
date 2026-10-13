# Using the continuous agent

Open **[http://127.0.0.1:3217/](http://127.0.0.1:3217/)**. The prototype now has one process, one numerical model and one interface. The LLM companion, separate car, research servers and old mode selector are removed.

## Start, inspect and stop

Run these commands in PowerShell from `C:\Users\Praneet Dhoolia\work\bob`:

```powershell
npm start
npm run status
npm stop
```

Start opens the interface and reuses a running agent. Loading the frozen sensory encoders can take time. The interface distinguishes loading, live, paused and failed states. If the browser closes, reopen the address; the model continues in the background.

Stop requests a full numerical checkpoint and normal shutdown. It verifies process and listener ownership before sending the request. Preview without stopping: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop-workspace.ps1 -WhatIf`.

The full state is saved automatically every few seconds in `artifacts/liquid-live-state.pt`: numerical coordinates, active time and output framing (about 0.1 MB). Model weights are written to `artifacts/liquid-live-state.weights.pt` only when they change. A failed autosave is reported in the interface and the agent keeps running. An unreadable saved state is moved aside as `liquid-live-state.corrupt-<time>.pt`, and the agent starts from its trained initial state. A restored agent starts paused and excludes shutdown downtime. Use the pause/resume control to continue. Pause retains state in RAM and stops this browser's capture/playback; shutdown saves and exits.

## Inputs and output

| Control | Action |
| --- | --- |
| **Camera** | Enable or stop the camera. Browser permission is required. |
| **Microphone** | Stream raw 16 kHz audio without speech recognition. |
| **Play video** | Read video frames and audio from a local recording. |
| **Play audio** | Feed a local audio file as raw samples. |
| **More → Test recording** | Run the existing tennis/cycling recording and initial spoken sample. |
| **More → Quiet input** | Stop the fixture/capture while numerical time continues. |
| **Chat** | Open or close text input and conversation history; closing preserves a draft. |
| **Speaker** | Read the custom translator's emitted text through browser speech synthesis. |
| Agent's hover microphone | Mute/unmute its speech immediately; also visible on touch and keyboard focus. |
| **Activity** | Show the agent's live visual detail, or hide it to leave the face centred in the same tile. |
| Desktop / sun / crescent | Cycle System, Light and Dark appearance. |
| Resize separators | Drag to resize tiles or chat. Use arrow keys after keyboard focus. |

Camera/video, audio and text arrive without a caption or transcript stage. One matrix retains their influence. The custom translator predicts bytes from that matrix and feeds emitted bytes back into it. It has no pretrained English decoder or conversational prompt. Its current training is narrow: silence or incorrect English is possible. Speaker synthesis is an output device, not the agent's reasoning engine.

The green rectangles approximate the current translator's sensitivity to image patches. They are not a certified gaze or evidence of awareness. Hardware camera/microphone behavior and generalization to ordinary human speech still need tests.

## Read the live display

The meeting interface uses transparent neutral grey glass over a rotating galaxy, icon controls and a collapsible chat. Bob's face animates during speech. Reply text remains in Chat. Turning Activity off hides detail without changing the tile layout. There is no JSON display, coordinate explorer or manual-save button. Refresh an older browser tab once to load interface changes.

Activity contains Memory and Flow graphs and red waveform bars, followed by a table of 46 sensors. Timing, input queues, memory, CPU, GPU allocations and output appear together. The active clock uses hh:mm:ss. Current, Minimum, Maximum and Average cover measured samples during this viewing session. Totals and fixed capacities use dashes for statistics. Select a sensor group heading to collapse its rows. Theme, group visibility and panel sizes persist locally. Scroll edges fade when more content remains.

Memory shows pooled strength from the 16,384-value matrix in RMS units. Flow shows recent changes in thousandths of an RMS unit. These are dimensionless neural measurements. Flow commits a three-observation average once per sample, then scrolls an eight-second history smoothly; earlier values retain their heights. Its scale expands only for a new high and does not repeatedly shrink or blend. The sensor table retains raw measurements. Audio displays five seconds of measured amplitude on a -60 to 0 dBFS scale. Bars stay at fixed horizontal positions, with half the previous density. Their heights gently rise and fall as the wave travels through them over approximately five seconds. Mirrored halves show amplitude magnitude. Display smoothing does not change the input stream.

Time and update rate use the runtime's physical clock. The visuals show processing, not certified understanding or thoughts. Canvas rendering uses the display's pixel density. Graphs and the state orbit interpolate between measurements at the display's animation-frame cadence, targeting 60 fps. The sensor table retains measured update rates. The galaxy uses procedural clouds, dust and orbital particles. It has a fixed view of a gently rotating horizontal disk, without a photograph. It is decorative and independent of the agent. Reduced-motion preferences stop galaxy motion and mouth animation.

The RAM meter reports used memory against actual physical capacity; its tooltip reports this agent's resident RAM. The configured 50 GB framework ceiling is kept out of the interface. If total system usage exceeds that ceiling, a running agent pauses, resuming is refused until memory is freed, and a starting agent waits before loading its encoders. VRAM and disk files are separate measurements. This guard cannot limit other applications' allocations.

The custom model currently has about 4.66 million parameters and a 64 KB transient matrix. Its sensory encoders use additional RAM/VRAM. It does not yet have a trained motor policy, broad English, validated autonomous deliberation or established subjective experience.

## Urge, stimulus and interpreter checks

The agent speaks only when its urge, a leaky accumulator of translator readiness and sensory surprise, crosses its threshold while the translator proposes a byte. Each onset releases a `stimulus` event: the numerical readout the translator then interprets. Activity shows urge, readiness, surprise and releases. See [the design](docs/live-core-design.md#urge-released-stimulus-and-interpreter).

Verify the interpreter and urge offline (CPU, shared research slot, no port, about two minutes):

```powershell
& .runtime/omni-python/Scripts/python.exe neural/evaluate_interpreter.py
```

It writes `artifacts/interpreter-verification-*.json` with per-report-type accuracy split by what is held out, plus null, identification, intervention and calibration controls, and urge onset tests. Validation video frames are identical to training frames, so video results cannot establish visual generalization.

## Evidence copies

`artifacts/` is not version-controlled. `scripts/snapshot_evidence.py` copies every receipt cited by the documentation into `docs/evidence/receipts/` with SHA-256 hashes in `docs/evidence/manifest.json`; `--check` verifies them. Run it after citing a new receipt.

## Ports and logs

Only **3217** is required. Ports **3210, 3213, 3214, 3215, 3218 and 11435** belong to retired project services and are stopped. The system Ollama service on 11434 is unrelated and is left alone. Browser test tooling may temporarily use 9223; it is not a model or extra interface.

Use `npm run status` to inspect the current listener and model status. Logs are `.runtime/agent.out.log` and `.runtime/agent.err.log`. The process descriptor is `.runtime/agent-server.json`; do not kill an unknown PID based on an old document.

## Research runtime and training

This checkout already has the local `.runtime/omni-python` environment and sensory weights. It uses PyTorch 2.7.1 with CUDA 12.8, Transformers 4.51.0 and the local MiniCPM-o sensory components. Its language trunk remains unmaterialized and is blocked from execution. Required local files, version checks and cold-install limitations are documented in [reproducibility](docs/reproducibility.md). Deleting sensory weight files arbitrarily breaks startup.

Retrain the custom core with:

```powershell
& .runtime/omni-python/Scripts/python.exe neural/train_liquid_core.py --updates 800
```

New training runs save their receipt and weights in a new `artifacts/liquid-core-*` directory. They do not change `artifacts/liquid-core-training.json`, which selects the current live checkpoint. A candidate must pass recorded-input, retention and timing checks before selection. Training data are existing raw sensory recordings; targets remain offline supervision. Familiar-case success is not general language or awareness. See [the model design and acceptance tests](docs/live-core-design.md).

Keep this guide current whenever the port, model, controls, checkpoint format or startup commands change.

Repository onboarding, handoff and project-report skills are available for ChatGPT/Codex and Claude Code. See [agent workflow commands and evidence sources](docs/agent-workflows.md).

Candidate language training uses the same core and raw audiovisual rehearsal:

```powershell
& .runtime/omni-python/Scripts/python.exe neural/train_liquid_language.py --updates 600 --pairs 160 --max-answer-bytes 160 --state-feedback
```

The pinned human-language corpus is already in `artifacts/english-corpus`. Candidate weights and `training.json` stay in a new `artifacts/liquid-language-*` directory. This command does not replace the running agent. `--resume <checkpoint>` continues a candidate; `--own-output-rate 0.2` increases training with its own byte predictions. These are experiments, and the current candidates still fail broad English evaluation.

Offline timing and language diagnostics use no additional UI port:

```powershell
& .runtime/omni-python/Scripts/python.exe neural/probe_liquid_time.py
& .runtime/omni-python/Scripts/python.exe neural/probe_liquid_language.py <candidate-checkpoint>
```

Timing readout calibration stays isolated from the live weights. The experiment supplies the event boundary; it does not establish an internal feeling of time. See [measured results and limitations](docs/live-core-results.md).

`probe_liquid_time.py --write-mode simultaneous` tests the experimental continuous input integration. `train_liquid_core.py --simultaneous --state-feedback --live-switches --resume <checkpoint>` trains that method on the existing recordings. The checkpoint stores its integration method. Legacy weights keep their ordered-write behavior during migration. Both commands keep the running agent's selected checkpoint unchanged.

Add `--continuous` to train full recorded interactions. One numerical state persists through both questions, its own output and intervening input. The current complete-interaction experiment uses 10 ms numerical ticks, 100 ms observations and a 30 ms byte readout. The running agent remains on the previously selected checkpoint while this experiment is assessed.

The spoken-input dataset is `artifacts/spoken-stream-corpus.json`. Use `--corpus artifacts/spoken-stream-corpus.json --continuous` for that experiment. Rebuild it with `neural/build_spoken_stream_corpus.py`; it writes synthetic test audio to files and encodes two existing recorded scenes. It does not play sound through your speakers or replace the live model.

`neural/test_live_inputs.py --checkpoint <candidate-checkpoint>` replays actual PCM and video through an isolated instance of the same architecture. Add `--spoken-query <wav-file>` to ask the question with audio instead of text. Its receipts are `artifacts/liquid-live-input-test-*.json`; no extra UI port is opened. The live agent and your camera selection are preserved.

`train_liquid_core.py --continuous --self-reports` runs the controlled own-code source-memory experiment. It injects a first feedback word and scores later generated reports of that word and the unchanged heard fact. The injected word itself is unscored. `--question-channel text --case-limit 8` limits this capacity experiment's data footprint. These flags affect experimental training and evaluation, not the live agent's conversation rules.

Large offline training, corpus encoding and validation share one research slot. A second workload waits before loading its data, which prevents overlapping fits and validation from exhausting RAM. The live agent continues separately. Core training reserves headroom for the selected files and possible visual-intervention copies before loading them. The total-system budget guard remains active throughout training.

New core-training checkpoints save optimizer and random states. Resuming the same corpus and configuration restores them. Older checkpoints and changed task configurations resume model weights with a fresh optimizer; the receipt reports `optimizerRestored`.

`neural/train_liquid_recurrence.py` tests internal processing after numerical inputs stop. It first learns one-hop memory, then freezes perception and translation while training internal dynamics for a four-hop decision. It saves an isolated `artifacts/liquid-recurrence-*` candidate and waits for the shared research slot. This capacity experiment does not replace Bob or establish consciousness.

`neural/train_liquid_copy.py --updates 1500` tests recall of unfamiliar content from its own feedback pathway. Random character sequences are injected as a controlled intervention, then recalled through the same neural state and byte translator while quiet native observations continue. Validation, delay and removed-feedback controls are written to `artifacts/liquid-copy-*/training.json`. This experiment does not replace the live checkpoint.

Add `--checkpoint <candidate>` to continue candidate weights; optimizer and random states are restored when the task configuration matches, otherwise the optimizer is fresh and the receipt says so. Add `--visual-inputs --rehearsal-rate 0.25` to train with changing recorded inputs and interleave ordinary audiovisual conversation on the same weights. `neural/probe_liquid_copy.py <candidate>` applies fixed-duration, terminated-output controls, including removed feedback, single-stream inference and visual distractions. `neural/probe_liquid_recurrence.py <candidate>` distinguishes nontrivial processing gains from damage to previously correct answers. Both diagnostics use the shared research slot and open no port.

Obsolete source files and services have been removed. About 47 GB of retired downloaded model folders remain in `.runtime` (`omni-gguf`, `omni-components`, `ollama-*`, `body-decoder`, `grounding-dino-tiny`, `whisper`, `llama-omni-engine`, `live-vision-model`). They are unused by this agent; delete them only after confirming `scripts/preflight.py` still passes. `.runtime/minicpm-omni`, `.runtime/omni-python` and `.runtime/python` are required.
