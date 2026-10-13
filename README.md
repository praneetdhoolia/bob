# Bob

A single continuous numerical agent with raw audiovisual/text input, persistent associative state and a custom byte translator. One local liquid-glass interface shows its inputs, output and live coordinates.

Open [http://127.0.0.1:3217/](http://127.0.0.1:3217/). Run `npm start`, `npm run status` and `npm stop`. The maintained [usage guide](USAGE.md) explains capture, recordings, speakers, training, checkpoints and current limits.

The agent does not use an LLM conversation path or pretrained English decoder. Frozen numerical sensory encoders supply input vectors. A custom model evolves a 64 KB matrix every 10 ms, assimilates audio, video and text every 100 ms, and reads it every 30 ms.

When its urge, a leaky accumulator of translator readiness and sensory surprise, crosses a threshold, it releases a numerical stimulus. Its custom byte interpreter turns that stimulus into English. The source architecture has 4,663,062 parameters. The selected live weights are a migrated version-1 checkpoint, in which 610,567 of those parameters are inert.

Its English is narrow:
- Trained fruit words: 8/8.
- Unseen fruit words: 0/4.
- Video reports: correct only on frames identical to training frames.

Consciousness, general self-awareness and broad language are unestablished.

The [full goal and evidence contract](docs/GOAL.md) preserves the requested outcome.
[Dependency checks and reproduction limits](docs/reproducibility.md) explain the
local runtime, asset requirements and experimental continual-learning command.

See [the architecture, research decisions and acceptance tests](docs/live-core-design.md). The original [research PDF](sentience-research-64gb.pdf) remains the starting reference. The older companion, separate motor UI and experimental services are retired.

[Measured results and remaining failures](docs/live-core-results.md) record the current translator, causal controls, raw input tests, exact checkpoint recovery and the 50 GB RAM budget.

Receipts cited by the documentation are copied into [docs/evidence](docs/evidence/manifest.json) because `artifacts/` is not version-controlled. Dated assessments are in [docs/reports](docs/reports/README.md).
