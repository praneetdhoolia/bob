# One continuous model

The revised user requirement removes the LLM companion, separate motor application and frozen language decoder. One process owns inputs, a numerical state, learning, translation and the interface. Its only port is 3217.

## Model contract

All arriving audio, video and literal text become vectors before they touch persistent state. Existing frozen numerical sensory encoders can supply these vectors; neither speech recognition nor a generated caption is an intermediate representation. Their pretrained language trunk must never execute. The first implementation retains the measured MiniCPM sensory frontend, not its language generator.

The persistent state is a learned associative matrix of 16,384 floating-point coordinates. Coordinates have no hand-assigned meanings such as self, other, scene or emotion. Learned delta writes integrate evidence. A bounded, state-dependent continuous flow evolves the same matrix between observations. Physical elapsed time enters the numerical dynamics. A faster tick is not by itself a thought or a feeling of time.

The custom translator has no pretrained language model, prompt, chat template, vocabulary model or separate recurrent decoder cache. It reads the matrix and predicts one UTF-8 byte, a blank or an utterance boundary. Emitted bytes feed back into that same matrix through learned writes. Blank is a learned output; the runtime does not manufacture questions or conversational greetings. Training labels are supervision, not input descriptions. Camera activity can change the next byte while a sentence is incomplete.

The duration readout uses the same matrix. Offline calibration tests recoverable interval information. The unused prediction head was removed from the new model; its obsolete weights are discarded explicitly during checkpoint migration. Native packet forecasting is now an isolated learning experiment in the existing shared context pathway; live predictive learning and prediction of own actions remain unimplemented. Self-model evidence still requires predicted consequences of its own emissions/actions, capability uncertainty and memory limitations. Persistent recurrence alone does not establish those abilities. Motor control can later be another read from this matrix; it is excluded from this first focused interface.

## Continual-learning experiment

`predict_sensory(S)` uses the same `input_context(read(S))` pathway that already
modulates sensory writes, own-byte feedback and internal flow. The frozen native
packet target is `tanh(mean(input_projection(input_norm(tokens))))`. The experiment
predicts before assimilating a packet, then updates only the shared pathway from
that packet's numerical target. It consumes no answer, caption or transcript
labels. Parameters have a measured RMS drift bound relative to their starting
anchor. Optimizer moments and this parameter anchor are learning machinery;
they store no extra recurrent world model or semantic self/other slots.

The residual variant (`--residual`) forecasts `x_t + input_context(read(S))`. It starts exactly at persistence, so learning has to earn every improvement over the last packet. Learned predictors beat copy-last-frame in this way ([PredNet](https://ar5iv.labs.arxiv.org/html/1605.08104)). Adam's per-weight steps act coherently across the 2,048-wide normalized readout, so tiny packet changes need learning rates near 1e-6. Multi-horizon targets (≥ 500 ms) and seeds with bootstrap intervals are the next tests.

The first strong-coupling run disrupted emissions and failed a constant-mean
forecast baseline. A weaker coupling of 0.02 preserved 6/10 held answers and
147/147 quiet states while beating that baseline. Both candidates lose to
persistence. This is evidence of shared-path adaptation, not useful anticipatory
world understanding or a deployment-ready learner. The runtime retains the
previous trained weights. See the dated results and isolated receipts.

The next isolated variant also trains the existing `write_strength` parameters.
Its next-packet loss propagates through one prior sensory write, actual chosen
byte feedback and intervening flow. The differentiable state is detached each
packet before rebuilding the next transition with updated weights. This is
one-packet truncated temporal credit, not an additional cognitive history or
an unbiased full-trajectory gradient. Every trained tensor receives its own
RMS trust bound, so a small write gate cannot borrow the context matrix's
entire drift allowance. This changed bound requires a matched context-only
control; earlier global-bound results are not an isolated gate ablation.

[UORO](https://arxiv.org/html/1702.05043v3) retains low-rank estimates of temporal
parameter sensitivity without storing an input history. An isolated rank-one
learner now applies this update to the same core and native predictive loss.
It estimates longer temporal credit within each recording. Sensitivity storage
is 2,184,208 bytes for context and write-strength learning, versus an estimated
34,712,322,048 bytes for their full float32 Jacobian. These figures exclude
optimizer state, activations and raw encoders. Exact sign-averaging recovers a
small linear reference update; actual-core steps remain finite in a short test.
Predictive benefit and long-run stability remain under evaluation. Its noisy
estimator, changing parameters, clipping, drift bounds and discrete fixed-code
transition derivatives do not establish a convergence theorem or awareness.
Its update occurs after assimilation; the earlier one-packet learner updates
before assimilation. Use a matched short-credit rank-one control before
attributing differences solely to the longer credit horizon.
[Astrocyte-gated multi-timescale plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC12886396/)
studies slower plasticity gating in spiking networks. Its classification and
continual-learning results do not establish generic audiovisual language or
sentience, and Bob does not implement that spiking rule. A retention/adaptation
tradeoff must be measured in this architecture before adopting such a mechanism.

[V-JEPA 2.1](https://arxiv.org/abs/2603.14482) supports investigating predictive
representations, but this experiment does not implement its dense-feature
architecture or inherit its results. [Continuous Thought Machines](https://pub.sakana.ai/ctm/)
use internal ticks and neural synchronization; neither ticking alone nor our
failed recurrence candidate reproduces their mechanism. These comparisons
motivate measurable tests rather than a claim of architectural equivalence.

## Urge, released stimulus and interpreter

The goal asks for an agent that, when probed, gets an urge to output a stimulus, and for a custom English interpreter of that stimulus whose accuracy is verified. `neural/urge.py` implements these as measurable mechanisms. The runtime and the offline evaluation share it.

* **Urge.** A leaky accumulator integrated exactly in physical time: `dU/dt = -U/τ + readiness + g·surprise`, with τ = 0.1 s, g = 0.1 and onset threshold 0.05. Readiness is the translator's non-blank probability from the common state. Surprise is a nonnegative running z-score of the change between consecutive frozen native packet targets. Any input can raise the urge, whether a question, a voice, a scene change or text; there is no question detector.
* **Onset.** An utterance may begin only when U ≥ threshold and the translator proposes a byte. Once begun, it continues to its learned boundary, and the boundary spends the urge. Surprise shortens the time to speak but cannot produce speech without a proposed code. A boundary proposed while silent is not an onset.
* **Stimulus.** At onset the runtime releases a `stimulus` event: the 2,048-value readout `read(S)` (RMS, maximum, a 32-bin projection) with the interpreter's first reading of it, which is the top byte probabilities. The stimulus is the agent's mathematical output.
* **Interpreter.** The custom byte translator interprets the stimulus. Its English is the utterance that follows, and every emitted byte feeds back into the same state.

Research motivation, without any claim of equivalence:
* leaky stochastic accumulators reproduce the timing of self-initiated movement ([Schurger et al. 2012](https://digitalcommons.chapman.edu/psychology_articles/175));
* dopamine ramps shift the hazard of acting without triggering it ([Hamilos et al. 2021](https://elifesciences.org/articles/62583v1));
* streaming video responders accumulate an informativeness score to a threshold ([MMDuet](https://arxiv.org/html/2411.17991v2)).

The urge is a functional variable. Its crossing is not evidence of felt desire.

`neural/evaluate_interpreter.py` verifies both mechanisms on the recorded corpus.

**Interpreter accuracy** is reported by what is actually held out. The controls follow decoder-verification practice ([Tang et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304553); [Hewitt and Liang 2019](https://arxiv.org/pdf/1909.03368)):
* **Null:** another episode's held state with a different answer, keeping the original later observations.
* **Identification:** is the produced report most probable under its own state among all same-kind states?
* **Visual intervention:** only later native visual rows change.
* **Calibration:** expected calibration error of mean emitted-code probability.

"Accurate" is claimed only for a report type that passes every applicable criterion:
* held accuracy ≥ 0.9
* null hit rate ≤ 0.2
* identification percentile ≥ 0.85
* intervention flip rate ≥ 0.9
* ECE ≤ 0.1

**The urge** is tested on complete streams with runtime timing, under four conditions:
* gate disabled
* gate enabled
* surprise clamped to zero
* doubled threshold

Each condition reports strict exact utterances, false starts, onset latency and releases.

## Physical timing

The GPU encodes 100 ms waveform chunks and contemporaneous frames. One CPU state owner integrates wall time in approximately 10 ms ticks, and commits numerical evidence as it arrives. A shared lock prevents competing updates. Hardware channels necessarily have different sampling rates; they converge numerically without waiting for captions or transcripts. Backpressure and missed deadlines are visible.

The translator reads at live ticks, including while sensory encoding is busy. Its feedback uses the same state transaction. Model weights are copied atomically after an accepted training update; the live state is not reset merely to hide drift. Training and inference share one model definition.

Actual live-state checkpoints must contain the full numerical matrix, emitted-output framing, model identity and active elapsed time. Downtime is excluded on restoration. A visible state projection is not a complete checkpoint.

## Research decisions

[Gated DeltaNet](https://arxiv.org/html/2412.06464v1) supplies prior art for selective associative writes. It does not supply consciousness or our translator. The existing mathematically verified delta write can be reused.

[Closed-form continuous-time networks](https://doi.org/10.1038/s42256-022-00556-7) show efficient elapsed-time-dependent recurrent dynamics. The new bounded flow is an engineering choice informed by this work, not an exact implementation of its theorem or evidence of a simulated brain.

[Connectionist temporal classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) demonstrates learned blank outputs and alignment of continuous observations with discrete labels. The first translator uses explicit timed-byte supervision to test causal output changes; it must not be called a completed CTC implementation. CTC is a possible later objective when aligned supervision is insufficient.

[V-JEPA 2.1](https://arxiv.org/abs/2603.14482) offers predictive image/video representations. It is not a ready-made unified audio/text/English agent. Switching encoders without retraining would not solve memory or translation.

The supplied PDF explicitly distinguishes functional self-awareness from subjective experience and describes its recommendations as hypotheses. This revision replaces its suggested language-model codec as the user requested.

[Continuous Thought Machines](https://pub.sakana.ai/ctm/) offer relevant prior art for internal recurrent ticks, neuron-level temporal processing and synchronization-based representations. The authors test memory and navigation. Their Q&A experiment uses digit/operator sequences, rather than unrestricted English conversation. Our next temporal experiment must compare trained recurrent computation against the current slow bounded flow. It must measure retention, live-input intervention and additional-compute benefit. Merely adding oscillations would not establish deliberation.

The stable recorded-task fit supplies future observations every five byte steps, or 150 ms. The live sensory worker supplies observations near 100 ms. The new language experiments retain observations at 100 ms while emitting bytes. A matched cadence probe did not change the stable fit's 6/10 held-out score between 80 and 150 ms. Candidate checkpoints stay separate from the running agent until regressions and transfer are assessed.

Human-written [OpenAssistant conversations](https://huggingface.co/datasets/OpenAssistant/oasst1) can supervise the existing byte translator. This is offline data, not a runtime pretrained language generator. The pinned export contains 88,838 messages. A first reviewed, short English subset has 382 root-response pairs. Conversation trees must remain separate across training and evaluation. Human ratings do not prove factual accuracy or absence of bias. The first experiment measures basic language transfer and causal dependence on the literal input.

Version 2 adds a learned projection from the retained matrix readout into the incoming feature encoder. Incoming write keys, values and strengths can therefore depend on the current state. The matrix remains the only recurrent state. This addresses a structural limitation of the original input-only writer. The projection initializes to zero. Legacy checkpoint migration preserves observations, byte probabilities and emitted-code feedback bit for bit. The candidate has 5,105,679 parameters. This is state-conditioned processing, not evidence of consciousness.

Candidate training recomputes complete observation traces with current weights and limits gradient history to the last eight observations. The readout must preserve incoming language while background video/audio continues at 100 ms intervals. One experiment updates all sensory writes. A follow-up freezes the established sensory backbone and trains its new state-feedback projection. Checkpoints remain isolated while English quality and recorded-task retention are tested.

Version 3 shares that state-summary projection with own-code feedback and between-input dynamics through three learned coupling strengths. It adds three parameters and no recurrent state. The two new couplings initialize to zero, preserving version 1 and 2 behavior exactly. The old feedback/dynamics context used only column means. A controlled probe constructs matrices with identical column means and different retained information. The full readout distinguishes them while the old context cannot. This identifies a structural bottleneck and an information path to test. It does not prove that the new couplings learn useful reasoning.

The biological timing reference is [Umbach et al.'s human time-cell study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7668099/). Its neural activity relates to temporal organization of episodic memory. It does not justify a universal update-frequency threshold for AI consciousness. The planned functional timing test varies physical intervals independently of observation counts. A learned readout must estimate elapsed intervals from the common state, without access to the host clock or a separate semantic timer.

## Continuous input integration and evaluation

Version 4 adds a checkpointed choice of observation integration. Existing checkpoints retain ordered delta writes exactly. The experimental method writes all input rows through one physical-time field. For fixed learned keys, values and rates, its equation is `dS/dt = B - S A`, where `A = Kᵀ diag(rate) K` and `B = Vᵀ diag(rate) K`. An augmented matrix exponential integrates the interval without inverting a possibly singular matrix. Input content uses the established reference-interval embedding; actual elapsed time drives integration. The matrix remains the only cognitive state. The unsuccessful intermediate rate-adjusted ordered method was removed from executable source; its results remain documented.

This method has an exact fixed-coefficient subdivision property within numerical precision. It does not promise that state-conditioned coefficients remain unchanged across intervals. Training and testing must include changing scenes, output feedback, and retention. The offline timing test calibrates a linear head after a supplied cue; it does not teach the agent to recognize or reset at an arbitrary event boundary.

Complete-interaction training now runs the same core through the entire 12-second recorded audiovisual episode. Numerical dynamics advance every 10 ms, sensory observations arrive every 100 ms, and the byte readout runs every 30 ms. One state carries early speech, the first answer, subsequent video and the second question. Its generated bytes feed back into that state. Offline targets include silence and utterance boundaries; class balancing prevents silent intervals from overwhelming speech supervision. These targets do not create a runtime topic router or canned reply mechanism.

Matched visual interventions occur during an unfinished common sentence prefix. They replace only native visual query rows, while preserving acoustic features, literal text and physical intervals. The next utterance word must follow the new visual input. Candidate testing scores complete conversations and onset timing, instead of only questions taken from independent prepared states.

The current-model silence objective recomputes quiet traces when state feedback is trainable. Cached quiet matrices from older weights can otherwise teach silence at states that no longer occur. This is a training consistency correction, not a rule deciding when to initiate a conversation.

Recent primary research informs tests rather than supplying a replacement LLM. [Professor Forcing](https://arxiv.org/abs/1610.09038) studies differences between teacher-driven and sampled recurrent dynamics. Our own-output diagnostic and scheduled sampling are related concerns; an adversarial Professor Forcing objective is not implemented. [OmniInteract](https://arxiv.org/abs/2605.26485) evaluates online audiovisual triggers, timing, interruptions and nested continuity. Those criteria are more relevant than isolated caption correctness. Its benchmark is not yet run here.

[StreamOV](https://arxiv.org/abs/2605.25621) studies bounded audiovisual memory and state-driven response triggering. [OneStreamer](https://arxiv.org/abs/2610.01762) studies proactive evidence recording and state-transition supervision. Their language-memory techniques are not adopted because this prototype retains a numerical common state. Their temporal evaluation and supervision designs remain useful reference points. Neither provides evidence that this prototype is conscious.

[Idei et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13475615/) test high-dimensional visuo-proprioceptive prediction in a hierarchical variational RNN. The reported evaluation is open-loop simulation on recorded robot data, rather than real-time physical control. This supports investigating prediction-error learning and multiple time scales within a shared model. It does not establish audiovisual language, a single undifferentiated neural module, or subjective experience.

## Own-output source control

Versions 5 and 6 tested selective internal writes and dynamic associative reads. Their four-hop capacity experiments failed to show useful processing; those active paths are retired. Current version 7 applies a learned factorized operator `L S R + S A S + b` to the whole common state. A bounded target and learned physical rates update that same state between inputs. Rates, alignment and projections are learned; there are no assigned self/world fields, internal utterances or extra recurrent cache. Legacy migration keeps this processing disabled and preserves prior behavior exactly. Zero-time, round-trip and 100-second finite-state checks pass. The first trained result is weak and tick-sensitive; it remains isolated from the live model.

The capacity experiment first trains one-hop retrieval from numerical graph packets. It then freezes perception, state read queries and the byte translator. Only internal processing parameters learn a four-hop decision. Labels are computed outside the model during dataset construction. Inference receives the graph and a starting-node cue, then advances the numerical state without graph traversal, further sensory input or emitted-code feedback. Immediate answers, additional elapsed processing, disabled internal writes and different tick resolutions are compared on unseen graphs. This distinguishes learned processing benefit from displaying a moving state or merely delaying an already-computed answer. Success on this task would still be intelligence evidence, not consciousness.

[Neural Turing Machines](https://arxiv.org/abs/1410.5401) show that learned copy and associative recall are useful capacity tests for differentiable memory. Their external-memory architecture is not adopted here. The current own-output test instead writes generated character feedback into the existing common matrix and requests later recall while native quiet inputs continue. Novel sequences, removal of the earlier feedback, longer delays and single-stream inference test content retention beyond a two-word classifier. These controls assess an engineering prerequisite, not subjective identity.

The next functional self-monitoring experiment keeps every sensory row and physical interval identical within a pair. Only an earlier own-code feedback sequence changes: `Apple.` or `Mango.`. The alternatives have equal length. Later literal questions request the first emitted word and the externally heard fruit. Their input embeddings and the external-fact target remain identical. One matrix carries the intervention, subsequent dialogue and ongoing audiovisual input.

Injected codes are an experimental intervention, not a voluntary statement generated by the model. Their spans contribute neither predictive loss nor accuracy. The model must generate the later reports itself. A pair passes only if both own-output reports change correctly and both external-fact reports remain correct. Scoring before 13 October 2026 accepted any matching utterance in a 0.5-second window and ignored extra utterances, so a model that said both alternatives could pass. Strict scoring (`match_targets`, `strict-first-utterance-v2`) now requires exactly one finished utterance in the window with the expected text and counts generated utterances outside every window as false starts. Earlier pair results must be rescored before they are cited. There is no transcript cache or assigned self/other memory slot.

[A corollary discharge circuit in human speech](https://pubmed.ncbi.nlm.nih.gov/39625978/) provides biological motivation for distinguishing self-generated and external signals. The experiment here tests memory of the model's own feedback pathway. It does not reproduce the paper's neural circuit or establish a sense of agency. [Unconscious reinforcement learning of hidden brain states supported by confidence](https://www.nature.com/articles/s41467-020-17828-8) also cautions against treating performance monitoring as sufficient evidence of consciousness. Capability calibration, predicted action consequences and useful autonomous control remain separate requirements.

## Interface direction

Use one meeting stage with camera/recorded video, green focus overlays, microphone and speaker toggles, a text field and live numerical activity. The current palette uses neutral grey glass with dark ink in light mode and pale ink in dark mode. A single desktop/sun/crescent button changes appearance. A procedural horizontal galaxy rotates gently behind translucent panels. System typography supports readable controls; tabular numerals and fixed sensor columns prevent layout movement. Keep the media central and diagnostics compact. Detailed values expand in place. Respect reduced motion and keyboard focus.

## Acceptance tests

1. Confirm no language-model forward or prompt path executes.
2. Replay raw speech and video; verify both alter the same state under matched interventions.
3. Change only incoming video during an incomplete byte stream; compare the unfinished output with held video.
4. Remove acoustic encoder history and vary earlier speech; future observations must be identical for a strict retention test.
5. Test idle intervals, pauses, irregular sampling and long runs. Report actual time separately from learned estimates.
6. Verify numerical state continues updating while input encoding and output playback run.
7. Evaluate novel voices, scenes and wording; familiar-case accuracy must not be described as general awareness.
8. Test checkpoint restoration and preserve failure cases. Do not certify conscience or consciousness from successful engineering tests.
