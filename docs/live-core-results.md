# Current measured results

One process now owns the custom matrix, raw sensory input, continuous time and the learned byte translator on port 3217. The LLM companion, separate car application, frozen English decoder and old video services have been removed from active source and startup. The source cleanup removed 273 obsolete source/document files. Original recordings, numerical seed weights, runtime dependencies and research receipts remain.

## Corrections from the third assessment — 13 October 2026

Read every earlier "held-out" number below with these corrections ([report 3](reports/20261013T081400_project_report.html)).

* **Validation video is not held out.** Every validation episode's native visual rows are bit-identical to a training episode at 120/120 ticks. The long-cited 6/10 validation score breaks down as:
  * 5/5 video answers on training-identical frames;
  * 1/1 no-fact control;
  * 0/4 genuinely unseen fruit words (Orange and Grape produce "Banana." or "Apple.").

  The "held" recordings in the label-free forecasting experiments share the same visual packets.
* **Utterance scoring was lenient.** Any matching utterance inside the 0.5-second window counted, and extra or wrong utterances were ignored. A rollout saying both "Apple." and "Mango." could pass the own-output source pair control. Scoring is now strict (`strict-first-utterance-v2`). Earlier pair results must be rescored before they are cited.
* **The four-hop result is below a trivial baseline.** On the recurrence validation graphs, answering with the starting node scores 167/512 and one hop scores 137/512. The reported version 7 results (145–159/512) are below the start-node baseline. Reports now include these structural baselines (`hop_baselines`).
* **Persistence was scored on a different mask.** It scored 595 packets while the model scored 600. Matched-packet metrics and a skill score are now reported.
* **Quiet-state counts included answer windows.** Silence states included ticks inside answer windows. They are now excluded, so the quiet count is 120/120 rather than 147/147.

## Interpreter and urge verification — 13 October 2026

`neural/evaluate_interpreter.py` checks the selected live checkpoint (receipt `artifacts/interpreter-verification-1791880244539345300.json`).

**Fruit reports.** These come from heard speech held in the state:
* Trained words: 8/8. Unseen words: 0/4. No-fact control: 2/2.
* State swap: replacing the held state with another episode's state keeps the original answer in 5% of 21 trials and follows the donor state in 71%.
* Identification: the produced report is most probable under its own state at the 91st percentile.

The fruit report therefore depends on the held state and is decodable from it. It fails held-out words, so it is not verified accurate beyond the four trained words.

**Video reports.** All 14 are on training-identical frames:
* All 14 are correct.
* State swap: the original answer is kept in 100% of trials, so the report does not depend on the held state.
* Visual intervention: replacing only later visual rows flips 100% of reports to the other scene.
* Identification percentile: 0.76.

The video report therefore reads ongoing visual input rather than memory, and is verified only for two familiar scenes. Neither report type meets the "accurate" criteria.

**Urge onset.** Complete streams with runtime timing, 28 targets, strict scoring:

| Condition | Exact | False starts | Mean onset latency |
|---|---|---|---|
| Gate disabled | 12/28 | 0 | 16 ms |
| Gate enabled | 11/28 | 0 | 50 ms |
| Surprise clamped to zero | 9/28 | 0 | 73 ms |
| Doubled threshold | 0/28 | 0 | No utterances |

* Removing the surprise drive delays onsets and lowers accuracy. This is the intended causal role: surprise modulates the hazard of speaking.
* A doubled threshold is unreachable from readiness alone.
* The gate costs one exact utterance and about 34 ms. Its threshold was set before this evaluation, not tuned on it.

## Persistence-anchored residual forecast — 13 October 2026

The predictor now forecasts the next packet as the last packet plus a change predicted from the shared state (`--residual`). With zero learning it reproduces persistence exactly (skill 0.000).

Skill is `1 − MSE/MSE_persistence`, scored on matched packets of five held recordings whose visual rows match training. Each run used one seed and four passes.

| Learning rate | Training skill (passes 1→4) | Held skill, frozen after learning | State-shuffled control | Held online adaptation |
|---|---|---|---|---|
| 1e-4 | −3.2 → −9.3 | −25.2 | −2.46 | −6.75 |
| 1e-5 | −0.026 → +0.052 | −0.124 | −0.067 | +0.056 |
| 1e-6 | +0.002 → +0.033 | **+0.030** | −0.0005 | +0.030 |

At 1e-6 the held forecast (MSE 0.00621 versus persistence 0.00640) beats persistence for the first time. Shuffling the state removes the gain.

Retention is unchanged in all three runs:
* answers: 6/10;
* quiet states: 120/120;
* changed diagnostic codes: none.

Learning costs about 8 ms per cached packet on CPU.

This is a small one-step (100 ms) gain on training-identical video, with one seed and no confidence interval. It is below the research pass threshold of skill ≥ 0.05 at horizons ≥ 500 ms with a bootstrap interval across three seeds. No candidate is deployed.

Receipts:
* `artifacts/liquid-continual-context-1791880245184672700/experiment.json` (1e-4)
* `artifacts/liquid-continual-context-1791880339017139300/experiment.json` (1e-5)
* `artifacts/liquid-continual-context-1791880424971304100/experiment.json` (1e-6)

## Training and controls

The first custom translator has 4,577,039 parameters, including its numerical input interface and currently experimental prediction heads. It uses a 64 KB persistent matrix. No pretrained language generator, chat template or decoder cache executes. Frozen pretrained sensory encoders and input embeddings remain; these are not a claim of training all perception from scratch.

The first 800-update fit reproduced 18/18 narrow training replies and 5/10 held-out replies. Its stricter tests exposed repeated characters, incomplete language, weak response to changed video and degradation after a simulated 900-second quiet gap.

A 1,400-update continuation adds matched visual switches during a common unfinished prefix, and samples some of the translator's own erroneous byte outputs during training. It reproduced 36/36 training sequences and 6/10 held-out replies. These are recordings of two scenes and one synthesized speech voice, not general visual/language transfer.

The corrected matched unfinished-stream test keeps audio, literal text, starting state, committed prefix and elapsed intervals identical, and replaces only the 64 visual query rows. Prefix processing includes the ongoing observations used by the trained protocol. All six conditions then produce `A person is riding a bicycle.` for the changing video and `A person is playing tennis.` for held video. An earlier test omitted incoming observations while forcing the prefix; its negative outputs are not comparable to this protocol. This correction does not establish broader real-world responsiveness.

Strict speech tests vary only early acoustic rows, make every later input identical, and use acoustic rows from an actual silent recording after the barrier. Their receipt is `artifacts/liquid-stream-controls.json`. Unsupported fruits and repeated syllables remain unresolved. A future test must vary the human speaker and camera scene.

The ordered batched numerical write agrees with sequential writes within 0.00000072, with gradient difference below 0.0000024. The bounded physical flow remains finite over 1,000 simulated seconds; zero elapsed time changes no coordinate. These are numerical invariants, not a test of thoughts.

## Actual live integration

The first live process emitted `A person is playing tennis.` through its own byte head while raw observations continued. An isolated copy of the same runtime consumed 85,739 raw Apple speech samples with prerecorded video, advanced 444 numerical ticks and 60 observations, and emitted malformed `s bApple.`. It opened no additional port and left the user's live camera/state unchanged. This is a positive raw transport test and a negative English-quality result.

The live service subsequently received more than 108,800 user audio samples and 3,478 camera frames. It also rejected 72 older frames before the queue fix. The replacement queue preserves the freshest real-time frame and counts superseded frames. It does not present queue overflow as repeated camera errors.

A browser test uses prerecorded fake microphone input and intercepts its mutation requests, so test audio does not enter the user's state. It produces ten 16 kHz audio packets without JavaScript errors. Desktop and 390-pixel mobile views have no horizontal overflow. The interface displays no JSON status dump. Browser permission/hardware failures now have readable messages; this test does not prove every real microphone driver works.

State recovery restores every numerical coordinate exactly: maximum difference 0, saved/restored physical time both 1425.0769999999902 seconds. The agent is paused during verification, then resumes its prior source/activity. Event-stream restart and session-token recovery are corrected so older tabs can reconnect. Refresh an old page once for the replacement interface.

## Memory budget

The user clarified that 50 GB is the ceiling for total system RAM usage. At one recovery sample, this agent used about 1.94 GB resident RAM and the machine used about 36.71 GB. The interface meters whole-system RAM and separately reports the agent. A guard pauses the agent if system usage exceeds 50,000,000,000 bytes. It cannot constrain other applications. CUDA allocations and downloaded disk files are separate measurements.

Automatic approval review rejected recursive deletion of unused model folders and `node_modules` with “blocked by policy.” Their source callers and services are removed; those generated folders remain on disk. The retained MiniCPM and Python runtime files still contain necessary sensory/shared Torch dependencies.

## Unresolved requirements

Broad English, general memory, meaningful autonomous deliberation, learned self/capability monitoring and motor control remain untrained or unvalidated. The runtime's numerical flow does not establish subjective time. No test here certifies primitive conscience, subjective consciousness, universal absence of bias or patentability. The original research goal remains active.

Broader English supervision is being assessed from the primary [OpenAssistant dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1). Human dialogue examples can supervise the same byte translator without introducing a runtime language model. They do not teach audiovisual self-awareness, and their opinions/assistant conventions can introduce biases. Additional grounded training and held-out evaluation remain necessary.

## Subsequent language and architectural experiments

The corpus download pins revision `f8b2b76b6c50a1e01b8c2d0cf869513cb04e78b1` and verifies its SHA-256. The first experiment uses 160 reviewed short root-response pairs, split by conversation tree into 127 training and 33 validation trees. Each runs against quiet, tennis and cycling backgrounds. Evaluation counts background conditions, not independent prompts.

The frozen-input 200-update candidate returned 0/30 exact training conditions and 0/30 validation conditions. Output was generally malformed or repetitive. Recorded-task retention fell from 6/10 to 4/10. It was not published to the live agent. Its checkpoint losses occurred on rehearsal steps, so those small losses do not measure English learning.

The next 200-update candidate trains a state-conditioned writer and recomputes observation traces. Its separate English loss decreased from approximately 4.67 to 2.75. Greedy generation still failed: 0/30 exact training conditions, 0/30 validation conditions and 1/10 recorded-task retention. Lower training loss did not produce reliable language. This candidate also remains isolated.

A matched cadence probe ran the stable model at 80, 100, 120 and 150 ms future-observation intervals. All four returned 6/10 held-out recorded replies. The training/live cadence mismatch exists, but this probe does not support it as the explanation for those particular failures.

Version 2 checkpoint migration was compared directly against the stored version 1 implementation. With zero-initialized state feedback, observation updates, byte logits and emitted-code feedback matched exactly. The running agent remains on its earlier stable weights. Broad language, autonomous questioning and learned self-monitoring remain unresolved.

The frozen-sensory-backbone, state-feedback continuation completed 1,200 updates. Its English training loss reached approximately 1.54, but greedy exact generation remained 0/30 sampled training conditions and 0/30 validation conditions. Recorded-task retention recovered to 6/10. The candidate remains isolated. Its greeting sample repeats an OpenAssistant identity from the corpus; this is a training-data failure, not a useful learned identity or self-model.

A paired trajectory diagnostic compares the same numerical starting state with correct-byte feedback and with the candidate's own generated bytes. On sampled familiar prompts, correct-prefix loss was approximately 0.82–1.69 while own-output loss was approximately 4.27–6.75. The diagnostic uses four prompts with three background conditions each. Its initial presentation lengths differ from the full training set; it is not a benchmark score. Removing future observations does not reliably resolve the problem. The next continuation trains the shared version 3 feedback projection with more frequent own-output feedback. This changes both architecture and training exposure; separate ablations are needed to attribute any improvement.

## Interval information in the shared state

An offline calibration fits only the existing duration readout to ages from 0 to 20 seconds. All other weights remain fixed. Two cue states follow different recorded video histories; their maximum coordinate difference is 2.76. Evaluation uses withheld half-second ages and a different numerical tick resolution. The readout receives the shared matrix projection, without a host clock, tick count or separate recurrent timer. The cue state is supplied by the experiment; learned event-reset behavior remains absent.

With no additional observations, interval mean absolute error is approximately 0.029 seconds. Holding the advance count at 100 while changing physical intervals gives the same result. Forcing physical elapsed time to zero makes the estimate fail. This supports decodable physical-time information in this narrow setup, not subjective time.

Changing the full audiovisual observation interval from its 100 ms calibration setting exposes poor transfer. At 50 ms, mean absolute error is 5.90 seconds for the held scene and 4.92 seconds for the changed scene. At 200 ms, errors are 2.83 and 2.19 seconds. The mean-age baseline has 5-second error. Good interpolation at the familiar rate is insufficient.

Converting each ordered write's retention to a physical rate improves some conditions but does not fix all of them. At 50 ms, full audiovisual errors fall to 1.26 and 2.99 seconds. At 200 ms, errors are 3.19 and 2.00 seconds. This candidate integration method is not installed. A simultaneous associative-flow experiment now integrates all input writes with an augmented matrix exponential and holds interval metadata out of input content. Its fixed-coefficient subdivision error is below 0.000003, including finite, matching state gradients. Recorded-task retention and real-time usefulness remain to be evaluated.

The simultaneous-write timing experiment completed on both cue scenes. Across 50, 100 and 200 ms full audiovisual intervals, mean absolute errors are 0.0047–0.0248 seconds on withheld ages from 0.5 to 19.5 seconds. This is much better rate transfer than the ordered-write calibration. It remains an offline, two-scene interval-decoding result with a supplied cue boundary. The candidate method is not installed in the live agent.

Version 4 stores the observation integration method in the checkpoint. Migration from a trained version 3 checkpoint retains observations, advance, byte logits and output feedback exactly at 80, 100 and 150 ms. A simultaneous-method checkpoint round trip also preserves its behavior. The unused, untrained prediction head is removed from new model source. Forecasting and motor control remain unimplemented.

The 600-update version 3 language continuation completed with English training loss approximately 1.30 and recorded-task retention 6/10. Greedy exact generation remained 0/30 sampled training conditions and 0/30 validation conditions. Corpus-derived OpenAssistant identity phrases remain in its outputs. This candidate was not selected for the live agent.

## Complete interaction, without per-question resets

The old stable model scores 3/10 expected utterances and 0/5 complete conversations when the same state carries both questions and its own generated output through a full recorded episode. This stricter protocol is different from its earlier 6/10 independently prepared-question score.

After 25 complete-interaction training updates, the simultaneous-write candidate scores 4/10 expected utterances and 0/5 complete conversations. Numerical states remain finite. A longer 400-update fit is running separately from the live agent. The result is still a failure of conversational continuity and unfamiliar fruit recognition; it is not primitive consciousness.

The completed 400-update fit reproduces 54/54 training utterances in 27/27 complete training conversations, including matched visual interventions. Held-out results remain 5/10 utterances and 0/5 complete conversations. Familiar-case learning has improved, but the changed question wording and unseen fruit names do not transfer reliably. Those two changes are confounded in the old validation split.

An isolated raw replay of the 200-update candidate receives 85,739 actual Apple waveform samples with ongoing prerecorded video. A typed fruit question then produces `Apple.`. The same fixed candidate receives 45,902 additional samples for a spoken version of the question and produces no completed reply. Both runs finish without runtime errors and open no listening port. These tests leave the selected live agent unchanged. They support raw transport and familiar typed-query retention, not spoken question understanding.

The new corpus contains 44 twenty-second episodes. Eighteen training episodes use the David synthetic voice; validation separates the Zira voice with familiar phrasing from David with held phrasing. Known fruit labels occur across splits so voice/wording tests do not require a completely unseen answer word. Questions occur in spoken and literal text forms, with both question orders. The native audio encoder history is cleared after the initial fact, before the questions. Answers and question annotations are never fed as text in spoken conditions. Four no-fact controls test whether the model invents a remembered fruit. Two recorded scenes and synthetic speech remain substantial limits. A continuation on this corpus is isolated from the live agent.

## Own-code source-memory baseline

The speech continuation completed another 200 updates after its interrupted checkpoint. Its old checkpoint lacked optimizer state, so this was a weight continuation with a fresh optimizer. Timed held-out accuracy is 18/52 expected utterances. Correct content sometimes starts during the spoken question, which the original completed-question window excludes. A separate audit preserves that strict score and recovers onsets from the actual fixture lengths. For a held synthetic speaker, 8/16 targets have correct content after question onset, with 7 starting before the question ends. Held phrasing in spoken form remains poor: 1/8 by this broader timing audit. These counts do not establish question comprehension; question-type and no-question controls remain necessary. The candidate is not installed.

The 400-update complete-interaction checkpoint was tested on two paired input histories. Within each pair, only the first injected own-code feedback word changes, between equal-length `Apple.` and `Mango.`. All sensory inputs, physical intervals, later questions and externally heard facts match exactly. The injected utterance is excluded from loss and accuracy. Subsequent self-report and external-report utterances are generated freely.

The baseline obtains 2/12 scored utterances, but 0/4 own-output reports and 0/4 external-fact reports. No pair passes source discrimination, and no complete conversation passes. All final states remain finite. A small training experiment now targets this missing ability through the same state and byte translator. This is a causal source-memory test, not a consciousness test or evidence of spontaneous agency.

## Research resource correction

Overlapping speech training, source-memory training and a full-gradient CPU validation exceeded the system RAM budget. Both fits stopped at their guards, with checkpoints retained at 200 and 50 updates. The live agent automatically paused. Its last error event confirmed the RAM guard, and it resumed its existing recorded source after usage returned to approximately 33 GB. Its selected weights and numerical state were not replaced.

Large offline workloads now share an operating-system file lock and wait for RAM headroom before loading data. A process test confirms that a second workload cannot enter while the first holds the lock and can enter after release. The guard remains enabled. New checkpoints also retain optimizer and random states for same-configuration resumption.

Training no longer synchronizes CUDA with the CPU for each scalar interval validation or generated-code conversion. Offline rollouts transfer generated codes once after the episode. A paired CUDA test returns identical losses, final states, emitted codes and gradients. The measured free-running forward time falls from 2.79 to 1.99 seconds, and the teacher-driven forward time from 3.11 to 2.51 seconds for one 20-second episode. These are workload measurements, not sustained real-time latency guarantees. CPU timing did not improve consistently.

An actual interruption test compares two one-update fits with an uninterrupted two-update fit under the same configuration. Restored optimizer/random states produce identical final model weights: maximum difference 0. The source-memory task still fails its paired criterion in that small test, despite 5/6 individually correct utterances. Its own-word report is correct in only one of the two intervention conditions. This prevents a high aggregate score from being mistaken for source discrimination.

## Active internal computation candidate

Version 5 adds four learned internal rates and selective associative writes between observations. Legacy disabled-path migration matches prior observations, advance and byte logits exactly. Zero elapsed time leaves the state unchanged. An active 100-second numerical run remains finite, with maximum coordinate approximately 0.10. A checkpoint round trip preserves active advancement exactly. These tests establish compatibility and initial numerical stability, not useful reasoning.

The capacity experiment freezes perception and translation after one-hop memory training and tests a four-hop decision on unseen numerical graphs. Its inference code advances the same neural state; it does not traverse the graph. Immediate output, extra processing time, disabled internal writes and changed tick resolution are compared. A 90% one-hop validation prerequisite prevents a deficient memory encoder from being misdiagnosed as failed recurrent reasoning.

The learned one-hop representation scores 512/512 held-out decisions. Version 5's single-key internal write does not improve four-hop accuracy: 137/512 with or without processing. Version 6 adds a dynamic associative read, learned transform and separate write. After 1,000 updates, it falls to 124/512 with 0.4 seconds of processing, versus 137/512 disabled. Both active paths are retired from the current source. Sequential queries retain 497/512 one-hop decisions, or 494/512 with generated-code feedback; complete loss of graph memory does not explain the four-hop failure.

Version 7 uses a factorized full-state operator, including a trainable bilinear term, rather than the failed selective paths. It adds no recurrent cache. After 1,000 isolated updates with perception/readout frozen, four-hop accuracy is 145/512 at 0.4 seconds, versus 137/512 disabled. At 0.2 seconds it is 151/512. Changing the integration tick from 10 ms to 5 ms changes the 0.4-second score to 159/512; 20 ms gives 145/512. This small, resolution-sensitive benefit does not establish reliable internal reasoning. The 137/512 baseline includes cases where one-hop and four-hop destinations coincide; it must not be presented as eight-way chance. Receipt: `artifacts/liquid-recurrence-1791854009336339200/training.json`. The candidate is not selected for the live agent.

## Source-memory transfer limitation

The controlled source-memory continuation generates 44/48 scored utterances in 16 familiar conditions. A stricter evaluation clamps all earlier feedback identically except for the first injected word. It then generates the self/external reports freely. Familiar Apple/Mango controls produce 16/16 own-output reports, 14/16 external reports and 7/8 completely correct pairs. Substituting previously untrained Banana/Pear feedback produces 0/16 own-output reports, 13/16 external reports and 0/8 pairs. This demonstrates a narrow learned source distinction, not general memory of its own language.

The isolated capacity experiment trains recall of random three-to-eight-character sequences through the same own-code feedback path. Validation strings do not overlap training strings. Native quiet acoustic observations continue at 100 ms intervals while the common state advances and the byte readout generates an answer. No transcript or string lookup reaches the model. The completed 1,500-update fit scores 80/128 exact strings in its original preparation protocol. That protocol inserts the query as an additional observation and varies the generation budget with the longest experimental string. Those choices do not feed the string as text, but they are removed in the stricter control.

The stricter protocol fixes nine feedback slots and twelve output slots, requires a generated boundary, and joins the query to the next ordinary 100 ms sensory packet. It recalls 74/128 strings at zero requested delay, 77/128 at 0.3 seconds, 77/128 at one second and 79/128 at three seconds. Removing the earlier feedback yields 0/128. Single-stream and batched outputs match. However, the same weights recall 0/32 strings against either recorded visual background. Receipt: `artifacts/liquid-copy-controls-1791854958859199400.json`. This supports novel content retention under quiet input and exposes a failure under visual interference. It does not establish broad English, voluntary intention or subjective self-awareness.

The continuation using changing native video and 25% audiovisual rehearsal completed at 1,500 updates in `artifacts/liquid-copy-1791855084855416000`. It is isolated. The fixed-budget strict protocol recalls 86/128 at zero delay, 81/128 at 0.3 seconds, and 77/128 at one and three seconds; removing own feedback gives 0/128. The first 32 words score 14/32 under tennis and 16/32 under cycling. Familiar rehearsal generates 16/16 expected utterances. This is narrow retention, not broad transfer. Receipt: `artifacts/liquid-copy-controls-1791856963453564700.json`.

Matched query controls expose a packet-layout confound: literal text starts at row 2 in quiet packets and row 68 with full video. Using the identical quiet query and adding visual input only during subsequent generation reduces exact recall from 81/128 to 42/128 with tennis or 52/128 with cycling. This establishes behavioral interference after a matched cue. It does not prove erased memory; a changed readout can also fail. Receipt: `artifacts/liquid-interference-controls-1791856955357811100.json`.

Mean emitted-token probability is not calibrated word confidence. At a mean probability of at least 0.9, the quiet condition still has 19 wrong words among 93 outputs. Under tennis, 10 of 22 high-confidence outputs are wrong. These scores are diagnostic only; the runtime does not use them as a self-confidence policy.

A paired version 7 processing audit separates nontrivial four-hop targets from cases already answered by one-hop retrieval. Active processing corrects 58/375 nontrivial decisions, versus zero with processing disabled, but breaks 50/137 previously correct unchanged-target decisions. Halving the numerical tick changes 195/512 predictions. Receipt: `artifacts/liquid-recurrence-controls-1791854951960581900.json`. There is a learned behavioral change, but not a dependable reasoning improvement.

The finer convergence audit fails as well. At 10, 5, 2.5, 1.25 and 0.625 ms,
accuracy is respectively 145, 159, 156, 157 and 153 out of 512. Relative state
RMS differences against the finest run remain about 0.44, 0.31, 0.22 and 0.22.
Finite values and a short positive accuracy change are insufficient. Runtime
startup, resume and hot-load reject active experimental processing.
Receipt: `artifacts/liquid-recurrence-controls-1791857035943900900.json`.

## Label-free shared-context learning — 13 October 2026

The new experiment predicts projected native packets from the same shared state
before their assimilation. Only the existing input-context weights learn, with
a bounded parameter drift. Nine recordings train the predictor; five historical
validation recordings provide 600 scored packets. Scene identities overlap;
these are not unseen-world recordings. Literal question tokens can arrive as
ordinary input, but answer labels never enter predictive learning.

With unit coupling, two training passes reduce held prediction MSE from 0.791
to 0.0549. The constant training-mean baseline is 0.0535 and persistence is
0.00640. Continuing online on the held stream gives 0.0110 pre-update MSE,
which is not a frozen held-out result. Generated codes change at 110/600 ticks.
Receipt: `artifacts/liquid-continual-context-1791865443541775400/experiment.json`.

With 0.02 coupling and four passes, frozen held MSE becomes 0.0398. Independent
answer retention remains 6/10, with 147/147 quiet states, and the one-code-per-
packet diagnostic emits identical code choices at all 600 ticks. Permuting prior
state coordinates worsens MSE to 0.658. Held-stream online adaptation scores
0.0127 before each update. Mean cached-packet processing is about 18 ms during
learning; this excludes raw encoder time. This candidate uses the shared state
but still fails persistence and is not deployed. Receipts retain both positive
controls and failures. `artifacts/liquid-continual-context-1791865745382463200/experiment.json`.

The diagnostic emits once per packet, unlike the runtime's approximately 30 ms
byte schedule. Pooled targets can hide modality-specific error. Broad English,
useful autonomous deliberation, general source memory, self-modeling and
subjective awareness remain unverified. Next tests need modality-specific
forecast errors, sustained raw-media adaptation, future horizons, retention
under concurrent output and matched state/history ablations.

The post-report write-learning variant uses per-tensor trust bounds and one
packet of temporal credit. Its frozen held MSE is 0.0347, versus 0.0454 for the
matched per-tensor context-only fit: about 24% lower. Both retain 6/10 held
answers and 147/147 quiet states. The write-learning regime changes 104/600
diagnostic output-code choices, while the context-only fit changes none. Mean
learning cost is about 34 versus 17 ms per cached packet. Both still lose to
persistence at 0.00640. Neither is selected for live operation.

This comparison initially changed both write-parameter learning and temporal
gradient flow. A further completed control retains one-packet temporal gradients
while freezing write strengths: frozen held MSE is 0.0456, with 6/10 answers,
147/147 quiet states and no changed diagnostic codes. Thus temporal context
credit alone does not explain the write-learning regime's improvement in this
single-seed experiment. Neither regime beats persistence. Receipt:
`artifacts/liquid-continual-context-1791867137830454000/experiment.json`.
Earlier matched receipts:
`artifacts/liquid-continual-context-1791866743688677200/experiment.json` and
`artifacts/liquid-continual-context-1791866943622652900/experiment.json`.

## Runtime and reproducibility verification after the second report

An isolated startup test loads real native sensory weights with no demonstration
images or speech file. It becomes ready, retains finite numerical state, accepts
camera as a source without opening hardware, rejects the missing built-in
recording clearly, stays within the total RAM ceiling and shuts down its worker.
No extra UI port is started and the user's live state is untouched. Receipt:
`artifacts/runtime-startup-1791867605222955400/validation.json`.

All 15 mandatory local sensory files match upstream revision
`503e754207c94da6bb26850b4469f367c9ea3582`. The content lock is recorded in
`config/sensory-assets.json`. This identifies configs, tokenizer/model code and required shards;
it does not publish custom weights or resolve original recording licenses.
Evidence: `artifacts/sensory-upstream-revision-audit.json` and
`artifacts/runtime-preflight-hashed.json`. Cold-install verification remains open.

## Rank-one temporal-credit experiment

The same core now has an isolated UORO-style learner, with 2,184,208 bytes of
rank-one sensitivity factors and no input-history replay. A sign-averaged linear
test recovers the exact reference sensitivity; short actual-core steps stay
finite. This tests the reduction and execution, not long-run gradient quality.

After four passes, frozen held forecast MSE is 0.0388, while persistence remains
0.00640. Held answer retention falls from 6/10 to 5/10, and 136/600 diagnostic
code choices change. Held-stream online adaptation is worse still, at 0.0554.
Cached learning takes about 80–92 ms per packet on CPU across training passes,
with some packets above 100 ms before raw encoder costs. This candidate is
rejected for live selection. Receipt:
`artifacts/liquid-continual-context-1791868402240603900/experiment.json`.

The rank-one scheme updates weights after assimilation, unlike the earlier
one-packet method. A same-phase rank-one control that resets credit each packet
is available to isolate longer history from update order and gradient noise.
It has not yet been trained. No subjective-awareness conclusion follows from
this experiment or the theoretical estimator.
