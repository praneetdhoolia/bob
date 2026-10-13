# Live meeting interface

The interface uses one meeting stage with video input and the agent as two participant tiles. Chat opens in a drawer without replacing either tile or interrupting inputs. On small screens the tiles stack and chat overlays them. Closing chat preserves the draft.

The controls use local SVG icons for camera, microphone, agent speaker, video/audio files, chat and activity. Test recording and quiet input move into the More menu. Mute/unmute is also available on the agent tile when hovered or focused, and stays visible on touch devices. Muting cancels browser speech immediately, discards queued playback text and leaves numerical processing active.

The design follows the separation of media controls, participant view and a chat pane described in Microsoft's [meeting controls](https://support.microsoft.com/en-us/teams/meetings/use-meeting-controls-in-microsoft-teams) and [meeting chat guidance](https://support.microsoft.com/en-us/teams/meetings/chat-in-microsoft-teams-meetings). This is a local prototype, not a Teams integration.

Glass material uses neutral grey tints, translucent surfaces, highlights and restrained shadows. The panels use background blur rather than opaque fills. White or charcoal text stays at full opacity. This follows Apple's [material guidance](https://developer.apple.com/design/human-interface-guidelines/materials) on transparency and legibility. The theme button cycles between desktop, sun and crescent icons for System, Light and Dark. System follows the operating system's appearance. The preference persists locally.

The galaxy uses procedural WebGL clouds, logarithmic spiral arms, dust lanes, a luminous core and 7,000 orbital particles. It uses no photograph or image texture. A fixed camera views the inclined, mostly horizontal disk. Only the disk revolves, at approximately 0.8 degrees per second. The viewport does not roll or zoom. This decorative animation is not an astrophysical simulation or neural activity. It draws at the display's animation-frame cadence, targeting 60 fps. It stops when hidden or under reduced-motion preferences. A canvas fallback renders orbital particles if WebGL is unavailable. Background failures leave the controls operational.

Scrollbars use rounded narrow thumbs with transparent padding and no native arrows. Scrollable activity and chat content fade at their upper and lower edges when more content remains. The first view has no JSON, coordinate explorer, manual numerical-save control or long explanatory copy.

System RAM is shown against measured physical capacity. The configured framework budget remains an internal limit. Memory shows pooled state strength in RMS units. Flow shows recent projection changes in milli RMS units. Both are dimensionless numerical measurements. They do not measure stored knowledge or understanding. The graphs use subtle horizontal guides. They omit scrolling seconds and Now labels.

Audio uses bars at fixed horizontal positions across a five-second window. Their heights sample and interpolate the recorded amplitude at each position. The wave travels through these stationary bars over five seconds. The bar count is half the previous density, with up to 50 bars on wide screens. Bar heights rise over 120 ms and fall over 220 ms. The runtime provides 40 amplitudes per 100 ms packet. The display combines these into measured 50 ms RMS segments. Its fixed amplitude scale runs from -60 to 0 dBFS. Mirrored halves show magnitude rather than positive and negative raw samples. Display smoothing does not change the samples delivered to the agent. The sensor table reports unsmoothed whole-packet RMS and dBFS level.

Activity places 46 sensor rows beneath the graphs. Current, Minimum, Maximum and Average align right, with subtle vertical dividers. The five sensor groups collapse independently and retain their preferences. Measurements cover timing, inputs, state, system/agent RAM, CPU, tensor/reserved VRAM and output. The elapsed clock uses hh:mm:ss.

The table uses fixed column widths and tabular numerals. Values changing digit count cannot move columns or row boundaries. Narrow panels scroll horizontally through the same stable layout. Truncated sensor labels retain their full text in tooltips and accessible content.

Minimum, maximum and average cover samples from the current viewing session. Repeated initial snapshots do not add duplicate samples. Totals and fixed capacities use dashes in statistical columns rather than misleading averages. Agent CPU is normalized against all logical processors. Tensor and reserved VRAM describe PyTorch allocations, not total GPU utilization. Missing measurements use dashes.

Every activity canvas uses device-pixel-density rendering and antialiased curves.
Memory and the state orbit blend between measured targets on each animation frame.
Flow averages three observations when each sample arrives, then preserves that
historical value. An eight-second physical-time window, buffered by 150 ms so
the newest segment can enter smoothly, scrolls horizontally at
the animation-frame cadence. Completed curve segments retain their endpoints;
new samples do not reshape earlier peaks. Its rounded vertical scale stays fixed
for the viewing session and expands only when a new high needs more range.
It does not animate or automatically shrink the axis. The sensor table retains
raw measurements. Rendering stops when the activity panel or page is hidden.
Update frequency uses differences in active physical time. Queued browser events
cannot inflate the rate. These visuals describe processing and influence. They
do not certify understanding or conscious thought. Green video regions remain
approximate current-output sensitivity.

Activity toggles the detail inside the existing agent tile. Turning it off leaves its face centred without changing participant columns or tile dimensions. Reply text stays in Chat. The face mouth animates only during playback. Desktop separators resize the participant tiles and chat. Drag them, or focus them and use arrow keys. Home and End select their limits. Sizes persist locally. Small screens stack the tiles and overlay chat. Text input uses a rounded focus treatment. Buttons retain visible keyboard focus and accessible names. Icon targets remain usable on touch screens.

Tracked automated checks are Node tests in `tests/`: audio-worklet resampling, Flow history, and media controls (camera and microphone stay independent, a second local video reuses its single media-element audio source, the video soundtrack reaches the audio graph, and the microphone replaces file audio). Earlier synthetic browser checks of drawers, drafts, speech cancellation, focus and overflow are recorded in `docs/evidence/receipts/report-ui-validation.json`; no browser automation is tracked, so those checks are not rerun by `npm test`.

Accessibility: streamed output is announced once per finished utterance through a separate polite live region rather than on every byte. Escape closes the More menu, then chat, and returns focus to the chat button; it no longer collapses Activity. The agent mute button keeps one name and reports its state through `aria-pressed`. Canvases, the face and the focus overlay expose `role="img"` with their labels. Light-mode secondary text uses #2a2a2a, about 4.9:1 against the base grey.

Activity's Output group includes the urge: its share of the onset threshold, translator readiness (non-blank probability), sensory surprise and the number of stimulus releases. These are numerical mechanism readings, not felt desire.
