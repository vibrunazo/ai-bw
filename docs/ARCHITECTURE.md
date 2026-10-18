# Gary — Architecture

Status: **draft v0.9** (2026-10-02)

Gary is a StarCraft: Brood War AI that pro and aspiring players can use as a **sparring partner**:
it plays with pro-level game sense, but with **human hands** (human APM, attention, reaction time
and mistakes), it can be **told what to play** ("go 2 hatch muta", "play the PvT build from this
replay"), and it can **explain why** it did what it did.

**What is built (2026-10-13).** This document is the design; [PROGRESS.md](PROGRESS.md) says what
exists, and [CONTROL.md](CONTROL.md) replaced the single-policy plan with layers of deciders and
skills. Per section:

| Section | Status |
|---|---|
| §3 principles | 2 (one pair of hands), 4 (symbolic layer learning by counting), 6 (swappable implementations), 7 (slots) and 8 (reproducible: games are deterministic per seed and replayed to any moment) built; 1 restated below as Gary works now; 3 partly (*z* is a build style, no prompts); 5 records exist, no narrator yet |
| §4 overview, §5 components | Replaced by CONTROL.md's layers: posture, macro, army, fight, motor, each a provider chosen by a manifest |
| §6.1 `StrategySpec` | Partly: *z* is a build-style cluster (`--style`, or the strategy selector's choice) |
| §6.2 `HumanProfile` | Built (pro and lower profiles); select-to-order times fitted to pro replays, the rest placeholders |
| §6.3 `HumanAction` | Built as the human interface every executor clicks through; no model emits these actions directly (deciders choose in their own spaces) |
| §6.4 `Intent` | Built as each layer's intents (`gary/agent/intents.py`); the single intent head with alternatives is deferred |
| §6.5 trace | Partly: decision and POV logs; no formal schema |
| §6.6 slots | Built as manifests and options (`gary/agent/manifests.py`) |
| §6.7 executors and arbiter | Built: one task at a time by fixed priority, production every other turn in fights; no minimum macro share by profile |
| §6.8 strategy selector | Built light: a win-rate table per map with a Beta posterior (v0.19), from Gary's own games; no pro statistics, no conditioning on what was scouted yet |
| §6.9 hidden enemy status | Partly: enemy HP hidden unless selected; no upgrade estimates |
| §6.10 fog memory | Built: enemy buildings, and the army seen in the last 3 minutes |
| §7.1-7.3 data, taxonomy | Built (taxonomy unreviewed) |
| §7.4 camera logger | Not started |
| §7.5 opponent model | Started: the belief model (the enemy's army, hidden or not); no inference of the enemy's *z* |
| §7.6 feature discovery, §7.7 guide mining | Deferred: unbuilt, they cost nothing |
| §8 explainability, §8.1 broadcast | The POV viewer and the decision log exist; no narrator, no broadcast |
| §9 experiments | Built: manifests, benchmarks against scripted and replayed Zergs, pro scenarios and episodes, branches, a run monitor |
| §10 deployment | The Remastered bridge for offline games |

---

## 1. Goals and non-goals

**Every race, every matchup.** Gary is one player that can be asked to play Terran, Zerg or
Protoss in any matchup, including mirrors, as a sparring partner for players of any race. Every
layer below is a provider per race over race-neutral contracts, trained from that race's side of
that matchup's replays. TvZ, with Gary as Terran, was the first instance built (§14), chosen in
October 2026 because it was convenient to start somewhere; the learned Zerg that plays it in
self-play (docs/SELF_PLAY.md) is Gary playing Zerg, not an opponent bot. Nothing in the design is
specific to Terran or to TvZ, and code that is counts as debt (PROGRESS.md tracks it).

### Goals

| # | Goal | What "done" looks like |
|---|------|------------------------|
| G1 | **Plays like a real opponent** | Survives and punishes harass like a human would. A 5-muta rush is neither free (built-in AI) nor hopeless (Pluto's 3-marine micro). |
| G2 | **Human constraints, not superhuman control** | Camera, selection limits, burst-limited APM, reaction delay, imprecision and lapses are enforced *in the action space*. |
| G3 | **Steerable** | A text prompt or a replay produces a strategy spec the bot follows reliably, game after game ("practice it 20 times in a row"). |
| G4 | **Adjustable skill** | One model, many skill levels: a "middle ground" between built-in AI and Pluto, set by a profile. |
| G5 | **Explainable** | Every game has a queryable trace: what it intended, what it believed, which alternatives it considered, and which human constraint got in the way. |
| G6 | **Modular for experiments** | Models, training recipes and pipeline variants can be swapped through config and compared on a shared evaluation harness. |

### Non-goals

- Beating Pluto or winning bot tournaments. Strength only matters up to "representative of the target human level".
- Playing on the Battle.net ladder. The bot is for offline practice and custom games only (see §10).
- Full transparency of the neural policy. We aim for *faithful, structured* explanations (§8), not a white box.

---

## 2. Background and key findings

**Pluto** (tscmoo, Computer Olympiad 2026 winner) shows full-game self-play RL for BW is feasible:
- A 315M-parameter network: a transformer over units, convolutions over the map, a GRU, and a slow memory module.
- One model step every 6 frames, roughly 240 steps/min at Fastest.
- **Why it feels superhuman: bandwidth per action, not raw APM.** A single action can command any number of units, with no 12-unit selection limit and no camera, so visible APM reads in the thousands. Capping APM alone would not humanize it.
- Its opening is chosen by a Thompson-sampling bandit and fed to the network as a 32-bit mask. That's a crude version of the strategy conditioning we want (G3).
- Binary-only release. We cannot retrain it or legally fork it. It's useful as an **upper reference opponent** in evaluation.

**Remastered (SC:R)**:
- BWAPI officially supports only 1.16.1.
- An unofficial bridge (Pluto-AI-Starcraft-Remaster) injects a DLL into SC:R build 1.23.10.13515. It translates SC:R memory into what BWAPI 4.4 expects and turns the bot's commands back into SC:R packets.
- Anything that speaks BWAPI 4.4 could, in principle, use the same approach. It's pinned to one build and unofficial.

**AlphaStar (SC2)** is the closest precedent for G2 and G3:
- Supervised learning from human replays, conditioned on a strategy statistic *z* (build order plus cumulative unit counts) taken from the same replay.
- RL fine-tuning with pseudo-rewards for following *z* and a KL penalty toward the supervised policy.
- A camera interface and APM limits. It was still criticized for superhuman *burst* precision, so we need **burst limits, not averages**.

**Decision models (the "System One" class, Sept 2026)**: models that pick from options you supply and return probabilities, with no text generation.

| Model | Access | Latency | Fine-tunable | Calibration |
|---|---|---|---|---|
| Jev (TypeSafe) | Hosted API | 70–500 ms | No | Trained for calibration (RLCD) |
| OpenAI Decisions API (GPT-6 Luna) | Hosted API | ~150 ms | — | Probabilities are self-reported by the LLM, not calibrated against outcomes |
| Laya (convaiinnovations) | Open weights, Apache-2.0, ModernBERT-large 421M | ~33 ms on a T4 GPU; 0.2–0.5 s on CPU | Yes (RLCD) | Over-confident out of the box; needs temperature scaling per domain |
| Kev / AnyJev / DiffusionGemma | Open | varies | varies | varies |

**Verdict:**
- Open, local, fine-tunable models (Laya) remove the hosted-API and no-fine-tuning objections. They can run on our own GPU, and they're fast enough for the Plan Supervisor (one decision every few seconds) and for labeling replays at scale.
- They are still **not** the per-step policy and **not** an RL teacher. A decision model fine-tuned on replay labels learns the same thing the policy's own belief and intent heads learn from the same data, but through a lossy text version of the game state.
- They are treated as **providers in decision slots** (§6.6), compared against scripts, LLMs and human labels on a gold set. Calibration (ECE) is measured by us, not taken from vendor claims.
- **To get a head start from a decision model, fine-tune it on replay labels. Don't distill its base version into a small network.**
  - Distilling base Laya copies only its answers on the queried states. Those answers are mostly guesses, since it has no BW training. The replay stage then overwrites most of what was copied.
  - Fine-tuning keeps its pretrained representations.
  - If a small, fast network is needed, **fine-tune first, then distill the fine-tuned model**.
  - Pretraining priors matter most for rare situations with few replay examples.
- **Open-vocabulary conditions** are a lasting niche for decision models. A user-written branch like "if he goes lurker, transition to mech" can be evaluated as text against current facts. A fixed-vocabulary network head can't do that, and a rule engine needs someone to write the rule (§6.6, `supervisor_branch`).

---

## 3. Design principles

1. **Constraints live in the action space, during training.** Adding a humanizer after training breaks policies that learned to rely on superhuman control.

   *As built:* the deciding layers choose in their own spaces (a build, a grid cell, a role per
   unit, a posture), and the executors and the arbiter turn that into clicks through one pair of
   hands: the human interface, the same in training and at runtime. Reinforcement learning on
   fights and roles plays through that real interface, so what it learns already pays the hands'
   costs. There are no action masks, and no deciding model sees the state of the hands (APM left,
   camera, selection, actions waiting): whether that costs Gary games is being measured
   (TODO.md item 2). The design as first written, for a single policy emitting actions:
   - **Hard limits are action masks inside the policy:** off-screen targets, more than 12 units selected, empty APM bucket. Illegal actions are never sampled. They are not silently dropped.
   - **Interface state is part of the observation:** APM tokens left, camera rectangle, current selection, pending delayed actions.
   - **The Human Interface only adds noise:** click scatter, misclicks, lapses. **Reaction delay is applied to observations**: the bot sees events late.
   - The **same implementation** of the interface is used in training and at runtime.
2. **One pair of hands.** Every executor that issues commands (the neural policy, a scripted macro executor, anything else) spends from the **same APM budget and the same camera**. Nothing gets free parallel actions. Human mistakes like a missed depot while microing mutas come from this shared budget, not from bolted-on randomness alone (§6.7).
3. **Steering goes through explicit conditioning.** What to play (*z*) and how well to play (*h*) are both policy inputs. Prompts and replays are compiled into *z*. They are never free-text instructions to the policy.
4. **Neural for skill and inference, symbolic for structure and decisions of record.** The neural side handles perception, micro, and inferring hidden information (the opponent's likely *z*). Macro execution is a swappable executor, scripted or neural (§6.7). Symbolic components provide:
   - the strategy language (*z*)
   - the intent bottleneck
   - the strategy layer: branch rules, plus **learned, inspectable win-rate statistics** (§6.8)
   - the trace

   **Symbolic does not mean static:** the strategy layer learns from replays and from games played, by counting outcomes. Neural models also help decide *which* context features those statistics should be conditioned on (§7.6).
5. **Explanations come from records, not stories.** Explanations are built from logged intents, beliefs, candidate probabilities and constraint events. The LLM only narrates. It never reconstructs reasoning after the fact.
6. **Contracts are stable, implementations are swappable.** Schemas are versioned. Everything else is a plugin selected by a run manifest.
7. **Every decision or label outside the policy is a slot.** Opening classification, intent labels, branch selection, camera inference and so on each have a typed interface. A deterministic script, a decision model, an LLM or a human can fill it, and they can be swapped and compared (§6.6).
8. **Everything is reproducible.** Seeded sampling and humanization, plus logged model versions, let any game be re-run and **decision-diffed** against another pipeline.

---

## 4. System overview

```
           ┌──────────────── Offline / authoring ────────────────┐
 prompt ──►│ Strategy Compiler (LLM)        ─┐                   │
 replay ──►│ Replay Extractor (screp+resim) ─┼──► z  StrategySpec │
 library ─►│ Strategy Library               ─┘                   │
 preset ──►│ Profile Library  ──────────────────► h  HumanProfile │
           └─────────────────────────────────────────────────────┘
                                   │
 ┌──────────────────────────────── Runtime (per game) ─────────────────────────────┐
 │                                                                                 │
 │  Game Adapter ──► Perception (fog-filtered obs + symbolic facts: what I SAW)    │
 │       ▲                     │                                                   │
 │       │                     ▼                                                   │
 │       │         Opponent Model (neural): P(enemy z | what I saw)  ── beliefs    │
 │       │                     │                                                   │
 │       │                     ▼                                                   │
 │       │         Strategy Layer (symbolic): Plan Supervisor rules +              │
 │       │           Strategy Selector (learned win-rate stats) → active z segment │
 │       │          ┌──────────┴──────────────┐                                    │
 │       │          ▼                         ▼                                    │
 │       │  ┌── Policy π(a | obs, z, h, iface) ──┐   ┌── Macro Executor ───────┐   │
 │       │  │ encoders → core → INTENT ► ACTION  │   │ scripted (queue from z) │   │
 │       │  │   (masked by interface state)      │   │  or neural (= policy)   │   │
 │       │  │ └► BELIEF heads (aux)              │   └───────────┬─────────────┘   │
 │       │  └─────────────────┬──────────────────┘               │                 │
 │       │                    ▼                                  ▼                 │
 │       │          Attention Arbiter: one APM bucket, one camera ◄── h            │
 │       │                    │          (interface state ──► obs, masks)          │
 │       │                    ▼                                                    │
 │       └──── Human Interface (click scatter, misclicks, lapses) ◄── h            │
 │              + observation delay (reaction time) on the Perception side         │
 │                                                                                 │
 │  Trace Bus ◄── facts · z switches · intents+probs · beliefs · arbiter/          │
 │                constraint events                                                │
 └─────────────────────────────────────────────────────────────────────────────────┘
                                   │
                    Trace Store ──► Query API ──► Narrator (LLM) ──► "why…?" answers

 Game backends:  OpenBW headless (training/eval)  |  BWAPI 1.16.1  |  SC:R via bridge
```

---

## 5. Components

| Component | Responsibility | Inputs → Outputs | Swappable implementations |
|---|---|---|---|
| **Game Adapter** | Step the game, read state, submit low-level commands. Hides the differences between backends. | backend ↔ `RawState`, `GameCommand[]` | OpenBW headless (vectorized, many games in parallel), BWAPI 4.4 client, SC:R bridge |
| **Perception** | Build observation tensors and symbolic facts under the **UI-visibility rule**: only what the BW interface would show this player (fog, plus hidden enemy HP/shields/energy/upgrades unless inspected, §6.9). Scramble unit IDs on fog entry; keep building snapshots; maintain the decaying group memory of fogged units (§6.10) and last-inspected values. | `RawState` → `Observation`, `Fact[]` | Feature sets v1/v2…; fact extractors |
| **Strategy Compiler** | Turn a natural-language request into a valid *z*: resolve build names via the taxonomy (§7.3) and sample a real human *z* from that cluster. Ask for clarification or reject if impossible. | text → `StrategySpec` | LLM model/prompt versions; template library |
| **Replay Extractor** | Turn a replay into *z* for one player: build order, timing targets, style statistics. Optionally suggest branches for human review. | `.rep` → `StrategySpec` | screp-only (commands) vs. resim (full state) |
| **Opponent Model** | Infers hidden information from what the player has actually seen. Main output: a probability distribution over the opponent's *z* cluster (from the build taxonomy, §7.3), plus other beliefs (army size, tech, proxy). **Never reads hidden game state.** | `Observation`, `Fact[]` → `Belief[]` | NN belief model (default; trained on replays with hindsight labels), Bayesian/symbolic, fine-tuned decision model |
| **Strategy Layer** | **Plan Supervisor:** hard branch rules and constraints from *z* (always on; e.g. no switching in strict mode). **Strategy Selector:** picks among allowed branches using learned win-rate statistics conditioned on beliefs and context (§6.8). Logs every switch with its evidence. | `Fact[]`, `Belief[]`, *z* → `z_active` | `supervisor_branch` slot providers: rules, stats table (§6.8), NN intent head, fine-tuned decision model |
| **Policy** | Perception → intent → human-interface actions, conditioned on *z*, *h* and interface state. Hard limits are applied as action masks. | obs, `z_active`, *h*, `InterfaceState` → `Intent`, `HumanAction` | Backbone size/architecture, training recipe, checkpoint |
| **Macro Executor** | Turns the active *z* segment into production, tech, supply and building-placement actions. | `z_active`, `Fact[]`, `InterfaceState` → `HumanAction` | `scripted` (build queue plus placement library; default early on), `neural` (the policy does macro itself) |
| **Attention Arbiter** | Gives every executor's actions **one** APM bucket and **one** camera. Decides whose action goes next. Emits arbiter events (e.g. "macro starved for 9 s during fight"). | candidate `HumanAction`s → one `HumanAction` per step | Priority rules (default), profile-weighted, learned (later) |
| **Human Interface** | The only path to the game. Maintains `InterfaceState` (tokens, camera, selection) for masks and observations. Adds click scatter, misclicks and lapses. Delays observations to model reaction time. Emits a `ConstraintEvent` for every perturbation. | `HumanAction` → `GameCommand[]` | **Single implementation** shared by training and runtime; parameterized only by *h* |
| **Trace Bus / Store** | Append-only, frame-stamped event log for every game, plus indexes for querying. | events → JSONL/Parquet | Storage backend |
| **Narrator** | Answer "why" questions by retrieving trace records and turning them into text, citing event IDs. | question + trace slice → answer | LLM model/prompt versions |
| **POV Viewer / Broadcaster** | Render the bot's own screen and cursor from its POV track, draw belief/memory/APM overlays, and voice live commentary chosen by the commentary director (§8.1). Doubles as the main debugging tool. | POV track + trace → video/stream | Offline renderer, live client capture; template vs. LLM commentary; TTS voice |
| **Eval Harness** | Run matches and scenarios at scale, compute metrics, keep the scoreboard. | manifests → metrics | Scenario suites, opponent pools |

---

## 6. Contracts (schemas)

All schemas are versioned protobuf definitions (`schemas/`). They are shown here as YAML for readability.
**Numbers in examples are placeholders** to be fitted from replay data (§7.2).

### 6.1 `StrategySpec` (*z*): what to play

Modeled on AlphaStar's *z*, extended with explicit branches and style.

> **In plain terms:** *z* is the bot's game plan, written down as data.
> - It's like a recipe card: "make these units and buildings, in roughly this order, by roughly these times."
> - The policy reads the card as an input every step and tries to play the game that matches it. Hand it a different card and you get a different game.
> - The card doesn't need a name. Names like "2 hatch muta" live in the build taxonomy (§7.3), which maps each name to many real recipe cards taken from human games.

```yaml
strategy:
  schema: strategy/v1
  id: zvt_2hatch_muta_std
  race: zerg
  matchup: ZvT
  source: {type: prompt, ref: "go 2 hatch muta"}   # prompt | replay | library
  adherence: strict            # strict | loose → weight of the adherence pseudo-reward / supervisor
  segments:
    - id: opening
      build_order:             # supply-anchored, with tolerance
        - {item: drone,     until_supply: 9}
        - {item: overlord,  at_supply: 9}
        - {item: hatchery,  at_supply: 12, where: natural}
        - {item: pool,      at_supply: 11}
        - {item: extractor, at_supply: 11}
        - {item: lair,      when: "minerals>=150 && gas>=100"}
        - {item: spire,     when: "lair_done"}
      tolerance: {supply: 1, seconds: 10}
      targets:                 # cumulative statistics by game time
        - {by: "7:30", units: {mutalisk: 9, zergling: 6}, bases: 3}
    - id: muta_harass
      style: {aggression: 0.7, harass_bias: high}
    - id: vs_early_bio_push
      build_order: [{item: sunken, count: 2, where: natural}]
  branches:                    # evaluated by the Plan Supervisor, not by the policy
    - from: opening
      when: "fact.enemy_marines_seen >= 8 && time < '5:30'"
      or_belief: {enemy_opening: bio_timing, p_min: 0.6}
      to: vs_early_bio_push
    - from: opening
      when: "fact.spire_done"
      to: muta_harass
  style:                       # continuous knobs, also policy inputs
    aggression: 0.5
    expand_tempo: standard     # fast | standard | greedy
    tech_bias: air
```

The policy encoder sees the **active segment** (build order tokens, targets, style vector), not the whole tree.

### 6.2 `HumanProfile` (*h*): how well to play

Two parts:
- `conditioning`: fed to the policy, so it learns to play like players at that level.
- `limits`: enforced by the Human Interface and the Attention Arbiter. `apm` and `interface` become **masks**; `reaction` becomes an **observation delay**; `precision` and `lapses` become **noise**.

```yaml
human_profile:
  schema: profile/v1
  id: b_rank_terran
  conditioning:
    race: terran
    rating_band: B             # learned from replay metadata
    style_tags: [bio, standard]
  limits:
    apm:
      bucket: {capacity: 10, refill_per_sec: 3.5}   # burst limit (token bucket)
      min_action_gap_ms: 55
    reaction:
      model: lognormal
      median_ms: 300
      sigma: 0.35
      attention_switch_ms: 180   # extra delay when the event is off-screen / off-focus
    interface:
      viewport: {w: 640, h: 400} # playable screen area in px (excl. HUD); confirm per backend
      max_selection: 12
      precise_commands_require_on_screen: true
      minimap_scatter_px: 48
    pointing:                    # Fitts's law, fitted from camera-logger mouse data
      fitts_a_ms: 90
      fitts_b_ms: 110
    precision:
      click_scatter_px: {base: 3, per_bucket_pressure: 6}  # sloppier when spamming
      misclick_rate: 0.008
    lapses:                      # probability per opportunity, scaled by load
      idle_production: 0.10
      missed_supply: 0.05
      forgotten_unit_group: 0.03
    memory:                      # fogged-unit memory (§6.10)
      max_groups: 8              # enemy groups tracked at once
      minimap_notice_prob: 0.6   # chance a minimap-only sighting is stored
      pos_blur: speed_x_time     # location uncertainty growth
      half_life_s: 45            # confidence half-life without refresh
      hp_reset_s: 5              # HP estimate returns to prior
  seed: 42
```

### 6.3 `HumanAction`: the policy's action space

Every action costs APM tokens. Screen-space targets are only valid inside the current viewport.

**Targets are pixels, never unit IDs.** The policy may *reason* about units, for example with a pointer network over the unit list, but every click it emits is a **screen point**. The Human Interface resolves what's under that point using BW's own hit-testing, so the topmost sprite wins. Consequences:
- Clicking a muta stack selects or targets whatever is on top, as it would for a human. Picking the lowest-HP muta only works if it's actually exposed.
- Clicking an enemy unit replaces the current selection, exactly as in the game (see §6.9).

| Action | Args | Notes |
|---|---|---|
| `CAMERA_MOVE` | map point | Costs a token. Edge scrolling is modeled as a series of moves. |
| `CAMERA_JUMP` | hotkey / base / last alert | |
| `SELECT_BOX` | screen rect, shift? | At most 12 units. Selection rules (type filters, buildings) follow BW. |
| `SELECT_CLICK` | screen point, shift?, double? | Resolved by hit-test. Double-click selects same-type units on screen, at most 12. Clicking an enemy unit selects it for inspection (§6.9). |
| `HOTKEY_RECALL` / `HOTKEY_SET` / `HOTKEY_ADD` | 0–9 | |
| `COMMAND` | type, target (screen point / minimap point), queued? | A screen point on a unit becomes a unit target via hit-test. Minimap targets get extra scatter. |
| `BUILD` | type, screen point | Applies to the selected worker. |
| `TRAIN` / `RESEARCH` / `UPGRADE` | type | Applies to selected buildings. |
| `NOOP` | | Explicit "do nothing this step". |

Unlike Pluto, one action affects **at most one selection**, and precise targets require the camera to be there.

**How limits are enforced:**

| Limit | Mechanism | Where |
|---|---|---|
| APM bucket empty | Mask everything except `NOOP` | Policy action head (logit → −∞) |
| Target outside viewport | Mask screen points outside the viewport; minimap targets stay legal | Policy action head |
| Selection > 12 / invalid selection | Mask in the selection head | Policy action head |
| Reaction time | Events enter the observation after a delay sampled from *h* (plus an attention-switch cost when off-screen) | Perception |
| Mouse travel | The click lands after a **Fitts's-law** delay from the current cursor position: `T = a + b·log₂(D/W + 1)`, where *D* is the distance and *W* the target's on-screen size | Human Interface |
| Click scatter, misclicks | Endpoint scatter grows when travel time is cut short (rushing) and shrinks for large targets; a scattered click may hit a different sprite or empty ground | Human Interface |
| Lapses | Occasionally skip or delay a due macro action, scaled by load | Human Interface / Arbiter |

**`InterfaceState`** (part of the observation): APM tokens left and refill rate, camera rectangle, **cursor position**, current selection (own units or one inspected enemy unit), hotkey groups, delayed actions still pending.

**Mouse model scope:** the cursor is *not* steered pixel by pixel. That would explode the action space and make training very slow. Each click is charged its travel time and scatter from the last cursor position. This captures why small, far, densely packed targets are slow and error-prone for humans. Fitts parameters are fitted per rating band from camera-logger mouse data (§7.4).

**Masks during supervised learning:** replay camera positions are *inferred* (§7.2), so a hard mask can wrongly mark a real human action as illegal. During BC, masks are **soft**: confidence-weighted from the camera-inference model, or relaxed to "near viewport". They become hard in RL and at runtime, where the camera is known exactly.

### 6.4 `Intent`: the symbolic bottleneck

Emitted by the intent head every K policy steps, or whenever it changes. The action head is conditioned on the current intent.

```yaml
intent_vocab:
  schema: intent/v1
  macro:   [EXPAND(base), TECH(item), PRODUCE(unit_mix), SUPPLY]
  army:    [ATTACK(region), HARASS(region, group), DEFEND(region),
            RETREAT(region), CONTAIN(region), REGROUP(region)]
  info:    [SCOUT(region), DENY_SCOUT]
  meta:    [TRANSITION(segment_id), ALL_IN]
```

Each intent record includes the **probabilities of the top-k alternatives**. That's what makes "why X instead of Y?" answerable.

### 6.5 Trace events

```yaml
# common envelope
{ game_id, frame, t_game, seq, kind, payload, refs: [event ids] }

kind: fact            # {key, value, source: seen|memory|inferred}
kind: belief          # {head, distribution}  e.g. enemy_opening: {bio_timing: .62, mech: .21, ...}; enemy_status: {unit, hp_est, ci, source: tracker|inspected}
kind: z_switch        # {from, to, rule, evidence: [fact/belief ids]}
kind: intent          # {chosen, params, alternatives: [{intent, p}], policy_ckpt}
kind: action          # {human_action, tokens_left}
kind: constraint      # {type: masked|obs_delayed|scattered|misclick|lapse, detail}
kind: strategy_select # {chosen, belief, context, options: [{id, ev, cells: [{enemy, ctx, wr, n, ci}]}], feature_library, table_version}
kind: arbiter         # {granted_to: micro|macro, waiting: [{executor, action, waited_ms}], reason}
kind: memory          # {op: create|refresh|merge|decay|drop, entry: {type?, count_est, ci, last_pos, last_seen, confidence}}
kind: outcome         # {engagement result, units lost/killed, ...}
kind: slot_decision   # {slot, provider, provider_version, answer, p, alternatives, cached?}
```

### 6.6 Decision slots: swappable providers

Any decision or label that isn't made by the policy network goes through a **slot**. A slot has:
- a typed question and answer schema
- a latency class
- any number of **providers** that implement it

Providers are chosen per run in the manifest (§9.1). This is what makes comparisons like "decision-model classifier vs. Python script vs. human label" a config change instead of a code change.

```python
class SlotQuestion(Protocol):     # one schema per slot, versioned
    slot: str                     # e.g. "opening_class/v1"
    context: dict                 # typed facts / state summary / replay window ref
    options: list[str] | None     # choice slots; None for score/bool

class SlotAnswer:
    answer: str | float | bool
    p: float | None               # probability; None if the provider has none (e.g. a script)
    alternatives: list[tuple[str, float]]
    provider: str                 # "rules_py", "laya_ft", "jev", "openai_decisions", "llm", "human"
    provider_version: str         # code hash / checkpoint / prompt version / annotator id
    latency_ms: float

class Provider(Protocol):
    def answer(self, qs: list[SlotQuestion]) -> list[SlotAnswer]: ...   # batched
```

**Provider types:**

| Provider | Notes |
|---|---|
| `script` | Deterministic Python rules. Free and reproducible. `p` is either 1.0 or a hand-set confidence. |
| `decision_model` | Laya/Kev (local, fine-tunable), Jev/OpenAI Decisions (hosted). A per-provider calibrator (temperature scaling) is fitted on the gold set. |
| `llm` | Prompted LLM returning a structured answer. Good for rare or complex cases. |
| `human` | Async **annotation queue**: questions are written to a queue and answered in a small labeling UI. Answers become the **gold set**. |
| `learned_head` | A head of the policy itself (e.g. a belief head) exposed as a provider, for comparison. |
| `cascade` | A composite: try A; if `p` < threshold, ask B; then C. For example script → laya_ft → human. Calibrated `p` makes this routing meaningful. |

**Label store:**
- Every answer is stored keyed by `(slot, item_id, provider, provider_version)`. Labels from different providers coexist and are never overwritten.
- A training dataset is defined by `slot → provider selection`, so you can train the same model on "script labels" vs. "laya labels" vs. "human labels" and compare.
- **Gold sets** (human-labeled, held out) are versioned per slot. Every provider is scored on them for accuracy, ECE/calibration, coverage at a fixed error rate, latency and cost.

**Slot inventory (v1):**

| Slot | Latency class | Candidate providers |
|---|---|---|
| `opening_class` (replay or live) | offline / ≤1 s | script (taxonomy rules, §7.3), laya_ft, jev, llm, human |
| `build_cluster_name` (cluster card → name or "new") | offline | llm, human |
| `build_cluster_review` (merge / split / accept) | offline | human |
| `intent_label` (replay windows) | offline | script, laya_ft, llm, human |
| `z_extract` (replay → StrategySpec) | offline | script (default), llm |
| `branch_suggest` (replay → branches) | offline | llm, human |
| `camera_infer` (replay → camera track) | offline | script, small learned model (validated against camera logs, §7.4) |
| `enemy_z_belief` (live opponent model) | ≤50 ms, every few seconds | nn_belief (default), bayes_symbolic, laya_ft |
| `supervisor_branch` (live z switch) | ≤50 ms, every few seconds | rules (constraints, always on), stats_table (§6.8), learned_head, laya_ft. laya_ft also handles **open-vocabulary conditions** written by the user. |
| `strategy_compile` (prompt → z) | interactive | llm, template script |
| `narrate` (trace → answer) | interactive | llm |
| `engagement_outcome` (eval labeling) | offline | script, human |
| `guide_extract` (guide/transcript → claims) | offline | llm, human |
| `claim_verdict` (claim → supported/contradicted/insufficient) | offline | script (replay stats, self-play), human |

**Live slots in training:** if a non-deterministic or slow provider fills a live slot (e.g. `supervisor_branch`), it must also be used during RL, or the policy will train against a different supervisor than the one it plays with. Hosted providers are therefore limited to offline slots in practice.

### 6.7 Executors and the Attention Arbiter

Decision slots answer questions. **Executors** issue commands. Two executors exist in v1:

| Executor | Provider options | Notes |
|---|---|---|
| `micro` | neural policy | Unit control, army posture, worker pulls. Always neural. |
| `macro` | `scripted` / `neural` | Production, supply, tech, expansion, building placement for the active *z* segment. |

**Why a scripted macro option:** practice drills need repeatable timings. A neural policy's build timings drift from game to game, and building placement (wall-offs, depots) is hard to learn but well solved by existing bot code and map-analysis libraries. `scripted` macro also gives a playable bot (P2b) long before full-game BC works.

**Why it must not bypass the human limits:** a scripted macro executor that runs "for free" alongside the micro policy is perfect multitasking, which is exactly the superhuman trait this project removes. So:
- The scripted executor emits ordinary `HumanAction`s (camera jump to base, select building, train…). Each one costs APM tokens and moves the camera away from wherever the fight is.
- The **Attention Arbiter** picks one action per step from all executors' candidates, using one APM bucket and one camera.
- When the micro policy is busy, macro actions wait. Supply blocks and idle production during fights then **emerge** from contention, the way they do for humans. `lapses` in *h* add extra mistakes on top.

**Arbiter policy (v1, rules):**
- Priority goes to the micro executor while an engagement fact is active, with a minimum macro share set by *h* (e.g. B-rank: macro gets ≥25% of actions during fights).
- Otherwise macro has priority.
- Every starvation interval is logged (`arbiter` trace events). That makes "why was I supply-blocked?" answerable.

How the code maps onto these executors, the finer layers below them (fight intents, motor skills) and the plan to make each one swappable: [CONTROL.md](CONTROL.md).

**Adherence modes** (in *z*):
- `adherence: strict` → scripted macro, low variance. For drills: "hit this timing 20 times".
- `adherence: loose` → neural macro. More human variety and adaptation.
- Both run through the same arbiter and interface, so they can be compared directly (§9.2).

### 6.8 Learned symbolic strategy: the Strategy Selector

A symbolic decision-maker that **learns by counting outcomes**. It's the same idea as a fighting-game AI that tries jump / block / attack against an unknown enemy move, records which response worked, and picks the best next time. Formally it's a **contextual bandit** over a lookup table. Every number it uses can be printed.

**Ingredients:**
- **Options:** the branches (*z* segments, or *z* clusters from the taxonomy) that the Plan Supervisor currently allows.
- **Belief:** `P(enemy_z | what I saw)` from the Opponent Model.
- **Context features:** a versioned **feature library** of symbolic facts, e.g. `spawn: close|cross`, map, rush distance, timings of scouted items. Which features matter is discovered, not guessed (§7.6).
- **Statistics:** win counts for `(my_option, enemy_z, context)` cells, with Beta posteriors (wins + 1, losses + 1).

**Decision rule:**

```
EV(option) = Σ_enemy_z  P(enemy_z | seen) × WinRate(option | enemy_z, context)
choose argmax EV          (or Thompson-sample from the posteriors when exploring)
```

**Sparse cells back off to coarser ones.** Each extra context feature splits the data. A cell with few games borrows strength from its parent: `(option, enemy_z, cross_spawn, map)` → `(option, enemy_z, cross_spawn)` → `(option, enemy_z)`. This is hierarchical Bayesian shrinkage, and in practice the table is stored as a tree. Every estimate carries its sample count and interval.

**Example trace record (what "why" questions read):**

```yaml
kind: strategy_select
chosen: t_allin_2fac_vult
belief: {p_nexus_first: 0.83, p_gate_core: 0.12, p_other: 0.05}
context: {spawn: cross, map: <map-id>}
options:
  - {id: t_std_timing,      ev: 0.09, cells: [{enemy: nexus_first, ctx: cross, wr: 0.06, n: 212, ci: [0.03, 0.10]}]}
  - {id: t_allin_2fac_vult, ev: 0.38, cells: [{enemy: nexus_first, ctx: cross, wr: 0.41, n: 87,  ci: [0.31, 0.51]}]}
feature_library: v3        # which features were available
table_version: tvz_stats_v2
```

**Sources of statistics, in layers:**
1. **Replays** (offline). Pros' choices and outcomes. Only needs screp-level data plus spawn and map info, available for all replays.
2. **Bot self-play** (offline). These are true *interventions*: the bot is made to play every option in every context. That fixes the confounding problem below.
3. **Games against this human** (online, optional). A per-opponent table, updated after every game, the way bots in BW tournaments already learn opponent-specific opening win rates.

**Confounding warning.** Replay win rates are observational. Pros all-in *when they judge it favorable*, so `WR(all-in)` from replays overstates what you'd get by always going all-in. Mitigations:
- condition on more context
- weight replays by how likely the option was to be chosen (propensity weighting)
- validate the table with self-play interventions before trusting it

**Modes:**

| Mode | Strategy Selector behavior | Use |
|---|---|---|
| `strict` | Off. The Plan Supervisor follows *z* exactly. | Drills |
| `pro` | Replay + self-play table, no per-opponent learning | "Play what a pro would do here" |
| `adaptive` | Plus a per-opponent table updated between games | Tournament prep: the bot learns your habits, like a real sparring partner would |

### 6.9 Hidden enemy status: HP, shields, energy, upgrades

In the standard BW interface, an enemy unit's HP, shields, energy and upgrade levels are **not shown** unless you select it. Pros click enemy units to check whether they're weak, whether a Science Vessel has irradiate energy, or which upgrades are done. Bots usually skip this by reading game memory. We don't.

**What the observation contains for enemy units:**

| Information | Available? |
|---|---|
| Position, type, visible animations and attacks | Yes, under fog rules |
| Visual damage cues (burning or bleeding buildings at damage thresholds, shield-hit flashes) | Yes, as discrete cues, not numbers |
| Exact HP / shields / energy / upgrades | **Only for the currently inspected unit**, plus a remembered value with its timestamp |

**Three parts:**

1. **Damage tracker (symbolic, always on).** A per-unit Bayesian filter, like a player keeping count in their head:
   - Prior: full HP, plus the expected upgrades given the enemy *z* belief and game time.
   - Subtract expected damage for each observed hit, using weapon damage, armor, the damage-type table and the uncertainty about unknown upgrades.
   - Add regeneration (Zerg HP, Protoss shields) over time.
   - Output: estimated HP with an interval per unit. It's cheap, transparent, and fed to the policy and the trace.
2. **Learned correction (NN, optional).** The policy, or a belief head, takes the tracker's estimates plus the observation history. It's trained with ground-truth HP from resim as an auxiliary target, so it learns what the tracker misses (e.g. medic healing it didn't see).
3. **Active inspection: clicking to check.** `SELECT_CLICK` on an enemy unit reveals its exact status in the observation. It has real costs, which the interface models:
   - Mouse travel and a click (Fitts cost, APM token).
   - It **replaces your current selection**, so getting your army back costs at least one more action (a hotkey recall).
   - The camera must be on the target.

**When to inspect is learned, not hand-coded.** The policy isn't given a separate value-of-information planner:
- **Supervised learning:** human inspection clicks in replays show when pros check (e.g. before committing to a fight).
- **RL:** the reward implicitly prices the information. Checking pays off when it changes the decision (kill the weak unit, fight or retreat), and it costs APM and selection otherwise.
- **Scripted executors:** they don't inspect in v1. An explicit value-of-information rule can be added later as a provider if needed.

**Explainability:** each inspection and each tracker estimate goes into the trace (`belief` events with `head: enemy_status`). That makes questions like "why did you focus that tank?" answerable: "estimated 40±15 HP after 3 volleys; inspected at 7:42: 31 HP".

**Verified: replays do record enemy inspections.** In a sample of 150 STARDATA games (observers excluded), players single-clicked units owned by their opponent in 130 games, about 29 times per game. Supervised learning can see when pros check enemy units. (Ownership was inferred from which player gave each unit orders; confirm with resim.)


### 6.10 Memory of fogged units

Perfect per-unit records of fogged enemies would be superhuman. Forgetting units the moment they enter the fog would be subhuman and easy to exploit with drops and harass. The target is **human-like memory: what was seen stays, but blurs.** Memory is about groups, not unit IDs, and how well something is remembered depends on how it was seen.

**Rules enforced in Perception:**

| Rule | Effect |
|---|---|
| **Unit IDs are scrambled when a unit enters the fog** | Visible units can be tracked continuously. Once fogged, identity is lost. On reappearance, "is that the same dropship?" must be inferred from type, location and timing, and can be wrong. |
| **Buildings persist as last seen** | BW's own UI shows enemy buildings frozen at their last-seen state until re-scouted, so no decay applies. |
| **Tech facts persist** | "Spire exists" or "lurker aspect researched" are facts that aren't forgotten mid-game. |

**Group memory model:** Perception maintains entries `{type?, count_est, count_ci, last_pos, heading, last_seen, confidence}`.
- **How well it's stored depends on how it was seen:**

  | Seen via | Stored |
  |---|---|
  | On screen | Type, count, rough HP |
  | Minimap only | Position and blob size, **no unit type** (the minimap doesn't show it) |
  | Off-screen, not on minimap, or during heavy load | May not be stored at all (missed) |

- **Blur over time:**
  - Location uncertainty grows with elapsed time × unit speed.
  - The count interval widens.
  - HP estimates return to the prior within seconds.
  - Confidence decays until the entry is dropped, unless something refreshes it.
- **Merging:** new sightings are matched to existing entries by plausibility (type, distance reachable since last seen), not by ID. That allows double-counting or merging mistakes, like a human makes.

**Parameters** live in *h* (`limits.memory`, §6.2) and are fitted per rating band where data allows.

**Keeping the policy's own memory honest.** The policy's recurrent core could learn to track fogged units more precisely than the memory model allows.
- The memory model is the primary memory input. The recurrent state is kept small and trained with noise or dropout, so the cheapest way to remember is to use the memory model.
- **Memory probe test (P3+):** train a simple probe to decode the true positions and counts of fogged units from the policy's hidden state. If the probe beats the memory model's precision, the policy is remembering more than allowed. Then shrink or add noise to the core, and re-test. Memory capacity becomes a measured number, not an assumption.

**Not too weak either: exploitability scenarios** (§9.4) such as repeated drops at the same spot, a muta flock leaving and returning, and fake retreats.
- The bot's responses must fall within the human range. Time spent keeping defense home after a drop, for example, is measured from replays.
- Adaptation across games ("he dropped me twice last game") belongs to `adaptive` mode (§6.8).

**Trace:** `memory` events record entry creation, refresh, merging, decay and drop. That answers "why didn't you see the drop coming?": *"last saw 2 dropships at 8:10 (40 s ago); location uncertainty covered all three bases; confidence 0.2."*

---

## 7. Training pipeline

### 7.1 Data sources

| Source | What it is | Notes |
|---|---|---|
| **STARDATA** | ~65k human games (1.16.1), already extracted into states, public S3 bucket | **Primary full-state dataset** (no resim needed). Older meta and mixed skill. Repo archived 2022. ~365 GB compressed: mirror a TvZ subset first. |
| **1.16.1-era archives** (ICCup-era packs, old pro packs) | `.rep` files from the 1.16.1 era | Resim in OpenBW should be exact for these. Second full-state source. |
| **RepMastered** (repmastered.app, by the author of screp) | Large SC:R database with pro, ladder and tournament games; filter by matchup, player, map, APM, date | Best source of modern pro TvZ. **screp-level data only (builds, timings, APM) until SC:R resim passes the P1 go/no-go (§7.2).** Downloads are blocked for unverified email domains unless you donate. **No bulk API: contact the maintainer rather than scraping.** |
| **Liquipedia replay packs** | Tournament packs (ASL and others) linked from event pages | Small, high quality. Good for gold sets and the strategy library. |
| **bwreplays.com, reps.ru, TL.net replay pack threads** | Community archives | Variable quality; many old links are dead. |
| **Camera-logged games** | Games played by contributors with the camera logger running (§7.4) | `.rep` files **never** store screen position, whoever recorded them. Camera ground truth only exists when a logger records it during play. Needed to validate camera inference. |
| Bot self-play games | Generated | RL stage only. |

Check each site's terms before bulk use, and keep provenance (source, URL, date) per replay in the dataset index.

### 7.2 Data processing

1. **Parse** with screp: commands, players, APM, metadata.
2. **Resim** to recover full state per frame.
   - **1.16.1 replays → OpenBW.** The default path.
   - **SC:R (1.18+) replays are a go/no-go measurement in P1, not an assumption.**
     - Blizzard kept the gameplay code: pathing is unchanged, 1.16 replays play in SC:R, and only a few bug fixes (e.g. sprite limits) changed.
     - But the replay format differs (OpenBW's viewer doesn't read 1.18+ files), and lockstep replays amplify any 1-frame difference into total desync.
   - **Go/no-go test:** convert and resim a few hundred SC:R TvZ replays in OpenBW. Check them against screp-derived facts: each build command's success, unit counts at fixed times, final score screen. Accept the replays that stay in sync and record the desync rate.
   - **If no-go, two fallbacks:**
     - (a) Use SC:R replays **only for screp-level data**: build orders and *z*, timings, APM profiles, results. That's enough for the build taxonomy (§7.3) and profile fitting.
     - (b) **Resim inside the SC:R client itself** with a memory reader (the same technique as the Pluto SC:R bridge). This is correct by construction but slower, and it's pinned to one SC:R build.
   - **Desync detection runs on every resim,** not just in P1. Replays that desync are excluded from full-state datasets automatically. Signal: the share of each player's replay commands the engine rejects (human spam keeps it at ~2%; after a desync it climbs past 35% and stays there).
   - **Results (October 2026, TvZ plus all Remastered-format replays), measured with the free StarCraft client's data:**

     | Replays | Desync rate | Notes |
     |---|---|---|
     | STARDATA (1.16-era) | 2.5% | **Go** for full-state training data |
     | Old-patch replays (1.08–1.15) | 77–100% | Usually from minute 6–14; the opening is often still usable |
     | Remastered, 1700-unit games (older SC:R builds) | 0% (24 games) | **Go** |
     | Remastered, 3400-unit games (recent SC:R builds, current ladder) | 19–32% | Partial: unit IDs translate correctly at first, but slot reuse diverges in some games. Under investigation |

     Remastered replays are decoded by `resim/scr_format.py`: zlib sections, the 1.21 command variants, map version 206 and `STRx` strings, and Remastered's larger unit table (1.21 games record its size in the `LMTS` section; unit IDs then use a 13-bit index). Games against the computer can't be simulated: the AI's decisions aren't in the replay.
3. **Per-player observations**, fog-filtered.
   - **v1 (macro), implemented:** `ingest/macro_dataset.py` stores, per clean game and second, each player's own state (resources, supply, units all/completed) and the opponent's units that player can see, plus every production decision.
   - Decisions are the accepted production commands from the resim. Build orders are matched to the buildings that really started, timed at the player's first order for them, so spam clicks and failed orders drop out. Town halls away from the player's other bases are labeled expansions.
   - Camera, positions and the action space below (steps 4–5) come later, with the army and micro models.
4. **Infer the camera.** Replays don't record screen position, so estimate it from click targets, selection boxes and hotkey jumps. Use a heuristic first, then a small model. Measure accuracy against camera-logged games (§7.4).
5. **Map labels to the action space.** Convert human commands into the `HumanAction` format (selections, hotkeys and commands are recorded).
6. **Extract *z*** (deterministic) from each player's own game. This gives hindsight conditioning for supervised learning. Assign each *z* to a build-taxonomy cluster (§7.3).
7. **Extract *h*-conditioning** (rating band, APM and burst statistics) from metadata and command timing.
8. **Fit `HumanProfile.limits`** per rating band from command timing: inter-action gaps, burst sizes, reaction to events (for example, time from first sight of an enemy unit to the first related command).
9. **Label intents.** Start with heuristic rules, then add LLM/Jev classification over windowed summaries. Keep labels above a confidence threshold, using calibrated probabilities. Spot-check by hand.
10. **Belief targets** come from ground truth in the full resim state (enemy opening, composition, tech).

### 7.3 Build taxonomy (automated discovery)

**Goal:** find the builds that humans actually play, name them, and map each name to real *z*'s, with a human only reviewing summaries, never labeling replays one by one.

**Division of labor:** algorithms find patterns, an LLM proposes names, a human approves. An LLM reading raw replays directly doesn't scale (they don't fit in its context) and pattern-matches loosely.

**Pipeline:**

1. **Extract build sequences** (`ingest/build_orders.py`). For each player, list production, tech and expansion items with game time.
   - Replays store commands, not results. A build can fail (blocked spot, no money), be spammed, retried or cancelled.
   - Without simulating, each item gets a **confidence** from evidence in the command stream:

     | Evidence | Example | Result |
     |---|---|---|
     | **Requirement** | Training zerglings proves a Spawning Pool; a Lair morph proves the pool too | `confirmed` |
     | **Gas** | Gas starts at 0 and the client refuses unaffordable orders, so any gas-costing order proves a gas building | `confirmed` |
     | **Ability use** | Using Stim, Siege, Irradiate, Storm, Scanner... proves the research or add-on finished | `confirmed` |
     | **Producer ID** | Select commands carry unit IDs (hotkeys tracked too). The k-th distinct Command Center ID that trains SCVs proves the k-th CC order. Same for Barracks, Gateways, Creep Colonies... | `confirmed` |
     | **Zerg drone ID** | A drone becomes the building and keeps its ID. The ID later used as a hatchery (rally, research, Lair, or as a **larva source**: selecting a hatchery and pressing S sends a one-unit Select, then a Select of its larva, then the morph) confirms it. The same drone later sent to build something else, or given a unit-only order (move, attack, harvest, burrow...), means it was never built | `confirmed` / `not built` |
     | **Cancel** | Cancel commands are recorded but don't name their target; attributed to the latest unconfirmed structure | `cancelled?` (overridden by any positive evidence) |
     | **Retry** | Same building re-ordered at the same spot within seconds | collapsed into one item (`attempts`) |
     | None of the above | | `ordered` |

   - **First measurement (all 15,402 TvZ games, first instance of each item per player):** tech-tree buildings, add-ons and used abilities are 84–100% `confirmed`; first expansion Command Center 94%; expansion Hatcheries 93% / 89% / 81% / 78% for the 1st to 4th (43% before the larva-source evidence). Items with no possible evidence stay `ordered`: supply depots, static defense, passive upgrades (speed, range, armor). The Extractor's 13% `cancelled?` rate is mostly the real "extractor trick".
   - This works for **all** replays, including SC:R and old-patch replays that can't be re-simulated, so the taxonomy doesn't depend on resim.
   - **Command cache** (`ingest/cmdcache.py`): screp parsing and JSON reading are ~90% of the cost, so each replay is parsed once and its relevant commands are stored compactly in `data/interim/cmdcache/` (~40 KB per game). Re-running the extractor after a rule change takes ~15 s for all TvZ instead of ~1 min.
   - Where resim is available (1.16.1 replays, or SC:R replays that pass the desync check), it gives the exact truth. It is used to **measure the extractor's error rate** (missing builds, phantom builds, timing error), which then applies approximately to replays that can't be re-simulated.
   - **Strict drills** (§6.8 modes) use *z*'s from re-simulated replays or high-confidence items only. A phantom hatchery in a drill is worse than one in a win-rate table.
2. **Discover patterns at several levels.**

   | Level | Window | Example labels |
   |---|---|---|
   | Opening | First ~4 min | 12 hatch, 9 pool, overpool |
   | Tech path | ~3–8 min | 2 hatch muta, 3 hatch lurker, 3 hatch muta |
   | Style | Mid/late game | Muta-ling-defiler, hive tech switch |

   Methods:
   - **Opening tree:** sequences share prefixes and then branch. Build a prefix tree and cut it where branches have enough games.
   - **Clustering:** timing-tolerant sequence distance (e.g. edit distance with supply/time windows), then HDBSCAN. This catches the same plan executed in a slightly different order.
3. **Make cluster cards.** For each cluster: the typical build (the medoid), timing ranges per item, game count, win rate by matchup, and links to example replays.
4. **Name clusters** (`build_cluster_name` slot). An LLM gets the card plus reference build definitions (e.g. Liquipedia strategy pages) and proposes a known name, or "unknown: suggested name X".
5. **Distill readable rules.** Train a shallow decision tree that separates the clusters. It produces definitions like *"pool before 3rd hatch, lair before 4:00, spire before 3rd hatch"*. These rules become the deterministic `script` provider for `opening_class`.
6. **Human review** (`build_cluster_review` slot). A person reviews ~30–50 cluster cards: accept, rename, merge, split. They also label a small held-out gold set. Expected effort is about an hour per matchup.
7. **Automatic quality checks:**
   - Cluster stability under bootstrap resampling.
   - Agreement between the readable rules and the cluster assignments.
   - Matchup and race sanity (no Protoss games in a Zerg cluster).
   - Gold-set accuracy of each `opening_class` provider (a slot bake-off, §9.1).

**Output: a versioned taxonomy file** per matchup, e.g. `taxonomy/tvz/v1.yaml`:

```yaml
taxonomy:
  schema: taxonomy/v1
  matchup: TvZ
  perspective: zerg
  version: 1
  method: {extract: resim, cluster: hdbscan_editdist_v1, namer: llm/<model-id>, reviewed_by: [<reviewer-id>]}
  builds:
    - id: z_2hatch_muta
      name: "2 Hatch Muta"
      aliases: ["2 hatch spire", "2h muta"]
      level: tech_path
      parent: z_12hatch          # from the opening tree
      rule: "hatch_count_at_spire_start == 2 && lair_before('4:00') && spire_before(third_hatch)"
      typical: {lair: "3:10–3:40", spire: "4:00–4:30", first_muta: "5:30–6:10"}
      members: {count: 412, replay_index: tvz_modern_v1/clusters/z_2hatch_muta.parquet}
      stability: 0.91
```

**How the policy uses it:** the policy never sees names. "Go 2 hatch muta" → the Strategy Compiler looks up `z_2hatch_muta` → samples a real human *z* from that cluster's members. Twenty practice games then give natural variety *within* the build, the way a human sparring partner would vary.

### 7.4 Camera ground truth: the camera logger

Replays don't record where the player's screen was, so camera inference (§7.2 step 4) needs a separate source of truth to be validated and trained against.

| Option | How | Status |
|---|---|---|
| **BWAPI camera-logger module (1.16.1)** | A BWAPI module loaded while a **human** plays normally. It issues no commands and only logs screen position and mouse position each frame, keyed to the replay file. | **Primary.** Clean ground truth, stable API. Ship it as a tool (`tools/camera_logger`) so community volunteers can build an open camera dataset. |
| SC:R memory reader | The same idea for Remastered, reading the client's memory as the Pluto SC:R bridge does | Optional. Breaks on every patch. Offline and custom games only. |
| Minimap computer vision | Detect the viewport rectangle on the minimap in screen recordings (e.g. player-POV streams) | Research only. Syncing video to replays is hard, and platform terms and copyright make scraping questionable. |

**Logger output** (one file per game, next to the `.rep`):

```yaml
camera_log:
  schema: camera_log/v1
  replay: <replay file hash>
  player: <in-game player slot>
  game_version: "1.16.1"
  frames:            # sampled every frame, stored columnar
    - {frame, screen_x, screen_y, mouse_x, mouse_y}
  clicks:            # every local click, including enemy-unit inspections
    - {frame, button, mouse_x, mouse_y, resolved_unit, selection_after}
```

The mouse and click data also fit the Fitts's-law parameters in *h* (§6.3) and record enemy inspections in case replays don't (§6.9).

**Target:** a few hundred logged TvZ games is enough to measure and train `camera_infer`. Logged games are **not** needed for every training game, only for the camera-inference model and its evaluation.

**Privacy:** logs contain no account data beyond what's already in the replay. Contributors opt in per upload.

### 7.5 Opponent Model training

Guessing the opponent's strategy from scouting is supervised learning with **free labels**: every replay says what the opponent actually did.
- **Input:** player A's fog-filtered observation and facts at time *t*: only what A had seen by then. This needs resim for visibility.
- **Label:** player B's taxonomy cluster (§7.3), known in hindsight from the replay.
- **Output:** `P(enemy_z | seen)` at every time step, plus auxiliary beliefs (enemy army size, tech, proxy yes/no).
- **Evaluation:** calibration (ECE) and accuracy as a function of game time ("how sure is it by 4:00?"), and the effect of scouting ("does seeing the natural change the belief correctly?").
- **Baselines:** a symbolic Bayesian model over scouted facts, and a fine-tuned decision model. Same slot (`enemy_z_belief`), same bake-off.

The policy may also have belief heads as auxiliary training targets. The Strategy Layer consumes the standalone Opponent Model, so beliefs are one versioned, testable component.

### 7.6 Feature discovery: neural networks tell the symbolic layer what to measure

**The problem:** the Strategy Selector's table only knows the context features someone thought to include.
- Example: `WR(standard timing | nexus first)` ≈ 50% overall, but ≈ 0% at cross spawn.
- If `spawn` isn't a feature, the table averages the two and recommends the wrong thing in both cases.
- A neural network trained on the same replays *would* pick up the spawn effect, but it can't say so.

**The loop:** use a neural network as a **detector of what the table is missing**. The table stays the decision-maker of record.

1. **Train a value network** `V(seen, my_option) → P(win)` on replays (and later self-play). It sees everything the player saw, including spawn positions, map and timings. It doesn't need to be explainable.
2. **Compare it to the table.** For every decision point, compute `residual = V − table_estimate`.
   - Near zero everywhere: the table's features capture what matters.
   - Large, systematic residuals: the network knows something the table doesn't.
3. **Find what explains the residuals.** In order of increasing cost:
   - **Existing candidate features:** fit a shallow decision tree on the residuals using the full feature library (including features not yet in the table). If it splits on `spawn`, that's the missing feature. This is the cheap, common case: compute many candidate facts up front and let the data decide which ones the table uses.
   - **Attribution:** compute feature attributions on the value network for high-residual cases (e.g. SHAP or integrated gradients over its structured inputs) to see which inputs drive the disagreement.
   - **New concepts:** if no existing feature explains it, cluster the high-residual situations and show an LLM (and then a human) example games from inside vs. outside the cluster. They propose a concept ("all of these are cross spawn on maps with long rush distance") that gets implemented as a new fact extractor.
4. **Validate before adopting.** A new feature is added only if it reduces residuals and improves the table's prediction of game outcomes **on held-out replays**. Many candidates are tested, so we correct for multiple comparisons. Self-play interventions confirm the effect when possible.
5. **Version everything.** Feature library v*N* → table v*N*. Every strategy decision in the trace records both versions, so explanations stay reproducible.

**The result:** the network does what it's good at (finding patterns in thousands of games), and the symbolic layer does what it's good at (stating them). The explanation for the all-in becomes "cross spawn + nexus first: standard timing wins 6% (n=212), all-in wins 41% (n=87)". That's a statement a pro can check and argue with.

**Precedents:** extracting decision trees from neural policies (VIPER), concept-bottleneck models (networks forced to predict named concepts), and SHAP-style attribution. The residual loop differs in keeping the network *outside* the decision path and using it only to grow the symbolic vocabulary.

### 7.7 Guide mining: pro knowledge as testable hypotheses

Replays show *what* happened. Written guides and video tutorials explain *why* and *when*. **Guides propose; replays and self-play decide.** Guides add three things replays don't provide directly:
1. **Causal claims.** "Turret at 6:30 *because* mutas arrive at 7:00." Replay statistics are confounded (§6.8), so a stated cause is a hypothesis we can test with self-play interventions.
2. **Coverage of rare situations.** Conditional advice ("if nexus first on cross spawn, then…") that has few replay examples.
3. **Named concepts.** Rush distance, gas timing, "did he scout my natural". These are candidate features for the feature library, *before* the residual loop (§7.6) has to discover them.

**Pipeline** (`guide_extract` slot, offline):
1. **Ingest** guides and video transcripts (many good ones are Korean; LLMs handle that). Keep the source URL, author, date and patch era.
2. **Extract claims** with an LLM into a structured format:

   ```yaml
   claim:
     id: <hash>
     source: {url, author, date, kind: article|video_transcript}
     matchup: TvP
     when: ["enemy_z == nexus_first", "spawn == cross"]   # mapped to feature-library facts
     do: t_allin_2fac_vult                                 # mapped to taxonomy/z options
     why: "nexus first is weakest before its first gateway units on long rush distance"
     claimed_effect: {metric: winrate, direction: up}
     new_concepts: ["rush_distance_long"]                  # not yet in the feature library
   ```

3. **Map to our vocabulary.** Conditions map to feature-library facts, and options map to taxonomy *z* clusters. Unmappable terms become candidate concepts for the feature library (same validation as §7.6 step 4).
4. **Test each claim:**
   - Replay statistics for the claim's condition and option, with the confounding caveats.
   - Later, self-play interventions.
   - Verdict: **supported / contradicted / insufficient data**, stored with the evidence.
5. **Use the results:**
   - Supported claims can seed Strategy Selector priors where data is sparse.
   - All claims feed the narrator, so explanations can say "guide X recommends this; data agrees (n=87)", or that the data disagrees.
   - Contradicted claims are reported as well. Guides go out of date, and advice for one skill level may not hold at another.

**Rules:**
- **Store extracted claims and links, not copies of the guide text.** Check each source's license; Liquipedia content is reusable with attribution, but many guides aren't.
- **Use video transcripts only where the platform allows it.** No bulk scraping.

### 7.8 Stages

| Stage | Method | Constraints on? | Output |
|---|---|---|---|
| **T0 Micro curriculum** | Supervised on replay snippets plus RL in short scenarios (15–60 s: muta vs marine/medic/turret, drops, ling runbys) with the scripted macro executor and arbiter active where macro matters | Yes (hard masks) | Validates the human interface. Micro-capable policy. **Fits on a reference workstation.** |
| **T1 Behavior cloning** | Supervised π(intent, action \| obs, z, h, iface) plus belief heads | Yes (soft masks: camera is inferred) | Humanlike, steerable, weak-ish |
| **T2a Segment RL** | RL from mid-game states sampled from replays or BC games, over short horizons (a few minutes), with the same reward as T2b | Yes (hard masks) | Better fights and harass response without full-game cost |
| **T2b Full-game RL** | League self-play against frozen supervised and earlier RL agents. Reward = win + λ_z·adherence + λ_KL·KL(π‖π_BC) + λ_D·discriminator | Yes (hard masks) | Stronger, still humanlike. **Rented compute only (B3).** |
| **T3 Export** | Quantize and distill for runtime latency | n/a | Runtime checkpoint |

The adherence pseudo-reward compares the bot's build order with *z* (supply/time-aligned edit distance) plus distance from the cumulative targets.

---

## 8. Explainability

| Question type | Answered from |
|---|---|
| "Why mutas?" | *z* (it was instructed) and `z_switch` events with their evidence |
| "Why did you all-in?" | `strategy_select` event: beliefs, context, and each option's win rate with sample size and interval (§6.8) |
| "What did you think I was doing?" (strategy) | `belief` history from the Opponent Model, with the facts seen at each point |
| "Why mutas instead of lurkers?" | `intent.alternatives` at the decision frames, plus beliefs at that time |
| "Why did you focus that unit?" | `belief` events with `head: enemy_status` (tracker estimate or inspected value) |
| "Why didn't you defend the drop?" | `constraint` events (masked off-screen, observation delay, empty APM bucket) and the intent at that time |
| "Why were you supply-blocked?" | `arbiter` events: macro actions waiting while micro had priority |
| "What did you think I was doing?" | `belief` history, compared with ground truth after the game |
| "Would you have done X if…?" | Counterfactual rerun: same frame, edited facts or beliefs, compare intent distributions |

The record layer under this (built, for debugging; no narrator on it yet): every layer's decisions of a
game, both sides, in one shape, as `gary/decision_log.py` records saved next to the replay
(`<game>.records.jsonl`). A decision: what the layer saw (in brief), its top alternatives with their
probabilities, the choice and who made it (the model, a draw, a rule, the posture), and what intervened
(each option set aside before the choice, and why: a rule, the money, a busy building, a parked
decision). An order: its fate at the hands (sent; refused by the interface for APM or off screen, or by
the game) and above them (retried, dropped, done, and why), stamped with who had the hands. And who had
the hands, whenever that changes (Gary-T's arbiter; Gary-Z's turns). A learned layer's decision also
carries what its odds rest on: each group of what it saw of the enemy, its money, its history and its plan
zeroed in turn, the chosen option's odds again (`gary/policy/attribution.py`; its own units, supply and the
time aren't zeroed: "no drones, so a drone" is true and useless). Paired branches keep each arm's records,
linked to the rows RL trains on and with the other model's odds of the same choice (`eval/why.py --pairs`,
`--pair N`: the first decision where the arms differ). `python -m eval.why <game> --minute
M` prints one window of it in time order: a why-question about a game is a query, not a script that
re-simulates it. A narrator comes only once the records are complete enough that it couldn't invent a
reason they don't hold.

Rules:
- Narrator answers must cite trace event IDs.
- Counterfactual consistency is tracked as a metric: if an explanation names fact F as the reason, removing F should change the choice.
- Records logged together can still invent causes: a belief logged next to an intent isn't why the intent was chosen. So before narrating a decision, perturb or zero each group of the deciding model's inputs at that moment (milliseconds for models this size) and give the narrator only the groups that changed the choice: counterfactual consistency by construction, not only as a metric.

### 8.1 POV broadcast mode

Bots that read game memory have no point of view: there's no screen to watch. This bot has one **by construction**. Its camera and cursor are part of its state, and every decision is in the trace. POV broadcast mode turns that into a video or stream: the bot's own screen, its mouse, live overlays of what it believes, and commentary on what it's doing **and why**.

The same viewer is also the **primary debugging tool**, so a rough version is built early (see the roadmap).

**1. POV track export.** At every step the bot writes its camera rectangle, cursor position and clicks in the **same format as the camera logger** (`camera_log/v1`, §7.4). Bot POV and human POV recordings are interchangeable, so the viewer can also show camera-logged human games, or a bot and a human side by side.

**2. Rendering:**

| Mode | How | Use |
|---|---|---|
| **Offline** (default) | Replay playback with the camera driven by the POV track; cursor drawn as an overlay | Videos, debugging, highest quality |
| **Live** | In the real client (BWAPI 1.16.1, or SC:R via the bridge), drive the in-game camera to the bot's camera position, draw the cursor, capture with OBS | Streams, sparring sessions with an audience |

**3. Cursor paths are rendered, not decided.** The interface models each click as a Fitts's-law travel time, not a pixel-by-pixel path (§6.3). For video, a smooth human-like path (e.g. minimum-jerk) is drawn between clicks with exactly that duration. **Where and when it clicks is real; the curve between clicks is cosmetic.** Say so in the video description.

**4. Commentary director.** It selects what's worth saying from the live trace, rate-limits it, and hands it to the narrator, then text-to-speech:

| Event source | Example line |
|---|---|
| `strategy_select` | "Nexus first on cross spawn. Standard play wins 6% here, so I'm going vulture all-in." |
| `belief` shift (Opponent Model) | "That's a third hatch, not 2-hatch muta. Relaxing the turrets." |
| `belief` `enemy_status` / inspection | "Checking that tank: 31 HP. Going in." |
| `memory` | "Lost track of the dropships 40 seconds ago. Keeping marines home." |
| `constraint` / `arbiter` | "Supply blocked. I was busy microing against the mutas." |
| `z_switch` | "He's going lurker, so per plan I'm transitioning to mech." |

- **Templates** cover frequent events. **The LLM** writes summaries and longer reasons.
- In live mode the expected lag is ~1–3 s (LLM plus TTS), which is acceptable for commentary.

**5. Honesty rules for commentary:**
- **No hindsight in live commentary.** Lines may use only trace events **up to the current frame**. Otherwise the bot narrates what it knows now as if it had known it then.
- **Post-game review is a separate mode** that may use hindsight, clearly labeled ("looking back, the all-in was wrong because…").
- **Explain only what the records support.** Strategy choices, beliefs, memory, inspections and constraints are explained from records. Micro is explained only at the intent level ("harassing the mineral line"). The narrator must not invent reasons below that.
- Every spoken line keeps its cited trace event IDs in a sidecar file, so a line can be checked after the fact.

**6. Overlays (the "chain of thought" made visual):**
- opponent-strategy belief bars
- current intent and active *z* segment
- APM-budget gauge
- fogged-unit memory drawn on the map (ghost markers with growing uncertainty circles)
- enemy HP estimates with intervals
- Strategy Selector option table at decision moments

Each overlay can be toggled; a clean POV mode shows none.

**7. Practical rules:**
- **Opponents:** bot vs. bot, or humans who agreed to be recorded, in custom games. **Never the ladder.**
- **Blizzard's game video policy:** follow the current terms for videos and streams, including monetization.
- **Platform disclosure:** label it clearly as an AI player with a synthetic voice, following each platform's synthetic-content rules.

---

## 9. Experiment framework

### 9.1 Run manifest

```yaml
run:
  name: bc_vs_rl_tvz_v3
  backend: openbw
  policy: {arch: unit_transformer_gru_m, ckpt: rl/tvz/0042}
  recipe: bc+rl+kl            # bc | bc+rl | bc+rl+kl | bc+rl+kl+disc
  profile: b_rank_terran
  executors:                  # §6.7
    micro: policy
    macro: {provider: scripted/v1, placement: <placement-lib>}   # or: neural
    arbiter: rules/v1
  slots:                      # §6.6 — provider per slot
    opening_class:     {provider: cascade, chain: [script/v2, laya_ft/0007@t=1.8, human]}
    intent_label:      {provider: laya_ft/0007}
    enemy_z_belief:    {provider: nn_belief/0003}
    supervisor_branch: {provider: stats_table, table: tvz_stats_v2, features: v3, mode: pro}
    strategy_compile:  {provider: llm, model: <model-id>, prompt: v3}
  data: {index: tvz_modern_v1, label_sources: {intent_label: laya_ft/0007}}
  budget: {tier: B0, wall_clock: 1h}    # §13
  seeds: [1, 2, 3, 4, 5]
  eval: [suite/core, suite/harass_tvz]
```

A **slot bake-off** is its own run type. It answers the gold set with several providers and reports accuracy, ECE, coverage at a fixed error rate, latency and cost side by side:

```yaml
bakeoff:
  slot: opening_class/v1
  gold: gold/opening_class/tvz_v1      # human-labeled
  providers: [script/v2, laya_zeroshot, laya_ft/0007, jev, openai_decisions, llm/<model-id>]
```

### 9.2 Experiment axes

- Policy architecture and size
- Macro executor (scripted vs. neural) and arbiter rules
- Hard vs. soft masks during BC
- Training recipe and λ weights
- Intent vocabulary and label source
- Supervisor rules vs. learned
- Humanizer parameters and profile fitting method
- Compiler model/prompt
- Data mix (STARDATA vs. modern replays)

### 9.3 Metrics

| Metric | Measures |
|---|---|
| Elo vs. opponent pool (classic bots, earlier checkpoints, Pluto as ceiling) | Strength per profile |
| **Adherence**: build-order edit distance and target error vs. *z*, plus variance over N runs | Steerability, repeatability (G3) |
| **Humanlikeness**: accuracy of a human-vs-bot classifier on replays (lower is better); APM/burst/reaction distribution distance from the target band | G2 |
| **Harass response suite** score (see below) | G1 |
| Explanation counterfactual consistency, % of narrator answers with valid citations | G5 |
| Runtime latency per step, cost | Feasibility |

### 9.4 Scenario suites

Scenarios are reproducible starting states with fixed opponent scripts. A scenario passes when the bot's response falls **within the human distribution**, not when it wins.
- `harass_tvz/muta_5_at_7m30`: 5 mutas hit the main while the Terran takes a third
- `harass_tvz/ling_runby_natural`
- `harass_pvz/muta_vs_cannon_sair`
- `drop_zvt/2_dropship_main_and_nat`
- `exploit/repeat_drop_same_spot`, `exploit/muta_leave_and_return`, `exploit/fake_retreat` (memory: neither always fooled nor perfectly prepared, §6.10)
- …

### 9.5 Human evaluation

Blind sessions: pros play a mix of the bot (various profiles) and real human sparring partners, then rate "was this a human?" and "was this useful practice?".

---

## 10. Deployment

| Target | Path | Notes |
|---|---|---|
| Headless eval | OpenBW | Primary dev loop |
| Play vs. humans (1.16.1) | BWAPI 4.4 client (32-bit module, IPC to 64-bit inference process, like Pluto) | Stable, official API |
| Play vs. humans (SC:R) | BWAPI-compatible bridge, same approach as Pluto-AI-Starcraft-Remaster | Pinned to one SC:R build and breaks on patches. Unofficial: **offline and custom games only, never ladder.** Check the bridge's license before reuse. |

---

## 11. Repository layout (proposed)

```
schemas/      protobuf contracts (strategy, profile, action, intent, trace) — versioned
adapters/     openbw_env (vectorized), bwapi_client (C++ shim), scr_bridge
interface/    human interface, masks, InterfaceState, observation delay, attention arbiter — ONE implementation (C++ core + pybind11)
executors/    scripted macro executor (build queue from z, building placement), executor API
perception/   observation encoding, fact extractors
ingest/       replay ingest (screp), resim, camera inference, labeling, profile fitting
taxonomy/     build discovery (prefix tree, clustering), cluster cards, rule distillation, versioned taxonomy files
tools/        camera_logger (BWAPI module), dataset utilities
compiler/     prompt→z (LLM), replay→z
supervisor/   plan supervisor (rules engine)
strategy/     strategy selector, win-rate tables, feature library, feature-discovery loop (value net, residual trees)
opponent/     opponent model (belief over enemy z), enemy-status damage tracker, baselines
guides/       guide/transcript ingest, claim extraction, vocabulary mapping, claim verdicts
slots/        slot schemas, providers (script, decision_model, llm, human, cascade), label store, calibrators, bake-off runner
annotate/     small labeling UI for the human provider queue
policy/       model definitions, encoders, heads
train/        bc, rl, league, rewards
trace/        bus, store, query API, narrator
broadcast/    POV viewer, overlays, commentary director, TTS, OBS/live-client integration
eval/         harness, scenarios, metrics, dashboards
configs/      run manifests, profiles, strategy library
docs/         this document, ADRs
data/         local datasets, gitignored except README and sources.yaml (provenance index)
tests/        tests; fixtures/replays/ holds a small committed set of redistributable replays
```

---

## 12. Roadmap

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **P0 Infra** | OpenBW vectorized env, human interface v1, trace schema, eval harness with classic bots | 1k headless games/hour on a reference workstation (to be confirmed by the benchmark); interface parity test (train vs. runtime) passes |
| **P1 Data** | Replay → (obs, HumanAction, z, h) pipeline; desync detection; camera logger tool and first logged games; camera inference; profile fitting | **SC:R resim go/no-go decided** (desync rate measured, fallback chosen if needed, §7.2); camera inference accuracy measured against camera-logged games (§7.4) |
| **P1b Taxonomy** | TvZ build taxonomy v1 (§7.3): discovery, naming, readable rules, human review | Reviewed `taxonomy/tvz/v1.yaml`; cluster stability and gold-set accuracy reported; `opening_class` bake-off done |
| **P1c Strategy stats** | Feature library v1 (seeded from guide mining, §7.7); replay win-rate table (§6.8) from screp-level data; first feature-discovery pass with decision trees on outcomes (§7.6); first batch of guide claims with verdicts | Table beats the no-context baseline at predicting held-out outcomes; at least one discovered feature validated (e.g. spawn) |
| **P2 Micro** | T0 curriculum in muta/marine/drop scenarios | Harass suite: responses within human distribution for ≥2 profiles |
| **P2b First playable** | Scripted macro executor + micro policy + arbiter, all under the shared human interface; strict-adherence *z* from the taxonomy; optional `pro` mode with the P1c table, a first Opponent Model (symbolic Bayesian or fine-tuned decision model), and open-vocabulary branch conditions via a fine-tuned decision model | Plays full TvZ games vs. humans on BWAPI 1.16.1; build timings within tolerance in ≥18/20 unharassed runs; supply blocks/idle production under harass within the human band |
| **P2c Debug viewer** | Rough POV viewer: replay + POV track + trace overlays (no voice) | Used routinely to debug P2b games |
| **P3 BC full game** | T1 policy with *z*, *h*, intent and belief heads; NN Opponent Model (§7.5); value network and residual feature-discovery loop (§7.6); supervisor; compiler | Adherence target met for 10 library strategies; humanlikeness classifier ≤ X%; memory probe within the memory model's precision (§6.10) |
| **P4 RL** | T2a segment RL (workstation), then T2b league (rented compute) | Elo gain at equal humanlikeness and adherence |
| **P5 Explain** | Query API, narrator, counterfactual tool | Counterfactual consistency ≥ target |
| **P5b Broadcast** | Commentary director, TTS, polished overlays; offline videos first, then live | Commentary passes the no-hindsight check; every spoken line traceable to event IDs |
| **P6 Deploy** | BWAPI client; SC:R bridge (offline) | Pro blind-test sessions |

---

## 13. Compute budgets

Start small and scale up. Every run manifest declares a budget tier, so results from different budgets are never compared as if they were equal.

**Reference workstation** (the baseline for B0–B2; budgets are stated in wall-clock hours on this class of machine):
- Modern desktop CPU with 12+ cores
- 64 GB RAM
- One consumer GPU with 16 GB VRAM, on a recent CUDA/PyTorch build
- 1–2 TB NVMe storage
- OpenBW and the training stack are Linux-first. On Windows, run them under **WSL2** with CUDA passthrough. The BWAPI/SC:R play path stays on Windows.

Contributors with different hardware should report their throughput benchmark (P0) alongside results.

| Tier | Budget | Purpose |
|---|---|---|
| **B0** | 1 h on a reference workstation | Smoke tests, bake-offs, micro-scenario learning curves |
| **B1** | Overnight (~10 h) on a reference workstation | Data preprocessing, small supervised runs, longer micro RL |
| **B2** | Weekend (~48 h) on a reference workstation | Small full-game supervised policy (≤15M params); segment RL (T2a) from mid-game states. **No full-game RL.** |
| **B3** | Rented multi-GPU (budget set later) | Full-game RL league (T2b), larger models |

**What fits in B0 (1 hour).** All throughput numbers are **estimates to be measured in P0**. OpenBW's headless speed in our setup is the key unknown.

| Experiment | Setup | Expected outcome |
|---|---|---|
| **B0-a Slot bake-off** | `opening_class` gold set of ~300–500 human labels. Providers: script, Laya zero-shot, Laya fine-tuned (421M fits on 16 GB with bf16 and small batches), hosted Jev/OpenAI (cents) | A complete comparison table in well under an hour. **The bottleneck is human labeling time, not compute.** |
| **B0-b Micro RL** | `harass_tvz/muta_5_vs_marines` scenario on a small map. Policy of ~1–5M params (small unit transformer). PPO with ~16–20 parallel OpenBW envs on CPU, learner on GPU. Human interface **on**. | Roughly 10⁶–10⁷ agent steps. Enough to see learning curves and compare 2–3 humanizer settings (e.g. C vs. B profile). Not enough for polished micro. |
| **B0-c Supervised smoke test** | ~10–30M param model on a preprocessed TvZ subset (1–2k games, prepared in B1) | Action-prediction accuracy curves. Confirms the data pipeline end to end. |

**Model size cap for prototypes (B0–B2): 5–15M parameters.** Scale up only after the pipeline and metrics are proven.

**Not in B0–B2:** full-game RL, leagues, Pluto-scale models (315M params). Full-length recurrent training on one 16 GB GPU runs into memory limits and is far too slow.

**Storage:** STARDATA alone is ~365 GB compressed. Extracted features for modern replays add more. Plan for 1–2 TB of fast disk (NVMe).

---

## 14. Risks and open questions

| Risk / question | Mitigation / next step |
|---|---|
| RL compute for full-game BW is unknown (Pluto's budget is unpublished) | Prove value in P2 (micro) first; scale model size gradually; supervised-only fallback |
| SC:R replays may not resim in OpenBW 1.16.1 (format differs; lockstep amplifies any difference) | P1 go/no-go with per-replay desync detection; fallbacks: screp-level data only for SC:R, or resim inside the SC:R client (§7.2) |
| Scripted macro reintroduces superhuman multitasking | Scripted executor emits `HumanAction`s through the shared arbiter and interface; never issues commands directly (§6.7) |
| Arbiter rules feel artificial (e.g. macro never slips, or slips too much) | Fit macro share during fights from replays per rating band; compare against neural macro in the harass suite |
| Soft masks in BC leak illegal actions into the policy | Hard masks in RL and at runtime; track masked-action rate during the transition |
| Camera inference quality limits how human the interface is | Collect camera-logged games with the BWAPI logger (§7.4) to train and evaluate the inference model |
| Too few volunteers for camera logging | A few hundred games is enough; recruit from the community and make the logger zero-config |
| Discovered clusters don't match community build names (or split them oddly) | Multi-level taxonomy, human review step, aliases; treat names as a UI layer over clusters |
| Replay win rates are confounded (pros pick options when they're favorable) | Propensity weighting, richer context, validation with self-play interventions (§6.8) |
| Win-rate table cells get too sparse as features are added | Hierarchical back-off to parent cells; report n and intervals; add features only when held-out prediction improves |
| Feature discovery finds spurious features | Held-out validation, multiple-comparison correction, self-play confirmation (§7.6) |
| Guide claims are outdated, wrong, or for a different skill level | Claims are hypotheses with verdicts, never rules; record date/patch era and author (§7.7) |
| Guide copyright / platform terms | Store claims and links, not text; check licenses; no bulk transcript scraping |
| ~~Enemy-unit inspections may not be recorded in replays~~ | Verified: they are recorded (§6.9) |
| Command-only build orders contain phantom or missing items | Evidence-based confidence per item; error rate measured against resim; strict drills use resim-validated *z* only (§7.3) |
| Fitts's-law parameters off for BW-specific habits (e.g. hotkeyed camera jumps) | Fit per rating band from logger mouse data; compare click-timing distributions with replays |
| Opponent Model leaks hidden information | Inputs restricted to the fog-filtered observation; test that beliefs don't change when unseen enemy state is altered |
| Policy's recurrent core rebuilds superhuman memory of fogged units | Small, noisy recurrent state; memory probe test (§6.10) |
| Memory model too weak → exploitable by repeated harass | Exploitability scenarios compared against human replay behavior; `adaptive` mode across games |
| Human memory parameters hard to measure | Fit indirectly from behavior (reaction to returning units, defense kept home after drops); treat as profile knobs |
| Humanlike vs. strong is a trade-off | Make it explicit via λ_KL, λ_D and profile conditioning; track both metrics together |
| Intent labels are noisy | Confidence filtering, manual audit set, and iterating on intent vocab versions |
| SC:R bridge fragility and ToS | Primary target is 1.16.1; SC:R is offline only; no ladder |
| Broadcast commentary sounds insightful but isn't faithful | No-hindsight rule, cited event IDs per line, explanations limited to recorded decisions (§8.1) |
| Video/streaming terms (game publisher, platforms) | Follow Blizzard's video policy; disclose AI player and synthetic voice; recorded opponents must consent |
| Pluto is closed source | Use only as an evaluation opponent. Do not distill from it (license, and it would import superhuman habits). |
| Decision-model probabilities aren't trustworthy out of the box (Laya ships over-confident; OpenAI's are self-reported) | Fit a calibrator per provider and slot on the gold set; report ECE in every bake-off |
| No bulk access to modern pro replays | Start with STARDATA plus Liquipedia packs; ask the RepMastered maintainer about research access |
| OpenBW headless throughput is unmeasured (native Linux vs. WSL2) | First P0 benchmark; it decides the B0-b numbers |
| OpenBW engine repo has no license file (default: all rights reserved) | Use it as an external dependency only; don't vendor or redistribute its source; ask the maintainers to clarify |
| **Decided:** first matchup | **TvZ**, Gary as Terran first, then as Zerg (self-play). The first instance only: Gary plays every race and matchup (§1); the other matchups' replays are already in the inventory (PvZ 19k, PvT 19k, PvP 7.6k, ZvZ 6.5k, TvT 3.1k games) |
| **Decided:** compute | Start at B0 (1 h on a reference workstation) and scale up through the tiers (§13) |
| **Open:** STARDATA subset strategy | Full dataset is ~365 GB compressed; decide whether to mirror it or stream a TvZ subset |

---

## 15. Role summary: which tool does what

| Tool | Used for | Not used for |
|---|---|---|
| **Neural policy (BC → RL)** | In-game perception, micro, intent selection; macro when `adherence: loose` | Bypassing the shared APM budget/camera |
| **Scripted macro executor** | Repeatable build execution and building placement for strict drills and the first playable bot | Issuing commands outside the arbiter/interface |
| **Symbolic layer** (*z*, supervisor, intents, facts, trace) | Steering, branch logic, accountability, explanations | Low-level control |
| **Strategy Selector** (learned symbolic, §6.8) | Strategy switches by counted win rates under beliefs and context; per-opponent adaptation | Inferring hidden information (that's the Opponent Model) |
| **Opponent Model** (NN) | `P(enemy z \| what I saw)` and other beliefs | Making the strategy decision itself |
| **Value network** (NN, offline) | Detecting what the win-rate table is missing (§7.6) | Being in the live decision path |
| **LLM** | Prompt → *z* compiler, replay branch suggestions, intent labeling, narrator | Real-time decisions |
| **Decision models** (Laya/Kev local; Jev/OpenAI Decisions hosted) | Providers in decision slots: replay labeling, opening classification, a learned alternative for `supervisor_branch` (local models only) | Per-step control (that's the policy's job), RL teacher (it would only re-learn the replay labels through a lossy text view) |
| **Python scripts** | Default provider for deterministic slots (z extraction, simple classifiers); baseline in every bake-off | — |
| **Humans** | Gold sets, branch approval, blind evaluation | Bulk labeling |
| **Classic bots / Pluto** | Evaluation opponents, strength reference | Training data |

---

## References

- Pluto: https://github.com/tscmoo/pluto
- Pluto reverse-engineering notes: https://github.com/hwkim3330/pluto-re
- Pluto SC:R bridge: https://github.com/chan22222/Pluto-AI-Starcraft-Remaster
- Pluto ladder incident: https://www.dexerto.com/gaming/ai-bot-destroys-top-starcraft-pros-after-invading-ladder-and-using-absurd-strategies-3410863/
- AlphaStar: Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", Nature 2019
- STARDATA: https://github.com/TorchCraft/StarData · https://arxiv.org/abs/1708.02139
- screp: https://github.com/icza/screp
- OpenBW / headless eval example: https://github.com/lukecameron/starcraft-ai
- Human-like agent constraints: https://arxiv.org/abs/2505.20011
- Jev: https://www.datacamp.com/blog/system-one-models-jev · https://www.mindstudio.ai/blog/jev-system-one-model-launch
- Laya (open decision model): https://huggingface.co/convaiinnovations/laya
- OpenAI Decisions API: https://startupfortune.com/openais-new-decisions-api-looks-a-lot-like-its-answer-to-a-rivals-jev-model/
- Open Jev rivals overview: https://trilogyai.substack.com/p/jev-open-decision-models
- RepMastered: https://repmastered.app
- Liquipedia replay packs: https://liquipedia.net/starcraft/Template:ReplayPack
- VIPER (decision-tree extraction from neural policies): Bastani, Pu, Solar-Lezama, "Verifiable Reinforcement Learning via Policy Extraction", NeurIPS 2018
- Concept Bottleneck Models: Koh et al., ICML 2020
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
- Fitts's law: Fitts, "The information capacity of the human motor system in controlling the amplitude of movement", 1954; MacKenzie's Shannon formulation, 1992
- OpenBW replay viewer (1.16.1 only): http://www.openbw.com/replay-viewer/
- SC:R keeps Brood War gameplay code: https://www.vice.com/en/article/starcraft-remastered-doesnt-fix-brood-wars-broken-perfection/ · https://starcraft.fandom.com/wiki/StarCraft:_Remastered
