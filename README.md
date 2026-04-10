# Higher Abstraction Vocabularies (HAV)

**An exhaustive vocabulary engine for agents and humans to communicate about complex concepts with precision.**

> "Stigmergy" compresses "indirect coordination through environment modification where agents communicate by leaving traces" into one word. The fleet needs thousands of these compressions.

## What It Is

HAV is a structured vocabulary of **248 terms** across **50 domains** spanning psychology, biology, economics, game theory, information theory, complexity science, ethics, epistemology, systems thinking, and more. Every term has:

- **Short definition** -- one line
- **Full description** -- with context, fleet integration, and real-world examples
- **Cross-domain bridges** -- how this term connects to concepts in other domains
- **Abstraction level** -- Concrete(0) -> Pattern(1) -> Behavior(2) -> Domain(3) -> Meta(4)
- **Antonyms** -- where applicable (e.g. robustness vs fragility, intrinsic vs extrinsic)
- **Tags** -- for discovery and filtering

## Why It Matters

### For Agents

Agents need shared vocabulary to communicate precisely. Instead of a 500-byte JSON payload explaining "I observed the path was blocked and therefore I am considering alternatives and will select the one with highest confidence", an agent sends one word: `deliberation`. Both agents look up `deliberation` in HAV and share the full semantics.

```python
from vocab import HAV

hav = HAV()

# Explain any term with full context
hav.explain("stigmergy")
# => "Indirect coordination through environment modification..."
#    Includes fleet integration, biological parallel, and examples.

# Search across all 50 domains
hav.search("memory that fades over time")
# => [("forgetting-curve", 0.3), ("episodic-decay", 0.3), ...]

# Natural language intent matching
hav.suggest("I need to handle too many requests")
# => [("backpressure", 0.2), ("rate-limit", 0.15), ...]

# Cross-domain bridges -- navigate between fields
hav.bridge("confidence", from_domain="uncertainty", to_domain="biological")
# => dopamine IS confidence (shared mathematical structure)
```

### For Humans

HAV is a field guide to the concepts that power autonomous agent systems. It connects biology to computer science, economics to game theory, philosophy to engineering.

### For Researchers

50 domains organized for systematic exploration. Each term's bridges create a cross-referenced knowledge graph. The CLI supports browsing, searching, and discovering connections between fields.

## The 50 Domains

| # | Domain | Terms | Core Concepts |
|---|--------|-------|---------------|
| 1 | **uncertainty** | 7 | Confidence, harmonic mean, trust, entropy, calibration |
| 2 | **memory** | 8 | Working, episodic, semantic, procedural, forgetting |
| 3 | **coordination** | 8 | Stigmergy, consensus, swarm, emergence, gossip |
| 4 | **biological** | 8 | Instinct, apoptosis, circadian, neurotransmitter |
| 5 | **psychology** | 8 | Confirmation bias, dunning-kruger, cognitive dissonance |
| 6 | **learning** | 7 | Exploration/exploitation, credit assignment, transfer |
| 7 | **architecture** | 7 | Actor model, circuit breaker, bulkhead, sidecar |
| 8 | **evolution** | 6 | Natural selection, fitness landscape, co-evolution |
| 9 | **networks** | 6 | Small-world, scale-free, hub, cascade failure |
| 10 | **pattern-recognition** | 6 | Feature extraction, clustering, anomaly detection |
| 11 | **design-patterns** | 6 | Observer, strategy, command, sidecar, adapter |
| 12 | **decision-theory** | 6 | Expected value, maximin, satisficing, pareto-optimal |
| 13 | **tradeoffs** | 6 | Exploration-exploitation, speed-accuracy, CAP |
| 14 | **epistemology** | 5 | Justified true belief, Gettier, epistemic humility |
| 15 | **biology** | 5 | Homeostasis, allostasis, metabolism, immune response |
| 16 | **control-theory** | 5 | Feedback loop, setpoint, hysteresis, dead zone |
| 17 | **game-theory** | 5 | Nash equilibrium, prisoner's dilemma, mechanism design |
| 18 | **optimization** | 5 | Gradient descent, local minimum, simulated annealing |
| 19 | **ecology** | 5 | Niche, keystone species, competitive exclusion |
| 20 | **creativity** | 5 | Analogy, divergent/convergent thinking, constraint relaxation |
| 21 | **failure-modes** | 5 | SPOF, robustness, anti-fragility, brittleness |
| 22 | **complexity** | 5 | Edge of chaos, self-organization, autopoiesis |
| 23 | **linguistics** | 5 | Compositionality, metaphor, grounding problem |
| 24 | **resilience** | 5 | Graceful degradation, circuit breaker, bulkhead |
| 25 | **information-theory** | 5 | Shannon entropy, mutual information, Kolmogorov complexity |
| 26 | **systems-thinking** | 5 | Emergent property, leverage point, compensating feedback |
| 27 | **concurrency** | 5 | Deadlock, race condition, livelock, eventual consistency |
| 28 | **measurement** | 5 | Latency, throughput, SLA, observability, tech debt |
| 29 | **perception** | 5 | Sensory adaptation, change blindness, object permanence |
| 30 | **philosophy-of-science** | 4 | Paradigm shift, falsifiability, Occam's razor |
| 31 | **communication-theory** | 4 | Shannon-Weaver model, information bottleneck |
| 32 | **security** | 4 | Least privilege, zero trust, confused deputy |
| 33 | **mathematics** | 5 | Harmonic mean, exponential decay, Welford's algorithm |
| 34 | **probability** | 4 | Base rate fallacy, conjunction fallacy, regression to mean |
| 35 | **economics** | 4 | Opportunity cost, marginal cost, externalities |
| 36 | **thermodynamics** | 4 | Entropy production, free energy principle, dissipative structure |
| 37 | **scaling** | 4 | Superlinear scaling, diminishing returns, tipping point |
| 38 | **philosophy-of-mind** | 4 | Functionalism, Chinese room, embodiment, extended mind |
| 39 | **motivation** | 4 | Intrinsic/extrinsic, goal hierarchy, drive reduction |
| 40 | **ethics** | 4 | Trolley problem, alignment problem, distributed responsibility |
| 41 | **time** | 4 | Real-time, time-to-live, causality, warm-up |
| 42 | **obsolescence** | 4 | Software rot, strangler pattern, bus factor |
| 43 | **temporal** | 3 | Temporal window, lead time, grace period |
| 44 | **emotion** | 3 | Valence-arousal, emotional contagion, anticipation |
| 45 | **metacognition** | 3 | Introspection, theory of mind, metacognitive monitoring |
| 46 | **semantics** | 3 | Reference, truth-conditional, compositionality |
| 47 | **identity** | 3 | Decentralized identity, provenance, attestation |
| 48 | **morphology** | 3 | Self-similarity, fractal, structural coupling |
| 49 | **decision** | 4 | Multi-armed bandit, minimax, paradox of choice |
| 50 | **spatial** | 4 | A* pathfinding, Manhattan distance, spatial hashing |

## Key Cross-Domain Bridges

The most powerful feature: terms connect across domains, revealing shared structure:

- **Dopamine IS confidence** -- `neurotransmitter` (biological) <-> `confidence` (uncertainty)
- **Serotonin IS trust** -- `neurotransmitter` (biological) <-> `trust` (uncertainty)
- **Stigmergy IS git** -- `stigmergy` (coordination) <-> `event-sourcing` (architecture)
- **Homeostasis IS setpoint** -- `homeostasis` (biology) <-> `setpoint` (control-theory)
- **CAP theorem IS exploration-exploitation** -- `consistency-availability` (tradeoffs) <-> `exploration-exploitation` (tradeoffs)
- **Free energy principle IS active inference** -- `free-energy-principle` (thermodynamics) <-> `perception` (perception)
- **Chinese room IS grounding problem** -- `chinese-room` (philosophy-of-mind) <-> `grounding-problem` (linguistics)
- **Anti-fragility IS learning from failure** -- `anti-fragility` (failure-modes) <-> `credit-assignment` (learning)

## Five Abstraction Levels

```
Level 0: CONCRETE    -- Implementation details (sensor types, config values)
Level 1: PATTERN     -- Reusable solutions (circuit breaker, feedback loop)
Level 2: BEHAVIOR    -- Observable phenomena (confirmation bias, overshoot)
Level 3: DOMAIN      -- Foundational concepts (natural selection, Nash equilibrium)
Level 4: META        -- Cross-domain patterns (autopoiesis, instrumentalism)
```

Search and filter by level to find the right granularity for your context.

## CLI

```bash
python3 src/cli.py search "how systems fail"
python3 src/cli.py explain anti-fragility
python3 src/cli.py suggest "too many messages overwhelming the system"
python3 src/cli.py bridge confidence from uncertainty to biological
python3 src/cli.py explore    # random term with related terms
python3 src/cli.py domains    # all 50 domains with term counts
python3 src/cli.py stats      # statistics
```

## Programmatic API

```python
from vocab import HAV

hav = HAV()

# Search across all 50 domains
for domain, term, score in hav.search("trust between agents"):
    print(f"{term.name} ({domain}): {term.short}")

# Full explanation with context
print(hav.explain("harmonic-mean-fusion"))

# Natural language intent matching
for ns, term, score in hav.suggest("handle overload"):
    print(f"  {term.name} -- {term.short}")

# Cross-domain bridge discovery
for ns, term in hav.bridge("confidence", from_domain="uncertainty", to_domain="biological"):
    print(f"{ns}/{term.name}: {term.short}")

# Filter by abstraction level
for ns, term in hav.all(lever=Level.META):
    print(f"{term.name}: {term.short}")
```

## Design Philosophy

1. **Bridges over boundaries** -- Terms connect across domains, revealing shared mathematical structure
2. **Fleet-integrated** -- Every term connects to actual cuda-* crates
3. **Multi-audience** -- Useful for agents, developers, and researchers
4. **Grounded in practice** -- Not academic definitions; terms include fleet-specific examples
5. **Growing** -- This is a living vocabulary. Contributions welcome.

## Part of the Lucineer Fleet

[The Fleet](https://github.com/Lucineer/the-fleet) | [Cocapn](https://github.com/Lucineer/cocapn-ai) | [Deckboss](https://github.com/Lucineer/deckboss)

## License

MIT
