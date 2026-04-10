# Higher Abstraction Vocabularies (HAV)

**A vocabulary engine for agents and humans to communicate about complex concepts with precision.**

> "Stigmergy" compresses "indirect coordination through environment modification where agents communicate by leaving traces" into one word. The fleet needs thousands of these compressions.

## What It Is

HAV is a structured vocabulary of **142 terms** across **28 domains** -- from confidence fusion to circadian rhythms, from circuit breakers to the paradox of choice, from free energy principle to autopoiesis. Every term has:

- **Short definition** -- one line
- **Full description** -- with context and fleet integration
- **Examples** -- real-world and fleet-specific
- **Cross-domain bridges** -- how this term connects to concepts in other domains
- **Abstraction level** -- from concrete implementation to meta-pattern
- **Tags** -- for discovery and filtering

## Why It Matters

### For Agents

Agents need shared vocabulary to communicate precisely. Instead of sending a 500-byte JSON payload explaining "I observed the path was blocked and therefore I am considering alternatives and will select the one with highest confidence", an agent sends one word: `deliberation`. Both agents look up `deliberation` in HAV and share the full semantics.

```python
from vocab import HAV

hav = HAV()
hav.explain("stigmergy")
# => "Indirect coordination through environment modification..."
#    Includes medium properties, examples (ant trails, git commits), and fleet integration.

hav.search("memory that fades")
# => [("forgetting-curve", 0.3), ("episodic-decay", 0.3), ...]

hav.suggest("I need to handle too many requests")
# => [("backpressure", 0.2), ("rate-limit", 0.15), ...]
```

### For Humans

HAV is a field guide to the concepts that power autonomous agent systems. Read it to understand:

- **Why** the fleet uses harmonic mean for confidence (not arithmetic mean)
- **What** biological metaphors map to software patterns (dopamine = confidence, serotonin = trust)
- **How** agents coordinate without central control (stigmergy, consensus, gossip)
- **When** to use satisficing vs optimizing (energy constraints)
- **Where** phase transitions emerge in fleet behavior (critical mass, percolation)
- **What** the free energy principle means for agent perception and action

## Domains (28)

| Domain | Terms | Covers |
|--------|-------|--------|
| **uncertainty** | 7 | Confidence, harmonic mean, trust, Bayesian update, entropy, calibration |
| **memory** | 8 | Working, episodic, semantic, procedural, forgetting, consolidation, chunking |
| **coordination** | 8 | Stigmergy, consensus, swarm, emergence, gossip, leader election, quorum |
| **biological** | 8 | Instinct, apoptosis, circadian, neurotransmitter, membrane, enzyme, Hebbian |
| **learning** | 7 | Exploration/exploitation, credit assignment, transfer, curriculum, overfitting |
| **architecture** | 7 | Actor model, circuit breaker, bulkhead, event sourcing, backpressure, sidecar |
| **evolution** | 6 | Natural selection, fitness landscape, punctuated equilibrium, co-evolution, speciation |
| **networks** | 6 | Small-world, scale-free, hub, percolation, cascade failure, clustering |
| **control-theory** | 5 | Feedback loop, setpoint, hysteresis, overshoot, dead zone |
| **game-theory** | 5 | Nash equilibrium, prisoner's dilemma, mechanism design, tragedy of commons |
| **optimization** | 5 | Gradient descent, local minimum, simulated annealing, convergence, multi-objective |
| **ecology** | 5 | Niche, keystone species, symbiosis, competitive exclusion, succession |
| **creativity** | 5 | Analogy, divergent/convergent thinking, combinatorial explosion, constraint relaxation |
| **failure-modes** | 5 | SPOF, robustness, anti-fragility, common-mode failure, brittleness |
| **complexity** | 5 | Edge of chaos, self-organization, autocatalysis, autopoiesis, phase transition |
| **linguistics** | 5 | Compositionality, metaphor, grounding problem, pragmatics, ambiguity |
| **mathematics** | 5 | Harmonic mean, exponential decay, Welford, topological sort, Hamming distance |
| **probability** | 4 | Prior, base rate fallacy, conjunction fallacy, regression to mean |
| **economics** | 4 | Opportunity cost, marginal cost, externalities, market equilibrium |
| **thermodynamics** | 4 | Entropy production, free energy principle, dissipative structure, negentropy |
| **scaling** | 4 | Superlinear, diminishing returns, critical mass, tipping point |
| **spatial** | 4 | Attention tiles, A*, Manhattan distance, spatial hashing |
| **communication** | 4 | Grounding, speech acts, information bottleneck, context window |
| **decision** | 4 | Satisficing, multi-armed bandit, minimax, paradox of choice |
| **emotion** | 3 | Valence-arousal, emotional contagion, anticipation |
| **metacognition** | 3 | Introspection, theory of mind, metacognitive monitoring |
| **temporal** | 3 | Deadline urgency, causal chains, heartbeats |
| **security** | 3 | Least privilege, sandbox, graceful degradation |

## CLI

```bash
python3 src/cli.py search "memory that fades"
python3 src/cli.py explain stigmergy
python3 src/cli.py suggest "gradually reduce options until one remains"
python3 src/cli.py bridge confidence from uncertainty to biological
python3 src/cli.py explore    # random term with related terms
python3 src/cli.py domains    # all domains with term counts
python3 src/cli.py stats      # statistics
```

## Programmatic API

```python
from vocab import HAV
hav = HAV()

# Search across all domains
for domain, term, score in hav.search("trust between agents"):
    print(f"{term.name} ({domain}): {term.short}")

# Full explanation
print(hav.explain("harmonic-mean-fusion"))

# Suggest terms for natural-language intent
for ns, term, score in hav.suggest("handle too many requests"):
    print(f"  {term.name} -- {term.short}")

# Cross-domain bridges
for ns, term in hav.bridge("confidence", from_domain="uncertainty", to_domain="biological"):
    print(f"{ns}/{term.name}: {term.short}")
```

## Key Design Decisions

### Harmonic Mean, Not Arithmetic Mean

When fusing confidence, the fleet uses harmonic mean: `1/(1/a + 1/b)`. Any uncertain source drags down the ensemble. Arithmetic mean would let a confident-but-wrong source overpower uncertain-but-correct ones.

### Five Abstraction Levels

Every term has a level from Concrete (implementation) to Meta (cross-domain pattern). Search and filter by level to find the right granularity for your needs.

### Cross-Domain Bridges

Terms reference each other across domains. "Dopamine IS confidence" isn't just metaphor -- the bridge connects `neurotransmitter` (biological) to `confidence` (uncertainty), indicating shared mathematical structure.

### Fleet-Integrated

Every term connects to actual fleet crates. `stigmergy` -> cuda-stigmergy. `circuit-breaker` -> cuda-circuit. `deliberation` -> cuda-deliberation. The vocabulary IS the documentation.

## Part of the Lucineer Fleet

[The Fleet](https://github.com/Lucineer/the-fleet) | [Cocapn](https://github.com/Lucineer/cocapn-ai) | [Deckboss](https://github.com/Lucineer/deckboss)

## License

MIT
