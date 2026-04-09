"""Higher Abstraction Vocabularies (HAV) — Domain vocabulary for complex concepts.

Provides vocabulary to talk about high-level patterns and behaviors:
compass constructions, fold operations, spin dynamics, tiling logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum


class AbstractionLevel(Enum):
    CONCRETE = "concrete"           # Implementation details
    PATTERN = "pattern"             # Design patterns
    BEHAVIOR = "behavior"           # Observable behaviors
    DOMAIN = "domain"               # Domain-specific concepts
    META = "meta"                   # Cross-domain abstractions


@dataclass
class Term:
    """A high-level vocabulary term."""
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    level: AbstractionLevel = AbstractionLevel.PATTERN
    domain: str = "general"
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, query: str) -> float:
        """Simple fuzzy match score."""
        q = query.lower()
        score = 0.0
        if q in self.name.lower():
            score += 0.5
        if q in self.description.lower():
            score += 0.3
        for ex in self.examples:
            if q in ex.lower():
                score += 0.1
        for rel in self.relationships:
            if q in rel.lower():
                score += 0.05
        return min(score, 1.0)


@dataclass
class Namespace:
    """A vocabulary namespace for a specific domain."""
    name: str
    description: str = ""
    terms: Dict[str, Term] = field(default_factory=dict)
    
    def define(self, name: str, description: str = "", **kwargs) -> Term:
        term = Term(name=name, description=description, domain=self.name, **kwargs)
        self.terms[name] = term
        return term
    
    def lookup(self, name: str) -> Optional[Term]:
        return self.terms.get(name)
    
    def search(self, query: str, threshold: float = 0.1) -> List[Tuple[Term, float]]:
        """Search terms by query string."""
        results = []
        for term in self.terms.values():
            score = term.matches(query)
            if score >= threshold:
                results.append((term, score))
        return sorted(results, key=lambda x: -x[1])
    
    def count(self) -> int:
        return len(self.terms)
    
    def all_terms(self) -> List[Term]:
        return list(self.terms.values())


class Vocabulary:
    """Collection of namespaces forming a complete vocabulary."""
    
    def __init__(self, name: str):
        self.name = name
        self._namespaces: Dict[str, Namespace] = {}
    
    def add_namespace(self, name: str, description: str = "") -> Namespace:
        ns = Namespace(name=name, description=description)
        self._namespaces[name] = ns
        return ns
    
    def get_namespace(self, name: str) -> Optional[Namespace]:
        return self._namespaces.get(name)
    
    def define(self, namespace: str, term_name: str, description: str = "", **kwargs) -> Term:
        if namespace not in self._namespaces:
            self.add_namespace(namespace)
        return self._namespaces[namespace].define(term_name, description, **kwargs)
    
    def search(self, query: str, threshold: float = 0.1) -> List[Tuple[str, Term, float]]:
        """Search across all namespaces."""
        results = []
        for ns_name, ns in self._namespaces.items():
            for term, score in ns.search(query, threshold):
                results.append((ns_name, term, score))
        return sorted(results, key=lambda x: -x[2])
    
    def abstract(self, concrete: Dict[str, Any]) -> Dict[str, str]:
        """Map concrete implementation to HAV vocabulary.
        
        Example: {"algorithm": "quick-sort", "pivot": "median-of-three"}
        -> {"pattern": "divide-and-conquer", "strategy": "median-pivot"}
        """
        result = {}
        for key, value in str(concrete).split(","):
            key = key.strip().strip("{}'")
            value = value.strip().strip("{}'")
            matches = self.search(value, threshold=0.05)
            if matches:
                ns, term, score = matches[0]
                result[term.name] = f"{ns}/{term.name}"
        return result
    
    def stats(self) -> dict:
        return {
            "name": self.name,
            "namespaces": len(self._namespaces),
            "total_terms": sum(ns.count() for ns in self._namespaces.values()),
            "by_namespace": {name: ns.count() for name, ns in self._namespaces.items()},
        }


# Built-in vocabulary: mathematical constructions
def math_vocabulary() -> Vocabulary:
    """Standard vocabulary for mathematical constructions."""
    v = Vocabulary("mathematical-constructions")
    
    geo = v.add_namespace("geometry", "Geometric constructions and transformations")
    geo.define("compass-construction", "Construction using only compass and straightedge",
               examples=["bisect angle", "construct perpendicular", "golden ratio"],
               relationships=["straightedge-construction", "classical-geometry"])
    geo.define("fold-operation", "Transformation through bending or folding",
               examples=["paper folding", "origami mathematics", "reflection"],
               relationships=["symmetry", "transformation"])
    geo.define("tiling-pattern", "How pieces fit together to fill space",
               examples=["penrose tiles", "tessellation", "periodic tiling"],
               relationships=["symmetry", "geometry"])
    
    dyn = v.add_namespace("dynamics", "Motion and transformation patterns")
    dyn.define("spin-dynamics", "Rotational behaviors and properties",
               examples=["angular momentum", "rotation group", "fermion spin"],
               relationships=["symmetry", "quantum"])
    dyn.define("drift-motion", "Slow systematic change in a parameter",
               examples=["parameter drift", "concept evolution", "paradigm shift"],
               relationships=["trend", "adaptation"])
    
    comp = v.add_namespace("computation", "Computational abstractions")
    comp.define("divide-and-conquer", "Split problem into subproblems",
               examples=["merge sort", "binary search", "quick sort"],
               relationships=["recursion", "parallel"])
    comp.define("memoization", "Cache computed results for reuse",
               examples=["dynamic programming", "Fibonacci cache", "LRU cache"],
               relationships=["optimization", "state"])
    
    return v


# Built-in vocabulary: agent behaviors
def agent_vocabulary() -> Vocabulary:
    """Vocabulary for agent behaviors and coordination."""
    v = Vocabulary("agent-behaviors")
    
    coord = v.add_namespace("coordination", "Multi-agent coordination patterns")
    coord.define("stigmergy", "Indirect coordination through environment modification",
                 examples=["ant trails", "wikipedia edits", "git commits"],
                 relationships=["decentralized", "swarm"])
    coord.define("consensus", "Agreement among agents on a shared state",
                 examples=["raft protocol", "paxos", "voting"],
                 relationships=["coordination", "distributed"])
    coord.define("deliberation", "Structured debate leading to decision",
                 examples=["agent deliberation", "jury process", "peer review"],
                 relationships=["consensus", "reasoning"])
    
    learn = v.add_namespace("learning", "Agent learning patterns")
    learn.define("exploration", "Trying new actions to discover better strategies",
                 examples=["epsilon-greedy", "curiosity-driven", "random walk"],
                 relationships=["reinforcement", "discovery"])
    learn.define("exploitation", "Using known best actions",
                 examples=["greedy policy", "local optimum", "best practice"],
                 relationships=["optimization", "convergence"])
    
    return v
