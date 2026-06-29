"""
question_generator.py
─────────────────────────────────────────────────────────────────
Production-grade Adaptive Interview Question Generator for AI HR Agent.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  CandidateProfile                                           │
  │     │  (skills, experience, JD, seniority tier)            │
  │     ▼                                                       │
  │  QuestionEngine (offline mode — always available)          │
  │     │                                                       │
  │     ├──► ConceptualGenerator   (Tier-aware theory Qs)      │
  │     ├──► CodingGenerator       (Pattern-based code Qs)     │
  │     ├──► ScenarioGenerator     (Rich situational Qs)       │
  │     ├──► SystemDesignGenerator (Architecture / design Qs)  │
  │     └──► BehavioralGenerator   (STAR-format soft-skill Qs) │
  │                                                             │
  │  LLMQuestionGenerator  (advanced, optional)                │
  │     │  (calls Anthropic Claude API or any OpenAI-compat)   │
  │     └──► falls back to QuestionEngine on error             │
  └─────────────────────────────────────────────────────────────┘

Seniority tiers (auto-detected from experience_years):
  JUNIOR  < 2 yrs  → fundamentals, conceptual
  MID     2-5 yrs  → applied, debugging, code quality
  SENIOR  5+ yrs   → architecture, system design, trade-offs

Drop-in replacement for hr_agent.QuestionGenerator.
"""

from __future__ import annotations

import hashlib
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════

class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class QuestionType(str, Enum):
    CONCEPTUAL   = "conceptual"
    CODING       = "coding"
    SCENARIO     = "scenario"
    SYSTEM_DESIGN = "system_design"
    BEHAVIORAL   = "behavioral"


class SeniorityTier(str, Enum):
    JUNIOR = "junior"    # < 2 yrs
    MID    = "mid"       # 2–5 yrs
    SENIOR = "senior"    # 5+ yrs


@dataclass
class CandidateProfile:
    skills:           List[str]
    experience_years: float    = 0.0
    jd_text:          str      = ""
    name:             str      = "Candidate"

    @property
    def tier(self) -> SeniorityTier:
        if self.experience_years < 2:
            return SeniorityTier.JUNIOR
        if self.experience_years < 5:
            return SeniorityTier.MID
        return SeniorityTier.SENIOR


@dataclass
class Question:
    skill:      str
    q_type:     QuestionType
    difficulty: Difficulty
    text:       str
    hint:       str = ""
    follow_up:  str = ""

    def to_dict(self) -> Dict:
        return {
            "skill":      self.skill,
            "type":       self.q_type.value,
            "difficulty": self.difficulty.value,
            "question":   self.text,
            "hint":       self.hint,
            "follow_up":  self.follow_up,
        }


# ═══════════════════════════════════════════════════════════════
# 2. KNOWLEDGE BANK
#    Keyed by canonical skill name (matches SkillRegistry output).
#    Each skill carries:
#      • concepts[tier] → list of concepts to ask about
#      • coding_patterns → list of problem patterns
#      • scenarios → production-realistic situational Qs
#      • design_topics → architecture / system-design prompts
# ═══════════════════════════════════════════════════════════════

_KNOWLEDGE: Dict[str, Dict] = {

    "Python": {
        "concepts": {
            "junior":  ["GIL", "list comprehensions", "mutability vs immutability",
                        "scope (LEGB)", "decorators basics", "built-in data types"],
            "mid":     ["generators vs iterators", "metaclasses", "context managers",
                        "memory management / gc", "asyncio fundamentals",
                        "type hints & mypy", "descriptors"],
            "senior":  ["CPython internals", "PyPy trade-offs", "extension modules (ctypes/Cython)",
                        "concurrency models (threads vs processes vs async)",
                        "performance profiling (cProfile, py-spy)"],
        },
        "coding_patterns": [
            "Implement a thread-safe LRU cache without using OrderedDict",
            "Write a decorator that retries a function N times with exponential back-off",
            "Implement a lazy, memory-efficient CSV processor for files > 10 GB",
            "Design a rate-limiter class using the token-bucket algorithm",
            "Write a context manager that temporarily patches environment variables",
            "Implement a topological sort using Python generators",
        ],
        "scenarios": [
            "A FastAPI service starts leaking memory after ~6 hours under sustained load. Walk me through your investigation.",
            "A pandas pipeline that processes 1 M rows takes 45 minutes. You need it under 3 minutes. How do you approach this?",
            "You inherit a 4,000-line Python monolith with no tests. A business-critical bug has been filed. How do you fix it safely?",
            "A background Celery worker silently drops tasks under heavy load with no errors logged. How do you diagnose and fix it?",
            "Your team's CI pipeline runs Python tests in 40 minutes. How do you cut it to under 5?",
        ],
        "design_topics": [
            "Design a Python-based data-ingestion microservice that handles schema evolution gracefully.",
            "How would you architect a high-throughput event-processing system in Python — what concurrency model and why?",
            "Design an internal Python library for feature engineering that 10 teams can use without breaking each other's code.",
        ],
    },

    "Machine Learning": {
        "concepts": {
            "junior":  ["bias-variance trade-off", "overfitting vs underfitting", "train/val/test split",
                        "precision vs recall", "gradient descent", "feature scaling"],
            "mid":     ["regularisation (L1/L2)", "cross-validation strategies", "class imbalance techniques",
                        "hyperparameter tuning (grid search vs Bayesian)", "feature importance",
                        "model calibration"],
            "senior":  ["distribution shift / concept drift", "model monitoring in production",
                        "experiment reproducibility", "responsible AI / fairness metrics",
                        "MLOps pipelines", "AutoML limitations"],
        },
        "coding_patterns": [
            "Implement k-fold cross-validation from scratch without sklearn",
            "Write code to detect and visualise feature collinearity",
            "Build a simple gradient boosting classifier step-by-step",
            "Implement SMOTE oversampling for a 1:100 class imbalance",
            "Write a model evaluation harness that detects data leakage",
        ],
        "scenarios": [
            "Your churn model has 94% accuracy in dev, but the business says it's not working in production. What do you investigate?",
            "A client wants real-time ML inference with < 50 ms P99 latency. Walk through your serving strategy.",
            "You're asked to improve a model but you can't get more data. What's your approach?",
            "Your new model outperforms the baseline in all offline metrics but performs worse after A/B test deployment. Why might this happen?",
            "A regulated industry needs an explainable model. Your team has a black-box XGBoost. How do you handle this?",
        ],
        "design_topics": [
            "Design an end-to-end ML pipeline for a recommendation system serving 10 M users.",
            "How would you build a feature store that supports both batch training and online serving?",
            "Design a model-monitoring system that detects drift and triggers retraining automatically.",
        ],
    },

    "Deep Learning": {
        "concepts": {
            "junior":  ["activation functions", "forward vs back propagation", "batch normalisation",
                        "dropout regularisation", "convolution operation"],
            "mid":     ["vanishing/exploding gradients", "transfer learning", "attention mechanism",
                        "learning rate schedules", "data augmentation strategies"],
            "senior":  ["transformer architecture internals", "mixture of experts", "distillation",
                        "quantisation and pruning", "distributed training (DDP, FSDP)"],
        },
        "coding_patterns": [
            "Implement a custom PyTorch Dataset and DataLoader for a streaming text corpus",
            "Write a training loop with gradient clipping, mixed precision, and checkpoint saving",
            "Build a simple self-attention head from scratch in NumPy",
        ],
        "scenarios": [
            "Your neural network's validation loss plateaus at epoch 5 while training loss keeps falling. What do you do?",
            "You need to classify images with only 300 labelled examples. Describe your complete modelling strategy.",
            "An LSTM model has low overall error but fails catastrophically on rare but business-critical events.",
        ],
        "design_topics": [
            "Design a cost-efficient fine-tuning pipeline for a 7B-parameter LLM on proprietary documents.",
            "How would you architect a real-time video classification system with < 100 ms latency?",
        ],
    },

    "Data Structures and Algorithms": {
        "concepts": {
            "junior":  ["arrays vs linked lists", "stacks and queues", "binary search",
                        "Big-O notation", "recursion basics", "hash maps"],
            "mid":     ["trees (BST, AVL, trie)", "graph traversal (BFS/DFS)",
                        "dynamic programming patterns", "sorting algorithms trade-offs",
                        "sliding window technique", "two-pointer approach"],
            "senior":  ["segment trees / Fenwick trees", "network flow algorithms",
                        "amortised analysis", "cache-oblivious algorithms",
                        "probabilistic data structures (Bloom filter, HyperLogLog)"],
        },
        "coding_patterns": [
            "Implement an LRU cache with O(1) get and put",
            "Find the longest substring without repeating characters",
            "Serialize and deserialize a binary tree",
            "Design an algorithm to find all strongly connected components",
            "Implement a min-heap from scratch",
            "Find the kth largest element in an unsorted stream",
        ],
        "scenarios": [
            "You have a graph of 100 M nodes representing a social network. How do you find shortest paths efficiently?",
            "A search feature needs to return autocomplete suggestions in < 10 ms. Which data structure do you use and why?",
            "Given an infinite stream of integers, how do you maintain the running median?",
        ],
        "design_topics": [
            "Design a URL shortener — what data structures underpin it and why?",
            "Design an in-memory key-value store with O(1) average time for all operations.",
        ],
    },

    "SQL": {
        "concepts": {
            "junior":  ["SELECT / WHERE / GROUP BY / ORDER BY", "JOINs (inner, left, right)",
                        "aggregate functions", "primary vs foreign keys", "indexes basics"],
            "mid":     ["window functions", "CTEs vs subqueries", "query optimisation",
                        "explain / execution plans", "transactions and isolation levels"],
            "senior":  ["partitioning strategies", "materialised views", "MVCC internals",
                        "query cost estimation", "replication lag and read consistency"],
        },
        "coding_patterns": [
            "Write a query to find the second highest salary in each department",
            "Identify customers who ordered every product in a given category",
            "De-duplicate a table that has no unique key",
            "Compute a 7-day rolling average of daily revenue using only window functions",
            "Write a recursive CTE to flatten an adjacency-list hierarchy",
        ],
        "scenarios": [
            "A query that ran in 2 s takes 12 minutes after a schema migration. Walk me through your diagnosis.",
            "Two analysts run the same report and get different totals. How do you find the root cause?",
            "You need to migrate a 500 GB production table to a new schema with zero downtime.",
        ],
        "design_topics": [
            "Design the schema for a multi-tenant SaaS billing system.",
            "How would you design a time-series database schema that supports fast range queries?",
        ],
    },

    "Docker": {
        "concepts": {
            "junior":  ["image vs container", "Dockerfile instructions", "port mapping",
                        "volume mounting", "docker run flags"],
            "mid":     ["multi-stage builds", "layer caching strategy", "Docker Compose networking",
                        "health checks", "image security scanning"],
            "senior":  ["OCI spec", "container runtimes (containerd, runc)",
                        "rootless Docker", "BuildKit optimisation", "image signing (cosign)"],
        },
        "coding_patterns": [
            "Write a multi-stage Dockerfile that compiles a Go binary and produces a distroless final image",
            "Write a Docker Compose file for a web app, PostgreSQL, and Redis with health checks",
            "Optimise a 3 GB Python image to under 200 MB while preserving all dependencies",
        ],
        "scenarios": [
            "A container works locally but crashes in production with exit code 137. What happened and how do you fix it?",
            "A containerised app can't reach its database in Compose. Walk through your troubleshooting.",
            "Your image build takes 15 minutes and pulls 2 GB each time. How do you fix the CI pipeline?",
        ],
        "design_topics": [
            "Design a container image build pipeline that enforces security scanning and provenance.",
        ],
    },

    "Kubernetes": {
        "concepts": {
            "junior":  ["Pods vs Deployments vs ReplicaSets", "Services (ClusterIP/NodePort/LoadBalancer)",
                        "ConfigMaps and Secrets", "kubectl basics"],
            "mid":     ["Rolling updates & rollbacks", "HPA / VPA", "RBAC",
                        "PersistentVolumeClaims", "Ingress controllers"],
            "senior":  ["etcd internals", "kube-scheduler extension points",
                        "service mesh (Istio/Linkerd)", "Operator pattern",
                        "multi-cluster federation"],
        },
        "coding_patterns": [
            "Write a Kubernetes Deployment YAML with readiness/liveness probes and resource limits",
            "Configure an HPA that scales on custom metrics from Prometheus",
            "Create an RBAC policy that gives a service account read-only access to secrets in a single namespace",
        ],
        "scenarios": [
            "A pod keeps entering CrashLoopBackOff. Walk me through your investigation step by step.",
            "A rolling deployment is causing a surge of 502 errors. How do you configure zero-downtime deployments?",
            "Your cluster nodes are at 90% CPU but only 3 pods are running. What's happening?",
        ],
        "design_topics": [
            "Design a multi-tenant Kubernetes platform for 50 independent engineering teams.",
            "How would you architect a disaster-recovery strategy for a Kubernetes-based production cluster?",
        ],
    },

    "AWS": {
        "concepts": {
            "junior":  ["EC2 instance types", "S3 storage classes", "IAM roles vs users",
                        "VPC subnets", "Route 53 basics"],
            "mid":     ["Lambda cold starts", "SQS vs SNS vs EventBridge",
                        "RDS multi-AZ vs read replicas", "CloudFront CDN",
                        "Cost Explorer and tagging"],
            "senior":  ["AWS Well-Architected pillars", "Transit Gateway",
                        "Service Control Policies", "PrivateLink",
                        "CDK vs Terraform trade-offs"],
        },
        "coding_patterns": [
            "Write a Lambda function triggered by S3 that resizes images and saves them to another bucket",
            "Design a Terraform module for a VPC with public/private subnets and NAT Gateway",
            "Write a CDK stack that deploys an ECS Fargate service with auto-scaling",
        ],
        "scenarios": [
            "Your AWS bill doubled this month. Walk me through identifying the cost driver.",
            "An S3-triggered Lambda is occasionally missing files. How do you make it reliable?",
            "Your EC2-based API is receiving 10× normal traffic. Walk through your scaling strategy with no downtime.",
        ],
        "design_topics": [
            "Design a globally available, multi-region e-commerce platform on AWS.",
            "How would you architect a real-time analytics pipeline ingesting 1 M events/sec on AWS?",
        ],
    },

    "React": {
        "concepts": {
            "junior":  ["JSX", "functional vs class components", "props vs state",
                        "useState / useEffect", "key prop in lists"],
            "mid":     ["useCallback / useMemo", "Context API vs Redux",
                        "code splitting / lazy loading", "controlled vs uncontrolled inputs",
                        "React reconciliation"],
            "senior":  ["Concurrent mode / Suspense", "Server Components (Next.js App Router)",
                        "micro-frontend architecture", "render performance profiling",
                        "custom renderer / Fabric in React Native"],
        },
        "coding_patterns": [
            "Implement an infinite scroll component that fetches the next page when the user reaches the bottom",
            "Build a debounced search input that calls an API and handles race conditions",
            "Create a custom hook that syncs state with localStorage",
            "Implement an accessible modal dialog from scratch (focus trap, Escape key, aria-*)",
        ],
        "scenarios": [
            "Users report the app freezes on scroll. How do you profile and fix the render cycle?",
            "A global state change is triggering unexpected re-renders everywhere. How do you trace and fix it?",
            "The bundle size has grown to 4 MB. Walk me through your optimisation plan.",
        ],
        "design_topics": [
            "Design a micro-frontend architecture for a platform with 5 independent teams.",
            "How would you build a design-system component library used across 20 React apps?",
        ],
    },

    "Cybersecurity": {
        "concepts": {
            "junior":  ["CIA triad", "common OWASP vulnerabilities", "phishing",
                        "symmetric vs asymmetric encryption", "password hashing"],
            "mid":     ["threat modelling (STRIDE)", "SIEM use cases",
                        "zero-trust principles", "TLS handshake", "JWT security"],
            "senior":  ["supply-chain attacks", "kernel exploitation basics",
                        "adversarial ML", "cloud-native security posture",
                        "red vs blue vs purple team"],
        },
        "coding_patterns": [
            "Write a Python script to scan a list of URLs for open redirects",
            "Implement a simple password strength checker following NIST SP 800-63B",
            "Write a Bash script that audits running processes for unsigned binaries",
        ],
        "scenarios": [
            "You detect unusual outbound traffic at 3 AM from an internal server. Walk me through incident response.",
            "A developer accidentally committed AWS credentials to a public GitHub repo. What are your immediate steps?",
            "Your web app passed last quarter's pentest but a new CVE just dropped for your framework. What do you do?",
        ],
        "design_topics": [
            "Design a zero-trust network architecture for a remote-first company.",
            "How would you build a secrets management system for 200 microservices?",
        ],
    },

    "Natural Language Processing": {
        "concepts": {
            "junior":  ["tokenisation", "bag-of-words", "TF-IDF", "stemming vs lemmatisation",
                        "n-grams", "word embeddings (Word2Vec)"],
            "mid":     ["attention mechanism", "BERT fine-tuning", "NER", "text classification pipelines",
                        "sentence embeddings", "RAG basics"],
            "senior":  ["pre-training objectives (masked LM, causal LM)", "instruction tuning",
                        "RLHF", "long-context handling", "hallucination mitigation"],
        },
        "coding_patterns": [
            "Implement TF-IDF from scratch without sklearn",
            "Write a text classification pipeline using HuggingFace Transformers",
            "Build a simple RAG system with a vector store and re-ranking step",
        ],
        "scenarios": [
            "Your NER model performs well in English but poorly on code-switched text. How do you fix it?",
            "A chatbot frequently hallucinates product facts. How do you reduce this in production?",
            "You need to search 10 M documents semantically in < 100 ms. Design the system.",
        ],
        "design_topics": [
            "Design a production RAG system that stays up to date with a live document corpus.",
            "How would you architect a multi-lingual customer support bot with strict PII handling?",
        ],
    },

    "DevOps": {
        "concepts": {
            "junior":  ["CI/CD pipeline stages", "blue-green vs canary deployments",
                        "log aggregation", "uptime SLA vs SLO"],
            "mid":     ["infrastructure as code (IaC)", "immutable infrastructure",
                        "observability pillars (metrics/logs/traces)",
                        "secrets rotation", "on-call runbooks"],
            "senior":  ["GitOps", "platform engineering",
                        "chaos engineering", "FinOps",
                        "internal developer platform design"],
        },
        "coding_patterns": [
            "Write a GitHub Actions workflow that runs tests, builds a Docker image, and deploys to EKS on merge to main",
            "Write a Terraform module for a self-healing EC2 Auto Scaling group",
        ],
        "scenarios": [
            "Your deployment pipeline takes 45 minutes and blocks 12 engineers. How do you cut it to 5 minutes?",
            "Production is down. Your monitoring shows nothing. How do you triage?",
            "The team wants to adopt GitOps but has 30 microservices in different states of maturity. How do you roll it out?",
        ],
        "design_topics": [
            "Design an Internal Developer Platform (IDP) for 200 engineers across 5 teams.",
            "How would you design an observability strategy for a 100-service microservice mesh?",
        ],
    },
}

# ── Default / generic knowledge for skills NOT in _KNOWLEDGE ──

def _generic_knowledge(skill: str) -> Dict:
    return {
        "concepts": {
            "junior":  [f"{skill} fundamentals", f"core {skill} syntax", f"basic {skill} tooling"],
            "mid":     [f"intermediate {skill} patterns", f"{skill} best practices", f"testing in {skill}"],
            "senior":  [f"advanced {skill} architecture", f"{skill} performance tuning", f"scalability in {skill}"],
        },
        "coding_patterns": [
            f"Implement a canonical {skill} pattern from memory and walk me through your design decisions.",
        ],
        "scenarios": [
            f"You encounter a critical production bug in your {skill} codebase. Walk me through your investigation.",
            f"You're asked to onboard a junior engineer to {skill}. What are the 3 most important concepts you'd cover first?",
        ],
        "design_topics": [
            f"Design a scalable system component using {skill} — what trade-offs would you make at 10× scale?",
        ],
    }


# ═══════════════════════════════════════════════════════════════
# 3. INDIVIDUAL QUESTION TYPE GENERATORS
# ═══════════════════════════════════════════════════════════════

class ConceptualGenerator:
    """Generates theory / knowledge-check questions."""

    _TEMPLATES = [
        "Explain {concept} in {skill} and describe a real-world scenario where it matters.",
        "What are the trade-offs of {concept} in {skill}? When would you choose an alternative?",
        "How does {concept} work under the hood in {skill}? What are the performance implications?",
        "A junior engineer on your team doesn't understand {concept}. How would you explain it with a concrete example?",
        "Describe a production bug or design mistake related to {concept} in {skill} that you've seen. How was it resolved?",
        "What changed between older and modern approaches to {concept} in {skill}? Why was the change needed?",
        "How do you test code that relies heavily on {concept} in {skill}?",
    ]

    def generate(self, skill: str, tier: SeniorityTier, n: int, used: Set[str]) -> List[Question]:
        kb       = _KNOWLEDGE.get(skill, _generic_knowledge(skill))
        concepts = list(kb["concepts"].get(tier.value, kb["concepts"]["junior"]))
        random.shuffle(concepts)

        questions = []
        templates = list(self._TEMPLATES)
        random.shuffle(templates)

        difficulty = {
            SeniorityTier.JUNIOR: Difficulty.EASY,
            SeniorityTier.MID:    Difficulty.MEDIUM,
            SeniorityTier.SENIOR: Difficulty.HARD,
        }[tier]

        for i in range(min(n, len(concepts))):
            text = templates[i % len(templates)].format(
                concept=concepts[i], skill=skill
            )
            if text in used:
                continue
            used.add(text)
            questions.append(Question(
                skill=skill,
                q_type=QuestionType.CONCEPTUAL,
                difficulty=difficulty,
                text=text,
                hint=f"Focus on {concepts[i]}.",
                follow_up=f"Can you give a concrete code example demonstrating {concepts[i]}?",
            ))
        return questions


class CodingGenerator:
    """Generates coding / implementation questions."""

    _WRAPPERS = [
        "{pattern}",
        "Live coding: {pattern}. Think aloud as you work.",
        "On a whiteboard: {pattern}. Analyse time and space complexity.",
        "Code review scenario: A colleague submitted code that {pattern}. Identify issues and refactor it.",
        "Pair-programming exercise: {pattern}. How would you approach TDD here?",
    ]

    def generate(self, skill: str, tier: SeniorityTier, n: int, used: Set[str]) -> List[Question]:
        kb       = _KNOWLEDGE.get(skill, _generic_knowledge(skill))
        patterns = list(kb.get("coding_patterns", []))
        random.shuffle(patterns)

        difficulty = {
            SeniorityTier.JUNIOR: Difficulty.EASY,
            SeniorityTier.MID:    Difficulty.MEDIUM,
            SeniorityTier.SENIOR: Difficulty.HARD,
        }[tier]

        questions = []
        wrappers  = list(self._WRAPPERS)
        random.shuffle(wrappers)

        for i, pattern in enumerate(patterns[:n]):
            text = wrappers[i % len(wrappers)].format(pattern=pattern)
            if text in used:
                continue
            used.add(text)
            questions.append(Question(
                skill=skill,
                q_type=QuestionType.CODING,
                difficulty=difficulty,
                text=text,
                hint="Start with the brute-force solution, then optimise.",
                follow_up="What's the time and space complexity? How would this change at 10× scale?",
            ))
        return questions


class ScenarioGenerator:
    """Generates production / debugging scenario questions."""

    def generate(self, skill: str, tier: SeniorityTier, n: int, used: Set[str]) -> List[Question]:
        kb        = _KNOWLEDGE.get(skill, _generic_knowledge(skill))
        scenarios = list(kb.get("scenarios", []))
        random.shuffle(scenarios)

        # Seniors get harder variants
        difficulty = Difficulty.HARD if tier == SeniorityTier.SENIOR else Difficulty.MEDIUM

        questions = []
        for scenario in scenarios[:n]:
            if scenario in used:
                continue
            used.add(scenario)
            questions.append(Question(
                skill=skill,
                q_type=QuestionType.SCENARIO,
                difficulty=difficulty,
                text=scenario,
                hint="Walk me through your process step by step.",
                follow_up="What monitoring or safeguards would you put in place to prevent this in future?",
            ))
        return questions


class SystemDesignGenerator:
    """Generates system design / architecture questions (mainly for seniors)."""

    _JUNIOR_WRAPPER = (
        "Even as a junior, you'll encounter design discussions. "
        "Describe at a high level: {topic}"
    )

    def generate(self, skill: str, tier: SeniorityTier, n: int, used: Set[str]) -> List[Question]:
        kb     = _KNOWLEDGE.get(skill, _generic_knowledge(skill))
        topics = list(kb.get("design_topics", []))
        random.shuffle(topics)

        if not topics:
            return []

        difficulty = {
            SeniorityTier.JUNIOR: Difficulty.MEDIUM,
            SeniorityTier.MID:    Difficulty.HARD,
            SeniorityTier.SENIOR: Difficulty.HARD,
        }[tier]

        questions = []
        for topic in topics[:n]:
            text = topic if tier != SeniorityTier.JUNIOR else self._JUNIOR_WRAPPER.format(topic=topic)
            if text in used:
                continue
            used.add(text)
            questions.append(Question(
                skill=skill,
                q_type=QuestionType.SYSTEM_DESIGN,
                difficulty=difficulty,
                text=text,
                hint="Start with requirements, then capacity estimation, then component design.",
                follow_up="Where would this design break under 100× load? What's your mitigation?",
            ))
        return questions


class BehavioralGenerator:
    """Generates STAR-format soft-skill and behavioural questions."""

    _TEMPLATES = [
        "Tell me about a time you had to debug a critical {skill} issue under a tight deadline. "
        "What was your process and what did you learn?",
        "Describe a project where {skill} was central and you significantly improved performance. "
        "What was the measurable impact?",
        "How do you stay current with {skill}? "
        "Describe something new you applied in the past 6 months.",
        "Tell me about a disagreement with a teammate over a {skill} design decision "
        "and how you resolved it.",
        "Describe a situation where you had to advocate for a {skill} best practice "
        "that your team was initially resistant to.",
        "Give an example of a mistake you made while working with {skill} "
        "and how you recovered from it.",
        "How do you onboard a new engineer to {skill} on your team? "
        "What resources or activities do you use?",
        "Describe the most complex {skill} problem you've solved. "
        "What made it hard and how did you break it down?",
        "Tell me about a time you had to make a build-vs-buy decision involving {skill}.",
    ]

    def generate(self, skill: str, tier: SeniorityTier, n: int, used: Set[str]) -> List[Question]:
        templates = list(self._TEMPLATES)
        random.shuffle(templates)
        questions = []
        for tmpl in templates[:n]:
            text = tmpl.format(skill=skill)
            if text in used:
                continue
            used.add(text)
            questions.append(Question(
                skill=skill,
                q_type=QuestionType.BEHAVIORAL,
                difficulty=Difficulty.MEDIUM,
                text=text,
                hint="Use the STAR method: Situation, Task, Action, Result.",
                follow_up="What would you do differently if you faced this situation again?",
            ))
        return questions


# ═══════════════════════════════════════════════════════════════
# 4. QUESTION ENGINE  (offline, always available)
# ═══════════════════════════════════════════════════════════════

class QuestionEngine:
    """
    Orchestrates all sub-generators.
    Tier-aware distribution:
      JUNIOR  → mostly conceptual + coding, light behavioral, no system design
      MID     → balanced across all types
      SENIOR  → heavy on scenario + system design, some conceptual
    """

    MAX_SKILLS = 8

    # Distribution:  { tier: { q_type: weight } }
    _DISTRIBUTION = {
        SeniorityTier.JUNIOR: {
            QuestionType.CONCEPTUAL:    3,
            QuestionType.CODING:        2,
            QuestionType.SCENARIO:      1,
            QuestionType.SYSTEM_DESIGN: 0,
            QuestionType.BEHAVIORAL:    1,
        },
        SeniorityTier.MID: {
            QuestionType.CONCEPTUAL:    2,
            QuestionType.CODING:        2,
            QuestionType.SCENARIO:      2,
            QuestionType.SYSTEM_DESIGN: 1,
            QuestionType.BEHAVIORAL:    1,
        },
        SeniorityTier.SENIOR: {
            QuestionType.CONCEPTUAL:    1,
            QuestionType.CODING:        1,
            QuestionType.SCENARIO:      2,
            QuestionType.SYSTEM_DESIGN: 2,
            QuestionType.BEHAVIORAL:    1,
        },
    }

    def __init__(self):
        self._conceptual   = ConceptualGenerator()
        self._coding       = CodingGenerator()
        self._scenario     = ScenarioGenerator()
        self._design       = SystemDesignGenerator()
        self._behavioral   = BehavioralGenerator()

    def generate(
        self,
        profile:      CandidateProfile,
        n_per_skill:  int = 4,
    ) -> List[Dict]:
        skills  = profile.skills[:self.MAX_SKILLS]
        tier    = profile.tier
        dist    = self._DISTRIBUTION[tier]
        used:   Set[str] = set()
        all_questions: List[Question] = []

        for skill in skills:
            n_con  = dist[QuestionType.CONCEPTUAL]
            n_cod  = dist[QuestionType.CODING]
            n_scen = dist[QuestionType.SCENARIO]
            n_des  = dist[QuestionType.SYSTEM_DESIGN]
            n_beh  = dist[QuestionType.BEHAVIORAL]

            # Scale down if n_per_skill is smaller than default (7)
            total_default = sum(dist.values()) or 1
            scale = n_per_skill / total_default
            n_con  = max(0, round(n_con  * scale))
            n_cod  = max(0, round(n_cod  * scale))
            n_scen = max(0, round(n_scen * scale))
            n_des  = max(0, round(n_des  * scale))
            n_beh  = max(0, round(n_beh  * scale))

            all_questions += self._conceptual.generate(skill, tier, n_con,  used)
            all_questions += self._coding.generate(    skill, tier, n_cod,  used)
            all_questions += self._scenario.generate(  skill, tier, n_scen, used)
            all_questions += self._design.generate(    skill, tier, n_des,  used)
            all_questions += self._behavioral.generate(skill, tier, n_beh,  used)

        return [q.to_dict() for q in all_questions]


# ═══════════════════════════════════════════════════════════════
# 5. LLM-BASED GENERATOR  (advanced, optional)
#    Calls Anthropic Claude (or any OpenAI-compatible API).
#    Falls back to QuestionEngine on any error.
# ═══════════════════════════════════════════════════════════════

class LLMQuestionGenerator:
    """
    Uses Claude to generate fully bespoke questions.

    Requirements:
      pip install anthropic          # for Anthropic Claude
      OR set OPENAI_API_KEY          # for OpenAI-compatible APIs

    Falls back to QuestionEngine transparently if:
      • No API key is available
      • The API call fails
      • Rate limits are hit
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "anthropic"):
        self._api_key  = api_key
        self._provider = provider
        self._fallback = QuestionEngine()

    def _build_prompt(self, profile: CandidateProfile) -> str:
        skills_str = ", ".join(profile.skills[:8]) or "general software engineering"
        tier_desc  = {
            SeniorityTier.JUNIOR: "junior (< 2 years experience) — focus on fundamentals",
            SeniorityTier.MID:    "mid-level (2-5 years) — focus on applied skills and problem solving",
            SeniorityTier.SENIOR: "senior (5+ years) — focus on architecture, trade-offs, and leadership",
        }[profile.tier]

        jd_section = (
            f"\nJob Description context:\n{profile.jd_text[:800]}"
            if profile.jd_text else ""
        )

        return f"""You are a senior technical interviewer at a top-tier technology company.
Generate a set of high-quality interview questions for a candidate with the following profile:

Candidate: {profile.name}
Experience: {profile.experience_years} years ({tier_desc})
Key skills: {skills_str}{jd_section}

Generate exactly 10 questions. Each question must be unique and non-repetitive.
Cover these types: conceptual (2), coding (2), scenario/debugging (3), system design (2), behavioral (1).
Difficulty must match the seniority tier.

Return ONLY a JSON array. No preamble, no markdown. Each element:
{{
  "skill": "<primary skill>",
  "type": "conceptual|coding|scenario|system_design|behavioral",
  "difficulty": "easy|medium|hard",
  "question": "<full question text>",
  "hint": "<one-line interviewer hint>",
  "follow_up": "<one follow-up question>"
}}"""

    def _call_anthropic(self, prompt: str) -> List[Dict]:
        import anthropic  # type: ignore
        import json

        client = anthropic.Anthropic(api_key=self._api_key)
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Strip markdown fences if present
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    def _call_openai(self, prompt: str) -> List[Dict]:
        import openai  # type: ignore
        import json

        client = openai.OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    def generate(self, profile: CandidateProfile, n_per_skill: int = 4) -> List[Dict]:
        """
        Try LLM generation; fall back to QuestionEngine on any failure.
        """
        if not self._api_key:
            logger.info("LLMQuestionGenerator: no API key — using offline engine.")
            return self._fallback.generate(profile, n_per_skill)

        prompt = self._build_prompt(profile)
        try:
            if self._provider == "anthropic":
                questions = self._call_anthropic(prompt)
            else:
                questions = self._call_openai(prompt)
            logger.info(f"LLMQuestionGenerator: {len(questions)} questions generated via {self._provider}.")
            return questions
        except Exception as e:
            logger.warning(f"LLMQuestionGenerator fallback triggered: {e}")
            return self._fallback.generate(profile, n_per_skill)


# ═══════════════════════════════════════════════════════════════
# 6. BACKWARDS-COMPAT SHIM  (keeps hr_agent.py working as-is)
# ═══════════════════════════════════════════════════════════════

class QuestionGenerator:
    """
    Drop-in replacement for hr_agent.QuestionGenerator.
    Existing call:  qgen.generate(skills, n_per_skill=3)
    """

    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        self._engine = (
            LLMQuestionGenerator(api_key=api_key)
            if use_llm and api_key
            else QuestionEngine()
        )

    def generate(
        self,
        skills:           List[str],
        n_per_skill:      int   = 3,
        experience_years: float = 0.0,
        jd_text:          str   = "",
        name:             str   = "Candidate",
    ) -> List[Dict]:
        profile = CandidateProfile(
            skills=skills,
            experience_years=experience_years,
            jd_text=jd_text,
            name=name,
        )
        return self._engine.generate(profile, n_per_skill=n_per_skill)


# ═══════════════════════════════════════════════════════════════
# 7. SMOKE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    profiles = [
        CandidateProfile(
            skills=["Python", "Machine Learning", "SQL"],
            experience_years=1.0,
            name="Priya (Junior)",
        ),
        CandidateProfile(
            skills=["React", "Node.js", "Docker", "AWS"],
            experience_years=3.5,
            name="Arjun (Mid)",
        ),
        CandidateProfile(
            skills=["Kubernetes", "DevOps", "Python", "Natural Language Processing"],
            experience_years=7.0,
            name="Zara (Senior)",
            jd_text="Senior ML Platform Engineer — owns reliability and scalability of ML infrastructure.",
        ),
        CandidateProfile(
            skills=["Data Structures and Algorithms", "C++"],
            experience_years=0.5,
            name="Kiran (Fresher — tests generic fallback)",
        ),
    ]

    engine = QuestionEngine()
    for profile in profiles:
        qs = engine.generate(profile, n_per_skill=3)
        print(f"\n{'='*60}")
        print(f"  {profile.name}  |  Tier: {profile.tier.value}  |  {len(qs)} questions")
        print(f"{'='*60}")
        for q in qs:
            print(f"  [{q['type'].upper()} | {q['difficulty']}] {q['skill']}")
            print(f"  Q: {q['question'][:120]}...")
            print()
