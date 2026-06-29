"""
skill_extractor.py
─────────────────────────────────────────────────────────────────
Production-grade Skill Extraction System for AI HR Agent.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  Raw Text                                                   │
  │     │                                                       │
  │     ▼                                                       │
  │  TextNormalizer  ──  lowercases, strips noise               │
  │     │                                                       │
  │     ▼                                                       │
  │  AliasResolver   ──  C++ → cpp, DSA → data structures …    │
  │     │                                                       │
  │     ▼                                                       │
  │  ExactMatcher    ──  token/phrase scan over taxonomy        │
  │     │                                                       │
  │     ▼                                                       │
  │  SemanticMatcher ──  sentence-transformer embeddings        │
  │     │                 (lazy-loaded, falls back gracefully)  │
  │     ▼                                                       │
  │  SkillRegistry   ──  single source of truth, dynamically   │
  │                       extensible, hierarchical taxonomy     │
  └─────────────────────────────────────────────────────────────┘

Drop-in replacement for ResumeRanker.extract_skills().
Also exported: SkillRegistry (for adding new skills at runtime).
"""

from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 1. GLOBAL SKILL TAXONOMY
#    Structure:  category → { canonical_name → [aliases] }
#
#    Design principles:
#      • Canonical name is the display name (e.g. "C++", "Node.js")
#      • Aliases cover abbreviations, alternate spellings, common
#        typos, and legacy names
#      • New entries can be added at runtime via SkillRegistry.add()
#      • File-based persistence: registry.save() / registry.load()
# ═══════════════════════════════════════════════════════════════

_RAW_TAXONOMY: Dict[str, Dict[str, List[str]]] = {

    # ── Programming Languages ────────────────────────────────
    "Programming Languages": {
        "Python":       ["py", "python3", "python2", "cpython", "micropython"],
        "Java":         ["jdk", "jvm", "j2ee", "java8", "java11", "java17"],
        "C":            ["c language", "c programming", "ansi c"],
        "C++":          ["cpp", "c plus plus", "cplusplus", "c++11", "c++14", "c++17", "c++20"],
        "C#":           ["csharp", "c sharp", "dotnet c#", ".net c#"],
        "Go":           ["golang", "go lang"],
        "Rust":         ["rust lang", "rustlang"],
        "Kotlin":       ["kotlin jvm", "android kotlin"],
        "Swift":        ["swift ios", "swiftui"],
        "TypeScript":   ["ts", "typescript js"],
        "JavaScript":   ["js", "es6", "es2015", "ecmascript", "vanilla js", "node js"],
        "PHP":          ["php7", "php8", "laravel php", "symfony php"],
        "Ruby":         ["ruby on rails", "ror", "rails"],
        "Scala":        ["scala spark", "akka"],
        "R":            ["r language", "r programming", "rstats", "r studio"],
        "MATLAB":       ["matlab simulink"],
        "Perl":         ["perl5"],
        "Shell":        ["bash", "shell scripting", "zsh", "ksh", "sh scripting", "powershell"],
        "Dart":         ["flutter dart"],
    },

    # ── Core CS Concepts ─────────────────────────────────────
    "Core CS": {
        "Data Structures and Algorithms": [
            "dsa", "data structures", "algorithms", "ds algo",
            "data structures & algorithms", "competitive programming",
            "algorithm design", "problem solving", "leetcode", "hackerrank",
        ],
        "Operating Systems":     ["os", "linux internals", "kernel", "process management", "concurrency"],
        "Computer Networks":     ["networking", "network protocols", "tcp/ip", "http", "dns", "osi model", "cn"],
        "Database Management":   ["dbms", "database systems", "rdbms", "sql fundamentals", "normalization"],
        "Object Oriented Programming": [
            "oop", "oops", "object oriented", "design patterns",
            "solid principles", "inheritance", "polymorphism", "encapsulation",
        ],
        "Compiler Design":       ["compilers", "parsing", "lexical analysis", "code generation"],
        "Computer Architecture": ["cpu architecture", "memory hierarchy", "cache", "pipelining"],
        "Discrete Mathematics":  ["discrete math", "combinatorics", "graph theory", "logic"],
        "Theory of Computation": ["automata", "turing machines", "formal languages", "toc"],
        "Software Engineering":  ["sdlc", "software design", "agile", "scrum", "waterfall", "tdd"],
    },

    # ── Web & Frontend ────────────────────────────────────────
    "Web & Frontend": {
        "React":         ["reactjs", "react.js", "react hooks", "react native", "nextjs", "next.js"],
        "Angular":       ["angularjs", "angular2+", "ng"],
        "Vue.js":        ["vuejs", "vue", "nuxt", "nuxtjs"],
        "HTML":          ["html5", "html/css", "markup"],
        "CSS":           ["css3", "sass", "scss", "less", "tailwind", "tailwindcss", "bootstrap"],
        "Node.js":       ["nodejs", "node", "expressjs", "express.js", "express"],
        "GraphQL":       ["gql", "graphql api", "apollo"],
        "REST APIs":     ["rest", "restful", "api design", "openapi", "swagger"],
        "WebSockets":    ["ws", "socket.io", "real-time"],
        "Redux":         ["redux toolkit", "state management", "mobx", "zustand"],
        "Django":        ["django rest framework", "drf"],
        "Flask":         ["flask python", "flask api"],
        "FastAPI":       ["fast api", "fastapi python"],
        "Spring Boot":   ["spring", "spring mvc", "spring framework", "java spring"],
        "ASP.NET":       ["asp.net core", "dotnet web", ".net mvc"],
    },

    # ── Data & AI/ML ─────────────────────────────────────────
    "Data & AI/ML": {
        "Machine Learning":      ["ml", "supervised learning", "unsupervised learning", "sklearn", "scikit-learn"],
        "Deep Learning":         ["dl", "neural networks", "ann", "cnn", "rnn", "lstm"],
        "Natural Language Processing": [
            "nlp", "text mining", "text analytics", "computational linguistics",
            "language models", "tokenization", "ner",
        ],
        "Computer Vision":       ["cv", "image processing", "object detection", "yolo", "opencv"],
        "Generative AI":         ["genai", "gen ai", "llm", "large language models", "gpt", "chatgpt", "stable diffusion", "dall-e", "diffusion models"],
        "LLMs":                  ["large language model", "language model", "foundation models", "gpt-4", "claude", "llama", "mistral"],
        "Reinforcement Learning":["rl", "q-learning", "policy gradient", "dqn"],
        "Data Science":          ["data scientist", "statistical modeling", "exploratory data analysis", "eda"],
        "Data Analysis":         ["data analytics", "business intelligence", "bi", "reporting"],
        "Data Engineering":      ["data pipelines", "etl", "elt", "data warehouse", "lakehouse"],
        "TensorFlow":            ["tensorflow2", "tf", "keras", "tf.keras"],
        "PyTorch":               ["torch", "pytorch lightning"],
        "Pandas":                ["pandas python", "dataframes"],
        "NumPy":                 ["numpy python", "numerical python"],
        "Scikit-learn":          ["sklearn", "scikit learn"],
        "Hugging Face":          ["transformers library", "huggingface", "hf transformers"],
    },

    # ── Databases ─────────────────────────────────────────────
    "Databases": {
        "SQL":           ["mysql", "postgresql", "sqlite", "ms sql", "sql server", "pl/sql", "t-sql"],
        "PostgreSQL":    ["postgres", "pg", "psql"],
        "MySQL":         ["mysql8", "mariadb"],
        "MongoDB":       ["mongo", "nosql mongo", "document database"],
        "Redis":         ["redis cache", "in-memory db", "redis sentinel"],
        "Cassandra":     ["apache cassandra", "wide-column"],
        "Elasticsearch": ["elastic", "es", "elk", "opensearch"],
        "DynamoDB":      ["aws dynamodb", "dynamo"],
        "Neo4j":         ["graph database", "cypher"],
        "Snowflake":     ["snowflake dwh", "snowflake cloud"],
        "BigQuery":      ["google bigquery", "gcp bigquery"],
        "Apache Spark":  ["spark", "pyspark", "spark sql", "databricks"],
        "Kafka":         ["apache kafka", "kafka streams", "event streaming"],
    },

    # ── DevOps & Cloud ────────────────────────────────────────
    "DevOps & Cloud": {
        "Docker":        ["containerization", "dockerfile", "docker compose", "docker-compose"],
        "Kubernetes":    ["k8s", "k8", "helm", "kubectl", "eks", "gke", "aks"],
        "AWS":           ["amazon web services", "ec2", "s3", "lambda", "aws cloud", "boto3"],
        "Azure":         ["microsoft azure", "azure cloud", "azure devops"],
        "GCP":           ["google cloud", "google cloud platform", "gcp cloud"],
        "Terraform":     ["terraform iac", "hashicorp terraform", "infrastructure as code"],
        "Ansible":       ["ansible playbooks", "configuration management"],
        "CI/CD":         ["continuous integration", "continuous delivery", "continuous deployment", "github actions", "gitlab ci", "jenkins", "circleci"],
        "Jenkins":       ["jenkins ci", "jenkinsfile"],
        "Linux":         ["ubuntu", "centos", "debian", "rhel", "unix", "linux admin"],
        "Git":           ["github", "gitlab", "version control", "git flow"],
        "DevOps":        ["site reliability", "sre", "platform engineering", "devsecops"],
        "Nginx":         ["reverse proxy", "nginx config"],
        "Prometheus":    ["prometheus monitoring", "grafana", "observability"],
        "Serverless":    ["lambda functions", "cloud functions", "serverless framework", "faas"],
        "Microservices": ["service mesh", "istio", "api gateway", "distributed systems"],
    },

    # ── Security ──────────────────────────────────────────────
    "Cybersecurity": {
        "Cybersecurity":        ["information security", "infosec", "cyber security"],
        "Network Security":     ["firewall", "ids", "ips", "vpn", "network hardening"],
        "Penetration Testing":  ["pen testing", "pentesting", "ethical hacking", "red team", "kali linux"],
        "Application Security": ["appsec", "owasp", "secure coding", "sast", "dast"],
        "Cloud Security":       ["iam policies", "zero trust", "cspm", "cloud hardening"],
        "SIEM":                 ["splunk", "qradar", "siem tools"],
        "Cryptography":         ["encryption", "pki", "tls", "ssl", "hash functions"],
        "Incident Response":    ["ir", "digital forensics", "blue team", "soc"],
    },

    # ── Emerging Tech ─────────────────────────────────────────
    "Emerging Tech": {
        "Blockchain":           ["distributed ledger", "web3", "dlt"],
        "Web3":                 ["web 3", "decentralized web", "dapps", "defi"],
        "Smart Contracts":      ["solidity", "ethereum", "evm"],
        "IoT":                  ["internet of things", "embedded systems", "mqtt", "arduino"],
        "Edge Computing":       ["fog computing", "edge ai"],
        "Quantum Computing":    ["quantum algorithms", "qiskit"],
        "AR/VR":                ["augmented reality", "virtual reality", "xr", "unity3d", "unreal engine"],
    },

    # ── Mobile ────────────────────────────────────────────────
    "Mobile": {
        "Android":      ["android sdk", "android studio", "android development"],
        "iOS":          ["ios development", "xcode", "objective-c"],
        "Flutter":      ["flutter sdk", "dart flutter"],
        "React Native": ["rn", "react-native"],
    },

    # ── Tools & Practices ─────────────────────────────────────
    "Tools & Practices": {
        "Agile":        ["scrum", "kanban", "sprint", "agile methodology"],
        "Git":          ["version control", "github", "gitlab", "bitbucket"],
        "Jira":         ["project management", "atlassian"],
        "Power BI":     ["powerbi", "microsoft bi", "power bi desktop"],
        "Tableau":      ["tableau desktop", "tableau server"],
        "Excel":        ["microsoft excel", "spreadsheets", "vba"],
    },
}


# ═══════════════════════════════════════════════════════════════
# 2. SKILL REGISTRY  (single source of truth)
# ═══════════════════════════════════════════════════════════════

class SkillRegistry:
    """
    Central skill store.  Flat look-up dicts are pre-built at
    construction time so every query is O(1) or O(n_tokens).

    Thread-safety note: mutating methods (add, load) are not
    thread-safe by design; call them once at startup.
    """

    def __init__(self, taxonomy: Optional[Dict] = None):
        # category → { canonical → [aliases] }
        self._taxonomy: Dict[str, Dict[str, List[str]]] = {}

        # alias_lower → canonical  (for fast look-up)
        self._alias_map: Dict[str, str] = {}

        # canonical_lower → canonical  (for direct match)
        self._canonical_lower: Dict[str, str] = {}

        # canonical → category
        self._skill_category: Dict[str, str] = {}

        raw = taxonomy or _RAW_TAXONOMY
        for category, skills in raw.items():
            for canonical, aliases in skills.items():
                self._register(category, canonical, aliases)

    # ── Internal ─────────────────────────────────────────────

    def _register(self, category: str, canonical: str, aliases: List[str]):
        if category not in self._taxonomy:
            self._taxonomy[category] = {}
        self._taxonomy[category][canonical] = aliases

        canon_low = canonical.lower()
        self._canonical_lower[canon_low] = canonical
        self._skill_category[canonical] = category

        # Also index every alias
        for alias in aliases:
            a_low = alias.lower().strip()
            if a_low and a_low not in self._alias_map:
                self._alias_map[a_low] = canonical

    # ── Public API ───────────────────────────────────────────

    def add(self, category: str, canonical: str, aliases: Optional[List[str]] = None):
        """
        Dynamically add a new skill (or aliases to an existing one).
        Safe to call at runtime — takes effect immediately.
        """
        self._register(category, canonical, aliases or [])

    def resolve(self, token: str) -> Optional[str]:
        """
        Return canonical skill name for a token (or None).
        Checks: exact canonical match → alias map.
        """
        t = token.strip().lower()
        if t in self._canonical_lower:
            return self._canonical_lower[t]
        return self._alias_map.get(t)

    def all_canonicals(self) -> List[str]:
        return list(self._canonical_lower.values())

    def category_of(self, canonical: str) -> Optional[str]:
        return self._skill_category.get(canonical)

    def aliases_of(self, canonical: str) -> List[str]:
        cat = self._skill_category.get(canonical)
        if not cat:
            return []
        return self._taxonomy.get(cat, {}).get(canonical, [])

    def save(self, path: str):
        """Persist the full taxonomy to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._taxonomy, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        """Merge an external JSON taxonomy file into the registry."""
        with open(path, encoding="utf-8") as f:
            extra = json.load(f)
        for category, skills in extra.items():
            for canonical, aliases in skills.items():
                self._register(category, canonical, aliases)


# Module-level singleton — import and use directly
SKILL_REGISTRY = SkillRegistry()


# ═══════════════════════════════════════════════════════════════
# 3. TEXT NORMALIZER
# ═══════════════════════════════════════════════════════════════

class TextNormalizer:
    """
    Cleans raw resume / JD text before skill matching.
    Preserves special tokens like "C++" and "C#".
    """

    # Protect these exact strings from being stripped
    _PROTECTED = {
        "c++": "__CPP__",
        "c#":  "__CSHARP__",
        "c/c++": "__C_CPP__",
        ".net": "__DOTNET__",
        "f#":  "__FSHARP__",
    }
    _PROTECTED_INV = {v: k for k, v in _PROTECTED.items()}

    def normalize(self, text: str) -> str:
        t = text.lower()

        # Protect special tokens
        for raw, placeholder in self._PROTECTED.items():
            t = t.replace(raw, placeholder)

        # Remove URLs
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)

        # Normalize separators (bullets, pipes, slashes between words)
        t = re.sub(r"[•·|/\\]", " ", t)

        # Remove possessives
        t = re.sub(r"'s\b", "", t)

        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()

        # Restore protected tokens
        for placeholder, raw in self._PROTECTED_INV.items():
            t = t.replace(placeholder, raw)

        return t


# ═══════════════════════════════════════════════════════════════
# 4. EXACT / PHRASE MATCHER
# ═══════════════════════════════════════════════════════════════

class ExactMatcher:
    """
    Scans normalised text for every alias + canonical name.

    Matching strategy (ordered by specificity):
      1. Multi-word phrases (e.g. "data structures and algorithms")
      2. Single tokens  (e.g. "dsa")
      3. Special protected tokens (e.g. "c++")

    Uses pre-compiled regex patterns for speed.
    """

    def __init__(self, registry: SkillRegistry):
        self._registry = registry
        self._patterns: List[Tuple[re.Pattern, str]] = []  # (pattern, canonical)
        self._build_patterns()

    def _build_patterns(self):
        # Collect ALL (text_variant, canonical) pairs, longest first
        pairs: List[Tuple[str, str]] = []

        for canonical in self._registry.all_canonicals():
            pairs.append((canonical.lower(), canonical))
            for alias in self._registry.aliases_of(canonical):
                pairs.append((alias.lower(), canonical))

        # Sort longest first so multi-word phrases match before sub-words
        pairs.sort(key=lambda x: len(x[0]), reverse=True)

        seen_patterns: Set[str] = set()
        for variant, canonical in pairs:
            if not variant.strip():
                continue
            # Escape for regex — preserves '+', '#', '.'
            escaped = re.escape(variant)
            if escaped in seen_patterns:
                continue
            seen_patterns.add(escaped)
            pattern = re.compile(r"(?<![a-z0-9_])" + escaped + r"(?![a-z0-9_])", re.IGNORECASE)
            self._patterns.append((pattern, canonical))

    def match(self, normalized_text: str) -> Set[str]:
        found: Set[str] = set()
        for pattern, canonical in self._patterns:
            if pattern.search(normalized_text):
                found.add(canonical)
        return found


# ═══════════════════════════════════════════════════════════════
# 5. SEMANTIC MATCHER  (optional, lazy-loaded)
# ═══════════════════════════════════════════════════════════════

class SemanticMatcher:
    """
    Uses sentence-transformers to find skills that are semantically
    implied but not explicitly named.

    Lazy-loads the model on first use.
    Falls back gracefully if the package is not installed.
    """

    MODEL_NAME  = "all-MiniLM-L6-v2"   # ~22 MB, fast, accurate
    THRESHOLD   = 0.42                  # cosine similarity cutoff

    def __init__(self, registry: SkillRegistry):
        self._registry = registry
        self._model    = None
        self._embeddings: Optional[Dict[str, object]] = None
        self._available: Optional[bool] = None

    def _try_load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np  # noqa
            self._model = SentenceTransformer(self.MODEL_NAME)
            canonicals  = self._registry.all_canonicals()
            vecs = self._model.encode(canonicals, convert_to_numpy=True,
                                      show_progress_bar=False, normalize_embeddings=True)
            self._embeddings = {c: v for c, v in zip(canonicals, vecs)}
            self._available  = True
            logger.info("SemanticMatcher: sentence-transformers loaded ✓")
        except Exception as e:
            logger.warning(f"SemanticMatcher unavailable: {e}. Falling back to exact matching only.")
            self._available = False
        return self._available

    def match(self, text: str, already_found: Set[str]) -> Set[str]:
        """
        Return additional skills semantically implied by 'text'
        that were not caught by ExactMatcher.
        """
        if not self._try_load():
            return set()

        import numpy as np  # type: ignore

        # Only consider skills not already found
        candidates  = [c for c in self._registry.all_canonicals() if c not in already_found]
        if not candidates:
            return set()

        # Encode the resume chunk (truncate to ~512 tokens worth of chars)
        text_vec = self._model.encode(text[:2000], convert_to_numpy=True,
                                      normalize_embeddings=True)

        # Cosine similarity (embeddings already normalised → dot product)
        found: Set[str] = set()
        cand_vecs = np.stack([self._embeddings[c] for c in candidates])
        sims = cand_vecs @ text_vec  # shape: (n_candidates,)

        for c, sim in zip(candidates, sims):
            if float(sim) >= self.THRESHOLD:
                found.add(c)

        return found


# ═══════════════════════════════════════════════════════════════
# 6. SKILL EXTRACTOR  (main API)
# ═══════════════════════════════════════════════════════════════

class SkillExtractor:
    """
    Primary entry-point.  Combines:
      • TextNormalizer
      • ExactMatcher  (always runs)
      • SemanticMatcher  (runs if sentence-transformers is installed)

    Usage:
        extractor = SkillExtractor()

        # Basic
        skills = extractor.extract("Python, C++ DSA TensorFlow Docker")

        # With categories
        skills, cats = extractor.extract_with_categories("Python ML Docker")

        # Add new skills at runtime
        extractor.registry.add("My Domain", "Rust", ["rustlang", "rust lang"])
    """

    def __init__(
        self,
        registry:        Optional[SkillRegistry] = None,
        use_semantic:    bool = True,
        taxonomy_file:   Optional[str] = None,
    ):
        self.registry    = registry or SKILL_REGISTRY
        if taxonomy_file and Path(taxonomy_file).exists():
            self.registry.load(taxonomy_file)

        self._normalizer = TextNormalizer()
        self._exact      = ExactMatcher(self.registry)
        self._semantic   = SemanticMatcher(self.registry) if use_semantic else None

    # ── Public API ───────────────────────────────────────────

    def extract(self, text: str) -> List[str]:
        """
        Return sorted list of canonical skill names found in text.
        Drop-in replacement for ResumeRanker.extract_skills().
        """
        normalized = self._normalizer.normalize(text)
        found = self._exact.match(normalized)

        if self._semantic:
            semantic_hits = self._semantic.match(normalized, found)
            found |= semantic_hits

        return sorted(found)

    def extract_with_categories(self, text: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Returns (skills, category_map) where category_map is
        { canonical_skill → category }.
        """
        skills = self.extract(text)
        cats   = {s: (self.registry.category_of(s) or "Unknown") for s in skills}
        return skills, cats

    def extract_structured(self, text: str) -> Dict[str, List[str]]:
        """
        Returns { category → [skills] } — useful for UI display.
        """
        skills, cats = self.extract_with_categories(text)
        grouped: Dict[str, List[str]] = {}
        for skill, cat in cats.items():
            grouped.setdefault(cat, []).append(skill)
        return {cat: sorted(skills) for cat, skills in grouped.items()}

    def coverage(self, resume_text: str, jd_text: str) -> Dict:
        """
        Compute JD coverage, breadth bonus, and matched/missing skills.
        Keeps compatibility with ResumeRanker.compute_jd_coverage().
        """
        resume_skills = set(self.extract(resume_text))
        jd_skills     = set(self.extract(jd_text))

        if not jd_skills:
            return {
                "jd_coverage": 0.5,
                "breadth_bonus": 0.0,
                "matched":  [],
                "missing":  [],
                "extra":    list(resume_skills),
            }

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        extra   = resume_skills - jd_skills

        jd_coverage   = round(len(matched) / len(jd_skills), 4)
        breadth_bonus = round(min(0.20, len(extra) * 0.04), 4)

        return {
            "jd_coverage":   jd_coverage,
            "breadth_bonus": breadth_bonus,
            "matched":       sorted(matched),
            "missing":       sorted(missing),
            "extra":         sorted(extra),
        }


# ═══════════════════════════════════════════════════════════════
# 7. BACKWARDS-COMPAT SHIM  (keeps hr_agent.py working as-is)
# ═══════════════════════════════════════════════════════════════

_default_extractor: Optional[SkillExtractor] = None


def _get_extractor() -> SkillExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = SkillExtractor()
    return _default_extractor


def extract_skills(text: str) -> List[str]:
    """
    Module-level convenience function.
    Replace calls to ResumeRanker.extract_skills() with this.
    """
    return _get_extractor().extract(text)


# ═══════════════════════════════════════════════════════════════
# 8.  SMOKE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ex = SkillExtractor(use_semantic=False)

    test_cases = [
        # Tricky / previously broken cases
        "Proficient in C++ with STL and OOP principles",
        "Strong DSA fundamentals — trees, graphs, dynamic programming",
        "Used data structures and algorithms in competitive programming",
        "Worked with Kubernetes (k8s) and Terraform for IaC",
        "Built LLM pipelines using Hugging Face transformers and LangChain",
        "Python, ML, Docker, AWS, CI/CD, React, PostgreSQL, Redis",
        "Experience in Gen AI, Stable Diffusion, and GPT-4 integrations",
        "Rust and Go microservices deployed on GCP with Helm",
        "Developed dApps using Solidity and Web3.js on Ethereum",
        "Shell scripting (bash), Linux admin, and Ansible playbooks",
    ]

    for text in test_cases:
        skills = ex.extract(text)
        print(f"\nINPUT : {text}")
        print(f"SKILLS: {', '.join(skills) if skills else '(none)'}")

    # Dynamic extension test
    ex.registry.add("Emerging Tech", "LangChain", ["langchain", "lang chain"])
    print("\n[Dynamic add] LangChain registered.")
    skills = ex.extract("We use LangChain with OpenAI APIs.")
    print(f"After add: {skills}")
