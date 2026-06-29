"""
hr_agent.py
Core AI HR Agent — resume parsing, ranking, scheduling,
pipeline management, and interview question generation.

Integration changelog:
  • ResumeRanker.extract_skills() removed → replaced by SkillExtractor (skill_extractor.py)
  • Old static QuestionGenerator class removed → replaced by QuestionGenerator (question_generator.py)
  • generate_questions() and run_full_pipeline() upgraded to pass experience-aware args

Fix changelog (production hardening):
  FIX-1  compute_jd_coverage: removed 0.5 fallback — now returns 0.0 when jd_skills empty
  FIX-2  rank_candidates: raises ValueError (fail-fast) if jd_skills is empty
  FIX-3  compute_experience_score: regex updated to capture decimals (e.g. "1.5 years")
  FIX-4  screen_resumes: duplicate prevention — candidates already in pipeline are NOT re-added
  FIX-7  run_full_pipeline: top_candidate correctly selected from ranked results
"""

import re
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Integration: new production-grade modules ─────────────────
from skill_extractor import SkillExtractor          # replaces ResumeRanker.extract_skills()
from question_generator import QuestionGenerator    # replaces old static QuestionGenerator


# ─────────────────────────────────────────────
# Constants & Taxonomy
# ─────────────────────────────────────────────

# Valid state transitions for the hiring pipeline.
# "rejected" includes "shortlisted" to allow manual re-opening of a rejected candidate.
VALID_TRANSITIONS = {
    "applied":             ["shortlisted", "rejected"],
    "shortlisted":         ["interview_scheduled", "rejected"],
    "interview_scheduled": ["interviewed", "rejected"],
    "interviewed":         ["selected", "rejected"],
    "selected":            [],
    "rejected":            ["shortlisted"],   # manual override: re-open a rejected candidate
}


def _generate_available_slots(
    days_ahead: int = 7,
    times: tuple = ("09:00", "11:00", "14:00", "16:00"),
    skip_sunday: bool = True,
) -> List[str]:
    slots: List[str] = []
    today = datetime.now().date()
    for offset in range(1, days_ahead + 1):
        day = today + timedelta(days=offset)
        if skip_sunday and day.weekday() == 6:
            continue
        for t in times:
            slots.append(f"{day.strftime('%Y-%m-%d')} {t}")
    return slots


# ─────────────────────────────────────────────
# 1. Resume Parser & Ranker
# ─────────────────────────────────────────────

class ResumeRanker:
    """
    Ranks candidates against a job description.

    Integration: extract_skills() removed. All skill extraction now
    delegates to self.skill_extractor (SkillExtractor from skill_extractor.py).
    """

    def __init__(self):
        self.vectorizer    = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        # Integration: initialise production-grade skill extractor
        self.skill_extractor = SkillExtractor()

    def compute_jd_coverage(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        # FIX-1: removed fallback 0.5 — return 0.0 when jd_skills is empty.
        # Callers must validate jd_skills is non-empty before calling this method.
        if not jd_skills:
            return 0.0
        resume_set = set(s.lower() for s in resume_skills)
        matched = sum(1 for s in jd_skills if s.lower() in resume_set)
        return round(matched / len(jd_skills), 4)

    def compute_breadth_bonus(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        jd_set = set(s.lower() for s in jd_skills)
        extra = sum(1 for s in resume_skills if s.lower() not in jd_set)
        return min(0.20, round(extra * 0.04, 4))

    def compute_keyword_similarity(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        if not jd_skills:
            return 0.0
        return len(set(resume_skills) & set(jd_skills)) / len(jd_skills)

    def compute_experience_score(self, exp_years: float, jd_text: str) -> float:
        jd_lower = jd_text.lower()
        required = 0
        # FIX-3: updated regex to capture decimal experience values (e.g. "1.5 years")
        patterns = [
            r'(\d+(?:\.\d+)?)\+?\s*years?\s*of\s*experience',
            r'minimum\s+(\d+(?:\.\d+)?)\s+years?',
            r'at\s+least\s+(\d+(?:\.\d+)?)\s+years?',
            r'(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?(?:work|professional|industry|relevant)',
        ]
        for pat in patterns:
            m = re.search(pat, jd_lower)
            if m:
                required = float(m.group(1))
                break
        if required == 0:
            return 0.7
        if exp_years >= required:
            return min(1.0, 0.7 + 0.06 * (exp_years - required))
        return max(0.0, 0.5 * exp_years / required)

    def rank_candidates(self, candidates: List[Dict], jd_text: str) -> List[Dict]:
        # Integration: use SkillExtractor instead of old extract_skills()
        jd_skills = self.skill_extractor.extract(jd_text)

        # FIX-2: fail-fast — do NOT rank if JD has no valid skills
        if not jd_skills:
            raise ValueError(
                "No valid skills found in Job Description. "
                "Please add explicit skill names (e.g. 'Python', 'Docker') and try again."
            )

        ranked = []
        for cand in candidates:
            # Integration: use SkillExtractor for each resume
            skills   = self.skill_extractor.extract(cand.get("resume_text", ""))
            coverage = self.compute_jd_coverage(skills, jd_skills)
            breadth  = self.compute_breadth_bonus(skills, jd_skills)
            kw_sim   = self.compute_keyword_similarity(skills, jd_skills)
            exp_sc   = self.compute_experience_score(cand.get("experience_years", 0), jd_text)
            score    = round(0.55 * coverage + 0.10 * kw_sim + 0.10 * breadth + 0.25 * exp_sc, 4)
            ranked.append({
                **cand,
                "skills":           skills,
                "jd_skills":        jd_skills,
                "jd_coverage":      coverage,
                "breadth_bonus":    breadth,
                "keyword_sim":      kw_sim,
                "experience_score": exp_sc,
                "match_score":      score,
            })
        ranked.sort(key=lambda x: x["match_score"], reverse=True)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        return ranked


# ─────────────────────────────────────────────
# 2. Interview Scheduler
# ─────────────────────────────────────────────

class InterviewScheduler:
    def __init__(
        self,
        days_ahead: int = 7,
        times: tuple = ("09:00", "11:00", "14:00", "16:00"),
        skip_sunday: bool = True,
    ):
        self.slots     = _generate_available_slots(days_ahead, times, skip_sunday)
        self.scheduled: Dict[str, str] = {}

    def schedule(self, candidate_id: str) -> Dict:
        if not self.slots:
            return {"candidate_id": candidate_id, "slot": None,
                    "status": "no_slots_available", "conflict": True}
        slot = self.slots.pop(0)
        self.scheduled[candidate_id] = slot
        return {"candidate_id": candidate_id, "slot": slot,
                "status": "scheduled", "conflict": False}

    def bulk_schedule(self, candidate_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
        scheduled, conflicts = [], []
        for cid in candidate_ids:
            result = self.schedule(cid)
            (scheduled if not result["conflict"] else conflicts).append(result)
        return scheduled, conflicts

    def available_slots(self) -> List[str]:
        return list(self.slots)


# ─────────────────────────────────────────────
# 3. Pipeline State Manager
# ─────────────────────────────────────────────

class PipelineManager:
    def __init__(self):
        self.pipeline: Dict[str, str] = {}

    def add_candidate(self, candidate_id: str) -> Dict:
        """Add a candidate at 'applied' state. No-op if already present."""
        if candidate_id not in self.pipeline:
            self.pipeline[candidate_id] = "applied"
        return {"candidate_id": candidate_id, "state": self.pipeline[candidate_id], "success": True}

    def upsert_state(self, candidate_id: str, desired_state: str) -> Dict:
        """
        Force-set a candidate's pipeline state, adding them if not present.

        Used exclusively for automatic bulk pipeline sync operations (e.g. after
        scheduling interviews). Bypasses VALID_TRANSITIONS intentionally so that
        bulk operations are always idempotent and never crash on edge-case paths.

        Manual UI transitions must still use transition() for validated state changes.
        """
        self.pipeline[candidate_id] = desired_state
        return {"candidate_id": candidate_id, "current_state": desired_state, "success": True}

    def transition(self, candidate_id: str, new_state: str) -> Dict:
        """Validated manual state transition. Enforces VALID_TRANSITIONS."""
        if candidate_id not in self.pipeline:
            return {"success": False, "error": f"Candidate {candidate_id} not found in pipeline."}
        current = self.pipeline[candidate_id]
        allowed = VALID_TRANSITIONS.get(current, [])
        if new_state not in allowed:
            return {
                "success":       False,
                "error":         f"Invalid transition: {current} → {new_state}. Allowed: {allowed}",
                "current_state": current,
            }
        self.pipeline[candidate_id] = new_state
        return {
            "success":       True,
            "candidate_id":  candidate_id,
            "previous":      current,
            "current_state": new_state,
        }

    def get_status(self, candidate_id: str) -> Optional[str]:
        return self.pipeline.get(candidate_id)

    def all_statuses(self) -> Dict[str, str]:
        return dict(self.pipeline)


# ─────────────────────────────────────────────
# 4. Master HR Agent
# ─────────────────────────────────────────────

class HRAgent:
    TEAM_ID = "ai_hr_team"
    TRACK   = "track_2_hr_agent"

    def __init__(self):
        self.ranker    = ResumeRanker()
        # Integration: QuestionGenerator is now imported from question_generator.py
        self.qgen      = QuestionGenerator()
        self.scheduler = InterviewScheduler()
        self.pipeline  = PipelineManager()

    def screen_resumes(self, candidates: List[Dict], jd: str) -> Dict:
        """
        Rank candidates and ensure every one is registered in the pipeline.

        FIX-2: rank_candidates now raises ValueError if JD has no valid skills.
               This propagates up to the UI layer which shows st.error + st.stop().
        FIX-4: add_candidate is idempotent — existing pipeline entries are NOT
               overwritten, preventing duplicate/reset on repeated screen calls.
        """
        ranked = self.ranker.rank_candidates(candidates, jd)
        for cand in ranked:
            # add_candidate is idempotent — skips if candidate already exists
            self.pipeline.add_candidate(str(cand["id"]))
        return {
            "ranked_candidates": [
                {
                    "id":          c["id"],
                    "name":        c.get("name", ""),
                    "rank":        c["rank"],
                    "match_score": c["match_score"],
                    "skills":      c["skills"],
                }
                for c in ranked
            ],
            "scores": [c["match_score"] for c in ranked],
        }

    def generate_questions(
        self,
        skills:           List[str],
        n_per_skill:      int   = 3,
        experience_years: float = 0.0,
        jd_text:          str   = "",
        name:             str   = "Candidate",
    ) -> List[Dict]:
        """
        Generate experience-aware interview questions.

        FIX-10: if skills list is empty, delegate to qgen which must return
        fallback generic questions rather than an empty list.

        Integration: forwards all context to the new QuestionGenerator so that
        seniority-tier routing and JD-aware question selection are enabled.
        """
        return self.qgen.generate(
            skills,
            n_per_skill=n_per_skill,
            experience_years=experience_years,
            jd_text=jd_text,
            name=name,
        )

    def schedule_interviews(self, candidate_ids: List[str]) -> Dict:
        """
        Schedule interviews for selected candidates.
        NOTE: pipeline state sync for ALL ranked candidates is handled by the
        UI layer (app.py) after this call, so that the full ranked list context
        is available. This method only transitions the scheduled candidates.
        """
        scheduled, conflicts = self.scheduler.bulk_schedule(candidate_ids)
        for s in scheduled:
            cid = s["candidate_id"]
            # Ensure candidate exists before transitioning
            self.pipeline.add_candidate(cid)
            # Use upsert for reliability — scheduled candidates must reach this state
            self.pipeline.upsert_state(cid, "interview_scheduled")
        return {"interviews_scheduled": scheduled, "conflicts": conflicts}

    def transition_candidate(self, candidate_id: str, new_state: str) -> Dict:
        return self.pipeline.transition(candidate_id, new_state)

    def run_full_pipeline(self, candidates: List[Dict], jd: str) -> Dict:
        screening  = self.screen_resumes(candidates, jd)
        top_ids    = [str(c["id"]) for c in screening["ranked_candidates"][:5]]
        scheduling = self.schedule_interviews(top_ids)

        # Sync remaining candidates to rejected
        scheduled_ids = {s["candidate_id"] for s in scheduling["interviews_scheduled"]}
        for cand in screening["ranked_candidates"]:
            cid = str(cand["id"])
            if cid not in scheduled_ids:
                current = self.pipeline.get_status(cid)
                if current in (None, "applied", "shortlisted"):
                    self.pipeline.upsert_state(cid, "rejected")

        # FIX-7: top_candidate is the highest-ranked candidate from ranked results,
        # NOT the first element of the raw input list.
        top_ranked = screening["ranked_candidates"][0] if screening["ranked_candidates"] else None
        top_candidate = next(
            (c for c in candidates if str(c["id"]) == str(top_ranked["id"])),
            {}
        ) if top_ranked else {}

        # Reuse skills already computed during ranking — no redundant extraction
        top_skills = top_ranked["skills"] if top_ranked else []

        # Integration: generate questions with full experience-aware context
        questions = self.generate_questions(
            top_skills,
            n_per_skill=3,
            experience_years=top_candidate.get("experience_years", 0.0),
            jd_text=jd,
            name=top_candidate.get("name", "Candidate"),
        )

        pipeline = self.pipeline.all_statuses()
        return {
            "team_id": self.TEAM_ID,
            "track":   self.TRACK,
            "results": {
                "resume_screening": screening,
                "scheduling":       scheduling,
                "questionnaire":    {"questions": questions},
                "pipeline":         {"candidates": pipeline},
            },
        }


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = HRAgent()
    result = agent.run_full_pipeline(
        candidates=[
            {
                "id":               1,
                "name":             "Alice",
                "resume_text":      "Python Machine Learning Docker AWS 5 years experience",
                "experience_years": 5,
            },
            {
                "id":               2,
                "name":             "Bob",
                "resume_text":      "JavaScript React Node.js 2 years experience",
                "experience_years": 2,
            },
        ],
        jd="Python ML engineer 3+ years experience cloud DevOps.",
    )
    print(json.dumps(result, indent=2, default=str))