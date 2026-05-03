"""Feature schema and reference distributions for CareerSight.

Single source of truth for which fields the model expects and how they map
between the wire format (resume JSON) and the model input vector.
"""
from __future__ import annotations

from dataclasses import dataclass


COURSE_TYPES = [
    "BTech-CS", "BTech-ECE", "BTech-Mech", "BTech-Civil",
    "MBA-Finance", "MBA-Marketing", "MBA-HR", "MBA-Operations",
    "BSc-Nursing", "MCA", "MTech-CS", "BCom",
]

REGIONS = ["Metro", "Tier2", "Tier3"]
INSTITUTE_TIERS = [1, 2, 3]

# 45 numeric/binary features the model is trained on.
# Categorical fields are one-hot expanded at feature-vector time.
NUMERIC_FEATURES = [
    "cgpa",
    "graduation_year_offset",
    "backlogs_count",
    "paper_publications",
    "semester_consistency",
    "internship_count",
    "internship_total_months",
    "internship_top_tier",
    "internship_relevance_score",
    "certifications_count",
    "relevant_certs_count",
    "github_projects",
    "github_stars_total",
    "coding_problem_count",
    "hackathon_wins",
    "languages_known",
    "leadership_roles_count",
    "soft_skills_score",
    "communication_score",
    "extracurriculars_count",
    "portal_activity_30d",
    "portal_activity_90d",
    "interview_invites_count",
    "interview_pass_rate",
    "resume_updates_30d",
    "mock_interviews_taken",
    "avg_response_time_hours",
    "last_login_days_ago",
    "institute_placement_rate_3m",
    "institute_placement_rate_6m",
    "institute_placement_rate_12m",
    "institute_recruiter_diversity",
    "institute_avg_salary_lpa",
    "placement_cell_active_score",
    "recruiter_visits_year",
    "sector_hiring_index",
    "local_job_density",
    "macro_unemployment_rate",
    "course_demand_index",
    "skill_gap_score",
    "salary_expectation_lpa",
    "has_target_company_referral",
    "competitive_exam_attempted",
    "tier",
    "is_metro",
    # NIRF 2024 per-institute priors. Replace tier-keyed constants with real
    # institute-level signal at inference time.
    "nirf_rank",
    "nirf_score",
    "nirf_median_salary_lpa",
]

CATEGORICAL_FEATURES = ["course_type"]

# Model expects course_type one-hot expanded.
COURSE_ONE_HOT = [f"course_{c}" for c in COURSE_TYPES]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + COURSE_ONE_HOT


@dataclass
class StudentProfile:
    """Wire format used by the resume-parsing endpoint.

    Free-form / friendly. The feature-extractor in features.py turns this
    into the 45-feature numeric vector the model consumes.
    """
    name: str = "Anonymous"
    course_type: str = "BTech-CS"
    institute_name: str = "Tier-2 Institute"
    institute_tier: int = 2
    region: str = "Tier2"
    graduation_year: int = 2024
    cgpa: float = 7.0
    backlogs_count: int = 0
    internships: list | None = None
    certifications: list | None = None
    skills: list | None = None
    projects_count: int = 0
    github_projects: int = 0
    coding_problem_count: int = 0
    hackathon_wins: int = 0
    leadership_roles_count: int = 0
    paper_publications: int = 0
    extracurriculars_count: int = 0
    languages_known: int = 1
    portal_activity_30d: int = 5
    interview_invites_count: int = 0
    salary_expectation_lpa: float = 6.0
    # Optional metadata — populated by the parser, used by predict.py to
    # widen ranges on low-confidence parses and bypass median-anchor on
    # elite outlier profiles. Backwards-compatible: defaults preserve the
    # pre-update behaviour for callers that don't set them.
    field_confidence: dict | None = None       # {field_name: 0..1}
    is_elite_outlier: bool = False             # see resume_parser.tail rule
    elite_outlier_reasons: list | None = None  # short bullets for the UI
    branch: str | None = None                  # explicit branch (for IPR)
    degree: str | None = None                  # explicit degree (for IPR)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "course_type": self.course_type,
            "institute_name": self.institute_name,
            "institute_tier": self.institute_tier,
            "region": self.region,
            "graduation_year": self.graduation_year,
            "cgpa": self.cgpa,
            "backlogs_count": self.backlogs_count,
            "internships": self.internships or [],
            "certifications": self.certifications or [],
            "skills": self.skills or [],
            "projects_count": self.projects_count,
            "github_projects": self.github_projects,
            "coding_problem_count": self.coding_problem_count,
            "hackathon_wins": self.hackathon_wins,
            "leadership_roles_count": self.leadership_roles_count,
            "paper_publications": self.paper_publications,
            "extracurriculars_count": self.extracurriculars_count,
            "languages_known": self.languages_known,
            "portal_activity_30d": self.portal_activity_30d,
            "interview_invites_count": self.interview_invites_count,
            "salary_expectation_lpa": self.salary_expectation_lpa,
            "field_confidence": self.field_confidence or {},
            "is_elite_outlier": self.is_elite_outlier,
            "elite_outlier_reasons": self.elite_outlier_reasons or [],
            "branch": self.branch,
            "degree": self.degree,
        }
