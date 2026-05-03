"""Synthetic dataset generator for CareerSight.

Produces a realistically-correlated cohort of education-loan borrowers with
placement outcomes at 3 / 6 / 12 months and final salary. The data-generating
process mixes:

  - Latent placement-readiness score from skills, internships, CGPA.
  - Institute-level multiplier (tier-1 metros vs tier-3 towns).
  - Sector demand shock (e.g., BFSI hiring slowdown).
  - Behavioural signal noise (portal activity, interview pass rate).

The Weibull survival model converts the latent score into a continuous
time-to-placement, which we then bucket into 3/6/12-month outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import COURSE_TYPES, REGIONS


def _course_base_demand() -> dict[str, float]:
    """Sector demand multipliers — IT/MBA-Finance hot, Mech/Civil cooling."""
    return {
        "BTech-CS": 1.30,
        "MTech-CS": 1.25,
        "MCA": 1.10,
        "BTech-ECE": 1.05,
        "MBA-Finance": 1.20,
        "MBA-Marketing": 1.05,
        "MBA-Operations": 1.00,
        "MBA-HR": 0.85,
        "BTech-Mech": 0.85,
        "BTech-Civil": 0.75,
        "BSc-Nursing": 1.10,
        "BCom": 0.80,
    }


def _course_salary_anchor() -> dict[str, float]:
    """Median LPA for the course at a tier-2 institute, before adjustments."""
    return {
        "BTech-CS": 9.0,
        "MTech-CS": 11.0,
        "MCA": 7.0,
        "BTech-ECE": 6.5,
        "MBA-Finance": 12.0,
        "MBA-Marketing": 9.5,
        "MBA-Operations": 8.5,
        "MBA-HR": 6.5,
        "BTech-Mech": 5.5,
        "BTech-Civil": 4.5,
        "BSc-Nursing": 4.0,
        "BCom": 4.0,
    }


def generate(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    course_type = rng.choice(COURSE_TYPES, size=n, p=_course_dist())
    tier = rng.choice([1, 2, 3], size=n, p=[0.15, 0.55, 0.30])
    region = np.where(tier == 1, "Metro", rng.choice(["Metro", "Tier2", "Tier3"], size=n, p=[0.20, 0.55, 0.25]))
    is_metro = (region == "Metro").astype(int)

    # Academic
    cgpa = np.clip(rng.normal(loc=np.where(tier == 1, 8.0, np.where(tier == 2, 7.2, 6.6)),
                              scale=0.9), 4.0, 10.0)
    graduation_year_offset = rng.integers(-2, 3, size=n)  # years from "now"
    backlogs = rng.poisson(lam=np.where(cgpa < 6.5, 1.5, 0.3), size=n).clip(0, 10)
    paper_pubs = rng.poisson(lam=np.where(tier == 1, 0.4, 0.1), size=n).clip(0, 5)
    semester_consistency = np.clip(rng.beta(5, 2, size=n), 0, 1)

    # Internships scale with tier and CGPA
    intern_count = rng.poisson(lam=np.where(tier == 1, 2.0, np.where(tier == 2, 1.2, 0.6)), size=n).clip(0, 5)
    intern_total_months = (intern_count * rng.uniform(1.5, 5.0, size=n)).clip(0, 24)
    intern_top_tier = (rng.uniform(size=n) < (0.50 * (tier == 1) + 0.20 * (tier == 2) + 0.05 * (tier == 3))).astype(int)
    intern_relevance = np.clip(rng.beta(np.where(intern_count > 0, 4, 1), 2, size=n), 0, 1)

    # Skills
    cert_count = rng.poisson(lam=np.where(cgpa > 7.5, 4, 2), size=n).clip(0, 15)
    relevant_cert_count = (cert_count * rng.uniform(0.3, 0.9, size=n)).round().astype(int)
    github_projects = rng.poisson(lam=np.where(np.isin(course_type, ["BTech-CS", "MTech-CS", "MCA"]), 6, 2), size=n).clip(0, 30)
    github_stars = (github_projects * rng.exponential(scale=8, size=n)).clip(0, 1000)
    coding_count = rng.poisson(lam=np.where(np.isin(course_type, ["BTech-CS", "MTech-CS", "MCA"]),
                                            np.where(tier == 1, 350, 180), 50), size=n).clip(0, 2000)
    hackathon_wins = rng.poisson(lam=np.where(tier == 1, 0.8, 0.3), size=n).clip(0, 10)
    languages = rng.integers(1, 6, size=n)
    leadership_roles = rng.poisson(lam=0.8, size=n).clip(0, 5)

    # Soft signals
    soft_skills = np.clip(rng.beta(4, 2, size=n), 0, 1)
    communication = np.clip(rng.beta(4, 2, size=n), 0, 1)
    extracurriculars = rng.poisson(lam=2.5, size=n).clip(0, 15)

    # Behavioural / real-time
    portal_30d = rng.poisson(lam=20 + 30 * (cgpa > 7.5), size=n).clip(0, 100)
    portal_90d = (portal_30d * rng.uniform(2.0, 3.5, size=n)).astype(int).clip(0, 300)
    interview_invites = rng.poisson(lam=portal_90d * 0.05, size=n).clip(0, 50)
    interview_pass_rate = np.clip(rng.beta(np.maximum(intern_count + 1, 1), 3, size=n), 0, 1)
    resume_updates_30d = rng.poisson(lam=2, size=n).clip(0, 10)
    mock_interviews = rng.poisson(lam=np.where(tier == 1, 4, 2), size=n).clip(0, 20)
    avg_response_hours = rng.exponential(scale=12, size=n).clip(0, 72)
    last_login_days = rng.exponential(scale=3, size=n).clip(0, 60)

    # Institute (correlated with tier)
    institute_pr_3m = np.clip(rng.normal(np.where(tier == 1, 0.55, np.where(tier == 2, 0.30, 0.15)), 0.08), 0, 1)
    institute_pr_6m = np.clip(institute_pr_3m + rng.uniform(0.10, 0.25, size=n), 0, 1)
    institute_pr_12m = np.clip(institute_pr_6m + rng.uniform(0.08, 0.20, size=n), 0, 1)
    institute_diversity = np.clip(rng.beta(np.where(tier == 1, 6, np.where(tier == 2, 3, 2)), 3, size=n), 0, 1)
    institute_avg_salary = np.clip(rng.normal(
        loc=np.where(tier == 1, 14, np.where(tier == 2, 7.5, 4.5)),
        scale=np.where(tier == 1, 3, 1.8)
    ), 3, 30)
    placement_cell_score = np.clip(rng.beta(np.where(tier == 1, 6, np.where(tier == 2, 3, 2)), 3, size=n), 0, 1)
    recruiter_visits = rng.poisson(lam=np.where(tier == 1, 60, np.where(tier == 2, 25, 8)), size=n).clip(0, 100)

    # Market signals
    course_demand = np.array([_course_base_demand()[c] for c in course_type])
    sector_hiring = np.clip(course_demand * rng.normal(1.0, 0.10, size=n), 0.4, 2.0)
    local_density = np.where(region == "Metro", rng.normal(80, 20, size=n),
                             np.where(region == "Tier2", rng.normal(40, 12, size=n),
                                      rng.normal(15, 6, size=n))).clip(0, 200)
    macro_unemp = np.clip(rng.normal(7.5, 1.0, size=n), 3, 15)
    course_demand_index = sector_hiring  # alias for explainability

    skill_gap = np.clip(rng.beta(3, 5, size=n) * (1 - 0.3 * (relevant_cert_count > 2)), 0, 1)

    salary_expectation = np.clip(np.array([_course_salary_anchor()[c] for c in course_type]) *
                                 rng.uniform(0.8, 1.4, size=n), 3, 30)

    referral = (rng.uniform(size=n) < 0.15).astype(int)
    competitive_exam = (rng.uniform(size=n) < 0.25).astype(int)

    # NIRF 2024 institute-level features. Real institute distribution sampled
    # so the model picks up rank/score/salary as continuous covariates rather
    # than collapsing onto three tier constants. Tier-1 -> rank 1-30 + score
    # 60-90 + salary 18-30 LPA. Tier-2 -> rank 30-100 + score 50-65 + salary
    # 8-15 LPA. Tier-3 -> rank > 200 + score 25-45 + salary 3-7 LPA.
    nirf_rank = np.where(
        tier == 1, rng.uniform(1, 30, size=n),
        np.where(tier == 2, rng.uniform(30, 100, size=n),
                 rng.uniform(150, 400, size=n)),
    )
    nirf_score = np.where(
        tier == 1, rng.uniform(60, 90, size=n),
        np.where(tier == 2, rng.uniform(48, 65, size=n),
                 rng.uniform(20, 45, size=n)),
    )
    nirf_median_salary = np.where(
        tier == 1, rng.uniform(18, 32, size=n),
        np.where(tier == 2, rng.uniform(8, 15, size=n),
                 rng.uniform(3, 7, size=n)),
    )

    # Latent placement-readiness score (higher = faster placement).
    # Coefficients are intentionally strong so the synthetic dataset has a
    # learnable signal — real-world models get reinforced via online learning.
    z = (
        0.30 * (cgpa - 7.0)
        + 0.25 * (tier == 1).astype(int)
        - 0.25 * (tier == 3).astype(int)
        + 0.18 * (intern_total_months / 12)
        + 0.20 * intern_top_tier
        + 0.10 * intern_relevance
        + 0.18 * (relevant_cert_count / 5)
        + 0.08 * (github_projects / 10)
        + 0.08 * (coding_count / 500)
        + 0.10 * hackathon_wins / 3
        + 0.18 * (sector_hiring - 1.0)
        + 0.20 * (institute_pr_6m - 0.4)
        + 0.12 * (nirf_score - 50) / 30        # NIRF score signal
        - 0.10 * (nirf_rank - 50) / 100        # lower rank = better
        + 0.10 * placement_cell_score
        + 0.08 * (institute_diversity - 0.4)
        + 0.10 * (interview_pass_rate - 0.3)
        + 0.10 * (portal_30d / 50)
        + 0.08 * referral
        + 0.06 * communication
        - 0.18 * (backlogs / 3)
        - 0.18 * skill_gap
        - 0.10 * (macro_unemp - 7.5) / 5
        + rng.normal(0, 0.10, size=n)
    )

    # Time-to-placement via Weibull. Higher z -> smaller scale (faster).
    # Higher shape concentrates probability mass, lowering aleatoric noise.
    weibull_shape = 2.2
    weibull_scale = np.clip(7.0 * np.exp(-z), 1.0, 36.0)
    t_placed_months = rng.weibull(weibull_shape, size=n) * weibull_scale

    # Some students never get placed in 12m horizon — censor at 13.
    never_placed_p = np.clip(0.05 + 0.15 * (z < -0.3) + 0.10 * (skill_gap > 0.7), 0, 0.5)
    never_mask = rng.uniform(size=n) < never_placed_p
    t_placed_months = np.where(never_mask, 14.0, t_placed_months)

    placed_3m = (t_placed_months <= 3).astype(int)
    placed_6m = (t_placed_months <= 6).astype(int)
    placed_12m = (t_placed_months <= 12).astype(int)

    # Final salary if placed (CQR target). Anchor by course + tier + skills.
    salary_anchor = np.array([_course_salary_anchor()[c] for c in course_type])
    salary_anchor = salary_anchor * np.where(tier == 1, 1.4, np.where(tier == 2, 1.0, 0.7))
    salary = (
        salary_anchor
        * (1.0 + 0.10 * z)
        * np.exp(rng.normal(0, 0.15, size=n))
    )
    salary = np.where(placed_12m == 1, salary, np.nan)
    salary = np.clip(salary, 2.0, 60.0)

    df = pd.DataFrame({
        "course_type": course_type,
        "institute_tier": tier,
        "tier": tier,
        "region": region,
        "is_metro": is_metro,
        "cgpa": cgpa,
        "graduation_year_offset": graduation_year_offset,
        "backlogs_count": backlogs,
        "paper_publications": paper_pubs,
        "semester_consistency": semester_consistency,
        "internship_count": intern_count,
        "internship_total_months": intern_total_months,
        "internship_top_tier": intern_top_tier,
        "internship_relevance_score": intern_relevance,
        "certifications_count": cert_count,
        "relevant_certs_count": relevant_cert_count,
        "github_projects": github_projects,
        "github_stars_total": github_stars,
        "coding_problem_count": coding_count,
        "hackathon_wins": hackathon_wins,
        "languages_known": languages,
        "leadership_roles_count": leadership_roles,
        "soft_skills_score": soft_skills,
        "communication_score": communication,
        "extracurriculars_count": extracurriculars,
        "portal_activity_30d": portal_30d,
        "portal_activity_90d": portal_90d,
        "interview_invites_count": interview_invites,
        "interview_pass_rate": interview_pass_rate,
        "resume_updates_30d": resume_updates_30d,
        "mock_interviews_taken": mock_interviews,
        "avg_response_time_hours": avg_response_hours,
        "last_login_days_ago": last_login_days,
        "institute_placement_rate_3m": institute_pr_3m,
        "institute_placement_rate_6m": institute_pr_6m,
        "institute_placement_rate_12m": institute_pr_12m,
        "institute_recruiter_diversity": institute_diversity,
        "institute_avg_salary_lpa": institute_avg_salary,
        "placement_cell_active_score": placement_cell_score,
        "recruiter_visits_year": recruiter_visits,
        "sector_hiring_index": sector_hiring,
        "local_job_density": local_density,
        "macro_unemployment_rate": macro_unemp,
        "course_demand_index": course_demand_index,
        "skill_gap_score": skill_gap,
        "salary_expectation_lpa": salary_expectation,
        "has_target_company_referral": referral,
        "competitive_exam_attempted": competitive_exam,
        "nirf_rank": nirf_rank,
        "nirf_score": nirf_score,
        "nirf_median_salary_lpa": nirf_median_salary,
        # outcomes
        "t_placed_months": t_placed_months,
        "placed_3m": placed_3m,
        "placed_6m": placed_6m,
        "placed_12m": placed_12m,
        "final_salary_lpa": salary,
    })
    return df


def _course_dist() -> list[float]:
    """Realistic mix of courses in an Indian education-loan portfolio."""
    p = [
        0.18,  # BTech-CS
        0.10,  # BTech-ECE
        0.07,  # BTech-Mech
        0.04,  # BTech-Civil
        0.10,  # MBA-Finance
        0.07,  # MBA-Marketing
        0.05,  # MBA-HR
        0.05,  # MBA-Operations
        0.10,  # BSc-Nursing
        0.07,  # MCA
        0.05,  # MTech-CS
        0.12,  # BCom
    ]
    s = sum(p)
    return [x / s for x in p]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/synthetic_students.csv")
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    print("Placement rates:",
          {k: round(df[k].mean(), 3) for k in ["placed_3m", "placed_6m", "placed_12m"]})
    print("Mean salary (placed):", round(df["final_salary_lpa"].mean(skipna=True), 2), "LPA")


if __name__ == "__main__":
    main()
