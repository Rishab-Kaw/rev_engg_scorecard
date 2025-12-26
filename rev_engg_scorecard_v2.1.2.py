# rev_engg_scorecard_v2.1.2.py
# Single-file Streamlit app: Programs / Fields-of-study (CIP4 + Credential) across schools.
#
# v2.0 changes vs v1.7:
# - UI: Added a hero header with query summary.
# - UI: Added progress + status UX for paginated API fetch.
# - UI: Refinement panel is inside the main flow (collapsible expander).
# - UI: Added a 'Sort + Quick filters' chip row (native Streamlit controls).
#
# Functional behavior retained:
# - Recommendation unit: Programs / Fields-of-study (CIP4 + Credential) across schools.
# - CIP-4 dropdown uses ONLY project-root ./cip_4_digit.json
# - No 'Only show schools flagged by DOL' filter functionality
# - No 'Local convenience filter (post-query)' functionality
#
# Requirements retained:
# - CIP-4 dropdown uses ONLY project-root ./cip_4_digit.json
# - Removed "Only show schools flagged by DOL" filter functionality
# - Removed "Local convenience filter (post-query)" functionality
#
# Required layout:
#   <project root>/
#     rev_engg_scorecard_v2.1.2.py
#     secrets.toml              (contains SCORECARD_API_KEY)
#     cip_4_digit.json          (CIP-4 catalog copied from the Scorecard repo)
    # No additional lookup JSONs required for v1.6 (mappings are embedded in code).
#
# Run:
#   pip install streamlit requests pandas
#   streamlit run rev_engg_scorecard_v2.1.2.py

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------

SCORECARD_BASE_URL = "https://api.data.gov/ed/collegescorecard/v1/schools"

FOS_DEGREES: List[Tuple[str, int]] = [
    ("UNDERGRADUATE (Level 1)", 1),
    ("ASSOCIATE (Level 2)", 2),
    ("BACHELOR'S (Level 3)", 3),
    ("BACCALAUREATE CERTIFICATE (Level 4)", 4),
    ("MASTER'S (Level 5)", 5),
    ("DOCTORATE (Level 6)", 6),
]

SORT_OPTIONS: List[Tuple[str, str]] = [
    ("School Name", "name"),
    ("Earnings", "fos_median_earnings"),
    ("Debt", "fos_debt"),
    ("Graduates", "fos_graduates"),
]

SORT_FIELD_MAP: Dict[str, str] = {
    "name": "school.name",
    "fos_median_earnings": "latest.programs.cip_4_digit.earnings.4_yr.overall_median_earnings",
    "fos_debt": "latest.programs.cip_4_digit.debt.staff_grad_plus.all.eval_inst.median",
    "fos_graduates": "latest.programs.cip_4_digit.counts.ipeds_awards2",
}

# Repo-like defaults
DEFAULT_FIXED_FILTERS: Dict[str, Any] = {
    "school.operating": 1,
    "school.degrees_awarded.predominant__range": "1..3",
    "latest.student.size_category": "1,2,3",
}

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM",
    "NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA",
    "WV","WI","WY",
]

DEFAULT_EARNINGS_K_RANGE = (0, 150)  # thousands of dollars
DEFAULT_DEBT_K_RANGE = (0, 50)       # thousands of dollars

# -----------------------------
# Repo-style label mappings (recreated from the Scorecard repo)
# -----------------------------
# Source reference (repo): composables/useFilters.js
# - control() / locale() / sizeCategory() / yearsText()
#
# We embed these mappings so you do NOT need extra JSON files beyond cip_4_digit.json.

OWNERSHIP_LABELS: Dict[str, str] = {
    "-1": "Unknown",
    "1": "Public",
    "2": "Private Nonprofit",
    "3": "Private For-Profit",
}

# NCES locale codes grouped in the Scorecard repo as City/Suburban/Town/Rural.
LOCALE_LABELS: Dict[str, str] = {
    "-1": "Locale Unknown",
    "11": "City", "12": "City", "13": "City",
    "21": "Suburban", "22": "Suburban", "23": "Suburban",
    "31": "Town", "32": "Town", "33": "Town",
    "41": "Rural", "42": "Rural", "43": "Rural",
}

# Scorecard repo uses student.size_category (1/2/3) as Small/Medium/Large.
SIZE_CATEGORY_LABELS: Dict[str, str] = {
    "1": "Small",
    "2": "Medium",
    "3": "Large",
}

# Scorecard repo uses school.degrees_awarded.predominant as "Certificate / 2-yr / 4-yr".
PREDOMINANT_DEGREE_YEARS: Dict[str, str] = {
    "1": "Certificate",
    "2": "2 Year",
    "3": "4 Year",
}

def _to_code_str(v: Any) -> str:
    if v is None:
        return "-1"
    try:
        # Preserve integers cleanly
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return str(int(v))
    except Exception:
        pass
    return str(v).strip() or "-1"

def ownership_label(v: Any) -> str:
    return OWNERSHIP_LABELS.get(_to_code_str(v), "Unknown")

def locale_label(v: Any) -> str:
    return LOCALE_LABELS.get(_to_code_str(v), "Locale Unknown")

def size_category_label(v: Any) -> str:
    return SIZE_CATEGORY_LABELS.get(_to_code_str(v), "size unknown")

def predominant_years_label(v: Any) -> str:
    return PREDOMINANT_DEGREE_YEARS.get(_to_code_str(v), "4 Year")

def _emoji_for_ownership(lbl: str) -> str:
    if lbl.lower().startswith("public"):
        return "🏛️"
    if "nonprofit" in lbl.lower():
        return "🏫"
    if "for-profit" in lbl.lower() or "for profit" in lbl.lower():
        return "💼"
    return "❓"

def _emoji_for_locale(lbl: str) -> str:
    m = lbl.lower()
    if m == "city":
        return "🏙️"
    if m == "suburban":
        return "🏘️"
    if m == "town":
        return "🏡"
    if m == "rural":
        return "🌾"
    return "❓"

def _emoji_for_size(lbl: str) -> str:
    m = lbl.lower()
    if m == "small":
        return "🟢"
    if m == "medium":
        return "🟡"
    if m == "large":
        return "🔵"
    return "❓"

def _emoji_for_years(lbl: str) -> str:
    m = lbl.lower()
    if "certificate" in m:
        return "📜"
    if "2" in m:
        return "🎓"
    if "4" in m:
        return "🎓"
    return "🎓"

# -----------------------------
# Secrets (project-root secrets.toml)
# -----------------------------


_TOML_KEY_RE = re.compile(r"^\s*SCORECARD_API_KEY\s*=\s*(?P<val>.+?)\s*$")


def _parse_toml_value(raw: str) -> str:
    s = raw.strip()
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1].strip()
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        return s[1:-1].strip()
    return s.strip()


def _load_api_key() -> str:
    # Optional env-var fallback (useful in deployments)
    env_key = os.environ.get("SCORECARD_API_KEY", "").strip()
    if env_key:
        return env_key

    candidates = [
        Path(__file__).resolve().parent / "secrets.toml",
        Path.cwd().resolve() / "secrets.toml",
    ]

    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _TOML_KEY_RE.match(line)
            if not m:
                continue
            key = _parse_toml_value(m.group("val"))
            if key:
                return key

    tried = ", ".join(str(c) for c in candidates)
    raise RuntimeError(
        "SCORECARD_API_KEY not found.\n\n"
        "Expected a secrets.toml (same folder as this script) containing:\n"
        '  SCORECARD_API_KEY = "PASTE_YOUR_API_KEY_HERE"\n\n'
        f"Tried: {tried}"
    )


def _api_key_hash(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


# -----------------------------
# CIP-4 catalog loader (CIP-4 only, from project root)
# -----------------------------

def _normalize_cip4_from_label(label: str) -> Optional[str]:
    """
    Repo CIP4 labels look like: "01.00", "11.07", etc.
    API expects a 4-digit code like "0100", "1107".
    """
    if not label:
        return None
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if len(digits) < 4:
        return None
    return digits[:4]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_cip4_catalog_from_project_root() -> List[Tuple[str, str, str]]:
    """
    Required by you: use ./cip_4_digit.json from project root.
    Returns list of tuples:
      (label_with_dot, title, code4digits)
    """
    base = Path(__file__).resolve().parent
    p = base / "cip_4_digit.json"
    if not p.exists():
        # Allow current working dir if Streamlit launched elsewhere.
        p2 = Path.cwd().resolve() / "cip_4_digit.json"
        if p2.exists():
            p = p2
        else:
            raise RuntimeError(
                "CIP-4 catalog not found.\n\n"
                "Expected:\n"
                "  ./cip_4_digit.json\n\n"
                "Tip: copy from the Scorecard repo:\n"
                "  assets/data/cip_4_digit.json\n"
            )

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise RuntimeError("cip_4_digit.json exists but is not a non-empty JSON list.")

    out: List[Tuple[str, str, str]] = []
    for item in data:
        label = str(item.get("label", "")).strip()
        title = str(item.get("value", "")).strip()
        code4 = _normalize_cip4_from_label(label)
        if not label or not title or not code4:
            continue
        out.append((label, title, code4))

    out.sort(key=lambda t: (t[1].lower(), t[0]))
    st.session_state["cip4_loaded_from"] = str(p)
    return out


def _fos_option_label(opt: Tuple[str, str, str]) -> str:
    label_with_dot, title, _ = opt
    return f"{title} ({label_with_dot})"


# -----------------------------
# Utilities
# -----------------------------

def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for seg in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, list):
            try:
                idx = int(seg)
                cur = cur[idx]
            except Exception:
                return default
        elif isinstance(cur, dict):
            if seg not in cur:
                return default
            cur = cur[seg]
        else:
            return default
    return cur


def _format_money(v: Any) -> str:
    if v is None:
        return "N/A"
    try:
        x = float(v)
        if math.isnan(x):
            return "N/A"
        return f"${x:,.0f}"
    except Exception:
        return "N/A"


def _format_int(v: Any) -> str:
    if v is None:
        return "N/A"
    try:
        x = int(round(float(v)))
        return f"{x:,}"
    except Exception:
        return "N/A"


# -----------------------------
# Query model
# -----------------------------

@dataclass
class QueryInputs:
    cip4: str
    credential_level: int

    location_mode: str  # None / State / ZIP / Lat-Long
    states: List[str]
    zip_code: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    distance_miles: Optional[int]

    fos_salary_range: Optional[Tuple[int, int]]  # dollars (None => no filter)
    fos_debt_range: Optional[Tuple[int, int]]    # dollars (None => no filter)
    sort_key: str
    sort_dir: str  # asc/desc


def _build_scorecard_params(inp: QueryInputs, page: int, per_page: int) -> Dict[str, Any]:
    per_page = min(max(int(per_page), 1), 100)

    q: Dict[str, Any] = dict(DEFAULT_FIXED_FILTERS)

    # Program filters (recommendation unit)
    q["latest.programs.cip_4_digit.code"] = inp.cip4
    q["latest.programs.cip_4_digit.credential.level"] = str(inp.credential_level)

    # Only apply range filters if user narrowed them (prevents excluding nulls by accident)
    if inp.fos_salary_range is not None:
        q["latest.programs.cip_4_digit.earnings.4_yr.overall_median_earnings__range"] = (
            f"{inp.fos_salary_range[0]}..{inp.fos_salary_range[1]}"
        )
    if inp.fos_debt_range is not None:
        q["latest.programs.cip_4_digit.debt.staff_grad_plus.all.eval_inst.median__range"] = (
            f"{inp.fos_debt_range[0]}..{inp.fos_debt_range[1]}"
        )

    # Location filters
    if inp.location_mode == "State" and inp.states:
        q["school.state"] = ",".join(inp.states)
    elif inp.location_mode == "ZIP" and inp.zip_code:
        q["zip"] = inp.zip_code
        if inp.distance_miles:
            q["distance"] = int(inp.distance_miles)
    elif inp.location_mode == "Lat-Long" and (inp.lat is not None and inp.lon is not None):
        q["lat"] = inp.lat
        q["lon"] = inp.lon
        if inp.distance_miles:
            q["distance"] = int(inp.distance_miles)


    # Sorting
    api_sort_field = SORT_FIELD_MAP.get(inp.sort_key, SORT_FIELD_MAP["name"])
    q["sort"] = f"{api_sort_field}:{inp.sort_dir}"

    # Pagination + response shape
    q["page"] = page
    q["per_page"] = per_page
    q["keys_nested"] = "true"
    q["all_programs_nested"] = "true"

    # Fields: keep response small but sufficient for table/cards
    
    # Fields: keep response small but sufficient for table/cards
    # v1.6 adds school ownership/control, locale, predominant degree, and enrollment size + size category.
    q["fields"] = ",".join([
        "id",
        "school.name",
        "school.city",
        "school.state",
        "school.zip",
        "school.school_url",
        "school.ownership",
        "school.locale",
        "school.degrees_awarded.predominant",
        "latest.student.size",
        "latest.student.size_category",
        "latest.hcm2",
        "latest.programs.cip_4_digit",
    ])
    return q


@st.cache_data(show_spinner=False, ttl=60 * 10)
def _scorecard_fetch_page(params: Dict[str, Any], api_key_hash: str) -> Dict[str, Any]:
    # api_key_hash varies cache keys without storing the real key in cache args.
    _ = api_key_hash

    api_key = _load_api_key()
    p = dict(params)
    p["api_key"] = api_key

    r = requests.get(SCORECARD_BASE_URL, params=p, timeout=40)

    try:
        data = r.json()
    except Exception:
        r.raise_for_status()
        raise

    if r.status_code >= 400:
        raise RuntimeError(f"Scorecard API HTTP {r.status_code}: {data}")

    if isinstance(data, dict) and data.get("errors"):
        raise RuntimeError(f"Scorecard API error(s): {data['errors']}")

    return data


def fetch_all_results(inp: QueryInputs, per_page: int = 100, hard_cap: int = 10000, progress_cb=None) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    """
    Show-all behavior: keep paging until all results are collected (or hard_cap is reached).
    Returns (schools, metadata, elapsed_seconds).
    """
    api_key = _load_api_key()
    kh = _api_key_hash(api_key)

    all_results: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    started = time.time()
    page = 0
    while True:
        params = _build_scorecard_params(inp, page=page, per_page=per_page)
        payload = _scorecard_fetch_page(params, kh)

        meta = payload.get("metadata", {}) or {}
        results = payload.get("results", []) or []
        all_results.extend(results)

        total = int(meta.get("total", len(all_results)) or len(all_results))

        if callable(progress_cb):
            # total can be unknown early; compute safe fraction
            try:
                frac = min(1.0, float(len(all_results)) / float(total)) if total else 0.0
            except Exception:
                frac = 0.0
            progress_cb(page=page, total=total, collected=len(all_results), fraction=frac)

        if not results:
            break
        if len(all_results) >= total:
            break
        if len(all_results) >= hard_cap:
            break

        page += 1

    return all_results, meta, (time.time() - started)


def condition_to_fos_rows(schools: List[Dict[str, Any]], cip4: str, cred_level: int) -> List[Dict[str, Any]]:
    """
    Convert /schools records into FoS rows:
      - one row per (school, program) match for the selected CIP4 + credential level.
    """
    rows: List[Dict[str, Any]] = []
    cip4_norm = str(cip4).zfill(4)

    for s in schools:
        unit_id = _safe_get(s, "id")
        school_obj = _safe_get(s, "school", {}) or {}
        latest_obj = _safe_get(s, "latest", {}) or {}
        programs = _safe_get(s, "latest.programs.cip_4_digit", []) or []
        if isinstance(programs, dict):
            programs = [programs]

        for prog in programs:
            code = str(_safe_get(prog, "code", "") or "").zfill(4)
            lvl = _safe_get(prog, "credential.level")

            if code != cip4_norm:
                continue
            if lvl is None:
                continue
            try:
                if int(lvl) != int(cred_level):
                    continue
            except Exception:
                continue

            row = dict(prog)
            row["unit_id"] = unit_id
            row["institution"] = {
                "id": unit_id,
                "school": school_obj,
                "latest": latest_obj,
            }
            rows.append(row)

    return rows


def fos_rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a display dataframe with commonly-used FoS metrics.
    """
    out = []
    for r in rows:
        school = _safe_get(r, "institution.school", {}) or {}
        latest = _safe_get(r, "institution.latest", {}) or {}

        out.append({
            "School": school.get("name"),
            "City": school.get("city"),
            "State": school.get("state"),
            "ZIP": school.get("zip"),

            # New (v1.6): school context needed for Scorecard-style cards
            "Ownership (code)": school.get("ownership"),
            "Ownership": ownership_label(school.get("ownership")),
            "Locale (code)": school.get("locale"),
            "Locale": locale_label(school.get("locale")),
            "Predominant degree (code)": _safe_get(school, "degrees_awarded.predominant"),
            "Years": predominant_years_label(_safe_get(school, "degrees_awarded.predominant")),
            "Enrollment": _safe_get(latest, "student.size"),
            "Size category (code)": _safe_get(latest, "student.size_category"),
            "Size category": size_category_label(_safe_get(latest, "student.size_category")),

            # Program (FoS) fields
            "Program": _safe_get(r, "title"),
            "Credential Level": _safe_get(r, "credential.level"),

            # Raw numeric metrics (keep numeric for st.metric + derived monthly values)
            "Median Earnings (4yr) RAW": _safe_get(r, "earnings.4_yr.overall_median_earnings"),
            "Median Earnings (5yr) RAW": _safe_get(r, "earnings.5_yr.overall_median_earnings"),
            "Median Debt RAW": _safe_get(r, "debt.staff_grad_plus.all.eval_inst.median"),
            "Monthly Loan Payment RAW": _safe_get(r, "debt.staff_grad_plus.all.eval_inst.median_payment"),
            "Graduates RAW": _safe_get(r, "counts.ipeds_awards2"),

            # Display-friendly columns (table)
            "Median Earnings (5yr)": _safe_get(r, "earnings.5_yr.overall_median_earnings"),
            "Median Earnings (4yr)": _safe_get(r, "earnings.4_yr.overall_median_earnings"),
            "Median Debt": _safe_get(r, "debt.staff_grad_plus.all.eval_inst.median"),
            "Monthly Loan Payment": _safe_get(r, "debt.staff_grad_plus.all.eval_inst.median_payment"),
            "Graduates": _safe_get(r, "counts.ipeds_awards2"),

            "School URL": school.get("school_url"),
            "UnitID": r.get("unit_id"),
            "CIP4": _safe_get(r, "code"),
        })

    df = pd.DataFrame(out)

    if not df.empty:
        df["Median Earnings (5yr)"] = df["Median Earnings (5yr)"].apply(_format_money)
        df["Median Earnings (4yr)"] = df["Median Earnings (4yr)"].apply(_format_money)
        df["Median Debt"] = df["Median Debt"].apply(_format_money)
        df["Monthly Loan Payment"] = df["Monthly Loan Payment"].apply(_format_money)
        df["Graduates"] = df["Graduates"].apply(_format_int)
        df["Enrollment"] = df["Enrollment"].apply(_format_int)

    return df


# -----------------------------
# Streamlit UI (native Streamlit, light theme, wide layout)
# -----------------------------

st.set_page_config(page_title="FoS Finder (v2.1.2)", layout="wide")

# -----------------------------
# Hero header (v2.1)
# -----------------------------
with st.container(border=True):
    st.markdown("## Programs / Fields of Study Finder")
    st.caption("College Scorecard API • Programs/Fields-of-study (CIP4 + Credential) across schools")

st.markdown(
    """
This app replicates the College Scorecard repo’s **Fields of Study** search flow using the public API.
It calls `/v1/schools` and conditions results into **program offerings across schools**.

- **Field of Study + Degree Type** are required inputs.
- Filters & sorting are visible **before** search and applied **before** querying the API.
- Results are fetched across all pages (per-page cap is 100).
"""
)

# Early validations (fail fast)
try:
    _ = _load_api_key()
except Exception as e:
    st.error(str(e))
    st.stop()

try:
    cip4_catalog = load_cip4_catalog_from_project_root()
except Exception as e:
    st.error(str(e))
    st.stop()

# --- Search + Filters + Sorting (v1.5: visible before initial search) ---
# v1.2-style behavior:
# - All filters/sorting are shown immediately.
# - API query executes only when user clicks SEARCH.
# - Filters are applied before querying the API (not post-query).

with st.form("search_form", clear_on_submit=False):
    # Row 1: Required inputs + Search button
    col_fos, col_deg, col_btn = st.columns([6, 4, 2], gap="large")

    with col_fos:
        st.markdown("**Search Fields of Study (Required)**")
        fos_selected = st.selectbox(
            " ",
            options=cip4_catalog,
            format_func=_fos_option_label,
            index=None,
            placeholder="Type to search",
            label_visibility="collapsed",
            key="fos_selected",
        )

    with col_deg:
        st.markdown("**Select Degree Type (Required)**")
        degree_label = st.selectbox(
            "  ",
            options=[lbl for (lbl, _) in FOS_DEGREES],
            index=None,
            placeholder="Select one",
            label_visibility="collapsed",
            key="degree_label",
        )

    with col_btn:
        st.markdown("&nbsp;")
        submitted = st.form_submit_button(
            "SEARCH",
            type="primary",
            use_container_width=True,
        )

    st.markdown("---")

    # Row 2: Sort + Quick filters (chips row)

    # Refinement panel inside main flow (collapsible)
    with st.expander("Refinement panel (advanced filters)", expanded=False):
        # Location
        st.markdown("#### Location")
        location_mode = st.radio(
            "Filter by location",
            options=["None", "State", "ZIP", "Lat-Long"],
            horizontal=True,
            key="location_mode",
        )

        if location_mode == "State":
            st.multiselect("States", options=US_STATES, default=st.session_state.get("states_sel", []), key="states_sel")
        elif location_mode == "ZIP":
            cc1, cc2 = st.columns([2, 2])
            with cc1:
                st.text_input("ZIP code", value=st.session_state.get("zip_code", ""), placeholder="e.g., 10001", key="zip_code")
            with cc2:
                st.number_input("Distance (miles)", min_value=1, max_value=500, value=int(st.session_state.get("distance_zip", 50) or 50), step=1, key="distance_zip")
        elif location_mode == "Lat-Long":
            cc1, cc2, cc3 = st.columns([2, 2, 2])
            with cc1:
                st.number_input("Latitude", value=float(st.session_state.get("lat", 0.0) or 0.0), format="%.6f", key="lat")
            with cc2:
                st.number_input("Longitude", value=float(st.session_state.get("lon", 0.0) or 0.0), format="%.6f", key="lon")
            with cc3:
                st.number_input("Distance (miles)", min_value=1, max_value=500, value=int(st.session_state.get("distance_latlon", 50) or 50), step=1, key="distance_latlon")

        # Outcomes
        st.markdown("#### Program outcomes")
        rc1, rc2 = st.columns([2, 2], gap="large")

        with rc1:
            st.slider(
                "Median earnings range (thousands of $) — 4-year measure",
                min_value=DEFAULT_EARNINGS_K_RANGE[0],
                max_value=DEFAULT_EARNINGS_K_RANGE[1],
                value=st.session_state.get("sal_k", DEFAULT_EARNINGS_K_RANGE),
                step=1,
                key="sal_k",
            )

        with rc2:
            st.slider(
                "Median debt range (thousands of $)",
                min_value=DEFAULT_DEBT_K_RANGE[0],
                max_value=DEFAULT_DEBT_K_RANGE[1],
                value=st.session_state.get("debt_k", DEFAULT_DEBT_K_RANGE),
                step=1,
                key="debt_k",
            )

        st.markdown("#### Sorting (manual override)")
        sc1, sc2 = st.columns([2, 1], gap="large")
        with sc1:
            st.selectbox("Sort by", options=[lbl for (lbl, _) in SORT_OPTIONS], index=0, key="sort_label")
        with sc2:
            st.selectbox("Direction", options=["asc", "desc"], index=0, key="sort_dir")

# After submit: validate required fields.

if submitted and (fos_selected is None or degree_label is None):
    st.error("Field of Study and Degree Type are required.")
    st.stop()

# If the user has not searched yet, do not fetch results.
# (Filters/sorting remain visible above.)
if not submitted and not st.session_state.get("has_searched", False):
    st.stop()

# If we reached here:
# - Either user just submitted, or a previous search was already performed and the app reran.
# We use the current session_state values to build the query.
st.session_state["has_searched"] = True

# Resolve required selection values from state (persistent)
fos_selected = st.session_state.get("fos_selected")
degree_label = st.session_state.get("degree_label")
if fos_selected is None or degree_label is None:
    st.error("Field of Study and Degree Type are required.")
    st.stop()

label_with_dot, fos_title, cip4_code = fos_selected
cred_level = dict(FOS_DEGREES)[degree_label]


# Build query inputs from current state
location_mode = st.session_state.get("location_mode", "None")
states_sel = st.session_state.get("states_sel", []) if location_mode == "State" else []
zip_code = (st.session_state.get("zip_code", "") or "").strip() if location_mode == "ZIP" else ""
zip_code = zip_code or None
lat = st.session_state.get("lat", None) if location_mode == "Lat-Long" else None
lon = st.session_state.get("lon", None) if location_mode == "Lat-Long" else None

distance_miles: Optional[int] = None
if location_mode == "ZIP":
    distance_miles = int(st.session_state.get("distance_zip", 50) or 50)
elif location_mode == "Lat-Long":
    distance_miles = int(st.session_state.get("distance_latlon", 50) or 50)

sal_k = st.session_state.get("sal_k", DEFAULT_EARNINGS_K_RANGE)
debt_k = st.session_state.get("debt_k", DEFAULT_DEBT_K_RANGE)

salary_range = None
if tuple(sal_k) != tuple(DEFAULT_EARNINGS_K_RANGE):
    salary_range = (int(sal_k[0]) * 1000, int(sal_k[1]) * 1000)

debt_range = None
if tuple(debt_k) != tuple(DEFAULT_DEBT_K_RANGE):
    debt_range = (int(debt_k[0]) * 1000, int(debt_k[1]) * 1000)


sort_label = st.session_state.get("sort_label", SORT_OPTIONS[0][0])
sort_key = dict(SORT_OPTIONS)[sort_label]
sort_dir = st.session_state.get("sort_dir", "asc")


inputs = QueryInputs(
    cip4=cip4_code,
    credential_level=int(cred_level),
    location_mode=location_mode,
    states=states_sel,
    zip_code=zip_code,
    lat=float(lat) if (location_mode == "Lat-Long" and lat is not None) else None,
    lon=float(lon) if (location_mode == "Lat-Long" and lon is not None) else None,
    distance_miles=distance_miles if location_mode in ("ZIP", "Lat-Long") else None,
    fos_salary_range=salary_range,
    fos_debt_range=debt_range,
    sort_key=sort_key,
    sort_dir=sort_dir,
)



# -----------------------------
# Query result caching (v1.5.1)
# -----------------------------
# Streamlit reruns the script on ANY widget interaction outside the form (e.g., Display Options).
# Without caching, the app would re-fetch the API on those reruns.
#
# Strategy:
# - Compute a stable query_key for the current "search inputs" (FoS + filters + sorting).
# - If we already fetched results for the same key, reuse them from session_state.
#
# NOTE: Filters/sorting widgets are inside the form, so their values only "commit" on SEARCH.
#       That means query_key changes only when SEARCH is clicked (as intended).

def _query_key_from_inputs(inp: QueryInputs) -> str:
    payload = {
        "cip4": inp.cip4,
        "credential_level": inp.credential_level,
        "location_mode": inp.location_mode,
        "states": inp.states,
        "zip_code": inp.zip_code,
        "lat": inp.lat,
        "lon": inp.lon,
        "distance_miles": inp.distance_miles,
        "fos_salary_range": inp.fos_salary_range,
        "fos_debt_range": inp.fos_debt_range,
        "sort_key": inp.sort_key,
        "sort_dir": inp.sort_dir,
    }
    return json.dumps(payload, sort_keys=True, default=str)

query_key = _query_key_from_inputs(inputs)

should_fetch = submitted or ("last_query_key" not in st.session_state) or (st.session_state.get("last_query_key") != query_key)


# --- Fetch + condition ---
if should_fetch:
    # Progress + status UX (v2.0)
    status = st.status("Fetching results from the College Scorecard API…", expanded=True)
    prog = st.progress(0, text="Starting…")

    def _progress_cb(page: int, total: int, collected: int, fraction: float):
        msg = f"Fetched page {page + 1} • collected {collected:,} of ~{total:,}"
        prog.progress(int(max(0.0, min(1.0, fraction)) * 100), text=msg)
        status.update(label=msg, state="running", expanded=True)

    try:
        school_results, meta, elapsed = fetch_all_results(inputs, per_page=100, hard_cap=10000, progress_cb=_progress_cb)
    except Exception as e:
        status.update(label="Fetch failed", state="error", expanded=True)
        st.error(str(e))
        st.stop()

    prog.progress(100, text="Done")
    status.update(label="Fetch complete", state="complete", expanded=False)

    fos_rows = condition_to_fos_rows(school_results, cip4=inputs.cip4, cred_level=inputs.credential_level)

    # Persist for non-search reruns (e.g., display options changes)
    st.session_state["last_query_key"] = query_key
    st.session_state["last_school_results"] = school_results
    st.session_state["last_meta"] = meta
    st.session_state["last_elapsed"] = elapsed
    st.session_state["last_fos_rows"] = fos_rows
else:
    school_results = st.session_state.get("last_school_results", [])
    meta = st.session_state.get("last_meta", {}) or {}
    elapsed = float(st.session_state.get("last_elapsed", 0.0) or 0.0)
    fos_rows = st.session_state.get("last_fos_rows", [])

st.markdown("---")
st.subheader("Results")

total_api = int((meta or {}).get("total", len(school_results)) or len(school_results))
st.caption(
    f"Field of Study: {fos_title} ({label_with_dot}) | Degree: {degree_label} | "
    f"API matched schools: {len(school_results):,} (metadata total={total_api:,}). "
    f"Conditioned FoS rows: {len(fos_rows):,}. "
    f"Fetched in {elapsed:.1f}s."
)


loaded_from = st.session_state.get("cip4_loaded_from")
if loaded_from:
    st.caption(f"CIP-4 catalog loaded from: {loaded_from}")

if not fos_rows:
    st.warning("No results for the selected Field of Study + credential with the chosen filters.")
    st.stop()

df = fos_rows_to_df(fos_rows)

# -----------------------------
# Display options (v1.4 fix)
# -----------------------------
with st.expander("Display options", expanded=True):
    n_rows = int(len(df)) if len(df) > 0 else 1
    max_allowed = max(1, min(5000, n_rows))
    default_val = min(200, max_allowed)
    step = 10 if max_allowed >= 10 else 1

    max_show = st.number_input(
        "Max results to render (table + cards)",
        min_value=1,
        max_value=max_allowed,
        value=default_val,
        step=step,
        key="max_show",
    )

df_show = df.head(int(st.session_state.get("max_show", max(1, min(200, len(df)))))).copy()

# 1) Table (always)
# Keep the table compact and user-facing (hide RAW/code helper columns used for cards).
TABLE_COLS = [
    "School", "City", "State",
    "Years", "Ownership", "Locale", "Size category",
    "Enrollment",
    "Program", "Credential Level",
    "Median Earnings (4yr)", "Median Debt", "Monthly Loan Payment", "Graduates",
    "School URL",
]
TABLE_COLS = [c for c in TABLE_COLS if c in df_show.columns]
st.dataframe(df_show[TABLE_COLS], use_container_width=True, hide_index=True)

# 2) Cards (always, below the table)
st.markdown("### Cards (Scorecard-style)")

def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None

def _monthly_from_annual(annual: Optional[float]) -> Optional[float]:
    if annual is None:
        return None
    try:
        return annual / 12.0
    except Exception:
        return None

for _, row in df_show.iterrows():
    school = row.get("School") or "Unknown School"
    city = row.get("City") or ""
    state = row.get("State") or ""
    zipc = row.get("ZIP") or ""

    years = row.get("Years") or "4 Year"
    ownership = row.get("Ownership") or "Unknown"
    locale = row.get("Locale") or "Locale Unknown"
    size_cat = row.get("Size category") or "size unknown"

    enrollment = row.get("Enrollment") or "N/A"

    program = row.get("Program") or ""
    url = row.get("School URL") or ""

    earn4_raw = _to_float(row.get("Median Earnings (4yr) RAW"))
    debt_raw = _to_float(row.get("Median Debt RAW"))
    pay_raw = _to_float(row.get("Monthly Loan Payment RAW"))
    grads_raw = row.get("Graduates") or "N/A"

    earn4 = _format_money(earn4_raw)
    debt = _format_money(debt_raw)
    monthly_earn = _format_money(_monthly_from_annual(earn4_raw))
    monthly_pay = _format_money(pay_raw)

    with st.container(border=True):
        # Layout: Left (school + icons), Middle (earnings), Middle (debt), Right (graduates)
        c_left, c_earn, c_debt, c_grads = st.columns([4, 3, 3, 2], gap="large")

        with c_left:
            st.markdown(f"**{school}**")
            st.caption(f"{city}, {state} {zipc}".strip())
            if program:
                st.caption(f"Program: {program}")
            # Icons row (use emojis per your instruction)
            icon_line = "  ".join([
                f"{_emoji_for_years(years)} {years}",
                f"{_emoji_for_ownership(ownership)} {ownership}",
                f"{_emoji_for_locale(locale)} {locale}",
                f"{_emoji_for_size(size_cat)} {size_cat}",
            ])
            st.markdown(icon_line)

            st.caption(f"Enrollment: {enrollment}")
            if url:
                st.caption(f"Website: {url}")

        with c_earn:
            st.metric("Median Earnings", earn4)
            st.caption(f"Monthly earnings: {monthly_earn}")

        with c_debt:
            st.metric("Median Total Debt", debt)
            st.caption(f"Monthly loan payment: {monthly_pay}")

        with c_grads:
            st.metric("Graduates", grads_raw)

