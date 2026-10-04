from pathlib import Path
import html

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="StackLens | Career Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown("""
<style>

/* ---------- BASE ---------- */

.stApp {
    background: #080B0F;
    color: #E8EDF2;
}

.block-container {
    max-width: 1180px;
    padding-top: 0.8rem;
    padding-bottom: 4rem;
}

#MainMenu,
footer,
header {
    visibility: hidden;
}


/* ---------- NAVBAR ---------- */

.navbar {
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #202731;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-logo {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #EDF2F6;
    color: #0A0D11;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 800;
}

.brand-name {
    color: #F2F5F8;
    font-size: 17px;
    font-weight: 750;
    letter-spacing: -0.4px;
}

.nav-label {
    color: #667381;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.4px;
}


/* ---------- HERO ---------- */

.hero-area {
    padding: 72px 0 70px;
}

.eyebrow {
    color: #7899B8;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 2px;
    margin-bottom: 17px;
}

.hero-title {
    color: #F3F6F9;
    font-size: 53px;
    font-weight: 780;
    line-height: 1.05;
    letter-spacing: -2.8px;
}

.hero-title span {
    color: #8EA9C2;
}

.hero-description {
    max-width: 570px;
    color: #7E8B99;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 22px;
}

.hero-tags {
    display: flex;
    gap: 7px;
    margin-top: 25px;
    flex-wrap: wrap;
}

.hero-tag {
    border: 1px solid #29333E;
    background: #0D1218;
    color: #788694;
    border-radius: 5px;
    padding: 6px 9px;
    font-size: 9px;
    letter-spacing: 0.5px;
}


/* ---------- ANALYSIS CONSOLE ---------- */

.console {
    background: #0E1319;
    border: 1px solid #28323D;
    border-radius: 13px;
    padding: 24px;
    margin-top: 8px;
}

.console-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 7px;
}

.console-title {
    color: #E7ECF1;
    font-size: 15px;
    font-weight: 700;
}

.console-status {
    color: #7795AF;
    font-size: 8px;
    font-weight: 750;
    letter-spacing: 1.3px;
}

.console-description {
    color: #707D8B;
    font-size: 11px;
    line-height: 1.6;
    margin-bottom: 18px;
}

.console-divider {
    border-top: 1px solid #202A34;
    margin: 19px 0;
}

.console-label {
    color: #6D7A88;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 9px;
}


/* ---------- STREAMLIT INPUT ---------- */

.stTextInput {
    margin-bottom: 0 !important;
}

.stTextInput input {
    background: #090D12 !important;
    border: 1px solid #303A46 !important;
    border-radius: 7px !important;
    color: #E8EDF2 !important;
    height: 45px !important;
    font-size: 12px !important;
}

.stTextInput input:focus {
    border-color: #7192AF !important;
    box-shadow: 0 0 0 1px #7192AF !important;
}

.stTextInput input::placeholder {
    color: #596572 !important;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    background: #E8EEF4 !important;
    color: #0A0E13 !important;
    border: none !important;
    border-radius: 7px !important;
    min-height: 44px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    transition: 0.15s ease !important;
}

.stButton > button:hover {
    background: #C9D7E3 !important;
    transform: translateY(-1px);
}


/* ---------- SKILL CHIPS ---------- */

.skills-label {
    color: #657280;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.skill-chip {
    display: inline-block;
    background: #151C24;
    border: 1px solid #293440;
    color: #8492A0;
    border-radius: 5px;
    padding: 5px 8px;
    margin: 2px 3px 2px 0;
    font-size: 9px;
}


/* ---------- ANALYSIS STEPS ---------- */

.analysis-steps {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-top: 20px;
}

.analysis-step {
    flex: 1;
}

.step-number {
    color: #7091AE;
    font-size: 8px;
    font-weight: 750;
}

.step-name {
    color: #788593;
    font-size: 8px;
    margin-top: 4px;
}


/* ---------- SECTION HEADER ---------- */

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #202731;
    padding-bottom: 14px;
    margin-bottom: 20px;
}

.section-title {
    color: #E7ECF1;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
}

.section-meta {
    color: #606D7A;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.2px;
}


/* ---------- RESULTS ---------- */

.result-card {
    background: #0E1319;
    border: 1px solid #26313C;
    border-radius: 11px;
    padding: 23px;
    min-height: 300px;
    transition: 0.18s ease;
}

.result-card:hover {
    border-color: #3B4A59;
    transform: translateY(-2px);
}

.result-rank {
    color: #6C7987;
    font-size: 9px;
    font-weight: 750;
    letter-spacing: 1.2px;
}

.result-role {
    color: #F0F3F6;
    font-size: 21px;
    font-weight: 720;
    letter-spacing: -0.5px;
    margin-top: 15px;
    min-height: 53px;
}

.score-row {
    display: flex;
    align-items: baseline;
    gap: 7px;
    margin-top: 13px;
}

.result-score {
    color: #DDE8F2;
    font-size: 32px;
    font-weight: 780;
    letter-spacing: -1px;
}

.score-caption {
    color: #657280;
    font-size: 8px;
    letter-spacing: 1px;
}

.score-track {
    height: 4px;
    width: 100%;
    background: #242D37;
    border-radius: 5px;
    overflow: hidden;
    margin: 13px 0 23px;
}

.score-fill {
    height: 100%;
    background: #83A6C5;
    border-radius: 5px;
}

.match-label {
    color: #657280;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

.match-skill {
    display: inline-block;
    background: #171F28;
    border: 1px solid #2A3541;
    color: #B8C5D2;
    border-radius: 5px;
    padding: 5px 8px;
    margin: 2px;
    font-size: 9px;
}


/* ---------- METHODOLOGY ---------- */

.method-card {
    background: #0E1319;
    border: 1px solid #242E38;
    border-radius: 9px;
    padding: 20px;
    min-height: 145px;
}

.method-number {
    color: #7698B6;
    font-size: 9px;
    font-weight: 750;
}

.method-title {
    color: #E4EAF0;
    font-size: 14px;
    font-weight: 700;
    margin-top: 15px;
}

.method-description {
    color: #707D8A;
    font-size: 10px;
    line-height: 1.6;
    margin-top: 7px;
}


/* ---------- EMPTY STATE ---------- */

.empty-state {
    border: 1px dashed #2A3540;
    background: #0C1117;
    border-radius: 10px;
    text-align: center;
    padding: 32px 20px;
    margin-top: 35px;
}

.empty-title {
    color: #CBD4DC;
    font-size: 14px;
    font-weight: 650;
}

.empty-text {
    color: #697683;
    font-size: 11px;
    margin-top: 7px;
}


/* ---------- FOOTER ---------- */

.footer {
    border-top: 1px solid #202731;
    margin-top: 70px;
    padding-top: 20px;
    display: flex;
    justify-content: space-between;
    color: #525E6A;
    font-size: 9px;
}


/* ---------- MOBILE ---------- */

@media (max-width: 800px) {

    .hero-area {
        padding-top: 45px;
    }

    .hero-title {
        font-size: 40px;
        letter-spacing: -1.8px;
    }

    .nav-label {
        display: none;
    }

    .section-meta {
        display: none;
    }

    .footer {
        flex-direction: column;
        gap: 8px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATASET
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_data():

    path = BASE_DIR / "data" / "raw_skills.csv"

    df = pd.read_csv(path)

    if not {"role", "skills"}.issubset(df.columns):
        raise ValueError(
            "raw_skills.csv must contain 'role' and 'skills' columns."
        )

    return df.dropna(subset=["role", "skills"]).copy()


try:
    data = load_data()

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# =========================================================
# TF-IDF MODEL
# =========================================================

vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w.+#-]+\b"
)

role_vectors = vectorizer.fit_transform(
    data["skills"].astype(str)
)


# =========================================================
# KNOWN SKILLS
# =========================================================

dataset_skills = set()

for skills in data["skills"]:

    for skill in str(skills).split(","):

        skill = skill.strip().lower()

        if skill:
            dataset_skills.add(skill)


# =========================================================
# SESSION STATE
# =========================================================

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "unknown_skills" not in st.session_state:
    st.session_state.unknown_skills = []


# =========================================================
# NAVBAR
# =========================================================

st.markdown(
    '<div class="navbar">'
    '<div class="brand">'
    '<div class="brand-logo">S</div>'
    '<div class="brand-name">StackLens</div>'
    '</div>'
    '<div class="nav-label">CAREER INTELLIGENCE PLATFORM</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# HERO + CONSOLE
# =========================================================

hero_col, console_col = st.columns(
    [1.08, 0.92],
    gap="large"
)


with hero_col:

    st.markdown(
        '<div class="hero-area">'
        '<div class="eyebrow">'
        'SKILL-BASED CAREER DISCOVERY'
        '</div>'
        '<div class="hero-title">'
        'Your skills have potential.<br>'
        '<span>Find where they belong.</span>'
        '</div>'
        '<div class="hero-description">'
        'StackLens analyzes your technical skills and compares '
        'them with career profiles to identify the paths that '
        'best align with your current stack.'
        '</div>'
        '<div class="hero-tags">'
        '<span class="hero-tag">TF-IDF</span>'
        '<span class="hero-tag">COSINE SIMILARITY</span>'
        '<span class="hero-tag">CONTENT-BASED</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with console_col:

    st.markdown(
        '<div class="console">'
        '<div class="console-header">'
        '<div class="console-title">'
        'Analyze your stack'
        '</div>'
        '<div class="console-status">'
        'READY'
        '</div>'
        '</div>'
        '<div class="console-description">'
        'Tell us what you know. StackLens will compare your '
        'skills against the available career profiles.'
        '</div>'
        '<div class="console-label">'
        'YOUR TECHNICAL SKILLS'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    user_input = st.text_input(
        "Skills",
        placeholder="Python, SQL, Machine Learning",
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="skills-label">'
        'COMMON SKILLS'
        '</div>'
        '<span class="skill-chip">Python</span>'
        '<span class="skill-chip">SQL</span>'
        '<span class="skill-chip">Java</span>'
        '<span class="skill-chip">Machine Learning</span>'
        '<span class="skill-chip">AWS</span>'
        '<span class="skill-chip">Docker</span>',
        unsafe_allow_html=True
    )

    st.write("")

    analyze = st.button(
        "Analyze my career fit →",
        use_container_width=True
    )

    st.markdown(
        '<div class="analysis-steps">'
        '<div class="analysis-step">'
        '<div class="step-number">01</div>'
        '<div class="step-name">INPUT</div>'
        '</div>'
        '<div class="analysis-step">'
        '<div class="step-number">02</div>'
        '<div class="step-name">TF-IDF</div>'
        '</div>'
        '<div class="analysis-step">'
        '<div class="step-number">03</div>'
        '<div class="step-name">SIMILARITY</div>'
        '</div>'
        '<div class="analysis-step">'
        '<div class="step-number">04</div>'
        '<div class="step-name">RANKING</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RUN RECOMMENDATION
# =========================================================

if analyze:

    entered_skills = [
        skill.strip()
        for skill in user_input.split(",")
        if skill.strip()
    ]

    unique_normalized = list(
        dict.fromkeys(
            skill.lower()
            for skill in entered_skills
        )
    )

    if len(unique_normalized) < 3:

        st.session_state.recommendations = None

        st.warning(
            "Please enter at least three different skills."
        )

    else:

        known_skills = []
        unknown_skills = []

        for skill in entered_skills:

            if skill.lower() in dataset_skills:
                known_skills.append(skill)
            else:
                unknown_skills.append(skill)

        if not known_skills:

            st.session_state.recommendations = []
            st.session_state.unknown_skills = unknown_skills

        else:

            user_vector = vectorizer.transform(
                [" ".join(known_skills)]
            )

            scores = cosine_similarity(
                user_vector,
                role_vectors
            )[0]

            results = data.copy()

            results["similarity"] = scores

            results = results.sort_values(
                "similarity",
                ascending=False
            )

            recommendations = []

            for _, row in results.head(3).iterrows():

                role_skills = [
                    skill.strip()
                    for skill in str(
                        row["skills"]
                    ).split(",")
                ]

                role_skill_set = {
                    skill.lower()
                    for skill in role_skills
                }

                matching = [
                    skill
                    for skill in known_skills
                    if skill.lower() in role_skill_set
                ]

                recommendations.append({
                    "role": str(row["role"]),
                    "score": float(row["similarity"]),
                    "matching": matching
                })

            st.session_state.recommendations = recommendations
            st.session_state.unknown_skills = unknown_skills


# =========================================================
# RESULTS
# =========================================================

if st.session_state.recommendations is not None:

    recommendations = st.session_state.recommendations
    unknown = st.session_state.unknown_skills

    if unknown:

        st.info(
            "Not represented in the current dataset: "
            + ", ".join(unknown)
        )

    if not recommendations:

        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-title">'
            'No matching career profiles found'
            '</div>'
            '<div class="empty-text">'
            'Try skills such as Python, SQL, Machine Learning, '
            'AWS, or Docker.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="section-header">'
            '<div class="section-title">'
            'Career fit analysis'
            '</div>'
            '<div class="section-meta">'
            'TOP 3 MATCHES'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        columns = st.columns(
            3,
            gap="medium"
        )

        for index, result in enumerate(recommendations):

            rank = index + 1

            role = html.escape(result["role"])

            percentage = max(
                0,
                min(
                    result["score"] * 100,
                    100
                )
            )

            if result["matching"]:

                skill_html = "".join(
                    '<span class="match-skill">'
                    + html.escape(skill)
                    + '</span>'
                    for skill in result["matching"]
                )

            else:

                skill_html = (
                    '<span class="match-skill">'
                    'No exact overlap'
                    '</span>'
                )

            card = (
                '<div class="result-card">'

                f'<div class="result-rank">'
                f'MATCH 0{rank}'
                f'</div>'

                f'<div class="result-role">'
                f'{role}'
                f'</div>'

                '<div class="score-row">'

                f'<div class="result-score">'
                f'{percentage:.1f}%'
                f'</div>'

                '<div class="score-caption">'
                'SIMILARITY'
                '</div>'

                '</div>'

                '<div class="score-track">'
                f'<div class="score-fill" '
                f'style="width:{percentage:.1f}%">'
                '</div>'
                '</div>'

                '<div class="match-label">'
                'MATCHING SKILLS'
                '</div>'

                f'{skill_html}'

                '</div>'
            )

            with columns[index]:

                st.markdown(
                    card,
                    unsafe_allow_html=True
                )

        st.caption(
            "Scores represent cosine similarity between your "
            "skill profile and the career profiles in the dataset."
        )


# =========================================================
# METHODOLOGY
# =========================================================

st.markdown(
    '<div class="section-header">'
    '<div class="section-title">'
    'How StackLens works'
    '</div>'
    '<div class="section-meta">'
    'RECOMMENDATION PIPELINE'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


methods = [
    (
        "01",
        "Skill Input",
        "Your technical skills form the input profile."
    ),
    (
        "02",
        "TF-IDF",
        "Skills are converted into weighted feature vectors."
    ),
    (
        "03",
        "Similarity",
        "Cosine similarity measures alignment with each role."
    ),
    (
        "04",
        "Ranking",
        "The strongest three career paths are returned."
    )
]


method_columns = st.columns(
    4,
    gap="medium"
)


for column, method in zip(
    method_columns,
    methods
):

    number, title, description = method

    card = (
        '<div class="method-card">'
        f'<div class="method-number">{number}</div>'
        f'<div class="method-title">{title}</div>'
        f'<div class="method-description">{description}</div>'
        '</div>'
    )

    with column:

        st.markdown(
            card,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '<div>StackLens · Career Intelligence</div>'
    '<div>Python · Scikit-learn · Streamlit</div>'
    '</div>',
    unsafe_allow_html=True
)