import streamlit as st
import requests
import datetime
import os

# ======================
# CONFIG
# ======================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

# ======================
# CONSTANTS (ENCODING)
# ======================
STATE_CODES = [
    'CA', 'MA', 'KY', 'NY', 'CO', 'VA', 'TX', 'WA', 'IL', 'NC', 'PA',
    'GA', 'NH', 'MO', 'FL', 'NJ', 'WV', 'MI', 'DC', 'CT', 'MD', 'OH',
    'TN', 'MN', 'RI', 'OR', 'UT', 'ME', 'NV', 'NM', 'IN', 'AZ', 'ID',
    'AR', 'WI'
]

CATEGORY_CODES = [
    'music', 'enterprise', 'web', 'software', 'games_video',
    'network_hosting', 'finance', 'mobile', 'education',
    'public_relations', 'security', 'other', 'photo_video', 'hardware',
    'ecommerce', 'advertising', 'travel', 'fashion', 'analytics',
    'consulting', 'biotech', 'cleantech', 'search', 'semiconductor',
    'social', 'medical', 'automotive', 'messaging', 'manufacturing',
    'hospitality', 'news', 'transportation', 'sports', 'real_estate',
    'health'
]

STATE_CITY_MAP = {
    'AR': ['Little Rock'],
    'AZ': ['Tempe'],
    'CA': ['San Diego', 'Los Gatos', 'Cupertino', 'San Francisco', 'Mountain View',
           'San Rafael', 'Palo Alto', 'Menlo Park', 'Los Altos', 'Burlingame',
           'Berkeley', 'Santa Ana', 'Moffett Field', 'San Jose', 'Sunnyvale',
           'San Mateo', 'South San Francisco', 'Aliso Viejo', 'Alameda',
           'Fremont', 'Santa Clara', 'Los Angeles', 'Santa Monica', 'Milpitas',
           'Redwood City', 'Campbell', 'Foster City', 'Oakland', 'Petaluma',
           'Newport Beach', 'Pasadena', 'Morgan Hill', 'Playa Vista',
           'Monterey Park', 'Freedom', 'Santa Barbara', 'Solana Beach',
           'San Bruno', 'Carlsbad', 'West Hollywood', 'San Franciso', 'Irvine',
           'Larkspur', 'Glendale', 'Beverly Hills', 'Napa', 'Calabasas',
           'North Hollywood', 'Carpinteria', 'Belmont', 'Pleasanton',
           'El Segundo', 'Sunnnyvale', 'Brisbane', 'Emeryville', 'Hollywood',
           'Chicago', 'Laguna Niguel', 'La Jolla', 'Thousand Oaks', 'Arcadia',
           'Yorba Linda', 'San Carlos', 'Torrance', 'El Segundo,',
           'Scotts Valley'],
    'CO': ['Denver', 'Boulder', 'Loveland', 'Centennial', 'Longmont',
           'Englewood', 'Broomfield', 'Greenwood Village', 'Louisville', 'Avon'],
    'CT': ['Bloomfield', 'Hartford', 'Farmington', 'Westport'],
    'DC': ['Washington'],
    'FL': ['Tampa', 'Weston', 'Altamonte Springs'],
    'GA': ['Atlanta', 'Alpharetta', 'Duluth', 'NW Atlanta', 'Lawrenceville'],
    'ID': ['Idaho Falls'],
    'IL': ['Chicago', 'Naperville', 'Warrenville', 'Evanston', 'Itasca',
           'Champaign', 'Lisle'],
    'IN': ['Indianapolis'],
    'KY': ['Louisville', 'Lexington'],
    'MA': ['Williamstown', 'Cambridge', 'Boston', 'Waltham', 'Wilmington',
           'Somerville', 'Needham', 'Marlborough', 'North Billerica',
           'Boxborough', 'Burlington', 'Maynard', 'Woburn', 'Lowell',
           'Littleton', 'Billerica', 'Lexington', 'North Reading', 'Bedford',
           'Dedham', 'Andover', 'Westford', 'Framingham', 'Acton', 'Newton',
           'Chelmsford'],
    'MD': ['Timonium', 'Bethesda', 'Columbia', 'Frederick', 'College Park',
           'Chevy Chase', 'Annapolis'],
    'ME': ['West Newfield', 'Tewksbury'],
    'MI': ['Canton', 'Bingham Farms', 'Zeeland'],
    'MN': ['Plymouth', 'Minneapolis', 'Minnetonka', 'Saint Paul',
           'Golden Valley'],
    'MO': ['Kansas City', 'Saint Louis'],
    'NC': ['Durham', 'Raleigh', 'Pittsboro'],
    'NH': ['Manchester', 'Nashua'],
    'NJ': ['Princeton', 'Somerset', 'Paramus', 'Red Bank', 'Jersey City',
           'Hillsborough', 'Hampton'],
    'NM': ['Albuquerque'],
    'NV': ['Henderson', 'Las Vegas'],
    'NY': ['Brooklyn', 'New York', 'Long Island City', 'NY', 'New York City',
           'Woodbury', 'NYC', 'Rye Brook', 'Kenmore'],
    'OH': ['Cincinnati', 'Cleveland', 'Columbus', 'Toledo'],
    'OR': ['Portland', 'Lake Oswego', 'Tualatin'],
    'PA': ['Pittsburgh', 'Conshohocken', 'Berwyn', 'Philadelphia', 'Allentown',
           'Bala Cynwyd', 'Bethlehem', 'New Hope', 'West Chester', 'Yardley',
           'Lancaster'],
    'RI': ['Providence'],
    'TN': ['Memphis', 'Nashville'],
    'TX': ['Austin', 'Dallas', 'Plano', 'The Woodlands', 'Richardson',
           'Addison', 'Waco', 'Houston'],
    'UT': ['Salt Lake City', 'Lindon', 'Provo'],
    'VA': ['Vienna', 'Charlottesville', 'Dulles', 'Reston', 'Sterling',
           'Arlington', 'Herndon', 'McLean', 'Chantilly', 'Viena',
           'Potomac Falls'],
    'WA': ['Seattle', 'Kirkland', 'Bothell', 'Bellevue', 'SPOKANE',
           'Puyallup', 'Vancouver', 'Redmond'],
    'WI': ['Middleton'],
    'WV': ['Kearneysville']
}


# ======================
# TITLE
# ======================
st.title("🚀 Startup Success Predictor")
st.markdown("Predict startup success based on early-stage signals.")

st.divider()


st.subheader("📍 Location")

state = st.selectbox(
    "State",
    options=sorted(STATE_CITY_MAP.keys()),
    help="State where the startup is headquartered."
)

cities = sorted(STATE_CITY_MAP[state])

city = st.selectbox(
    "City",
    options=cities,
    help="City where the startup is headquartered."
)

state_code = sorted(STATE_CITY_MAP.keys()).index(state)
city_code = cities.index(city)


# ======================
# INPUT FORM
# ======================
with st.form("startup_form"):
    st.subheader("🏢 Startup Information")
  

    founded_at = st.date_input(
        "Founded Date",
        help="Date when the startup was officially founded."
    )

    first_funding_at = st.date_input(
        "First Funding Date",
        help="Date when the startup received its first external funding."
    )

    last_funding_at = st.date_input(
        "Last Funding Date",
        help="Most recent funding date received by the startup."
    )

    st.caption(
        "⏱️ The system automatically calculates how old the startup was at the time of funding."
    )

    relationships = st.number_input(
        "Number of Relationships",
        min_value=0,
        step=1,
        help=(
            "The number of professional or strategic relationships the startup has. "
            "This may include investors, mentors, vendors, accelerators, accountants, "
            "or other business partners."
        )
    )

    st.caption(
        "🤝 Relationships represent the startup’s business network and external support."
    )

    funding_rounds = st.number_input(
        "Funding Rounds",
        min_value=0,
        step=1,
        help="Total number of funding rounds the startup has completed (Seed, A, B, etc.)."
    )

    funding_total_usd = st.number_input(
        "Total Funding (USD)",
        min_value=0.0,
        step=10000.0,
        help="Total amount of funding raised by the startup in U.S. dollars."
    )

    milestones = st.number_input(
        "Milestones Achieved",
        min_value=0,
        step=1,
        help=(
            "Milestones track a startup’s progress over time. "
            "Examples include product launches, major customer acquisitions, "
            "revenue targets reached, or market expansion."
        )
    )

    st.caption(
        "📍 A milestone is a measurable progress marker that shows how far the startup has grown."
    )

    category_code = st.selectbox(
        "Startup Category",
        CATEGORY_CODES,
        help="Primary industry or sector in which the startup operates."
    )

    avg_participants = st.number_input(
        "Average Participants per Funding Round",
        min_value=0.0,
        step=1.0,
        help="Average number of investors or participants involved in each funding round."
    )


    has_RoundABCD = st.checkbox(
        "Has Any Round A/B/C/D",
        help="Check if the startup has raised any institutional funding (Series A–D)."
    )

    has_Investor = st.checkbox(
        "Has Investors",
        help="Indicates whether the startup has external investors."
    )

    has_Seed = st.checkbox(
        "Has Seed Funding",
        help="Check if the startup has raised seed funding."
    )

    is_top500 = st.checkbox(
        "Top 500 Startup",
        help="Indicates whether the startup has been ranked among the top 500 startups."
    )

    submitted = st.form_submit_button("🔮 Predict")

# ---- Auto-compute ages ----
age_first_funding_year = (
    (first_funding_at - founded_at).days / 365
    if first_funding_at >= founded_at else 0
)

age_last_funding_year = (
    (last_funding_at - founded_at).days / 365
    if last_funding_at >= founded_at else 0
)

# ======================
# PREDICTION
# ======================
if submitted:

    payload = {
    "state_code": state,
    #"zip_code": zip_code,
    "city": city,
    # "first_funding_at": first_funding_at.isoformat(),
    # "last_funding_at": last_funding_at.isoformat(),
    "age_first_funding_year": age_first_funding_year,
    "age_last_funding_year": age_last_funding_year,
    "relationships": relationships,
    "funding_rounds": funding_rounds,
    "funding_total_usd": funding_total_usd,
    "milestones": milestones,
    "category_code": category_code,
    "avg_participants": avg_participants,
    "is_top500": int(is_top500),
    "has_RoundABCD": int(has_RoundABCD),
    "has_Investor": int(has_Investor),
    "has_Seed": int(has_Seed)
    }
    


    #payload = {"features": features}
    #st.write("🔍 Payload being sent:")
    #st.json({"features": payload})

    try:
        response = requests.post(BACKEND_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"][0]
        probabilities = result["probability"][0]

        st.divider()
        st.subheader("📈 Prediction Result")

        if prediction == 1:
            st.success("✅ **Startup is likely to SUCCEED**")
        else:
            st.error("❌ **Startup is likely to FAIL**")

        st.markdown("### 🔢 Confidence")
        st.progress(probabilities[1])

        col1, col2 = st.columns(2)
        col1.metric("Success Probability", f"{probabilities[1]*100:.2f}%")
        col2.metric("Failure Probability", f"{probabilities[0]*100:.2f}%")

    except requests.exceptions.RequestException as e:
        st.error("⚠️ Backend API error")
        st.text(str(e))
