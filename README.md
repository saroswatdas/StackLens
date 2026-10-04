# StackLens

### Skill-Based Career Intelligence

StackLens is a machine-learning powered career recommendation tool that analyzes a user's technical skills and identifies the career paths that best match their current skill set.

Instead of asking:

> "Which career should I choose?"

StackLens asks:

> "Based on what I already know, which career paths fit me best?"

---

## ✦ What it does

Enter your technical skills and StackLens:

- Analyzes your skill profile
- Compares it with different career profiles
- Calculates skill similarity
- Ranks the strongest career matches
- Shows your top 3 recommendations
- Displays the skills contributing to each match

### Example

**Input**

```text
Python, SQL, Machine Learning

Possible recommendations
1. Data Scientist
2. AI Engineer
3. Machine Learning Engineer

⚙ How it works
StackLens uses a simple content-based recommendation pipeline:
User Skills
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity
     ↓
Career Ranking
     ↓
Top 3 Matches

01 — Skill Input
The user enters their technical skills.
Python, SQL, Machine Learning

02 — TF-IDF
The skills are converted into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency).
This allows the system to represent skill profiles mathematically.
03 — Cosine Similarity
The user's skill vector is compared against every career profile using cosine similarity.
A higher score means the skill profiles are more closely aligned.
04 — Ranking
The career profiles are sorted by similarity and the three strongest matches are displayed.
🧠 Recommendation Example
For:
Python, SQL, Machine Learning

StackLens might produce:
Career	Similarity
Data Scientist	69.5%
AI Engineer	61.9%
Machine Learning Engineer	58.4%


The application also shows which skills matched each career.
Similarity is a measure of skill-profile alignment, not a prediction of career success.

🖥️ Interface
StackLens uses a minimal dark interface designed around a technical analysis workflow.
The application contains:
Skill Analysis
Enter your current technical stack.
Career Fit Analysis
View the strongest career matches and their similarity scores.
Recommendation Pipeline
See how the system moves from skills → vectors → similarity → ranking.
🛠 Tech Stack
Technology	Purpose
Python	Core programming
Pandas	Dataset processing
Scikit-learn	TF-IDF & cosine similarity
Streamlit	Web interface
CSV	Career skill dataset


📁 Project Structure
StackLens/
│
├── data/
│   └── raw_skills.csv
│
├── app.py
├── recommendation.py
├── requirements.txt
├── README.md
└── .gitignore

Main files
app.py
The Streamlit application and recommendation interface.
recommendation.py
Command-line version of the recommendation system.
data/raw_skills.csv
Career profiles and their associated technical skills.
🚀 Run Locally
Clone the repository:
git clone https://github.com/saroswatdas/StackLens.git
cd StackLens

Install the dependencies:
python -m pip install -r requirements.txt

Start the application:
python -m streamlit run app.py

Then open:
http://localhost:8501

🧪 Try These Inputs
Data / AI
Python, SQL, Machine Learning

Frontend
HTML, CSS, JavaScript, React

Cloud / DevOps
AWS, Docker, Kubernetes

Backend
Java, Python, SQL, APIs

Software Development
C++, Java, SQL

📊 Current Career Profiles
The dataset currently includes roles such as:
- Data Scientist
- Data Analyst
- AI Engineer
- Machine Learning Engineer
- Backend Developer
- Frontend Developer
- Full Stack Developer
- Software Developer
- DevOps Engineer
- Cloud Engineer
- Cloud Developer
- Cybersecurity Analyst
- Network Engineer
- Database Administrator
- Mobile App Developer
⚠️ Limitations
StackLens is an educational recommendation prototype.
The quality of the recommendations depends on the career profiles and skills available in the dataset.
The current system does not consider:
- Work experience
- Education
- Salary expectations
- Location
- Personal interests
- Soft skills
- Current job-market demand
Therefore, the similarity score should be interpreted as skill similarity, not career suitability.
🔭 Future Improvements
Some possible extensions are:
- Larger career and skill dataset
- Skill aliases such as JS → JavaScript
- Skill-gap analysis
- Personalized career profiles
- Searchable skill selection
- Real job-market data
- Experience-level based recommendations

👨‍💻 Author
Saroswat Das
Computer Science Student
Interested in AI/ML, software development, and building practical technology projects.
Built with
Python · Scikit-learn · Pandas · Streamlit
