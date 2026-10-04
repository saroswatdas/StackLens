# StackLens

### Skill-Based Career Intelligence

StackLens is a content-based career recommendation system that analyzes a user's technical skills and identifies career paths that best match their current skill set.

Instead of asking users to choose a career first, StackLens works in reverse:

> **Give me your skills → I'll show you where they fit.**

The system uses **TF-IDF vectorization** and **cosine similarity** to compare user-provided skills against predefined career profiles and returns the top three career matches.

---

## 🚀 Live Demo

**Try StackLens:**  
[Open the Live Application](https://saroswatdas-stacklens-app-odvz4x.streamlit.app/)

---

## ✨ Features

- Enter technical skills in a simple comma-separated format
- Validates skills against the career dataset
- Uses TF-IDF to represent skills numerically
- Calculates similarity using cosine similarity
- Ranks career paths based on skill compatibility
- Displays the top 3 career recommendations
- Shows similarity scores for each recommendation
- Highlights skills that match each career
- Clean, responsive Streamlit interface
- Works entirely from a predefined career-skill dataset

---

## 🧠 How It Works

StackLens follows a content-based recommendation approach.

```text
User Skills
     ↓
Skill Validation
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity
     ↓
Career Ranking
     ↓
Top 3 Recommendations

1. Skill Input
The user enters technical skills such as:
Python, SQL, Machine Learning

A minimum of three skills is required.
2. Skill Validation
The entered skills are compared against the skills available in the dataset.
Unknown skills are identified and reported rather than silently affecting the recommendation.
3. TF-IDF Vectorization
Each career profile is represented as a text document containing its required skills.
TF-IDF converts these skill profiles into numerical vectors based on the importance of each term.
4. Cosine Similarity
The user's skill vector is compared with every career profile using cosine similarity.
A higher similarity score indicates a stronger match between the user's skills and the career profile.
5. Career Ranking
Career profiles are sorted by similarity score, and the three highest-scoring careers are displayed.
📊 Example
Input
Python, SQL, Machine Learning

Possible Recommendations
1. Data Scientist
2. Machine Learning Engineer
3. AI Engineer

Each recommendation includes:
- Career role
- Similarity percentage
- Matching skills
The exact ranking depends on the skills entered and the career profiles available in the dataset.
🛠️ Tech Stack
Technology	Purpose
Python	Core programming language
Streamlit	Web application interface
Pandas	Dataset handling
Scikit-learn	TF-IDF and cosine similarity
Git & GitHub	Version control and source hosting
Streamlit Community Cloud	Deployment


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

app.py
Contains the Streamlit application, interface, user input handling, validation, and recommendation display.
recommendation.py
Contains the recommendation logic using TF-IDF vectorization and cosine similarity.
data/raw_skills.csv
Contains the career profiles and their associated technical skills.

🧪 Current Career Profiles
The current dataset includes career paths such as:
- Data Scientist
- DevOps Engineer
- Backend Developer
- Frontend Developer
- Cloud Engineer
- Data Analyst
- Cybersecurity Analyst
- AI Engineer
- Machine Learning Engineer
- Software Developer
- Full Stack Developer
- Database Administrator
- Cloud Developer
- Network Engineer
- Mobile App Developer
The recommendation quality depends on the coverage and quality of these career profiles.

💻 Run Locally
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd StackLens

2. Create a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate

macOS/Linux:
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the application
python -m streamlit run app.py

The application will open locally at:
http://localhost:8501

🧪 Test Inputs
You can test the application with examples such as:
Python, SQL, Machine Learning

HTML, CSS, JavaScript, React

AWS, Docker, Kubernetes, Linux

Java, C++, SQL

The last example is useful for testing how the system handles skills that may have different coverage across career profiles.
📐 Recommendation Method
StackLens uses a simple content-based recommendation model.
For each career:
Career Skills → TF-IDF Vector

The user's input is also converted into a TF-IDF representation:
User Skills → TF-IDF Vector

The system then calculates:
Cosine Similarity(User Vector, Career Vector)

The resulting similarity scores are used to rank the available careers.
This approach allows recommendations to be generated without requiring a large historical user dataset.

⚠️ Limitations
StackLens is designed as a skill-matching prototype rather than a complete career guidance platform.
Current limitations include:
- Recommendations depend on the predefined dataset
- Skills outside the dataset have limited influence
- Similarity does not represent actual job-market demand
- The system does not consider work experience or seniority
- It does not currently consider salary expectations
- It does not analyze soft skills
- It does not use real-time job-market data
- Career profiles are manually defined rather than learned from job postings
The recommendations should therefore be treated as skill-based suggestions, not definitive career advice.

🔮 Future Improvements
Possible improvements include:
- Expand the career and skill dataset
- Add skill synonyms and normalization
- Introduce skill weighting
- Include experience level
- Add salary and industry preferences
- Incorporate real-world job posting data
- Add learning-path recommendations for missing skills
- Allow users to compare multiple career paths
- Track recommendation history
- Experiment with semantic embeddings and transformer-based models

🎯 Project Objective
The goal of StackLens is to demonstrate how a relatively simple machine learning pipeline can be turned into a practical recommendation system.
The project combines:
Data Processing
      +
Machine Learning
      +
Similarity Analysis
      +
Recommendation Logic
      +
Interactive Web Application

It serves as a practical implementation of content-based filtering using technical skill profiles.
👨‍💻 Author
Saroswat Das
Built as a machine learning and web application project using Python, Scikit-learn, Pandas, and Streamlit.

📌 Project Status
Status: Deployed 🚀
StackLens is currently available as a live Streamlit application and can be run locally using the instructions above.

Built with
Python · Scikit-learn · Pandas · Streamlit