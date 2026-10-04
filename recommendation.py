import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "data" / "raw_skills.csv")


# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
role_vectors = vectorizer.fit_transform(data["skills"])


# Get user skills
while True:
    user_input = input(
        "\nEnter at least 3 skills separated by commas: "
    )

    user_skills = [
        skill.strip()
        for skill in user_input.split(",")
        if skill.strip()
    ]

    if len(user_skills) >= 3:
        break

    print("Please enter at least 3 skills.")


# Find skills known by our dataset
all_dataset_skills = set()

for skills in data["skills"]:
    for skill in skills.split(","):
        all_dataset_skills.add(skill.strip().lower())


known_skills = []
unknown_skills = []

for skill in user_skills:
    if skill.lower() in all_dataset_skills:
        known_skills.append(skill)
    else:
        unknown_skills.append(skill)


# Convert user skills into a TF-IDF vector
user_vector = vectorizer.transform(
    [" ".join(known_skills)]
)


# Calculate cosine similarity
scores = cosine_similarity(
    user_vector,
    role_vectors
)[0]


# Add scores to dataset
data["similarity"] = scores


# Sort results
recommendations = data.sort_values(
    by="similarity",
    ascending=False
)


# Display header
print("\n========================================")
print("       AI TECH STACK RECOMMENDER")
print("========================================")

print("\nYour skills:")
print(", ".join(user_skills))


# Display unknown skills
if unknown_skills:
    print("\nSkills not found in our dataset:")
    print(", ".join(unknown_skills))
    print("\nThese skills were not used for similarity calculation.")


# Display recommendations
print("\nTop 3 Recommended Career Paths:")
print("----------------------------------------")

for rank, (_, row) in enumerate(
    recommendations.head(3).iterrows(),
    start=1
):
    percentage = row["similarity"] * 100

    print(
        f"\n{rank}. {row['role']} - "
        f"{percentage:.1f}% match"
    )

    # Find matching skills
    role_skills = [
        skill.strip()
        for skill in row["skills"].split(",")
    ]

    matching_skills = []

    for skill in known_skills:
        if skill.lower() in [
            role_skill.lower()
            for role_skill in role_skills
        ]:
            matching_skills.append(skill)

    if matching_skills:
        print(
            "   Matching skills: "
            + ", ".join(matching_skills)
        )

print("\n========================================")