import csv
from collections import defaultdict

# Define the sets for keyword scoring
SDC_equal_keywords = set(['Backprop'])
SDC_better_keywords = set(['IS', 'STREAM', 'PuReMD', 'Kmeans', 'Lulesh', "MG", "LU", "Bfs-rodinia", "CG", "NW", "BT"])
SDC_worse_keywords = set(['DC', 'Blackholes', 'hotspot', 'Lud', "SP", "Nn", "Pathfinder"])

def load_keywords_and_scores(csv_path: str):
    """Load keywords from a CSV and score them based on predefined groups."""
    keyword_scores = defaultdict(int)
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) < 2:
                continue  # Skip rows that do not have enough data
            filename, keywords_str = row
            keywords = keywords_str.split(',')
            for keyword in keywords:
                if keyword in SDC_equal_keywords:
                    score = 0
                elif keyword in SDC_better_keywords:
                    score = 1
                elif keyword in SDC_worse_keywords:
                    score = -1
                else:
                    continue  # If the keyword doesn't match any group, skip it
                keyword_scores[keyword] += score
    return keyword_scores

def get_top_keywords(keyword_scores, top_n=20):
    """Return the top N keywords sorted by their scores."""
    # Sort keywords by score in descending order and return the top N
    return sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

# Path to the CSV file containing keywords and files
csv_path = 'keyFeature.csv'

# Process the CSV file to score each keyword
keyword_scores = load_keywords_and_scores(csv_path)

# Get the top 20 keywords
top_keywords = get_top_keywords(keyword_scores)

print("Top 20 Keywords by Score:")
for keyword, score in top_keywords:
    print(f"{keyword}: {score}")