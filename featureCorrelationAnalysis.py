import pandas as pd

# Creating a DataFrame from the provided data

df = pd.read_csv("keyFeature.csv")

# Splitting keywords into keys by ','
df['Keywords'] = df['Keywords'].apply(lambda x: x.split(','))

# Creating a dictionary to hold the initial score of all keys
key_scores = {}

# SDC group sets
SDC_better_keywords = set(['IS', 'STREAM', 'PuReMD', 'Kmeans', 'Lulesh', "MG", "LU", "Bfs-rodinia", "CG", "NW"])
SDC_worse_keywords = set(['DC', 'Blacksholes', 'Hotspot', 'Lud', "SP", "Nn", "Pathfinder"])

# Increment and decrement scores based on the SDC groups
for index, row in df.iterrows():
    bench = row['Filename']
    keywords = row['Keywords']
    for key in keywords:
        if key not in key_scores:
            key_scores[key] = 0
        if bench in SDC_better_keywords:
            key_scores[key] += 1
        if bench in SDC_worse_keywords:
            key_scores[key] -= 1

# Sorting the scores
sorted_scores = sorted(key_scores.items(), key=lambda item: item[1], reverse=True)

# Getting the top 20 and last 20 keys based on scores
top_20 = sorted_scores[:20]
last_20 = sorted_scores[-20:]

print(top_20, last_20)
