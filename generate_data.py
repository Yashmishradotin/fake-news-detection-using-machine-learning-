import csv
import random

# Categories for generating news
locations = ["New York", "London", "Tokyo", "Mumbai", "Paris", "Berlin", "Sydney", "the Amazon", "Antarctica", "Mars", "the Pacific Ocean", "California", "Delhi", "Beijing"]
people = ["Elon Musk", "The President", "A famous actor", "A local politician", "The CEO of a tech giant", "A group of scientists", "A renowned doctor", "An anonymous hacker"]
years = ["2020", "2021", "2022", "2023", "2024", "2025", "1999", "2008", "last year", "a decade ago"]
times = ["yesterday", "this morning", "last night", "over the weekend", "on Monday", "today", "last month", "in the early 2000s"]
organizations = ["United Nations", "WHO", "NASA", "SpaceX", "Google", "Apple", "The local government", "The Supreme Court"]
discoveries = ["species of frog", "exoplanet", "ancient ruin", "subatomic particle", "cure for a rare virus", "method to recycle plastic 100%"]
fake_creatures = ["reptile", "vampire", "time traveler", "alien in disguise", "robot"]
fake_actions = ["giving away free money", "stealing internet cables", "controlling the weather", "building a secret underground city"]

real_templates = [
    "The {organization} announced a new policy affecting millions starting {time}.",
    "{person} won a major award for their incredible work in {year}.",
    "A major earthquake struck {location} {time}, causing widespread power outages.",
    "The stock market saw a massive drop of 5% due to global trade tensions {time}.",
    "Scientists at a top university discovered a new {discovery} in {location}.",
    "The unemployment rate in {location} dropped significantly in {year}.",
    "NASA's latest rover sent back stunning high-resolution images of {location}.",
    "{organization} released their latest financial report {time}, showing record profits.",
    "A new trade agreement was signed between two major countries {time}.",
    "Global temperatures reached a new record high in {year}, experts warn."
]

fake_templates = [
    "Aliens were seen landing in {location} and were {fake_action}.",
    "{person} secretly admitted to being a {fake_creature} in a leaked video from {year}.",
    "Drinking 5 liters of bleach cures all known diseases instantly, claims internet doctor.",
    "The government is planning to replace all birds in {location} with spy drones by {year}.",
    "A new study proves that eating rocks gives you superpowers.",
    "The Earth is actually shaped like a donut, according to a former {organization} employee.",
    "Zombies have been spotted walking the streets of {location} {time}.",
    "{person} has purchased the moon and plans to build a casino on it.",
    "A man from {location} claims he lived in {year} and traveled to the future using a microwave.",
    "Water is actually poisonous and the government has been hiding it since {year}."
]

def generate_news(is_real=True, count=300):
    news_list = []
    templates = real_templates if is_real else fake_templates
    label = "REAL" if is_real else "FAKE"
    
    for _ in range(count):
        template = random.choice(templates)
        news_text = template.format(
            location=random.choice(locations),
            person=random.choice(people),
            year=random.choice(years),
            time=random.choice(times),
            organization=random.choice(organizations),
            discovery=random.choice(discoveries),
            fake_creature=random.choice(fake_creatures),
            fake_action=random.choice(fake_actions)
        )
        # Add some random variations to make them slightly unique
        news_text = news_text.replace("5%", str(random.randint(1, 15)) + "%")
        news_text = news_text.replace("5 liters", str(random.randint(2, 10)) + " liters")
        news_list.append({"text": news_text, "label": label})
        
    return news_list

print("Generating 600 mixed news articles (300 REAL, 300 FAKE)...")
real_news = generate_news(is_real=True, count=300)
fake_news = generate_news(is_real=False, count=300)

all_news = real_news + fake_news
random.shuffle(all_news) # Mix them up randomly!

csv_file = "dataset.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["text", "label"])
    writer.writeheader()
    for news in all_news:
        writer.writerow(news)

print(f"SUCCESS: Generated {len(all_news)} news articles and saved to {csv_file}")
