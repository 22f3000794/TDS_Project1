import json
from collections import defaultdict

# Load the scraped Discourse data (full thread with questions and replies)
with open("discourse_full_threads.json") as f:
    posts = json.load(f)

# Map: (topic_id, post_number) → post
post_lookup = {(post["topic_id"], post["post_number"]): post for post in posts}

# Build: topic_id → list of posts (ordered by post_number)
threads = defaultdict(list)
for post in posts:
    threads[post["topic_id"]].append(post)
for topic_id in threads:
    threads[topic_id].sort(key=lambda x: x["post_number"])

# Official users (TA, instructors)
OFFICIAL_USERS = {"s.anand", "Jivraj", "Saransh_Saini", "carlton"}

# Extract Q&A pairs
qa_pairs = []
for topic_id, thread_posts in threads.items():
    topic_title = thread_posts[0]["topic_title"] if thread_posts else ""

    for post in thread_posts:
        if post["reply_to_post_number"] is not None:
            continue  # Skip replies, only look at top-level questions

        # This is a question (original or follow-up)
        question = post
        question_number = post["post_number"]

        # Find replies to this question from official users
        answers = [
            reply for reply in thread_posts
            if reply.get("reply_to_post_number") == question_number
            and reply["author"] in OFFICIAL_USERS
        ]

        if not answers:
            continue  # Skip if no official answer

        main_answer = answers[0]
        extra_answers = answers[1:]

        qa_pairs.append({
            "question": question["content"],
            "answer": main_answer["content"],
            "url": question["url"],
            "topic_title": topic_title
        })

# Save the extracted QA pairs
with open("qa_pairs.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"✅ Extracted {len(qa_pairs)} Q&A pairs.")
