# # save this as data/download_papers.py

# import arxiv
# import os
# import time

# # ---------------- CONFIG ----------------
# DOMAINS = {
#     "rag": [
#         "retrieval augmented generation architecture",
#         "agentic AI systems LLM agents tool use",
#         "hybrid search dense sparse retrieval"
#     ],
#     "ml": [
#         "supervised machine learning classification regression",
#         "unsupervised learning clustering dimensionality reduction"
#     ],
#     "predictive": [
#         "predictive analytics time series forecasting machine learning",
#         "customer churn prediction machine learning"
#     ],
#     "deep_learning": [
#         "deep learning neural networks CNN RNN transformers",
#         "training deep neural networks optimization techniques"
#     ],
#     "computer_vision": [
#         "computer vision image classification object detection",
#         "YOLO Faster R-CNN vision transformers"
#     ],
#     "nlp": [
#         "transformers BERT GPT architectures NLP",
#         "text classification sentiment analysis NLP"
#     ],
#     "mlops": [
#         "MLOps model deployment monitoring drift detection",
#         "feature stores data pipelines machine learning systems"
#     ]
# }

# PER_DOMAIN_LIMIT = 10
# SAVE_DIR = "data/papers"

# # ---------------- SETUP ----------------
# client = arxiv.Client()
# os.makedirs(SAVE_DIR, exist_ok=True)

# seen = set()
# domain_counts = {d: 0 for d in DOMAINS}

# # ---------------- CORE LOOP ----------------
# for domain, queries in DOMAINS.items():
#     print(f"\n=== Domain: {domain} ===")

#     for query in queries:
#         if domain_counts[domain] >= PER_DOMAIN_LIMIT:
#             break

#         search = arxiv.Search(
#             query=query,
#             max_results=20,  # fetch extra to compensate duplicates
#             sort_by=arxiv.SortCriterion.Relevance
#         )

#         for paper in client.results(search):

#             if domain_counts[domain] >= PER_DOMAIN_LIMIT:
#                 break

#             if paper.entry_id in seen:
#                 continue

#             try:
#                 filename = f"{domain}_{domain_counts[domain]:02d}_{paper.entry_id.split('/')[-1]}.pdf"

#                 paper.download_pdf(
#                     dirpath=SAVE_DIR,
#                     filename=filename
#                 )

#                 seen.add(paper.entry_id)
#                 domain_counts[domain] += 1

#                 print(f"[{domain_counts[domain]}/{PER_DOMAIN_LIMIT}] {paper.title}")

#                 # rate limiting
#                 time.sleep(1)

#             except Exception as e:
#                 print(f"Failed: {paper.title} | Error: {e}")
#                 continue

# # ---------------- SUMMARY ----------------
# total = sum(domain_counts.values())

# print("\n=== SUMMARY ===")
# for d, c in domain_counts.items():
#     print(f"{d}: {c}")

# print(f"\nTotal downloaded: {total}")