Reddit Mental Health LLM Classification

A comprehensive project on hierarchical mental-health classification using Large Language Models (LLMs) on a custom Reddit dataset of 34,216 posts.
Each Reddit post is labeled across a 3-level DSM-style hierarchy:

Level 1: Category

Level 2: Subcategory

Level 3: Specific Disorder

The goal is to evaluate how well mental-health–specialized LLMs perform zero-shot and one-shot classification across these levels.

📌 Project Overview

34k Reddit posts collected from mental-health subreddits
(depression, anxiety, BPD, ADHD, OCD, PTSD, autism, etc.)

Full preprocessing, EDA, visualizations, token analysis, and dataset cleaning

Evaluation of 5 mental-health-focused LLMs:

CPSYCoun

EmoLLaMA-7B

EmoLLaMA-Chat-7B

Mental-Alpaca

Mental-LLaMA-Chat-7B

Two prompting strategies:

✔ Zero-shot

✔ One-shot

Three prediction tasks:

✔ Category

✔ Subcategory

✔ Specific Disorder

All outputs, results, and Jupyter notebooks are included.

📂 Repository Structure
reddit-mental-health-llm/
│
├── dataset/                  
├── cpsycoun/                 
├── Emollama-7b/              
├── Emollama-chat-7b/         
├── mental-alpaca/            
├── MentaLLaMA-chat-7B/       
│
├── output/                   
├── results/                  
│
├── dataset processing.ipynb
├── max length of words in dataset.ipynb
├── reddit_crawler.ipynb
│
└── README.md

🧠 Dataset Description (34,216 Reddit Posts)

All posts are mapped to a DSM-aligned 3-level disorder taxonomy.

Level 1 — Category

Mood Disorders

Anxiety Disorders

Obsessive-Compulsive & Related Disorders

Trauma & Stressor-Related Disorders

Personality Disorders

Neurodevelopmental Disorders

Schizophrenia Spectrum & Psychotic Disorders

Eating Disorders

Sleep Disorders

Substance-Related & Addictive Disorders

Level 2 — Subcategory

Mood Disorders

Depressive Disorders

Bipolar Disorders

Anxiety Disorders

Generalized Anxiety Disorder

Phobias

Panic Disorders

Obsessive-Compulsive & Related Disorders

OCD Spectrum

Trauma & Stressor-Related Disorders

PTSD Spectrum

Personality Disorders

Cluster A

Cluster B

Cluster C

Neurodevelopmental Disorders

Autism Spectrum Disorders

ADHD

Schizophrenia Spectrum

Psychotic Disorders

Eating Disorders

Restrictive/Compensatory Disorders

Sleep Disorders

Insomnia Spectrum

Substance-Related Disorders

Substance Use Disorders

Level 3 — Specific Disorders

Includes 50+ disorders such as:

Major Depressive Disorder

Bipolar I/II Disorder

Generalized Anxiety Disorder

Social Anxiety Disorder

OCD

PTSD

Borderline Personality Disorder

Autism Spectrum Disorder (ASD)

Schizophrenia

Anorexia Nervosa

Insomnia

Alcohol Use Disorder

…and many more.

(Full list available inside dataset.)

⚙️ Methods
1. Data Preprocessing

URL, emoji, punctuation removal

Lowercasing & cleaning Reddit markup

Removing empty/short posts

Deduplication

Token length analysis

Mapping each subreddit → hierarchical disorder label

2. Exploratory Data Analysis (EDA)

Category and subcategory distributions

Disorder frequency analysis

Word clouds

Post length statistics

3. Model Evaluation

Each LLM is evaluated in:

Zero-shot

One-shot

On all 3 label levels:

Category

Subcategory

Specific Disorder

Metrics Used

Precision (macro & weighted)

Recall (macro & weighted)

F1 Score (macro & weighted)

📊 Results Summary
CPSYCoun
One-shot
Label	Precision (macro)	Recall (macro)	F1 (macro)	Precision (weighted)	Recall (weighted)	F1 (weighted)
Category	43.60%	35.87%	36.79%	46.17%	39.19%	40.05%
Subcategory	39.06%	23.87%	24.81%	42.34%	27.19%	26.97%
Specific Disorder	58.42%	29.34%	31.42%	62.42%	29.28%	32.02%
Zero-shot

Category F1 (macro): 43.54%
Subcategory F1 (macro): 23.92%
Specific Disorder F1 (macro): 32.27%

EmoLLaMA-7B
One-shot

Category F1 (macro): 45.91%
Subcategory F1 (macro): 34.62%
Specific Disorder F1 (macro): 28.27%

Zero-shot

All metrics 0% (instruction failure)

EmoLLaMA-Chat-7B
One-shot

Category F1 (macro): 52.24%
Subcategory F1 (macro): 26.98%
Specific Disorder F1 (macro): 29.86%

Zero-shot

Category F1: 19.99%
Subcategory F1: 7.09%
Specific Disorder F1: 5.29%

Mental-Alpaca
One-shot

Category F1: 1.57%
Subcategory F1: 0.92%
Specific Disorder F1: 0.15%

Zero-shot

All metrics 0%

Mental-LLaMA-Chat-7B
One-shot

Category F1: 39.79%
Subcategory F1: 21.39%
Specific Disorder F1: 31.28%

Zero-shot

Very poor across all levels.

🔍 Key Observations

Zero-shot prompting is weak for most models (except CPSYCoun).

One-shot examples significantly improve performance.

Level-3 classification (33 disorder classes) is the most difficult task.

Instruction-following ability heavily affects results.

Base models struggle compared to chat-tuned versions.

🚀 Future Work

Fine-tuning on the 34k Reddit dataset

Larger context window models

Improved prompt engineering

Ranking-based evaluation

Creating a public benchmark leaderboard

📜 License

MIT License (or specify another license).

🙌 Acknowledgments

Reddit community contributors

Developers of CPSYCoun, EmoLLaMA, Mental-Alpaca, and Mental-LLaMA

Open-source community supporting mental-health NLP research
