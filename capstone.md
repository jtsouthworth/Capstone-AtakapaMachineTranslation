# Project Title: A Multi-Agent Translation Pipeline for Low-Resource Polysynthetic Languages (Atakapa)

## Phase 1: Project Scoping & Business Alignment
*Defining the project’s value and translating the linguistic challenge into a formal data problem.*
* **Business Context & Problem:** Mainstream translation models fail on dormant, polysynthetic languages due to complex morphology and "data starvation." The business problem is figuring out how to build an accurate translation engine when millions of training tokens are unavailable.
* **Stakeholder Value:** Delivering a low-data, high-accuracy translation architecture provides immense value to the Atakapa-Ishak Nation (cultural revitalization), computational linguists, and NLP researchers looking for scalable frameworks for other endangered languages.
* **Translating to a Data Question:** "How can we use deterministic Python pipelines to generate synthetic morphological data, and use multi-agent AI orchestration to translate texts?"

## Phase 2: Data Sourcing, Exploration, and Feature Engineering
*Understanding the data's limitations and preparing it for machine learning.*
* **Data Sources & Quality:** The primary sources are Kimball’s grammar, 400 isolated sentences, and 9 historical texts. The quality is highly controlled but limited in volume. Future sourcing will rely on archival recovery or adapting the pipeline to neighboring languages.
* **Exploratory Data Analysis (EDA) & Visualization:** * *Code Structure:* Conducted in isolated Jupyter Notebooks (keeping EDA cleanly separated from production modeling code).
  * *Visualizations:* Plotting the distribution of verb valency (transitive vs. intransitive), affix frequencies, and sentence lengths. Mapping the relationships between key entities (Roots $\rightarrow$ Affixes).
* **Feature Engineering:** Using linguistic domain knowledge to define critical features (e.g., tagging nouns as `animate` or `inalienable`, verbs as `stative`). These categorical features will dictate the downstream routing logic in the pipeline.

## Phase 3: The Data Generation & Preparation Pipeline
*Building the reproducible infrastructure to solve the "data starvation" problem.*
* **The Deterministic Pipeline (The Switchboard):** Building a reproducible Python script that ingests the Lexicon and Morphology CSVs and combines them mathematically. 
* **Data Generation:** Automatically generating 50,000+ perfectly segmented synthetic sentences to train the model.
* **Encoding & Formatting:** Formatting the generated data into strictly structured `JSONL` files (Instruction/Input/Output) required for parameter-efficient fine-tuning (PEFT). 
* **Data Splitting:** Carving out a strictly isolated "holdout" test set from the 400 historical Kimball sentences to prevent data leakage during model evaluation.

## Phase 4: Machine Learning Architecture & Orchestration
*Selecting and coordinating multiple ML algorithms to solve different facets of the problem.*
* **Model 1: The Lexicographer (Retrieval/RAG)**
  * *Algorithm:* Lightweight text-embedding model (e.g., `all-MiniLM-L6-v2`) linked to a FAISS vector database.
  * *Role:* Retrieves grammatical rules and cultural context from the digitized textbook based on the input sentence.
* **Model 2: The Morphologist (SLM)**
  * *Algorithm:* Fine-tuned Small Language Model (e.g., Google Gemma 2B or Meta Llama-3 8B).
  * *Role:* Trained exclusively on the synthetic Switchboard data to output mathematically rigid, morpheme-by-morpheme literal glosses.
* **Model 3: The Ethnographic Editor (Commercial LLM)**
  * *Algorithm:* Zero-shot/Few-shot Large Language Model via API (e.g., Gemini or Claude).
  * *Role:* Synthesizes the retrieved grammar rules and the literal gloss into a fluent, culturally accurate English sentence.
* **Orchestration:** Utilizing **CrewAI** and **LangChain** to programmatically route data between these three models, creating an automated "Translation Committee."

## Phase 5: Evaluation, Metrics, and End-to-End Delivery
*Testing the models, building the UI, and deploying the final solution.*
* **Model Evaluation & Metrics:** * *Morphological Accuracy:* Evaluating Model 2 against the 100-sentence historical holdout set using **chrF** (character-level F-score, which is standard for polysynthetic languages, avoiding the flaws of word-level BLEU).
  * *Discourse Consistency:* Using the 9 unbroken historical texts to test Model 3's ability to maintain narrative flow and pronoun resolution across paragraphs.
* **End-to-End UI Construction:** Wrapping the entire multi-agent pipeline in a **Streamlit** or **Gradio** web interface. Users input an Atakapa sentence and see the real-time processing of the retrieval, the morphological segmentation, and the final translation.

## Phase 6: Project Management & Next Steps
*Documenting the effort and planning for production.*
* **Effort Allocation:** Documenting the split of labor (e.g., 40% Linguistic Data Engineering, 30% SLM Fine-Tuning, 20% CrewAI Orchestration, 10% UI/Evaluation).
* **Documentation:** Maintaining clean, heavily commented `.py` scripts for the pipeline, markdown files for the linguistic methodology, and a `requirements.txt` for environment reproduction.
* **Next Steps to Production:** Plan to containerize the final multi-agent application using Docker and deploy it to a cloud host (like AWS or Hugging Face Spaces) for live access by stakeholders.

---

### Why this hits the rubric:
* **Business/Stakeholders:** Covered in Phase 1.
* **Data Sources/Quality:** Covered in Phase 2.
* **EDA/Viz:** Covered in Phase 2.
* **Feature Engineering:** Covered in Phase 2.
* **Reproducible Pipeline & Code Separation:** Covered in Phases 2 and 3.
* **Multiple Algorithms & Metrics:** Covered in Phases 4 and 5.
* **End-to-End (UI & Tools):** Covered in Phases 4 and 5.
* **Documentation & Planning:** Covered in Phase 6.