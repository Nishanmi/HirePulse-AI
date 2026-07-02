# HirePulse AI 🚀

An intelligent, completely offline candidate discovery and ranking platform built for the **Redrob Intelligent Candidate Discovery & Ranking Hackathon**. 

HirePulse AI is designed to read a Job Description exactly like an expert human recruiter. It parses strict technical requirements, soft skills, and core responsibilities, and then evaluates 100,000 candidates across multiple independent axes using a highly optimized, CPU-only **Hybrid Retrieval and Multi-Signal Ranking Engine**.

---

## 🧠 Core Architecture

HirePulse AI is built on three major pillars:

1. **Domain-Driven Parsing Engine**  
   We utilize strict Pydantic domain models to validate data integrity. The parser calculates timeline consistency, detects overlapping roles, and extracts critical career trajectory features before the ranking even begins.

2. **Offline Hybrid Retrieval**  
   We combine the precision of **BM25 (lexical matching)** with the semantic understanding of **Jina Embeddings v2**. To achieve lightning-fast CPU performance, our embedding models are quantized into ONNX format. This hybrid approach ensures we enforce hard technical requirements while capturing candidates who use synonymous terminology.

3. **Multi-Signal Ensemble Ranking**  
   Candidates are evaluated across 5 independent signals:
   - **Role Alignment Score**: Similarity between candidate's past titles and the target role.
   - **Technical Match Score**: Direct overlap of required skills vs. candidate skills.
   - **Semantic Fit Score**: Deep embedding similarity of responsibilities and achievements.
   - **Experience Compatibility**: Timeline alignment against minimum and maximum required years.
   - **Redrob Behavioral Signals**: A multiplicative modifier evaluating profile completeness, endorsements, and validation scores.

---

## 📂 Repository Structure

```
HirePulse-AI/
├── backend/                  # Core ranking logic
│   ├── jd/                   # Job Description parsing and understanding
│   ├── candidate/            # Candidate data parsing and validation
│   ├── models/               # Pydantic domain models
│   ├── embeddings/           # ONNX-based fast semantic encoders
│   ├── retrieval/            # Hybrid Retrieval engine (BM25 + Semantic)
│   ├── ranking/              # Multi-Signal Scoring Engine
│   ├── explainability/       # Candidate explanation generation
│   ├── pipeline/             # End-to-end orchestration
│   └── submission/           # Final CSV export and validation
├── data/raw/                 # Dataset directory
│   ├── candidates.jsonl.gz   # Full 100k candidate dataset
│   ├── sample_jd.txt         # Target Job Description
│   └── sample_100.jsonl      # 100-candidate sample for Sandbox testing
├── scripts/                  # Utility scripts for offline preparation
│   ├── download_model.py     # Downloads ONNX models for offline use
│   └── build_indexes.py      # Pre-computes FAISS indexes (8-hour build)
├── indexes/                  # Pre-computed FAISS indexes (generated via script)
├── main.py                   # Main CLI entrypoint
├── requirements.txt          # Python dependencies
└── submission_metadata.yaml  # Hackathon compliance metadata
```

---

## ⚙️ Setup & Installation

The pipeline requires **Python 3.12+**. 

```bash
# 1. Clone the repository
git clone https://github.com/Nishanmi/HirePulse-AI.git
cd HirePulse-AI

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Download the ONNX models for offline semantic search
# (Internet must be ON for this specific preparation step)
HF_HUB_OFFLINE=0 python3 scripts/download_model.py
```

---

## 🏆 Stage 3: Reproducing the Submission

As per Section 10 of the `submission_spec.md`, the pipeline is designed to run entirely offline on a CPU. 

**Pre-computation Note:** Because calculating 100,000 embeddings on a CPU takes ~8 hours, the `indexes/` directory must be generated prior to ranking using `python3 scripts/build_indexes.py --candidates data/raw/candidates.jsonl.gz`. 

To generate the final Top 100 `submission.csv` strictly offline within 5 minutes, execute the following single command:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python3 main.py \
  --candidates data/raw/candidates.jsonl.gz \
  --jd data/raw/sample_jd.txt \
  --index-dir indexes \
  --out final_submission.csv
```

*Note: The `--index-dir indexes` flag skips the 8-hour embedding process and immediately loads the pre-computed vectors for lightning-fast retrieval.*

---

## 🧪 Running the Sandbox (Colab)

If you wish to test the end-to-end functionality of the pipeline without waiting 8 hours for the full dataset to index, you can run our 100-candidate sample in a Google Colab Sandbox. 

The pipeline will instantly build the FAISS index for 100 candidates in memory and execute the complete retrieval and ranking flow.

Run this block in a fresh Google Colab environment:

```python
# 1. Clone fresh
!git clone https://github.com/Nishanmi/HirePulse-AI.git
%cd HirePulse-AI

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Prepare the offline AI models (Requires Internet)
!HF_HUB_OFFLINE=0 python3 scripts/download_model.py

# 4. Run the Sandbox Pipeline STRICTLY OFFLINE on the 100-candidate sample
!TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python3 main.py \
  --candidates data/raw/sample_100.jsonl \
  --jd data/raw/sample_jd.txt \
  --out sandbox_results.csv

# 5. Display the Top Ranked Candidates
import pandas as pd
pd.read_csv("sandbox_results.csv").head(10)
```

---

## 🤖 AI Usage Declaration
* **Planning & Architecture:** ChatGPT 
* **Debugging & Log Review:** DeepSeek 
* **Implementation & Offline Integration:** Google Antigravity Agent
* **Candidate Data Privacy:** No candidate data (neither PII nor anonymized) was fed to any external LLMs during the development or execution of this pipeline. All semantic embeddings are computed locally.

*Refer to `submission_metadata.yaml` for official Hackathon metadata constraints.*
