# 🖥️ Code Generation Using LLMs and Evaluation

## 📌 Project Overview
This project explores the capabilities and limitations of **Large Language Models (LLMs)** in **automated code generation**, focusing on **SQL query generation**. We introduce a **Hierarchical Decomposition Framework** to improve **modularity, maintainability, and accuracy** in generated code.

### 🎯 Objectives:
- **Develop an LLM-based SQL code generation model**.
- **Improve code security, modularity, and maintainability**.
- **Evaluate performance using metrics like Fuzz Ratio, BLEU Score, and ROUGE Score**.

---

## 📂 Project Files
- **📄 Final Report:** `AIT726_Team5_MR_PR_UD_final_report.pdf`
- **📊 Presentation:** `Code-Generation-Using-LLMs-and-Evaluation.pptx`
- **📓 Jupyter Notebook (Model Implementation):** `Code_Gen_LLM.ipynb`

---

## 🔑 Key Features
- **🤖 LLM-Based Code Generation:** Leverages T5 and Hierarchical Decomposition for SQL query generation.
- **🔍 Modular Query Processing:** Breaks SQL queries into stages for improved **accuracy & debugging**.
- **🛡️ Security & Maintainability:** Addresses vulnerabilities like **SQL injection** and ensures reusable code.
- **📈 Performance Metrics:** Evaluates accuracy using **Fuzz Ratio, BLEU Score, and ROUGE Score**.

---

## 🗃️ Dataset Information
- **Source:** [SQL-Create-Context Dataset (Hugging Face)](https://huggingface.co/datasets/b-mc2/sql-create-context)
- **Components:**
  - `Question`: Natural language question input.
  - `Context`: Additional schema-related information.
  - `Answer`: Expected SQL query output.
- **Preprocessing:**
  - **Subset of 10,000 samples** used due to memory constraints.
  - **8,000 samples** for training and **2,000 samples** for testing.

---

## 🛠️ Technology Stack
- **Model:** T5 Transformer
- **Frameworks:** PyTorch, Hugging Face Transformers
- **Libraries:** Pandas, Scikit-Learn, PyLint, Bandit (Security Analysis)
- **Hardware:** Google Colab (T4 GPU)

---

## 🚀 Implementation Details

### **1️⃣ Baseline Model (T5 Transformer)**
- **Approach:** Text-to-text **sequence generation**.
- **Steps:**
  - Data Preprocessing: Encode input **("question" + "context")** and output **("answer")**.
  - Train **T5 model** on SQL query generation.
  - Validate performance using **Fuzz Ratio & BLEU Score**.
- **Limitations:**
  - High memory usage restricts dataset size.
  - Struggles with **complex multi-clause queries**.

### **2️⃣ Enhanced Model (Hierarchical Decomposition)**
- **Approach:** Multi-stage modular code generation.
- **Stages:**
  1. **Component Identification:** Extract **SQL elements** (SELECT, FROM, WHERE, etc.).
  2. **SQL Detail Filling:** Predict attributes for each **SQL component**.
  3. **Query Construction:** Assemble **final SQL query**.
- **Advantages:**
  - **Error Localization:** Fixes incorrect query components without regenerating the entire query.
  - **Improved Modularity:** Enhances **code readability and debugging**.
  - **Better Scalability:** Handles **complex SQL queries & diverse schemas**.

---

## 📋 Sample Queries
| Query Type | Description |
|------------|------------|
| **Simple Queries** | Convert natural language to **basic SELECT queries**. |
| **Complex Queries** | Generate SQL queries with **WHERE, GROUP BY, ORDER BY**. |
| **Multi-Table Joins** | Construct **JOIN operations** between tables. |
| **Parameterized Queries** | Prevent **SQL injection attacks**. |

---

## 📊 Model Evaluation

### **1️⃣ Performance Metrics**
| Metric | Baseline (T5) | Enhanced (Hierarchical) |
|--------|--------------|----------------|
| **Fuzz Ratio** | 78.4% | **91.2%** |
| **ROUGE Score** | 0.62 | **0.81** |
| **BLEU Score** | 0.58 | **0.75** |

### **2️⃣ Observations**
- **Baseline T5 struggles with complex SQL queries**.
- **Hierarchical Decomposition improves modularity & error handling**.
- **Memory efficiency**: Enhanced model is **lighter & scales better**.

---

## 🔄 Automation & Enhancements
- **📅 Dynamic Subtask Identification:** Automate query segmentation.
- **⚡ Optimized LLMs:** Explore **Codex, BERT, and GPT-4** for performance improvements.
- **🛠️ Multi-Agent Collaboration:** Use multiple models for **specialized query components**.

---

## ⚙️ How to Run the Project

### **1️⃣ Prerequisites**
- **Python 3.x**
- **Jupyter Notebook**
- **Google Colab (for GPU acceleration)**

### **2️⃣ Install Required Libraries**
```bash
pip install torch transformers pandas scikit-learn
pip install pylint bandit
```

---

## 📚 References

- **[SQL-Create-Context Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context)**: Hugging Face  
- **[T5 Model Documentation](https://huggingface.co/docs/transformers/en/model_doc/t5)**: Hugging Face Transformers  
- **[Fuzzy String Matching](https://www.analyticsvidhya.com/blog/2021/07/fuzzy-string-matching-a-hands-on-guide/)**: Analytics Vidhya  
- **[BERT for SQL Generation](https://arxiv.org/pdf/1910.07179v5)**: ArXiv  
- **[Codex for SQL Processing](https://arxiv.org/pdf/2204.08941)**: ArXiv  
- **[GitHub Copilot AI Pair Programmer: Asset or Liability?](https://arxiv.org/pdf/2206.15331)**: ArXiv  
- **[A Review on Code Generation with LLMs: Application and Evaluation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10403378)**: IEEE Xplore  
- **[Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning](https://www.semanticscholar.org/reader/cbd569036fc72ae7ff747350b91816440282596b)**: Semantic Scholar  

---

## 🤝 Contributors

### 🏫 George Mason University  
**AIT 726: NLP with Deep Learning**  

- 👤 **Mukesh Rajmohan**  
- 👤 **Praneeth Ravirala**  
- 👤 **Utkarsh Desai** 

### 👨‍🏫 Professor  
- **Dr. Lindi Liao** - *Instructor & Project Mentor*  

---
### ✅ Key Features:
- **Detailed breakdown** of **baseline vs. enhanced model**.
- **Hierarchical Decomposition framework** for **modular SQL generation**.
- **Evaluation with real-world benchmarks** (Fuzz Ratio, BLEU, ROUGE).
- **Execution guide** for **Google Colab & Jupyter Notebook**.

Let me know if you need any modifications! 🚀
