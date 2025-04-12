# **Git & Branching Strategy for Sentiment Analysis ( [IE7500 Repository](https://github.com/Northeastern-MSDAE/IE7500))**  

---

## **Introduction**  
This document outlines a **Git & Branching Strategy** using the structure and best practices demonstrated in the [IE7500 repository](https://github.com/Northeastern-MSDAE/IE7500). Our goal is to maintain an organized, scalable, and collaborative workflow for the **Sentiment Analysis on Twitter Dataset** project.

---

## **Repository Structure for this project**  
A well-structured GitHub repository makes collaboration easier and keeps the codebase maintainable. Below is a **recommended structure** based on the IE7500 repository:

```
📂 Sentiment-Analysis-Twitter
│── 📁 data                # Datasets (raw & processed)
│── 📁 notebooks           # Jupyter notebooks for EDA, model training
│── 📁 src                 # Python scripts for modular processing
│── 📁 results             # Evaluation results and reports
│── 📁 docs                # Documentation files (README, setup guide)
│── 📁 tests               # Unit tests for code reliability
│── requirements.txt       # Dependencies list
│── README.md              # Overview of the project
│── .gitignore             # Ignore unnecessary files (e.g., logs, cache)
│── LICENSE                # License information
```

###  **Why Follow This Structure?**
- **Clear Separation of Concerns**: Code, data, and documentation are logically organized.
- **Reusability & Maintainability**: Encourages modular and scalable development.
- **Easy Collaboration**: Makes it simpler for multiple contributors to work on different aspects of the project.

---

## **Branching Strategy**
We will use a **feature-branch workflow**:

### **Main Branches**
| Branch Name | Purpose |
|------------|---------|
| `main` | Stable, production-ready branch. Only tested and reviewed code is merged here. |
| `dev` | Active development branch. New features and bug fixes are merged here before going into `main`. |

### **Feature & Hotfix Branches**
| Branch Name | Purpose |
|------------|---------|
| `feature/<feature-name>` | New features (e.g., `feature/text-cleaning`) |
| `bugfix/<bug-name>` | Fixing bugs (e.g., `bugfix/remove-null-values`) |
| `hotfix/<issue-name>` | Urgent fixes in `main` (e.g., `hotfix/model-bug`) |
| `experiment/<experiment-name>` | Experimental changes (e.g., `experiment/bert-implementation`) |

---

## **Git Workflow Process**
This process ensures an **efficient and collaborative development cycle**

### ** Step 1: Clone the Repository**  
```bash
git clone https://github.com/Northeastern-MSDAE/IE7500.git
cd IE7500
```

### **Step 2: Create a New Branch for Development**  
Before making changes, create a **new branch** based on `dev`:

```bash
git checkout -b feature/text-cleaning dev
```

 **Branch Naming Guidelines:**
- **Use prefixes**: `feature/`, `bugfix/`, `hotfix/`, `experiment/`
- **Use hyphens (`-`)** to separate words
- **Keep names descriptive**:  
  ✅ `feature/text-cleaning`  
  ❌ `feature/tc`

### **Step 3: Make Changes & Commit**  
Once you’ve implemented your changes, **stage and commit** them:

```bash
git add src/preprocessing.py
git commit -m "Added text preprocessing steps"
```

 **Commit Message Best Practices**  
-  **Use present tense**: `"Implement new tokenizer"` (not `"Implemented tokenizer"`)
-  **Be concise but descriptive**
-  **Group related changes into a single commit**

---

### **Step 4: Push Changes to GitHub**
```bash
git push origin feature/text-cleaning
```

---

### **Step 5: Open a Pull Request (PR)**
Navigate to **GitHub → Repository → Pull Requests** and click **"New Pull Request"**.

#### PR **Title Example**
```
[Feature] Implement text preprocessing pipeline
```

#### PR **Description Example**
```markdown
### Description
- Implemented text tokenization, stopword removal, and lemmatization.
- Added `preprocessing.py` script under `src/` folder.
- Tested preprocessing steps on sample tweets.

### Checklist
- [x] Code follows PEP8 guidelines.
- [x] Unit tests added for text cleaning functions.
- [x] Documentation updated.

### Related Issues
Closes #10
```

---

### **Step 6: Review & Merge the PR**
Once the PR is approved:
1. **Merge into `dev`** for testing:
   ```bash
   git checkout dev
   git merge feature/text-cleaning
   ```
2. **Delete the feature branch** (locally & remotely):
   ```bash
   git branch -d feature/text-cleaning
   git push origin --delete feature/text-cleaning
   ```

3. **Once tested, merge `dev` → `main`**:
   ```bash
   git checkout main
   git merge dev
   ```

---

## **Handling Hotfixes & Bug Fixes**
For **urgent bug fixes**, create a `hotfix/` or `bugfix/` branch directly from `main`:

```bash
git checkout -b hotfix/model-bug main
```
- Fix the issue, commit, and push:
  ```bash
  git add .
  git commit -m "Fix model loading issue"
  git push origin hotfix/model-bug
  ```
- Open a PR to merge **hotfix → `main`**.
- Once approved, also merge it into `dev` to keep both branches updated.

---

## **Resolving Merge Conflicts**
If a **merge conflict** occurs:
1. Identify conflicting files:
   ```bash
   git status
   ```
2. Open the file and **manually resolve conflicts** (`<<<<<<<`, `=======`, `>>>>>>>` markers).
3. Stage and commit the resolved file:
   ```bash
   git add <conflicted-file>
   git commit -m "Resolved merge conflict in <file>"
   ```

---

## **Best Practices for Collaboration**
Follow these best practices for a smooth Git workflow:

* **Always create a new branch before working**  
* **Commit frequently with meaningful messages**  
* **Pull latest changes before starting new work**  
```bash
git checkout dev
git pull origin dev
```
* **Use `.gitignore`** to exclude unnecessary files  
* **Perform code reviews before merging PRs**  

---


## **Conclusion**
This Git & Branching Strategy ensures:
- **Efficient version control** using `feature`, `bugfix`, `hotfix` branches.
- **Clear organization** similar to the **IE7500 repository**.
- **Streamlined collaboration** with PRs, reviews, and branch merging.

By following this workflow, our **Sentiment Analysis on Twitter Dataset** project will stay **well-structured, scalable, and collaboration-friendly**. 

---

🔗 **Reference:**  
- [IE7500 Repository](https://github.com/Northeastern-MSDAE/IE7500)  
- [GitHub Docs: Working with Branches](https://docs.github.com/en/get-started/using-git/about-branches)