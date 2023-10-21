# MaliciousLLMs_Demo

## Overview
This is the demo used for paper "" ...

---
## Installation

```python
gitclone https://github.com/404notfound4321/MaliciousLLMs_Demo.git
cd MaliciousLLMs_Demo
pip install -r requirements.txt
```
---

## Run the Demos

### API key management
Create a .env file and save your API key there:
`OPENAI_API_KEY = `

### Run demo seperately

```python
python standard_demo.py
```
or 
```python
python malicious_demo.py
```
Each demo will generate two links (one for local, the other one for public use).

---
## Customize Your Own Demo
+ If embedding an open source LLM such as LLama, uncomment the code block marked in the script and load the model's checkpoints.
+ If embedding a gpt model, simply change the model type variable (`engine`) in the script.

---
## Contact
