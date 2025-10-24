# Spoken-SQuAD Question Answering using BERT

Fine-tuned **BERT models** for extractive Question Answering (QA) on **ASR-transcribed Spoken-SQuAD** data.  
The project evaluates model robustness under different noise levels (Word Error Rates).

---

## Models

- **Base:** `bert-base-uncased` – fine-tuned on Spoken-SQuAD  
- **Pretrained:** `bert-large-uncased-whole-word-masking-finetuned-squad` – QA-optimized model from Hugging Face  

Both models were trained for **3 epochs** using **AdamW (lr=2e-5)** with linear learning rate decay and evaluated on noisy ASR test sets.

---

##  Results

| Test Set | **Base (bert-base-uncased)** | **Pretrained (bert-large-uncased-whole-word-masking-finetuned-squad)** |
|-----------|------------------------------|------------------------------------------------------------------------|
| **No Noise** | F1 = 68.20 EM = 49.86 | **F1 = 76.65 EM = 62.74** |
| **WER 44** | F1 = 48.76 EM = 29.88 | **F1 = 54.89 EM = 37.34** |
| **WER 54** | F1 = 36.58 EM = 20.46 | **F1 = 41.62 EM = 25.92** |

> **Observation:** The pretrained QA model outperforms the baseline under all conditions, showing greater robustness to ASR noise.

---

## Highlights

- Evaluated on **Spoken-SQuAD** with clean and noisy ASR transcriptions  
- Pretrained QA BERT provides higher accuracy and generalization  
- Includes **training loss/accuracy curves** and **evaluation plots**

---

## Tech Stack

- **Framework:** PyTorch + Hugging Face Transformers  
- **Optimizer:** AdamW  
- **Tokenizer:** BertTokenizerFast  
- **Metrics:** F1 and Exact Match (EM)  




