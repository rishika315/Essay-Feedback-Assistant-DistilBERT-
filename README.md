# Essay Feedback Assistant (DistilBERT)

This project is an explainable essay evaluation system that uses a fine-tuned DistilBERT regressor to predict essay quality and provide detailed feedback on grammar, readability, vocabulary, and writing style. It aims to make automated writing assessment interpretable, consistent, and educational.

## Overview

The application accepts an essay as input and generates:

* A predicted score between 0 and 6 (similar to standard essay scoring rubrics)
* Readability metrics such as Flesch Reading Ease, SMOG Index, and Flesch-Kincaid Grade
* Grammar and clarity feedback
* Detection of vague phrases and clichés
* Lexical richness and style suggestions
* Actionable recommendations for improvement

The system is deployed as an interactive Streamlit web app.

## Features

* Automated essay scoring using a fine-tuned DistilBERT model
* Grammar and spelling feedback via LanguageTool
* Readability analysis using Textstat
* Lexical and stylistic evaluation
* Context-aware feedback rules for writing improvement
* Fast, interpretable results suitable for education or research use

## Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit
* NLTK
* Textstat
* LanguageTool

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/essay-feedback-assistant.git
   cd essay-feedback-assistant
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your fine-tuned DistilBERT model to the project folder (or specify the path in `MODEL_PATH` inside `essay_coach_bert.py`).

## Usage

Run the Streamlit application:

```bash
streamlit run essay_coach_bert.py
```

Then open the local URL displayed in the terminal.
Paste your essay into the text box and click **Analyze Essay** to view results.

## Model

The system uses a fine-tuned DistilBERT model trained for essay scoring regression.
You can replace it with your own fine-tuned model compatible with `AutoModelForSequenceClassification`.

## License

This project is released under the MIT License.
You are free to use, modify, and distribute it with appropriate attribution.
