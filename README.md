
# Embedding Accuracy Evaluation with Visual Representation

This project provides a comprehensive evaluation framework for comparing embeddings generated by different language models. Specifically, it computes accuracy rates (Top-1 to Top-5) for both "question-to-answer" and "answer-to-question" matching methods and visualizes the embeddings using `t-SNE` for a better understanding of model performance.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Accuracy Calculation Details](#accuracy-calculation-details)
- [Visualization of Results](#visualization-of-results)
- [Results Interpretation](#results-interpretation)
- [Contact](#contact)

## Overview

The goal of this project is to assess the accuracy of embeddings generated by various models by evaluating their ability to match questions with correct answers. It also provides a visual representation of embedding distributions, helping us analyze the model's effectiveness in different scenarios.

This project:
1. Computes Top-1 to Top-5 accuracy rates for two methods:
   - **Question-to-Answer Matching**: Checks if the correct answer is within the top k closest embeddings to each question.
   - **Answer-to-Question Matching**: Checks if the correct question is within the top k closest embeddings to each answer.
2. Generates visualizations for each model's embeddings using `t-SNE`, with distinct colors representing question-to-answer and answer-to-question methods.

## Installation

Ensure you have Python installed (Python 3.7 or later is recommended). Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencies
This project requires the following Python libraries:
- `torch`: For loading and saving embeddings.
- `matplotlib`: For creating accuracy graphs.
- `sklearn`: For dimensionality reduction (`t-SNE`).

## Project Structure

The project has the following structure:

```
.
├── embeddings_q2a/                   # Directory for question-to-answer embeddings
├── embeddings_a2q/                   # Directory for answer-to-question embeddings
├── main.py                           # Script for generating embeddings and saving them
├── evaluate.py                       # Script for calculating accuracy and generating graphs
├── requirements.txt                  # List of dependencies
└── README.md                         # Project documentation
```

## Usage

To use this project, first run `main.py` to generate and save embeddings in the appropriate folders (`embeddings_q2a` for question-to-answer embeddings and `embeddings_a2q` for answer-to-question embeddings).

```bash
python main.py
```

Then, run `evaluate.py` to calculate Top-1 to Top-5 accuracy for each model and generate visualizations for embedding distributions:

```bash
python evaluate.py
```

This script will display the Top-1 and Top-5 accuracy values in the console for each model and method and save the visualization graphs as `.png` files.

## Accuracy Calculation Details

The project calculates Top-1 through Top-5 accuracy as follows:
1. **Top-1 Accuracy**: Measures if the correct answer/question is the closest match.
2. **Top-5 Accuracy**: Measures if the correct answer/question is within the five closest matches.

These scores are computed separately for both `question-to-answer` and `answer-to-question` methods and are printed in the console.

## Visualization of Results

After accuracy calculations, the script generates a `t-SNE` visualization for each model:
- Each model has a unique graph, representing two methods (`question-to-answer` and `answer-to-question`) in distinct colors.
- The graphs help to understand how well each model clusters relevant question-answer pairs in a 2D space.

Visualizations are saved as `.png` files in the project directory, named after the respective model.

## Results Interpretation

Higher accuracy in `Top-k` scores suggests better performance of the embeddings in matching questions to answers or vice versa. Clear clustering in the `t-SNE` plots indicates that the model has effectively encoded the semantic relationships between questions and answers.

## Contact

For any questions or support regarding this project, please contact [Your Name/Email].
