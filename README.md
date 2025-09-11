# torch2jax_openLLMs
# T2J: Prompt Augmentation Framework for PyTorch-to-JAX Translation

## Overview

T2J is a prompt augmentation framework designed to improve Large Language Model (LLM)-based PyTorch-to-JAX code translation using open-source, freely available language models. This repository contains experimental implementations and evaluations of different prompt augmentation strategies for enhancing the accuracy and reliability of automated code translation between PyTorch and JAX frameworks.

## Motivation

The transition from PyTorch to JAX is becoming increasingly important as the machine learning community embraces JAX's functional programming paradigm and its superior performance for high-performance computing. However, manual translation between these frameworks is time-consuming and error-prone. T2J addresses this challenge by leveraging the capabilities of open LLMs with carefully designed prompt augmentation techniques.

## Key Features

- **Prompt Augmentation Framework**: Systematic approach to enhancing LLM prompts for better code translation
- **Open LLM Integration**: Utilizes freely available language models for accessibility compared with our framework with openAI's LLMs
- **Comparative Analysis**: Direct comparison between baseline and augmented prompt strategies
- **PyTorch-to-JAX Focus**: Specialized for translating between these two popular ML frameworks

## Experimental Setup

### Baseline Approach (`mistral_7B_NO_JSON/`)
- Standard prompt-based translation without augmentation
- Serves as a baseline for comparison
- Uses direct instruction-following approach

### Augmented Approach (`mistral_7B_with_augmented_JSON/`)
- Implements JSON-based prompt augmentation
- Enhanced context and structured output format
- Improved translation accuracy through better prompt engineering

### Prerequisites

- Python 3.8+
- Access to open LLM APIs or local model deployment
- PyTorch and JAX frameworks for testing translations

## Methodology

T2J employs several key techniques:

1. **Structured Prompting**: Uses JSON-based output formatting for more reliable parsing
2. **Context Augmentation**: Provides additional context about PyTorch-JAX differences
3. **Example-Based Learning**: Includes relevant translation examples in prompts
4. **Error Handling**: Incorporates common translation pitfalls and solutions

## Evaluation

The framework includes comprehensive evaluation metrics:

- **Syntactic Correctness**: Valid JAX syntax generation
- **Semantic Equivalence**: Functional equivalence between original and translated code
- **Performance Metrics**: Translation accuracy and success rates
- **Human Evaluation**: Manual assessment of translation quality

## Results

[Results and findings will be documented here as experiments are completed]


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project is part of ongoing research in automated code translation and LLM-based programming assistance.*
