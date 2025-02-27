# Adversarial Decoding

## Overview

Adversarial Decoding is a project designed to explore and implement various decoding strategies and scoring mechanisms for language models. The project includes experiments such as jailbreak, LlamaGuard, and naturalness experiments, leveraging advanced language models like Meta-Llama.

## Features

- **Decoding Strategies**: Implementations of different decoding strategies including Jailbreak and LlamaGuard.
- **Scoring Mechanisms**: Various scoring mechanisms such as Perplexity, Naturalness, and Cosine Similarity.
- **Experimentation Framework**: A framework to run experiments with different configurations and models.
- **Dataset Handling**: Includes handling of datasets like HarmBench for testing and evaluation.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/adversarial-decoding.git
   cd adversarial-decoding
   ```

2. **Install dependencies:**

   Ensure you have Python installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file in the root directory and configure any necessary environment variables as per your setup.

## Usage

To run an experiment, use the `main.py` script with the appropriate arguments. For example, to run a jailbreak experiment:

```bash
python adversarial_decoding/main.py --experiment jailbreak --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Available Experiments

- **Jailbreak**: Test the model's ability to bypass restrictions.
- **LlamaGuard**: Evaluate the model's safety mechanisms.
- **Naturalness**: Assess the naturalness of generated text.
- **RAG**: Run retrieval-augmented generation experiments.

## Testing

To run the tests, execute the following command:

```bash
python -m unittest test.py
```

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please fork the repository and submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback, please contact [your-email@example.com]. 