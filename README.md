# SMS Spam Classification

This project involves building a machine learning model to classify SMS messages as spam or not spam using the Naive Bayes algorithm.

## Project Overview

The aim of this project is to create an effective spam filter for SMS messages. The dataset used consists of a collection of SMS messages labeled as 'spam' or 'ham' (not spam). The Naive Bayes algorithm is employed to train a model that can predict whether a new SMS message is spam or ham.

## Dataset

The dataset used for this project is the 'SMSSpamCollection' dataset. It consists of two columns:
- `label`: Indicates whether the message is 'spam' or 'ham'.
- `message`: The content of the SMS message.

## Project Structure

The repository contains the following files:

- `SMSSpamCollection`: The dataset file containing SMS messages and their labels.
- `sms_spam_classification.ipynb`: Jupyter notebook containing the data preprocessing, model training, and evaluation code.
- `README.md`: This file, providing an overview of the project.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/codeMaverick7770/SMS-Spam-Classification-Model.git
    ```

2. Install the required dependencies:
    ```bash
    pip install pandas scikit-learn
    ```

3. Run the Jupyter notebook to preprocess the data, train the model, and evaluate its performance.

## Usage

1. Load the dataset:
    ```python
    import pandas as pd
    msg = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
    ```

2. Preprocess the data, train the model, and evaluate its performance using the provided notebook.

## Model

The Naive Bayes algorithm is used for training the model. This algorithm is particularly well-suited for text classification tasks due to its simplicity and effectiveness.

## Evaluation

The model's performance is evaluated using common metrics such as accuracy, precision, recall, and F1-score. The results demonstrate the model's ability to accurately classify SMS messages as spam or ham.

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or need further information, please feel free to contact me at Mtauqeer7770@gmail.com.

---

Feel free to customize this README file as per your specific project details and requirements.
