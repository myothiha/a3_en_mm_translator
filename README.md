# NLP Assignment 2 (AIT - DSAI)

- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [Task 1 - Dataset](#task-1---dataset)
- [Task 2 - Model Training](#task-2---model-training)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Architecture](#model-architecture)
    - [Training Process](#training-process)
    - [Hyperparameters](#hyperparameters)
- [Task 3 - Web Application](#task-3---web-application)
    - [How to run](#how-to-run)
    - [Usage](#usage)
    - [Documentation](#documentation)
- [Result](#result)

## Student Information
 - Name: Myo Thiha
 - ID: st123783

## Files Structure
 - In the code folder, The Jupytor notebook files (training) can be located.
 - The 'app' folder include 
    - `app.py` file for the entry point of the web application
    - Dockerfile and docker-compose.yaml for containerization of the application.
    - `template` folder to hold the HTML pages.
    - `models` folder which contains LSTM model exports and its metadata file.

## Task 1 - Dataset
- Source: https://www.kaggle.com/datasets/thedevastator/short-jokes-dataset
- The dataset is a csv file containing a collections of 231,657 short jokes.
- Then, I split it into training, validation and testing sets: 187641, 20850 and 23166 records respectively.

## Task 2 - Model Training

### Data Preprocessing

#### Tokenizing

- The text data is tokenized using the `torchtext` library's basic English tokenizer.
- The `tokenize_data` function is applied to tokenize the text in the dataset, and the result is stored in a new field named 'tokens'.
- The entire dataset is tokenized using the `map` function, and the 'text' field is removed, keeping only the 'tokens' field.
- The tokenized dataset is stored in the variable `tokenized_dataset`.

#### Numericalizing

- A vocabulary is built from the tokenized data using `torchtext.vocab.build_vocab_from_iterator`.
- The minimum frequency for a token to be included in the vocabulary is set to 3.
- Special tokens `<unk>` (unknown) and `<eos>` (end of sequence) are inserted into the vocabulary at indices 0 and 1, respectively.
- The default index for the vocabulary is set to the index of the `<unk>` token.
- The resulting vocabulary is stored in the variable `vocab`.

### Model Architecture
The Joke Generation Language Model is built using PyTorch and consists of the following components:

1. **Embedding Layer:**
   - Input: Word indices
   - Output: Word embeddings of dimension `emb_dim`
   - Implemented using `nn.Embedding`

2. **LSTM Layer:**
   - Input: Word embeddings
   - Output: Hidden states for each time step
   - Implemented using `nn.LSTM` with `num_layers` LSTM layers, each having `emb_dim` input features and producing `hid_dim` output features. `batch_first=True` is used.

3. **Dropout Layer:**
   - Applied after the embedding layer and LSTM layer
   - Implemented using `nn.Dropout`

4. **Linear Layer:**
   - Input: Hidden states from LSTM layer after dropout
   - Output: Scores for each word in the vocabulary
   - Implemented using `nn.Linear`

5. **Initialization:**
   - Weights of embedding layer and linear layer are initialized with uniform values.
   - LSTM weights are initialized similarly for input-to-hidden and hidden-to-hidden connections.

6. **Hidden State Initialization:**
   - `init_hidden` method initializes the hidden state and cell state for the LSTM.

7. **Forward Method:**
   - Takes an input sequence (`src`) and initial hidden state.
   - Embeds the input sequence, passes through LSTM, applies dropout, and passes through linear layer.
   - Returns output scores and updated hidden state.

### Training Process
- Adam optimizer with learning rate (`lr`).
- CrossEntropyLoss used as the loss function.
- Total trainable parameters printed.

### Hyperparameters:

- `vocab_size`: Size of the vocabulary.
- `emb_dim`: Dimensionality of word embeddings.
- `hid_dim`: Number of features in LSTM hidden state.
- `num_layers`: Number of LSTM layers.
- `dropout_rate`: Dropout probability.
- `lr`: Learning rate.

## Task 3 - Web Application

### How to run?
 - Run the `docker compose up` in the app folder.
 - Then, the application can be accessed on http://localhost:8000
 - You will directly land on the "Home" page.

### Usage:
- Input: After you run the web app, there will be a textbox where you can type your prompt. However, the prompt should not contain a period such as '.' and '?'. In most cases, the model will think it is a complete sentence and generate nothing but a sequence of full stops or question marks.
- Output: after that, you can hit the 'generate' button and you will see the generated result below.

### Documentation:

#### Model Loading:
- The web application loads the language model from the exported state dictionary file (`best-val-lstm_lm.pt`).
- Model parameters and vocabulary are loaded from the metadata file (`meta.pkl`).

#### Tokenizer:
- The `basic_english` tokenizer from `torchtext` is used for processing text inputs.

#### Integration Steps:

1. **Loading Model and Metadata:**
    - The `LSTMLanguageModel` class is instantiated with parameters loaded from the metadata file.
    - The model is moved to the appropriate device (CPU or GPU).
    - State dictionary is loaded into the model.

    ```python
    model = LSTMLanguageModel(**args).to(device)
    model.load_state_dict(torch.load('models/best-val-lstm_lm.pt', map_location=device))
    ```

2. **Loading Vocabulary:**
    - Vocabulary is built using the tokenized data from the training set in the metadata file.

    ```python
    vocab = meta['vocab']
    ```

3. **Flask Integration:**
    - Flask routes (`/` and `/generate`) are defined to handle user interaction and text generation.
    - User input is taken from the form, and the `generate` function is called for text generation.

    ```python
    prompt = request.form['query'].strip()
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device)
    ```

4. **Result Rendering:**
    - The generated text is then rendered on the web page using the Flask template.

    ```python
    return render_template('index.html', result=generation, old_query=prompt)
    ```

## Result

The language model was evaluated using perplexity scores on different datasets. Perplexity is a measure of how well the model predicts the data.

- **Train Perplexity:** 42.071
  - The perplexity score on the training dataset.

- **Valid Perplexity:** 49.605
  - The perplexity score on the validation dataset.

- **Test Perplexity:** 49.457
  - The perplexity score on the test dataset.

Lower perplexity scores indicate better performance, with the model making more accurate predictions on the given datasets.

These scores provide an insight into the language model's ability to understand and generate text, with lower perplexity values indicating better performance.