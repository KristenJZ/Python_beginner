# [Using Transformers](https://huggingface.co/learn/nlp-course/chapter2/1?fw=pt)

## 1. Behind the pipeline

![pipeline behind](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline.svg)

1）tokenize

- splitting the input into words, subwords, or symbols that are called tokens.
- mapping each token to an integer
- adding additional inputs that may be useful to the model

2）preprocessing - to have the tokenizer

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

3）convert the list of input IDs to tensors

- specy the type of tensors we want to get back (PyTorch, TensorFlow, or plain Numpy)

```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

4）Going through the model

![full model](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg)

## 2. Creating a model 

```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

# In this way, the model is randomly initialized (with random values)
```

so we need to train it first.

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

Saving a model

```python
model.save_pretrained("directory_on_my_computer")
```

```python
ls directory_on_my_computer

config.json pytorch_model.bin
#architecture: `config.json` file is the attributes necessary to build the model architecture (meta data).
#model's parameters: The `pytorch_model.bin` file is known as the state dictionary; it contains all your model’s weights. 
```

## 3. Using a model for inference

```python
sequences = ["Hello!", "Cool.", "Nice!"]
```

```python
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
```

```python
import torch

model_inputs = torch.tensor(encoded_sequences)
```

## 4. Tokenizer

1）word-based

```python
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
```

`dog` is different than `dogs` in the tokenizer.

for some unknown token, it will be `[UNK]` or `<unk>`

2）character-based: depending on language

3）subword tokenization

annoyingly = annoying + ly

(two tokens that have a semantic meaning while being space-efficient)

4）loading and saving

e.g., loading the BERT tokenizer

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokenizer("Using a Transformer network is simple")

{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.save_pretrained("directory_on_my_computer")
```

## 5. Encoding: translating text to numbers

two-step process: the tokenization + the conversion to input IDs.

- The tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

the output:

```python
['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
```

- From tokens to input IDs

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

```python
[7993, 170, 11303, 1200, 2443, 1110, 3014]
```

## 6. Decoding: from the vocabulary indices, get a string

```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```

`'Using a Transformer network is simple'`

## 7. Multiple sequences

The model always expect a batch of inputs. So if only one sequences is sent, it will fail.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

output:

```python
Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
Logits: [[-2.7276,  2.8789]]
```

1）multiple sequences sent batched

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
# To take all tokens of a sequence, it take account the padding tokens.

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

2）For longer sequences

- Most models handle sequences of up to 512 or 1024 tokens.
- Will crash when asked to process longer sequences

Way 1: Use a model with a longer supported sequence length. E.g., [longformer](https://huggingface.co/docs/transformers/model_doc/longformer), [LED](https://huggingface.co/docs/transformers/model_doc/led)

Way 2: truncate the sequence by specifying the max_sequence

`sequence = sequence[:max_sequence_length]`

## 8. Put it all together

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

It could also handle multiple sequences at a time, with no change in the API:

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)
```

It can pad based on several objectives:

```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

It can also truncate sequences:

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

Can return different tensors:

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

finally, output:

`output = model (**tokens)`

If we put it all together, example:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```



