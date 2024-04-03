Hugging Face Introduction Sector can be found [here](https://huggingface.co/learn/nlp-course/chapter1/1)

## 1. Natural Language Processing

NLP: 
- Classifying whole sentences: Is it a spam? 
- Classifying each word in asentence: NER
- Generating text content: Completing a prompt with auto-generated text
- Extracting an answer from a text
- Generating a new sentence from the input.

## 2. Transformers

pipeline: connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer (pretrained model will be used)

``` python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.", "I hate this so much!")
```

The result could be:

```python
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

#### Available pipelines:

- `feature-extraction` (get the vector representation of a text)

- `fill-mask`: fill the blanks in a given text

  - ```python
    from transformers import pipeline
    
    unmasker = pipeline("fill-mask")
    unmasker("This course will teach you all about <mask> models.", top_k=2)
    ```

  - ```python
    [{'sequence': 'This course will teach you all about mathematical models.',
      'score': 0.19619831442832947,
      'token': 30412,
      'token_str': ' mathematical'},
     {'sequence': 'This course will teach you all about computational models.',
      'score': 0.04052725434303284,
      'token': 38163,
      'token_str': ' computational'}]
    ```

- `ner` (named entity recognition)

  - ```python
    from transformers import pipeline
    
    ner = pipeline("ner", grouped_entities=True)
    ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    ```

  - ```python
    [{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
     {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
     {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
    ]
    ```

- `question-answering`

  - ```python
    from transformers import pipeline
    
    question_answerer = pipeline("question-answering")
    question_answerer(
        question="Where do I work?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )
    ```

  - ```python
    {'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
    ```

- `sentiment-analysis`

- `summarization`

  - ```python
    from transformers import pipeline
    
    summarizer = pipeline("summarization")
    summarizer(
        """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.
    
        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
    )
    ```

  - ```python
    [{'summary_text': ' America has changed dramatically during recent years . The '
                      'number of engineering graduates in the U.S. has declined in '
                      'traditional engineering disciplines such as mechanical, civil '
                      ', electrical, chemical, and aeronautical engineering . Rapidly '
                      'developing economies such as China and India, as well as other '
                      'industrial countries in Europe and Asia, continue to encourage '
                      'and advance engineering .'}]
    ```

  - A `max_length` or `min_length` can also be set.

- `text-generation`: give a prompt and the model will autho-complete it by generating the remaining text.

  - ```python
    from transformers import pipeline
    
    generator = pipeline("text-generation")
    generator("In this course, we will teach you how to")
    ```

  - ```python
    [{'generated_text': 'In this course, we will teach you how to understand and use '
                        'data flow and data interchange when handling user data. We '
                        'will be working with one or more of the most commonly used '
                        'data flows — data flows of various types, as seen by the '
                        'HTTP'}]
    ```

- `translation`

  - ```python
    from transformers import pipeline
    
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    translator("Ce cours est produit par Hugging Face.")
    ```

  - ```python
    [{'translation_text': 'This course is produced by Hugging Face.'}]
    ```

- `zero-shot-classification`:specify which labels to use for the classification, so you don’t have to rely on the labels of the pretrained model.

  - ```python
    from transformers import pipeline
    
    classifier = pipeline("zero-shot-classification")
    classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    ```

  - ```python
    {'sequence': 'This is a course about the Transformers library',
     'labels': ['education', 'business', 'politics'],
     'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
    ```

#### if we want to keep it direct and any pipeline can be used?

The [model hub](https://huggingface.co/models) can be useful.

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

```python
[{'generated_text': 'In this course, we will teach you how to manipulate the world and '
                    'move your mental and physical capabilities to your advantage.'},
 {'generated_text': 'In this course, we will teach you how to become an expert and '
                    'practice realtime, and with a hands on experience on both real '
                    'time and real'}]
```

#### Categories of Pipelines

- GPT-like: auto-regressive transformer models.
- BERT-like: auto-encoding transformer models.
- BART/T5-like: sequence-to-sequence transformer models.

#### Transfer Learning

- pretraining
- fine-tuning

## 3. Encoder and Decoder

Encoder: The encoder receives an input and builds a representation of it (its features). 

- encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.

Decoder:  The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence.

- decoder-only models:  Good for generative tasks such as text generation.

- encoder-decoder models/ sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

#### 1) encoder models: sentence classification, NER, extractive question answering

- [ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)
- [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)
- [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)

#### 2) decoder models: text generation

- [CTRL](https://huggingface.co/transformers/model_doc/ctrl)
- [GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)
- [GPT-2](https://huggingface.co/transformers/model_doc/gpt2)
- [Transformer XL](https://huggingface.co/transformers/model_doc/transfo-xl)

#### 3) Sequence-to-sequence models: summarization, translation, generative question answering

- [BART](https://huggingface.co/transformers/model_doc/bart)
- [mBART](https://huggingface.co/transformers/model_doc/mbart)
- [Marian](https://huggingface.co/transformers/model_doc/marian)
- [T5](https://huggingface.co/transformers/model_doc/t5)

