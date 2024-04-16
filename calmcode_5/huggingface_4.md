Sharing Model and Tokenizers can be found [here](https://huggingface.co/learn/nlp-course/chapter4/1?fw=pt)

To find all the models - main website: https://huggingface.co/ (Every model show their tasks)

## 1. Using pretrained model

Initiate a model with the `pipeline( )` function

```python
from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
```

Also, we can instntiate the checkpoint using the model architecture directly:

```python
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
```

p.s., if we have model to upload, refer to this page: https://huggingface.co/learn/nlp-course/chapter4/3?fw=pt

(I don't have any model to upload)

- The models uploaded can be managed with git and git-lfs
- On the hugging face hub web interface, you can:
  - Create a new model repository.
  - Manage and edit files.
  - Upload files.
  - See diffs across versions.
- Common git operations here:
  - `git_commit()`
  - `git_pull()`
  - `git_push()`





