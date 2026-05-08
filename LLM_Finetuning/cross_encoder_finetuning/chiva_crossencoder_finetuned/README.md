---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:539
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Surgical steps for Unknown ligation', 'Type 2B perforator ligation surgical steps: 1. Ultrasound identify perforator entry point (N2->N2). 2. Location depends on posYRatio: SFJ-Knee level, Hunterian, or calf location. 3. Longitudinal or transverse incision at ligation site. 4. Dissect to expose perforator vein. 5. Ligate perforator away from entry into GSV. 6. Preserve GSV trunk patent. 7. Multiple perforators: repeat for each based on hemodynamics.'],
    ['How to classify Type 1 venous shunt?', 'Type 1 venous shunt ligation: SFJ incompetent with N1->N2 entry and N2->N1 GSV reflux. Primary ligation site: high tie at saphenofemoral junction. Secondary: ligate below each RP N2->N1 except most distal. Approach: under local anesthesia, high ligation to prevent thrombosis. Consider venous diameter and multiple reflux points.'],
    ['Type 2A ligation: step-by-step procedure', 'Type 2A tributary entry ligation surgical steps: 1. Identify highest EP at N2->N3 junction via ultrasound guidance. 2. Small incisions at ligation levels for tributary branches. 3. Ligate GSV branch feeding tributary at junction level. 4. For multiple tributaries: assess each for size and location. 5. Preserve GSV trunk if diameter normal. 6. Technique: ligation at junction allows branch preservation. 7. Echo-guided marking essential for accurate localization.'],
    ['Type 3 ligation technique and approach', 'Type 1+2 complex venous shunt ligation: Dual entry with EP N1->N2 (SFJ incompetent) AND EP N2->N3 (tributary). RP patterns: RP N2->N1 AND RP N3 present. Ligation strategy depends on RP N2->N1 diameter and elimination test result. Small RP N2->N1: CHIVA 2 approach (ligate EP N2->N3 first, assess, then SFJ). Large/multiple RP N2->N1: simultaneous ligation of SFJ and tributaries. Key: RP diameter assessment determines treatment sequence.'],
    ['How to classify Type 3 venous shunt?', 'Type 3 venous shunt ligation (SFJ incompetent with tributary involvement): Dual EP at N1->N2 and N2->N3. Staged approach recommended. Stage 1: Ligate EP N2->N3 tributaries at their junctions. Stage 2: Follow-up at 6-12 months assess SFJ reflux development. If N2 reflux develops during follow-up: then perform SFJ ligation. Conservative initial approach to avoid unnecessary SFJ intervention.'],
]
scores = model.predict(pairs)
print(scores)
# [-1.8529  9.483   5.1104 -0.8741  8.9244]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Surgical steps for Unknown ligation',
    [
        'Type 2B perforator ligation surgical steps: 1. Ultrasound identify perforator entry point (N2->N2). 2. Location depends on posYRatio: SFJ-Knee level, Hunterian, or calf location. 3. Longitudinal or transverse incision at ligation site. 4. Dissect to expose perforator vein. 5. Ligate perforator away from entry into GSV. 6. Preserve GSV trunk patent. 7. Multiple perforators: repeat for each based on hemodynamics.',
        'Type 1 venous shunt ligation: SFJ incompetent with N1->N2 entry and N2->N1 GSV reflux. Primary ligation site: high tie at saphenofemoral junction. Secondary: ligate below each RP N2->N1 except most distal. Approach: under local anesthesia, high ligation to prevent thrombosis. Consider venous diameter and multiple reflux points.',
        'Type 2A tributary entry ligation surgical steps: 1. Identify highest EP at N2->N3 junction via ultrasound guidance. 2. Small incisions at ligation levels for tributary branches. 3. Ligate GSV branch feeding tributary at junction level. 4. For multiple tributaries: assess each for size and location. 5. Preserve GSV trunk if diameter normal. 6. Technique: ligation at junction allows branch preservation. 7. Echo-guided marking essential for accurate localization.',
        'Type 1+2 complex venous shunt ligation: Dual entry with EP N1->N2 (SFJ incompetent) AND EP N2->N3 (tributary). RP patterns: RP N2->N1 AND RP N3 present. Ligation strategy depends on RP N2->N1 diameter and elimination test result. Small RP N2->N1: CHIVA 2 approach (ligate EP N2->N3 first, assess, then SFJ). Large/multiple RP N2->N1: simultaneous ligation of SFJ and tributaries. Key: RP diameter assessment determines treatment sequence.',
        'Type 3 venous shunt ligation (SFJ incompetent with tributary involvement): Dual EP at N1->N2 and N2->N3. Staged approach recommended. Stage 1: Ligate EP N2->N3 tributaries at their junctions. Stage 2: Follow-up at 6-12 months assess SFJ reflux development. If N2 reflux develops during follow-up: then perform SFJ ligation. Conservative initial approach to avoid unnecessary SFJ intervention.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 539 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 539 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               | float                                                          |
  | details | <ul><li>min: 7 tokens</li><li>mean: 10.63 tokens</li><li>max: 14 tokens</li></ul> | <ul><li>min: 94 tokens</li><li>mean: 109.68 tokens</li><li>max: 146 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.71</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                            | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | label            |
  |:------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Surgical steps for Unknown ligation</code>      | <code>Type 2B perforator ligation surgical steps: 1. Ultrasound identify perforator entry point (N2->N2). 2. Location depends on posYRatio: SFJ-Knee level, Hunterian, or calf location. 3. Longitudinal or transverse incision at ligation site. 4. Dissect to expose perforator vein. 5. Ligate perforator away from entry into GSV. 6. Preserve GSV trunk patent. 7. Multiple perforators: repeat for each based on hemodynamics.</code>                                                   | <code>0.1</code> |
  | <code>How to classify Type 1 venous shunt?</code>     | <code>Type 1 venous shunt ligation: SFJ incompetent with N1->N2 entry and N2->N1 GSV reflux. Primary ligation site: high tie at saphenofemoral junction. Secondary: ligate below each RP N2->N1 except most distal. Approach: under local anesthesia, high ligation to prevent thrombosis. Consider venous diameter and multiple reflux points.</code>                                                                                                                                        | <code>1.0</code> |
  | <code>Type 2A ligation: step-by-step procedure</code> | <code>Type 2A tributary entry ligation surgical steps: 1. Identify highest EP at N2->N3 junction via ultrasound guidance. 2. Small incisions at ligation levels for tributary branches. 3. Ligate GSV branch feeding tributary at junction level. 4. For multiple tributaries: assess each for size and location. 5. Preserve GSV trunk if diameter normal. 6. Technique: ligation at junction allows branch preservation. 7. Echo-guided marking essential for accurate localization.</code> | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 10
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 10
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 2.7 minutes

### Framework Versions
- Python: 3.14.4
- Sentence Transformers: 5.4.1
- Transformers: 5.7.0
- PyTorch: 2.11.0+cpu
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->