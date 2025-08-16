# **Fine-Tuning a Multimodal Vision Transformer with LoRA(Beans Dataset)**


## 1) Overview of the Issue

  To classify plant diseases, we apply a pre-trained multimodal Vision Transformer (CLIP ViT-
  B/32) to a small, domain-specific dataset (Hugging Face Beans). Instead of using full-model
  training, parameter-efficient fine-tuning (LoRA) aims to achieve high accuracy on a GPU with
  limited resources.
  
  The goal is to train a small percentage of parameters while improving task performance
  (classification) over zero-shot CLIP Preprocessing into image-text pairs, LoRA adaptation,
  training with contrastive loss, and an evaluation/analysis utilizing parameter counts and
  (optional) zero-shot vs. fine-tuned accuracy are all necessary deliverables.

## 2) Dataset Hugging Face

  Smartphone photos of bean leaves belonging to three classes: healthy, rusty, and angular
  leaf spots splits (as given): test, validation, and training.
  Why this dataset? For CLIP's contrastive objective (image text alignment), it is trivial to pair
  each image with a descriptive caption.each image with a descriptive caption.
  
## 3) Overview of the Method

### 3.1 Preprocessing

  Text prompts for caption mapping
  - angular_leaf_spot a picture of an angular leaf spot on a bean leaf
  - bean_rust a picture of a bean leaf covered in bean rust
  - A picture of a healthy bean leaf
  - CLIPProcessor turns (caption, image) into tensors:
  Images have been resized and normalized to meet CLIP specifications (ViT-B/32,224×224).
  For stable batching, the text was tokenized to 77 tokens (max_length=77).

### 3.2 LoRA & Model

  OpenAI/clip-vit-base-patch32 is the base model (image & text encoders).
  LoRA adapters were used for the q_proj, k_proj, v_proj, and out_proj attention projections in both encoders. Only LoRA parameters are trained; base weights are frozen.

### 3.3 Training-Loss

  symmetric cross-entropy over image text similarities, or CLIP contrastive loss.AdamW; LR = 5e-5, warmup = 5%; fp16 if GPU is available, is the optimalconfiguration.
  - Total batch size: 16; approximately 3 Colab epochs.
  Utilising get_image_features and get_text_features to calculate loss, the Custom Trainer circumvents the PEFT×CLIP inputs_embeds problem.

### 3.4 Evaluation Protocol-Zero-shot

  Prediction = argmax cosine similarity; Base CLIP encodes test images and class prompts.
  - Adjusted: Repeat the same classification on the test set after merging the LoRA base.
  Report accuracy; per-class metrics and a confusion matrix are optional.

## 4) Findings
   
  Final training loss: 2.0799; Zero-shot accuracy (base CLIP): 0.3047; Fine-tuned accuracy (LoRA, merged): 0.9141; Accuracy (fine-tuned zero-shot): +0.6094
  Efficiency of parameters: 152,260,353 total model parameters; 983,040 logical LoRA trainable parameters, or 0.6456% of the total
  (Note: The logical LoRA size reflects what was trained; all parameters are frozen during inference.)

## 5) Evaluation and Conversation
     
  Why LoRA is useful in this situation
  - Small dataset plus big prior: Strong visual text semantics are already encoded by CLIP. LoRA limits overfitting and lowers memory/compute by nudging this representation in the direction of the Beans classes with less than 1% trainable parameters.
  Rapid convergence: In about three epochs, accuracy increased from 30% to 91%.
  Although more epochs can be tried, overfitting on approximately 1,000 images is a risk and gains frequently diminish.
  What got better the most?
  Instead of learning generic textures, the model learnt more precise disease cues unique to bean leaves, such as rust versus angular leaf spot.
  Restrictions & Upcoming Projects
  For robustness, include stronger augmentations and early stopping.
  Experiment with prompt ensembling (using different templates for each class).


## 6) Artefacts & Reproducibility

  Torch 2.6.0 (CUDA), Transformers 4.55.0, PEFT (current), and Colab GPU are examples of the environment.
  Outputs saved:
  LoRA adapters: clip_beans_lora_adapters_ep3/ (small, for resume)
  - Inference of the merged model: clip_beans_merged_ep3/ + tokenizer/processor+class_prompts.json
  How to run (Colab)
  1) Launch the notebook that is ready for grading, Runtime Run everything.
  2) Once finished, download the file in HTML format for exporting the results.
  3) Verify predictions using the single-image inference cell.
  4) Use the printed FINAL RESULTS table to report the final metrics.
    
## 7) Conclusion
   
  Using LoRA, we were able to successfully adapt a multimodal Vision Transformer (CLIP
  ViT-B/32) to a small plant-disease dataset, training only about 0.65% of the model. This
  achieved an accuracy gain of +61 points over zero-shot (0.30 0.91), fulfilling the objectives
  of the assignment while adhering to resource limitations. The method is easy to use, quick,
  and deployable using the combined model.
