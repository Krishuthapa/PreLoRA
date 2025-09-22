### PreLORA: PreLoRA: Hybrid Pre-training of Vision Transformers with Full Training and Low-Rank Adapters

Optimizing Pre-training of ViT models on the ImageNet datasets.

Model used: https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTForImageClassification (```Huggingface ViT```)

Dataset used: https://www.image-net.org/download.php (```ImageNet```)

### For ImageNet Scratch Training Configuration

```ViT AugReg```: https://arxiv.org/abs/2106.10270


### Overall flow:

1. Start with full model training.  **(```PreLoRA/train/peft/vit-large-final.py```)**
2. Run the model in full parameter setup until the user-defined warmup steps.
3. Run partial convergence test after warmup is completed, based on weight norms and losses. **(```PreLoRA/helper_functions/convergence.py```)**
4. Once the convergence is passed, initiate LoRA adapters to the selected modules.  **(```PreLoRA/helper_functions/dora_instantiate.py```)**
5. Run the rank-assignment algorithm to assign dynamic ranks to each layer based on their convergence at the switch point.  **(```PreLoRA/helper_functions/rank_assignment_algorithm.py```)**
6. Train full model and LoRA parameters together for some epochs, to ensure LoRA is guided by full model during the intial learning.
7. Freeze full model and only train LoRA adapters for remaining part of training.

### Overall outcome:
In 64 GPU, a speed-up of 9hrs, 3x throughput improvement and 20% lesser GPU utilization was obtained.

### Results and Plots

- Result are stored inside the corresponding exp folder inside **```PreLoRA/peft_final_plots```**
- Code to generate plots can be found in **```PreLoRA/peft_final_plots/visualize_results_final.ipynb```** 





