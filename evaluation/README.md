# Setup
### Download the models
```
chmod +x download_models.sh
./download_models.sh
```
### Generate Merged Models
```
python generate_merged_models.py 
    --rl_model models/AFM-CodeAgent-7B-rl \
    --sft_model models/AFM-CodeAgent-7b-sft \
    --base_model modles/AFM-CodeAgent-7B-sft \
    --alphas -2.0 -1.0 1.0 2.0
```
### Evaluate a Model
```
lm_eval --model hf --model_args pretrained=merges/alpha1.0,dtype=bfloat16,trust_remote_code=True --tasks humaneval --device cuda:0 --batch_size 1 --gen_kwargs temperature=0.5,max_gen_toks=2048 --log_samples --output_path results.json --confirm_run_unsafe_code
```
