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
### Evaluate Models
```
python evaluate.py models/
```

### Export the results .tar
Run a webserver
```
python -m http.server 8000
```
Go to your computer/runpods port 8000 over http and download the tar.
