# How to download the trajecotires for swebench
```
cd data
chmod +x download_trajectories.sh
./download_trajectories.sh
```

# Convert the trajectory to a sft friendly dataset
```
python trajectory_to_dataset.py <PATH-TO-TRAJECTORIES-FOLDER>
```
# Run SFT on the dataset
```
python sft.py <PATH-TO-DATSET>
```
 