# How to download the trajecotires for swebench
Use this repo: `https://github.com/SWE-bench/experiments` to download the swe benc trajectories.

# Convert the trajectory to a sft friendly dataset
```
python trajectory_to_dataset.py <PATH-TO-TRAJECTORIES-FOLDER>
```
# Run SFT on the dataset
```
python sft.py <PATH-TO-DATSET>
```
 