# nsys_recipes

## Usage
clone this repository and replace the original one
```
git clone ssh://git@gitlab-master.nvidia.com:12051/congliangx/nsys_recipes.git
rm /opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/python/packages/nsys_recipe/recipes
rsync -aPp nsys_recipes/nccl_gpu_overlap_trace  /opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/python/packages/nsys_recipe/recipes
```

