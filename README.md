# nsys_recipes

## Usage
clone this repository and add it to the package
```
git clone ssh://git@gitlab-master.nvidia.com:12051/congliangx/nsys_recipes.git
sudo rsync -aPp nsys_recipes/compute_comm_trace /opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/python/packages/nsys_recipe/recipes/
sudo rsync -aPp nsys_recipes/lib/*  /opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/python/packages/nsys_recipe/lib/
```

"--subtimerange" is optional, if you only want to show the static of the total profile or only 1 part of the total profile, you can choose to use "--start" "--end",  if you want to show the subrange statics as well as the total range, you can use "--subtimerage", it's time unit is 's',  you can also put multi time range, for example, you can put --subtimerange 10.452-10.587,10.601-10.882, it means you want to stat the time range from 10.452s to 10.587s , and also 10.601s to 10.882s. 
```
nsys recipe compute_comm_trace --input profile_1465_node0_rank0.nsys-rep --output compute_comm_trace --subtimerange 10.452-10.587,10.601-10.882
```

