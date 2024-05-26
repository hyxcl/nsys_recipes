# nsys_recipes

## Usage
clone this repository and replace the original one
```
git clone ssh://git@gitlab-master.nvidia.com:12051/congliangx/nsys_recipes.git
rm /opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/python/packages/nsys_recipe/recipes
rsync -aPp nsys_recipes/nccl_gpu_overlap_trace  /opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/python/packages/nsys_recipe/recipes
```

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab-master.nvidia.com/congliangx/nsys_recipes.git
git branch -M main
git push -uf origin main
```
