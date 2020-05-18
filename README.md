# QueryProcessing-GPU
Algorithms for top-k query processing on GPUs.

# Instructions

Below were created some folders for group members to post their codes, slides, spreadsheets, etc.
To post material do the following:

###### How to update the repository?

###### [Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the [repository](https://github.com/HPCSys-Lab/QueryProcessing-GPU.git).

###### Create a local clone of your fork:

`$ git clone https://github.com/HPCSys-Lab/QueryProcessing-GPU.git`

###### Inside the repository you just cloned, configure Git to sync your fork with the original StencilCodes4GPUs repository:

`$ git remote add upstream https://github.com/HPCSys-Lab/QueryProcessing-GPU.git`

###### Verify if everything is ok: 

```    
$ git remote -v
> origin    https://github.com/YOUR_USERNAME/QueryProcessing-GPU.git (fetch)
> origin    https://github.com/YOUR_USERNAME/QueryProcessing-GPU.git (push)
> upstream  https://github.com/HPCSys-Lab/QueryProcessing-GPU.git (fetch)
> upstream  https://github.com/HPCSys-Lab/QueryProcessing-GPU.git (push)
```

###### Now, you can keep your fork [synced](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) with the upstream repository with a few Git commands:

`$ git fetch upstream`
`$ git checkout master`
`$ git merge upstream/master`

###### Once you update your local clone/fork, you must create a new branch for your update:

`$ git checkout -b name-of-your-branch`

###### Update the branch with all information you want, add the files and commit:

`$ git add .`
`$ git commit -m "some comment about the commit"`

###### Now you must push the commit:

`$ git push -u origin name-of-your-branch` 

###### Now you can go the github and open a pull request (PR) to merge the branch to master in https://github.com/HPCSys-Lab/QueryProcessing-GPU.git.
