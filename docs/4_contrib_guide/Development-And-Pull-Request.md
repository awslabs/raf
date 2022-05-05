<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Developmenet and Pull Request

This tutorial introduces the developmenet flow and the way of filing a pull request to let your contribution be a part of RAF.

## Step 0. 🍽 Fork me @ GitHub
The very initial step is to click the the "Fork" button on the upper right corner of [RAF](http://github.com/awslabs/raf) like below:

![Fork](https://user-images.githubusercontent.com/22515877/68988345-a29a8a00-07ea-11ea-9061-fcf001a1fff3.png)

This will create a mirror repository at your own account, on which you have all permissions, including adding/removing branches, commits or anything else.

## Step 1. 🚀 Work on your own fork
**Download from GitHub.** You may download the project using the following command:
```bash
# HTTPS
git clone git@github.com:YourAccount/raf.git --recursive
# SSH
git clone https://github.com/YourAccount/raf.git --recursive
```

**Create a new branch.** To avoid possible mess-ups, it is suggested to work on a new branch, instead of `main`. You may create a new branch using the following commands.
```bash
# Create a branch on your local machine
git checkout -b AwesomeName
# Push the branch to GitHub
git push --set-upstream origin AwesomeName
# ⬆️ `origin` is a default name for the remote GitHub server.
# You may run `git remote -v` to list all remote servers Git knows.
```

**Commit your code.** In case a disaster happened and destroyed everything, you might be interested in incrementally doing the coding, and letting Git and GitHub remembers your change. The following commands will help:

```bash
# Add files to git stash
git add path/or/directory/to/your/files
# Do a 'commit' so that Git will remember your code
git commit -m "Your warm messages or curses"
# Push the commits to GitHub, so you can view them in browsers
git push
```

## Step 2. 🎉 Upstreaming your change
Here we come to the most exiting time: you have completed the coding on your fork. So please let us know - we are happy to review and potentially merge your awesome change into our latest codebase! 😊

On Git/GitHub, the process is called "pull request", which means "requesting maintainers to pull your branch into the mainline". So, please jump to the [Pull Request](https://github.com/awslabs/raf/pulls) page, and click the little green button.

<img src="https://user-images.githubusercontent.com/22515877/68989884-83f3bd80-0801-11ea-9580-4f0a87fbd6b6.png" width=180/>

Then, our continuous integration (CI) system will automatically check your code, including some notorious coding styles as well as unit tests. Once the CI system passes, you may request active community members to review by directly @ them on GitHub. Note that we do not merge code that cannot pass CI.

Last but not least, as a community, the most important thing is to be happy together. Even if you feel annoyed by something, please be polite and respect each other :-)

## Step 3. 🕙 Keep up to date
**Conflicts.** Sometimes, your fork might conflict with our repositories, because both of them change over time. In this case, when you submitting a pull request, it is possible that Git/GitHub says they are unable to automatically test the code, because of these conflicts.

**Rebase is required.** In this case, sad it is, you may have to manually "rebase" to our most recent commit. During rebase, git will replay all your commits on top of our latest commit, forcing you to make every decision on the conflict part of the code.

```bash
# Adding a remote git address called `upstream`.
# You may actually call it anything.
# `git remote -v` will list all known remotes
git remote add upstream https://github.com/awslabs/raf
# Fetch the latest information from all remotes
git fetch --all
# And...Do the rebase.
git rebase upstream/main
```

**Rebasing process.** Then Git will replay your commits on top of `upstream/main`, and ask you to manually resolve conflicts on each of your commit. This is tedious but you have to edit each of the files it reports as conflicted.

```bash
# Git will tell where conflict happens
git status
# After editing the conflicted file, tell Git about this
git add path-to-conflict-file
# When there is no conflicts, let's move to the next commit
git rebase --continue
```

**Squash commits into one** If you are really annoyed by that Git repeatedly reports conflicts, squashing commits might be a solution that might save your time. There are two ways to do it:

```bash
# Solution 1.
# This will prompts you to choose which commit to be squashed
git rebase -i HEAD~3
# And force push to let GitHub know that you change the commit history
git push -f 
```

```
# Solution 2.
# This will revert back to 3 commits before, and all changes will be uncommitted
git reset HEAD~3
# Commit the changes again
git add .
git commit -m "something"
# And force push to let GitHub know that you change the commit history
git push -f 
```

## Other resources to learn Git/GitHub

- TVM has a nice tutorial on certain topics: https://docs.tvm.ai/contribute/git_howto.html
- GitHub itself collects resources for beginners: https://try.github.io
- Google and Stack Overflow are often super helpful!

