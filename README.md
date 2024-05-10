# Notes
- All notebooks and scripts in this repo are just some samples from other projects I'm working on. Some of them will not work. DO NOT RUN THME without understanding them!
- You can commit without running `pre-commit` by adding `--no-verify` to the commit command (e.g. `git commit --no-verify -m "some message"`).
- You can also just run `pre-commit` without `git commit` to check what will happen by using `pre-commit run --all-files`
- jupyter notebooks are ignored in the repo in the "notebooks" folder, so to recreate them after pulling, use `jupytext --to notebook --output-path notebooks/ scripts/*.py`. This will create the `.ipynb` files from the `.py` files in the `notebooks` folder.
- Since jupytext is not yet being run on the notebooks in the `notebooks` folder, they will be added to the repository. Run `git rm -r --cached .` to remove all ignored files before continuing on testing them.


# Todos
[x] fix some .ipynb breaking <br/>
[x] ignore notebooks in "notebooks" folder in `.gitignore` <br/> 
[x] ONLY clear the notebooks' output not in the "notebooks" folder <br/>
[x] make converted notebooks (`.py`) as light instead of percent <br/>
[x] make the `.py` appear one level up in "_notebooks" folder <br/>
[x] `nbstrip` run on the outer notebooks only <br/>
[x] `nbstrip` fail when run on files so that you add the notebooks without output again to git. <br/>
[x] Change the "_notebooks" to something better <br/>
[ ] run jupytext on the notebooks ignored in `.gitignore` ONLY <br/>
[ ] Add ‘--sync’ to the entry point and see if `.ipynb` is in 2-way sync with `.py` <br/>
[ ] How to move notebooks and linked scripts? <br/>


## Try this (AI generated) :
The updated .pre-commit-config.yaml file that looks into all subfolders for .ipynb files and syncs them with corresponding .py files in the scripts directory:
```ymal
  - repo: local
    hooks:
      - id: sync-py-files
        name: Sync .py files with .ipynb files
        entry: bash -c '
          new_files=0
          for notebook in $(find . -name "*.ipynb" -not -path "*/notebooks/*"); do
            script="scripts/$(echo "$notebook" | sed 's/\.ipynb$/.py/' | sed 's/^\.\///')"
            mkdir -p "$(dirname "$script")"
            if [ ! -e "$script" ] || [ "$notebook" -nt "$script" ]; then
              jupytext --to py:light "$notebook" -o "$script"
              echo "Synced $script with $notebook"
              new_files=1
            fi
          done
          if [ "$new_files" -eq 1 ]; then
            echo "New or updated .py files were synced. Please add them and commit again."
            exit 1
          fi'
        language: system
        pass_filenames: false
        always_run: true
```
The changes made to the script are:

- The find command now starts from the current directory (.) instead of the src directory. This ensures that it searches for .ipynb files in all subfolders.
- The script variable assignment has been updated to handle the new file paths: `script="scripts/$(echo "$notebook" | sed 's/\.ipynb$/.py/' | sed 's/^\.\///')"`
  - The first sed command replaces the .ipynb extension with .py.
  - The second sed command removes the leading ./ from the file path, since the find command now starts from the current directory.
- The rest of the script remains the same, syncing the .py files with their corresponding .ipynb files and breaking the commit if any new or updated .py files were synced.

With these modifications, the pre-commit hook will:

1. Find all .ipynb files in the current directory and its subfolders, excluding those located in notebooks directories.
2. For each .ipynb file, check if the corresponding .py file doesn't exist or if the .ipynb file is newer than the .py file.
3. If the condition is met, sync the .py file with the .ipynb file using jupytext.
4. Set new_files to 1 if any .py files were created or updated.
5. Break the git commit if new_files is equal to 1, indicating that new or updated .py files were synced.

This updated script will look into all subfolders for .ipynb files and sync them with corresponding .py files in the scripts directory while maintaining the same directory structure.
