# Notes
- You can commit without running `pre-commit` by adding `--no-verify` to the commit command (e.g. `git commit --no-verify -m "some message"`).
- jupyter notebooks are ignored in the repo, so to recreate them after pulling, use `jupytext --sync` to sync the `.py` and `.ipynb` 
  files. This will create the `.ipynb` files from the `.py` files in the `notebooks` folder.
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
