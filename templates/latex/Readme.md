# Usage

## Copy templates to user folder

Copy this directory `latex` and it's contents to your user template folder, e.g. `~/.local/share/jupyter/nbconvert/templates/`.

To find out the correct template path in `data`, call the following command:

```
$ jupyter --paths

config:
    /home/bk/.jupyter
    /home/bk/jupyter-env/etc/jupyter
    /usr/local/etc/jupyter
    /etc/jupyter
data:
    /home/bk/.local/share/jupyter
    /home/bk/jupyter-env/share/jupyter
    /usr/local/share/jupyter
    /usr/share/jupyter
runtime:
    /home/bk/.local/share/jupyter/runtime
```

## Call `nbconvert` command

Call `nbconvert` command to produce LaTeX or PDF output either:

```
$ jupyter nbconvert jupyter_notebook.ipynb --to latex
```

Or:

```
$ jupyter nbconvert jupyter_notebook.ipynb --to pdf
```
