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

## Including figures

The handling of figures was inspired by and adapted from http://blog.juliusschulz.de/blog/ultimate-ipython-notebook#figures.

## Citing and literature references

Install packages `texlive-bibtex-extra` and `biber` for using biblatex first.

% Insert citations in markdown as e.g.
%    <cite data-cite="DevoretS2013">[DevoretS2013]</cite>
% requires file notebook.bib in current directory (or the file set as "bib" in the latex_metadata)

%IMPORTANT: 
% - for biblatex "biber" should be set as preprocessor (bibtex does not work).
% - Encoding should be set the same in JabRef and Docear (e.g. UTF8) => otherwise there will be problems!
