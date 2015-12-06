# Emojineering
## Proposed collaboration workflow with IPython notebook
Inspired from http://www.svds.com/jupyter-notebook-best-practices-for-data-science/

### Folders:
- ./data
- ./deliver : final notebook for consumption
- ./develop : exploratory notebooks stored here
- ./src     : Scripts/modules stored here

### Notebooks naming convention:
`\<date in format yyyy-mm-dd\>-\<author initials\>-\<title\>.ipynb`

e.g.: 2015-12-06-hr-data-cleaning.ipynb

### Enable autosave of notebooks to .py
In `~/.ipython/profile_default/ipython_notebook_config.py` replace:
    
    $ c.FileContentsManager.post_save_hook = None
by 
