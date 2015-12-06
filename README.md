# Emojineering
## Proposed collaboration workflow with IPython notebook
Inspired from http://www.svds.com/jupyter-notebook-best-practices-for-data-science/

### Folders:
- ./data
- ./src     : Scripts/modules stored here

### Notebooks naming convention:
`\<date in format yyyy-mm-dd\>-\<author initials\>-\<title\>.ipynb`

e.g.: 2015-12-06-hr-data-cleaning.ipynb

Note:
- No spaces
- all lowercase with words seperated by -

### Enable autosave of notebooks to .py
In `~/.ipython/profile_default/ipython_notebook_config.py` replace:
    
    $ c.FileContentsManager.post_save_hook = None

by 

```
import os
from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['ipython', 'nbconvert', '--to', 'script', fname], cwd=d)

c.FileContentsManager.post_save_hook = post_save
```
Then restart your ipython notebook. 

If `~/.ipython/profile_default/ipython_notebook_config.py` doesn't exist run:

        $ ipython profile create

### Workflow
- Create notebook with above mentioned convention on the exploratory branch
- Make changes
- Save changes, .py file is automatically updated
- If you want to share some code with someone else, make it a function
- To import the function from someone else import the .py file in your notebook
