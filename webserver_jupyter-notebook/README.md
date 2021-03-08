# webserver_jupyter-notebook  
### This project is to acheieve 5 requirements.  
1. Use conda or virtualenv to build an environment of python 3.5, and install jupyter notebook.  
2. Design an easy webUI to let user set virtual environment name and python version. User can get the URL of jupyter notebook after clicking the button which will automatically build the virtual environment.  
3. Click the URL of jupyter notebook then use directly instead of entering the Token. The webUI need the function of deleting environment.  
4. Set a button for each virtual environment, then after clicking and it will show the python packages and version number on the new windows.  
5. Set a button for each virtual environment for exporting and need to transfer into zip file.  
  
### keypoint  
1. Need to set the kernel in the jupyter notebook. If not, the 'pip list' will show the packages in the host.  
2. I set time.sleep to wait the jupyter notebook information.  
3. IP I use can connect google.com, maybe there are more solutions to set in the future.  
