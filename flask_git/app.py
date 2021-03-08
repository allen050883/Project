#%%
from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
#from flaskext.mysql import MySQL 
from datetime import datetime

from config import Config
from flask_wtf import FlaskForm
from wtforms import BooleanField, SelectField, RadioField, StringField, TextField, IntegerField, FieldList, FormField, SubmitField

import os
import socket
import time
#%%

""" global variable"""
tasks = []
""" flask config """
app = Flask(__name__)
app.config.from_object(Config)
app.config["SECRET_KEY"] = '123456'
db = SQLAlchemy(app)

IP = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]


class taskInfo(db.Model):
    __tablename__ = "task_info"
    name = db.Column(db.String(40), primary_key=True, unique = True, nullable=False)
    ver = db.Column(db.String(20), nullable=False)
    pid = db.Column(db.String(40))
    url = db.Column(db.String(300))
    pkg_lst = db.Column(db.String(1000))
    exp_cmd = db.Column(db.String(100))
    del_cmd = db.Column(db.String(100))

    def __init__(self, name, ver, pid, url, pkg_lst, exp_cmd, del_cmd):
        self.name = name
        self.ver = ver
        self.pid = pid
        self.url = url
        self.pkg_lst = pkg_lst
        self.exp_cmd = exp_cmd
        self.del_cmd = del_cmd

db.create_all()

class OptionForm(FlaskForm):
    py_ver = RadioField('Python version', choices = [('python3.5','python 3.5'),('python3.6','python 3.6'),('python3.7','python 3.7')], default='python3.5')
    env_name = TextField('Virtual environment name')

class StatsForm(FlaskForm):
    exp_st = SubmitField()
    del_st = SubmitField()

def otherinfo(name, ver):
    os.system("virtualenv "+name+" --python="+ver)
    #os.system(". ./"+name+"/bin/activate")
    os.system("./"+name+"/bin/pip install ipykernel")
    os.system("python -m ipykernel install --user --name="+name)
    os.system("nohup jupyter-notebook --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password='' --no-browser > notebook_info.log &")
    time.sleep(1)
    
    #url       
    f = open("notebook_info.log", "r")
    content = f.read()
    port = ((content.split("http:")[1]).split(":")[1]).split("/")[0]
    url = IP+":"+port
    f.close()
    os.system("rm notebook_info.log")
    
    #PID
    os.system("lsof -t -i4TCP:"+port+" > notebook_pid.log &")
    time.sleep(1)
    f = open("notebook_pid.log", "r")
    pid = (f.read()).split("\n")[0]
    print("pid:", pid)
    f.close()
    os.system("rm notebook_pid.log")

    #pkg_lst
    os.system("./"+name+"/bin/pip freeze > requirements_"+name+".txt")
    pkg_lst = []
    with open("requirements_"+name+".txt", "r") as file:
        pkg_lst = file.read()
    os.system("rm requirements_"+name+".txt")
        
    #exp_cmd
    exp_cmd = "zip -r "+name+".zip "+name
    
    #del_cmd
    del_cmd = "rm -r "+name
    
    return pid, url, pkg_lst, exp_cmd, del_cmd

@app.route('/', methods=['POST', 'GET'])  #root directory --> IP:port/
def index():
    ## make task setting object
    opt_form = OptionForm()
    if opt_form.validate_on_submit():
        print("SUBMIT")
        ## get task setting
        name = opt_form.env_name.data
        ver = opt_form.py_ver.data
        
        #check name is unique
        sql_cmd = """
            select name from task_info
            """
        name_list = db.engine.execute(sql_cmd).fetchall()
        name_list = [n[0] for n in name_list]
        print("name_list: ", name_list)
        
        if name in name_list:
            print("Duplicate name, please change another.")
        else:
            pid, url, pkg_lst, exp_cmd, del_cmd = otherinfo(name, ver)

            ## push data to DB
            if len(name)>0:
                add_env = taskInfo(name, ver, pid, url, pkg_lst, exp_cmd, del_cmd)
                db.session.add(add_env)
                db.session.commit()           
                query_data = db.engine.execute(sql_cmd) 
                return redirect('/')
            else:
                print("Environment name can not be empty")
                
    ## print all task in database
    sql_cmd = """
        select *
        from task_info
        """
    query_data = db.engine.execute(sql_cmd).fetchall()
    print("DB data: ", query_data)
                
    return render_template('index.html', opt_form=opt_form, tasks = query_data)

@app.route('/package_list/<path:target_env>', methods=['POST', 'GET'])  #root directory --> IP:port/
def package_list(target_env):
    #pkg_lst
    name = target_env
    os.system("./"+name+"/bin/pip freeze > requirements_"+name+".txt")
    pkg_lst = []
    with open("requirements_"+name+".txt", "r") as file:
        pkg_lst = file.read()
    os.system("rm requirements_"+name+".txt")

    ## get package list
    sql_cmd = """
        select pkg_lst from task_info 
        where name = "%s"
        """ % (target_env)
    pkg_lst = db.engine.execute(sql_cmd).fetchone()
    print("DB package list: ", pkg_lst)
    pkg_ver = [pkg.split("==") for pkg in pkg_lst[0].split("\n")]

    return render_template('package_list.html', pkg_lst=pkg_ver)

@app.route('/del_env/<path:del_target_env>', methods=['POST', 'GET'])  #root directory --> IP:port/
def del_env(del_target_env):
    ## close jupyter notebook
    sql_cmd = """
        select pid from task_info 
        where name = "%s"
        """ % (del_target_env)
    pid_cmd = db.engine.execute(sql_cmd).fetchone()
    try: os.system("kill "+pid_cmd[0])
    except: print("kill pid problem")
    
    #remove kernel
    #os.system("jupyter kernelspec uninstall "+del_target_env)
    os.system("rm -r /home/$USER/.local/share/jupyter/kernels/"+del_target_env)

    ## get delete comamnd and delete enviroment
    sql_cmd = """
        select del_cmd from task_info 
        where name = "%s"
        """ % (del_target_env)
    del_cmd = db.engine.execute(sql_cmd).fetchone()
    print("DB delete command: ", del_cmd[0])
    try: os.system(del_cmd[0])
    except: print("delete problem")

    ## delete task in database
    sql_cmd = """
        delete from task_info 
        where name = "%s"
        """ % (del_target_env)
    db.engine.execute(sql_cmd)
    print("delete", del_target_env)
    return redirect('/')
    
@app.route('/exp_env/<path:exp_target_env>', methods=['POST', 'GET'])  #root directory --> IP:port/
def exp_env(exp_target_env):

    ## get export comamnd and delete enviroment
    sql_cmd = """
        select exp_cmd from task_info 
        where name = "%s"
        """ % (exp_target_env)
    
    exp_cmd = db.engine.execute(sql_cmd).fetchone()
    print("DB export command: ", exp_cmd[0])
    try: os.system(exp_cmd[0])
    except: print("export problem")

    return redirect('/')


if __name__ == "__main__":
    app.run(host=IP, port=8888, debug=False)
