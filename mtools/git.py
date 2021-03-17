import git
from datetime import datetime

def gitcommit(path, msg=None):
    if message==None:
        now = datetime.now()
        message = now.strftime("%m/%d/%y %H:%M:%S")
    repo = git.Repo.init(path=path)
    if repo.is_dirty():
        repo.git.add('--all')
        repo.index.commit('autocommit:%s'%msg)