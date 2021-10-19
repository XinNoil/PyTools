from git import Repo
from datetime import datetime
from .io import load_json

def gitcommit(path, msg=None):
    if type(path)==str:
        repo = Repo(path=path)
    else:
        repo = path
    if repo.is_dirty():
        if msg==None:
            now = datetime.now()
            msg = now.strftime(" at %m/%d/%y %H:%M:%S")
        repo.git.add('--all')
        repo.index.commit('autocommit :%s'%msg)
        print('git auto commit: %s'%msg)

def get_repos(git_config):
    repo_names = [_lib['name'] for _lib in git_config['Dependent libraries']]
    repos = [Repo(_lib['path']) for _lib in git_config['Dependent libraries']]
    for name, repo in zip(repo_names, repos):
        repo.name = name
    return repos

def get_repos_info(repos):
    if type(repos)!=list:
        repos = [repos]
    return [{'name':repo.name ,'branch':repo.heads[0].name, 'commit':str(repo.heads[0].commit), 'dirty':repo.is_dirty()} for repo in repos]

def get_git_info(config_file):
    git_config = load_json(config_file)
    repos = get_repos(git_config)
    return get_repos_info(repos)

# print(get_git_info(join_path('configs','git.json')))
def gitcommit_repos(config_file):
    git_config = load_json(config_file)
    repos = get_repos(git_config)
    # map(gitcommit, repos)
    for repo in repos:
        gitcommit(repo)