"""Publish TransitBox code and a data-free GitHub Pages frontend.

The source tree's .git is never touched. Publication uses an isolated checkout.
Credentials come from an authenticated gh CLI, environment or a hidden prompt;
they are never written to a file or passed in command-line arguments.
"""
from pathlib import Path
import argparse
import base64
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.check_public_source import check
from tools.build_public import sanitize_html


def stage(repository):
    checkout=ROOT/'.publish'/'code-checkout';checkout.mkdir(parents=True,exist_ok=True)
    directories=['transitbox','tools','tests','vendor/transreid','vendor/ghr_vlm','.github']
    files=['README.md','requirements-reid.txt','launch.sh','serve.py','index.html','.gitignore','.gitattributes']
    if (ROOT/'package.json').is_file():files.append('package.json')
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','progress.txt','encoding.json','.build-cache.json')
    for directory in directories:
        destination=checkout/directory
        if destination.exists():shutil.rmtree(destination)
        shutil.copytree(ROOT/directory,destination,ignore=ignore)
    for filename in files:shutil.copy2(ROOT/filename,checkout/filename)
    # Always remove local datasets from the HTML copied into the public repository.
    (checkout/'index.html').write_text(sanitize_html((ROOT/'index.html').read_text()))
    def git(*args,**kwargs):return subprocess.run(['git','-C',str(checkout),*args],check=True,text=True,**kwargs)
    if not (checkout/'.git').is_dir():git('init','-b','main',stdout=subprocess.DEVNULL)
    owner=repository.split('/')[0]
    git('config','user.name',owner)
    git('config','user.email',f'{owner}@users.noreply.github.com')
    remote=subprocess.run(['git','-C',str(checkout),'remote','get-url','origin'],capture_output=True,text=True)
    url=f'https://github.com/{repository}.git'
    if remote.returncode:git('remote','add','origin',url)
    elif remote.stdout.strip()!=url:raise ValueError('Publication checkout points to another repository')
    git('add','--all')
    check(checkout)
    changed=subprocess.run(['git','-C',str(checkout),'diff','--cached','--quiet']).returncode
    if changed:git('commit','-m','Publish TransitBox code without private video data',stdout=subprocess.DEVNULL)
    commit=git('rev-parse','HEAD',capture_output=True).stdout.strip()
    tracked=git('ls-files',capture_output=True).stdout.splitlines()
    check(checkout,history=True)
    print(f'Prepared {repository} at commit {commit}; {len(tracked)} files',flush=True)
    return checkout,commit


def authentication():
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if not token and shutil.which('gh'):
        auth=subprocess.run(['gh','auth','token'],capture_output=True,text=True)
        if auth.returncode==0:token=auth.stdout.strip()
    if not token:
        if not sys.stdin.isatty():raise RuntimeError('GitHub is not authenticated. Connect GitHub or run this script in an authenticated terminal.')
        token=getpass.getpass('GitHub access token (input hidden): ').strip()
    if not token:raise RuntimeError('GitHub authentication is required')
    return token


def publish(repository,checkout,commit,token):
    def api(path,method='GET',body=None):
        request=urllib.request.Request('https://api.github.com'+path,
            data=json.dumps(body).encode() if body is not None else None,method=method,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                     'Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'TransitBox'})
        with urllib.request.urlopen(request,timeout=30) as response:
            content=response.read();return json.loads(content) if content else {}
    owner,name=repository.split('/')
    account=api('/user')
    if account['login'].lower()!=owner.lower():raise RuntimeError(f'Authenticate as {owner} to create this repository')
    try:repo=api('/repos/'+repository)
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        repo=api('/user/repos','POST',{'name':name,'description':'TransitBox: shared passenger tracking, TransReID and payment visualization',
                                     'private':False,'auto_init':False})
    env=os.environ.copy();env.update({'TRANSITBOX_GIT_TOKEN':token,'GIT_TERMINAL_PROMPT':'0',
                                     'GIT_ASKPASS':str(ROOT/'tools/github_askpass.py')})
    subprocess.run(['git','-C',str(checkout),'push','-u','origin','main'],env=env,check=True)
    try:pages=api('/repos/'+repository+'/pages')
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        pages=api('/repos/'+repository+'/pages','POST',{'build_type':'workflow'})
    if pages.get('build_type')!='workflow':api('/repos/'+repository+'/pages','PUT',{'build_type':'workflow'})
    # A newly pushed workflow may take a moment to appear in the API. The first
    # push can race Pages enablement, so track a fresh manual dispatch explicitly.
    workflow_path='/repos/'+repository+'/actions/workflows/pages.yml'
    prior_ids=set()
    for attempt in range(12):
        try:
            prior_ids={run['id'] for run in api(workflow_path+'/runs?per_page=10')['workflow_runs']}
            api(workflow_path+'/dispatches','POST',{'ref':'main'})
            break
        except urllib.error.HTTPError as error:
            if error.code!=404 or attempt==11:raise
            time.sleep(5)
    print('Pushed source and requested GitHub Pages deployment',flush=True)
    deadline=time.monotonic()+900
    run=None
    while time.monotonic()<deadline:
        runs=api('/repos/'+repository+'/actions/workflows/pages.yml/runs?per_page=10')['workflow_runs']
        matching=[item for item in runs if item['head_sha']==commit and item['event']=='workflow_dispatch' and item['id'] not in prior_ids]
        if matching:
            run=matching[0]
            if run['status']=='completed':
                if run['conclusion']!='success':raise RuntimeError(f'GitHub Pages workflow ended with {run["conclusion"]}: {run["html_url"]}')
                pages=api('/repos/'+repository+'/pages')
                url=pages.get('html_url')
                if not url:raise RuntimeError('Deployment succeeded but no Pages URL was returned')
                print(json.dumps({'repository':repo['html_url'],'website':url,'workflow':run['html_url'],
                                  'commit':commit,'status':'published'},indent=2),flush=True)
                return
            print(f'GitHub Pages: {run["status"]}',flush=True)
        time.sleep(15)
    raise RuntimeError('Deployment is still running; check '+(run['html_url'] if run else repo['html_url']+'/actions'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',default='ccoopq/TransitBox')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repository):parser.error('Use owner/repository')
    try:
        checkout,commit=stage(args.repository)
        if not args.prepare_only:publish(args.repository,checkout,commit,authentication())
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        print(f'GitHub publication failed: {error}',file=sys.stderr);sys.exit(1)
