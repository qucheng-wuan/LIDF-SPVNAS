import json
from pathlib import Path
from urllib.request import Request, urlopen
out=Path(__file__).resolve().parents[2]/'runs/spvnas_fusion'
for repo in ['spvnas','torchsparse']:
 url=f'https://api.github.com/repos/mit-han-lab/{repo}/issues?state=all&per_page=100'
 try:
  data=json.load(urlopen(Request(url,headers={'User-Agent':'LIDF-research'}),timeout=30))
  (out/f'{repo}_issues.json').write_text(json.dumps(data,indent=2))
  for issue in data:
   text=issue['title']+' '+(issue.get('body') or '')
   if any(w in text.lower() for w in ['pretrain','download','hanlab','model_zoo']):
    comments=json.load(urlopen(Request(issue['comments_url'],headers={'User-Agent':'LIDF-research'}),timeout=30))
    (out/f'{repo}_issue_{issue["number"]}.json').write_text(json.dumps(comments,indent=2))
    print(issue['html_url'],issue['title'])
    for c in comments:
     if 'hanlab' in c['body'].lower(): print(c['html_url'],c['body'])
 except Exception as e: print(repo,repr(e))
