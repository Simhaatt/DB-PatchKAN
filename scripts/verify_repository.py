"""Standard-library integrity/syntax checks; no research inference or training."""
import ast
import hashlib
import json
from pathlib import Path
import re


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'PUBLIC_FILES_SHA256.json').read_text(encoding='utf-8'))
    problems=[];py_count=0;nb_count=0
    secret=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{40,})')
    for name,record in manifest['files'].items():
        path=root/name
        if not path.is_file():problems.append('Missing: '+name);continue
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=record['sha256']:problems.append('Hash mismatch: '+name)
        if len(content)!=record['bytes']:problems.append('Size mismatch: '+name)
        if path.suffix in ['.py','.md','.yaml','.yml','.toml','.cff','.json','.ipynb']:
            text=content.decode('utf-8-sig')
            if secret.search(text):problems.append('Possible credential: '+name)
            if re.search(r'[A-Za-z]:[/\\]+Users[/\\]+',text,re.I):problems.append('Private user path: '+name)
        if path.suffix=='.py':ast.parse(content.decode('utf-8-sig'));py_count+=1
        if path.suffix=='.ipynb':
            nb=json.loads(content);nb_count+=1
            for cell in nb['cells']:
                if cell['cell_type']=='code':
                    source=''.join(cell['source'])
                    # Historical IPython shell/magic lines are retained as notebook source.
                    python_only='\n'.join(line for line in source.splitlines()
                                          if not line.lstrip().startswith(('!','%')))
                    ast.parse(python_only)
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        problems.append('Notebook execution data retained: '+name)
    if problems:raise SystemExit('\n'.join(problems))
    print(f'PASS: {len(manifest["files"])} public files; {py_count} Python sources; {nb_count} clean notebooks.',flush=True)
    print('No model training or scientific analysis was performed by this checker.',flush=True)


if __name__=='__main__':main()
