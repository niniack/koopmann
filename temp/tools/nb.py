import sys,json
nb=json.load(sys.stdin)
for i,c in enumerate(nb.get('cells',[])):
    src=''.join(c.get('source',[]))
    if c['cell_type']=='markdown':
        print(f'[md {i}]',src[:400])
    else:
        print(f'[code {i}]',src[:260].replace('\n',' ; '))
        for o in c.get('outputs',[]):
            t=''.join(o.get('text',[])) if 'text' in o else ''.join(o.get('data',{}).get('text/plain',[]))
            if t.strip(): print('   ->',t[:260].replace('\n',' / '))
            if o.get('output_type')=='error': print('   !! error',o.get('ename'),str(o.get('evalue'))[:120])
