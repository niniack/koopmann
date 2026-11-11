import sys,re
L=open(sys.argv[1]).read().split('\n'); a,b=int(sys.argv[2]),int(sys.argv[3])
row=[];
for l in L[a-1:b]:
    s=l.strip()
    if s.startswith('|'):
        v=s.lstrip('| ').strip()
        if v: row.append(v)
    else:
        if row: print(' ; '.join(row)[:230]); row=[]
        if s: print(s[:330])
if row: print(' ; '.join(row)[:230])
