import sys,re,html
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(s):
        super().__init__(convert_charrefs=True); s.o=[]; s.skip=0
    def handle_starttag(s,t,a):
        if t in('script','style','annotation','annotation-xml'): s.skip+=1
        if t in('p','div','h1','h2','h3','h4','li','tr','figcaption','section','table'): s.o.append('\n')
        if t in('td','th'): s.o.append(' | ')
    def handle_endtag(s,t):
        if t in('script','style','annotation','annotation-xml'): s.skip=max(0,s.skip-1)
    def handle_data(s,d):
        if not s.skip: s.o.append(d)
p=P(); p.feed(open(sys.argv[1],encoding='utf8',errors='replace').read())
t=''.join(p.o); t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t); open(sys.argv[2],'w').write(t); print(sys.argv[2],len(t))
