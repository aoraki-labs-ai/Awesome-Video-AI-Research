import yaml, io, urllib.request, urllib.parse, re, sys, time, difflib

d = yaml.safe_load(io.open('data/papers.yaml', encoding='utf-8'))
papers = d['papers']

ids = [p['id'] for p in papers]
assert len(ids) == len(set(ids)), "重复 id: " + str([i for i in ids if ids.count(i)>1])
known = set(ids)
for p in papers:
    for q in p.get('prereq') or []:
        if q not in known: print(f"!! {p['id']} 的 prereq '{q}' 不存在")

arx = [(p['id'], str(p['arxiv']), p['title']) for p in papers if p.get('arxiv')]
print(f"待核 {len(arx)} 条 arXiv 编号\n")

def norm(s):
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    return ' '.join(s.split())

results = {}
B = 25
for i in range(0, len(arx), B):
    chunk = arx[i:i+B]
    url = "http://export.arxiv.org/api/query?id_list=" + ",".join(c[1] for c in chunk) + f"&max_results={len(chunk)}"
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=60).read().decode()
            break
        except Exception as e:
            if attempt == 2: raise
            time.sleep(5)
    entries = re.findall(r'<entry>(.*?)</entry>', raw, re.S)
    for e in entries:
        aid = re.search(r'<id>http://arxiv.org/abs/([0-9.]+)v?\d*</id>', e)
        ti  = re.search(r'<title>(.*?)</title>', e, re.S)
        if aid and ti:
            results[aid.group(1)] = ' '.join(ti.group(1).split())
    time.sleep(3)

bad, fixed = [], []
for pid, aid, mytitle in arx:
    real = results.get(aid)
    if real is None:
        bad.append((pid, aid, mytitle, "<API 无返回>")); continue
    r = difflib.SequenceMatcher(None, norm(mytitle), norm(real)).ratio()
    if r < 0.72:
        bad.append((pid, aid, mytitle, real))
    elif norm(mytitle) != norm(real):
        fixed.append((pid, aid, mytitle, real))

print("═══ 编号疑似错误(标题对不上)═══")
for pid, aid, mine, real in bad:
    print(f"  [{pid}] arXiv:{aid}\n      我写的: {mine}\n      实际是: {real}\n")
print("═══ 编号正确、标题需要按官方订正 ═══")
for pid, aid, mine, real in fixed:
    print(f"  [{pid}] {mine}\n       -> {real}")
print(f"\n核对完成: 错误 {len(bad)}, 待订正 {len(fixed)}, 完全正确 {len(arx)-len(bad)-len(fixed)}")

# ── link liveness ────────────────────────────────────────────────────────────
def check_links():
    import concurrent.futures, urllib.request, re, io
    md = io.open('README.md', encoding='utf-8').read()
    urls = sorted(set(re.findall(r'\((https?://[^)\s]+)\)', md)))
    def probe(u):
        req = urllib.request.Request(u, method='HEAD',
              headers={'User-Agent':'Mozilla/5.0 (awesome-video-ai link check)'})
        try:
            return u, urllib.request.urlopen(req, timeout=25).status
        except Exception as e:
            code = getattr(e, 'code', None)
            if code in (403, 405):   # HEAD refused; try GET
                try:
                    req = urllib.request.Request(u, headers={'User-Agent':'Mozilla/5.0'})
                    return u, urllib.request.urlopen(req, timeout=25).status
                except Exception as e2:
                    return u, getattr(e2, 'code', type(e2).__name__)
            return u, code or type(e).__name__
    print(f"\n═══ 链接存活检查({len(urls)} 条)═══")
    bad = []
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        for u, st in ex.map(probe, urls):
            if st != 200:
                bad.append((u, st))
    for u, st in bad:
        print(f"  {st}  {u}")
    print(f"存活 {len(urls)-len(bad)}/{len(urls)}")
    return bad

if __name__ == '__main__' or True:
    import sys
    if '--links' in sys.argv:
        check_links()
