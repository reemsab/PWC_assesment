from bs4 import BeautifulSoup 
import requests 
import json
# This scraper simply visits the desired webpages and exctracts all hyper-references that belong to the targted domain then iteraitivly scrape them

def is_subpage(main_url, href):
    if not href or  len(href) <1:
        return False
    if main_url in href[:len(main_url)]:
        return True
    if href[0] == '/' and main_url.split('/')[-1] in href and 'https' not in href:
        return True
    return False

def main():
    corpus = []
    visited = set()
    not_visited= set()
    URL = "https://u.ae/en/information-and-services" 
    not_visited.add(URL)
    while len(not_visited)!=0:
        url = not_visited.pop()
        visited.add(url)
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
        corpus.append((url, '\n'.join([p.text for p in soup.find_all('p')])))
        for a in soup.find_all('a', href=True):
            u = a['href'] if 'http' in a['href'] else 'https://u.ae'+ a['href']
            if is_subpage(URL, u) and u not in visited:
                not_visited.add(u)

    open('corpus.txt','w').write('\n'.join([x[1] for x in corpus]))
    open('corpus.jsonl', 'w').write('\n'.join([json.dumps({'url': x[0], 'text':x[1]}) for x in corpus]))

if __name__ == '__main__':
    main()