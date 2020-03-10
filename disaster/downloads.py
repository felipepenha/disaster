# MISC
import requests
import zipfile

# DOWLOAD: glove6B

url = 'http://nlp.stanford.edu/data/glove.6B.zip'
r = requests.get(url, allow_redirects=True)
path = 'glove6B.zip'
open(path, 'wb').write(r.content)

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall('./')
zip_ref.close()
