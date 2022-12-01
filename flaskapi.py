import urllib
# import urllib2
import json
from pprint import pprint
import sys
from urllib import request

print(sys.argv)
ip      = sys.argv[1]
fromm   = int(sys.argv[2])
size    = int(sys.argv[3])
F = sys.argv[4].upper().strip()
L = sys.argv[5].upper().strip() if(len(sys.argv)>5) else None
P = sys.argv[6].upper().strip() if(len(sys.argv)>6) else None

# url = "http://localhost:9250/astynomia/anazitisi"
# url = "http://10.1.69.15:9250/astynomia/anazitisi"
url = "http://{}:9250/astynomia/anazitisi".format(ip)
data = {
        "first_name"    : F,
        "last_name"     : L,
        "father_name"   : P,
        "size"          : size,
        "from"          : fromm
}

req = request.urlopen(url, data=bytes(json.dumps(data)))
req.add_header('Content-type', 'application/json')
response = request.urlopen(req)
ret_data = json.loads(response.read())
pprint(ret_data)

for r in ret_data['results']:
    print(u"{} | {} | {} | {} | {}| {}".format(r['first_name'], r['father_name'], r['last_name'], r['score'], ret_data['max_score'], r['id']))

print('search :')
pprint(data)


