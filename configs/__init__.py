import os

flist = []
files = os.listdir("configs")
for f in files:
    if os.path.splitext(f)[1] == '.py':
        file = f.split('.')[0]
        flist.append("configs." + file)
for f in flist:
    __import__(f)