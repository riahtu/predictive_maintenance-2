import os, sys
def append_id(filename, uid):
    name, ext = os.path.splitext(filename)
    return "{name}_{uid}{ext}".format(name=name, uid=uid, ext=ext)
print(
append_id("hard.csv", "new"))