import os


class Bunch(object):
  def __init__(self, **kwds):
    self.__dict__.update(kwds)

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

  def __str__(self):
    string = "BUNCH {" + str(self.__dict__)[2:]
    string = string.replace("': ", ":")
    string = string.replace(", '", ", ")
    return string

  def __repr__(self):
    return str(self)

  def to_file_name(self, folder=None, ext=None):
    res = str(self.__dict__)[2:-1]
    res = res.replace("'", "")
    res = res.replace(": ", ".")
    parts = res.split(', ')
    res = '_'.join(sorted(parts))
    
    if ext is not None:
      res = '%s.%s' % (res, ext)
    if folder is not None:
      res = os.path.join(folder, res)
    return res


if __name__ == '__main__':
    b = Bunch(x=5, y='something', other=9.0)
    print(b)
    print(b.to_file_name())
    print(b.to_file_name('./here', 'txt'))


