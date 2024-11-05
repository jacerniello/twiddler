class ValueTracker:
    __array_ufunc__ = None  # Numpy 13.0 and above

    def  __init__(self, obj, attribute_name, attr_val=None):
        self.obj = obj
        self.attr = attribute_name
        self.attr_val = attr_val  # if the variable is static and should be returned as a static value

    def get_value(self):
        if self.attr_val:
            return self.attr_val
        else:
            return getattr(self.obj, self.attr)

    def __getitem__(self, item):
        return self.get_value()[item]

    def __iadd__(self, n):
        return self.get_value() + n

    def __add__(self, n):
        return self.get_value() + n

    def __radd__(self, n):
        return n + self.get_value()

    def __sub__(self, n):
        return self.get_value() - n

    def __rsub__(self, n):
        return n - self.get_value()

    def __mul__(self, n):
        return self.get_value() * n

    def __rmul__(self, n):
        return n * self.get_value()

    def __div__(self, n):
        return self.get_value() / n

    def __rdiv__(self, n):
        return n / self.get_value()
