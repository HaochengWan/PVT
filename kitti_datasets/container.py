__all__ = ['G']

class G(dict):
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError(k)
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]
