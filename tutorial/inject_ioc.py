"""
Import the inject module.
https://github.com/ivankorobkov/python-inject
"""
import inject
from typing import (
    Any,
)

class Cache:
    """ doc """
    def __init__(self):
        raise Exception('can not be instanciated')


class RamCache:
    """doc"""
    def __init__(self):
        print('ram cache inited')

    def save(self, k, v):
        """doc"""
        print('RamCache::save(%s, %s)' % (k, v))


class DiskCache:
    """doc"""
    def __init__(self):
        print('disk cache inited')

    def save(self, k, v):
        """doc"""
        print('DiskCache::save(%s, %s)' % (k, v))


class User(object):
    """ doc """
    # `inject.attr` creates properties (descriptors)
    # which request dependencies on access.
    cache = inject.attr(Cache)

    @inject.params(cache=Cache)
    def __init__(self, cache=None):
        # print(cache)
        cache.save('test init inject', True)
        self.cache.save('test class attr inject', True)

    def save(self):
        """ doc """
        self.cache.save('users', self)

    @classmethod
    def load(cls, id):
        """ doc """
        return cls.cache.load('users', id)


# Create an optional configuration.
def ioc(binder) -> None:
    """ doc """
    # binder.install(my_config2)  # Add bindings from another config.
    binder.bind(Cache, DiskCache())


# Configure a shared injector.
inject.configure(ioc)

user = User()
