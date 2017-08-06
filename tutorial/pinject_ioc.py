#!/usr/bin/env python2
"""
https://github.com/google/pinject
"""
import pinject


class OuterClass(object):
    # @pinject.copy_args_to_internal_fields
    @pinject.copy_args_to_public_fields
    def __init__(self, inner_class):
        # print(inner_class)
        pass


class InnerClass(object):
    def __init__(self):
        self.forty_two = 42


ioc = pinject.new_object_graph()
outer_class = ioc.provide(OuterClass)
# outer_class = OuterClass()
print(outer_class.inner_class.forty_two)
