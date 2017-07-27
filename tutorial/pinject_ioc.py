#!/usr/bin/env python2
"""
https://github.com/google/pinject
"""
import pinject

class OuterClass(object):
    # @pinject.copy_args_to_internal_fields
    @pinject.copy_args_to_public_fields
    def __init__(self, inner_class):
        pass

class InnerClass(object):
    def __init__(self):
        self.forty_two = 42

obj_graph = pinject.new_object_graph()
outer_class = obj_graph.provide(OuterClass)
print(outer_class.inner_class.forty_two)

