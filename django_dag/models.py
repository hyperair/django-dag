"""
A class to model hierarchies of objects following
Directed Acyclic Graph structure.

Some ideas stolen from: from https://github.com/stdbrouw/django-treebeard-dag

"""


from django.db import models
from django.core.exceptions import ValidationError


class NodeNotReachableException (Exception):
    """
    Exception for node distance and path
    """
    pass


class NodeBase(object):
    """
    Main node abstract model
    """

    class Meta:
        ordering = ('-id',)

    traverse_sql_tpl_lvl2 = """
        WITH RECURSIVE traverse(id, path, cycle) AS (
            SELECT edge.{{to_column}},
                   ARRAY[edge.{{from_column}},
                         edge.{{to_column}}],
                   false
            FROM {edge_table} edge
            WHERE edge.{{from_column}} = %s
                UNION ALL
            SELECT edge2.{{to_column}},
                   edge1.path || edge2.{{to_column}},
                   edge2.{{to_column}} = ANY(edge1.path)
            FROM traverse edge1, {edge_table} edge2
            WHERE edge1.id = edge2.{{from_column}} AND NOT edge1.cycle)
        SELECT {{select_columns}} FROM {node_table} node INNER JOIN traverse
        ON traverse.id = node.id
    """

    traverse_sql_template = None

    @staticmethod
    def get_column_names(direction):
        if direction == 'descendant':
            return {'from_column': 'parent_id',
                    'to_column': 'child_id'}

        elif direction == 'ancestor':
            return {'from_column': 'child_id',
                    'to_column': 'parent_id'}

        else:
            raise ValueError("direction has to be 'ancestor' or 'descendant'")

    @classmethod
    def get_traverse_sql(cls, direction, **kwargs):
        """
        Constructs a traversal SQL based on direction (ancestor, or descendant)
        """
        if cls.traverse_sql_template is None:
            cls.traverse_sql_template = cls.traverse_sql_tpl_lvl2.format(
                node_table=cls._meta.db_table,
                edge_table=cls.children.through._meta.db_table)

        kwargs2 = kwargs.copy()
        kwargs2.update(cls.get_column_names(direction))

        return cls.traverse_sql_template.format(**kwargs2)

    def __unicode__(self):
        return "# %s" % self.pk

    def __str__(self):
        return self.__unicode__()

    def add_child(self, descendant, **kwargs):
        """
        Adds a child
        """
        args = kwargs
        args.update({'parent' : self, 'child' : descendant })
        cls = self.children.through(**kwargs)
        return cls.save()


    def add_parent(self, parent, *args, **kwargs):
        """
        Adds a parent
        """
        return parent.add_child(self, **kwargs)

    def remove_child(self, descendant):
        """
        Removes a child
        """
        self.children.through.objects.get(parent = self, child = descendant).delete()

    def remove_parent(self, parent):
        """
        Removes a parent
        """
        parent.children.through.objects.get(parent = parent, child = self).delete()

    def parents(self):
        """
        Returns all elements which have 'self' as a direct descendant
        """
        return self.__class__.objects.filter(children = self)

    def build_tree(self, direction):
        q = (self.get_traverse_sql(direction,
                                   select_columns=('node.*, '
                                                   'path')) +
             "ORDER BY array_length(traverse.path, 1)")

        qs = self.__class__.objects.raw(q, [self.id])

        # Lookup table of nodes by id
        nodes = dict((node.id, node) for node in qs)
        nodes[self.id] = self
        tree = {}
        seen_paths = {}

        for node in qs:
            current_path = tuple(node.path[1:-1])

            try:
                current_tree = seen_paths[current_path]
            except KeyError:
                current_tree = tree
                for n in current_path:
                    current_tree = current_tree[nodes[n]]
                seen_paths[current_path] = current_tree

            seen_paths[tuple(node.path)] = current_tree[node] = {}

        return tree

    def descendants_tree(self):
        """
        Returns a tree-like structure with progeny
        """
        return self.build_tree('descendant')

    def ancestors_tree(self):
        """
        Returns a tree-like structure with ancestors
        """
        return self.build_tree('ancestor')

    def get_descendants(self):
        """Returns a RawQuerySet of descendants"""
        q = self.get_traverse_sql('descendant', select_columns='node.*')

        return self.__class__.objects.raw(q, [self.id])

    def get_ancestors(self):
        """
        Returns a RawQuerySet of ancestors
        """
        q = self.get_traverse_sql('ancestor', select_columns='node.*')

        return self.__class__.objects.raw(q, [self.id])

    def descendants_set(self):
        """
        Compatibility wrapper for get_descendants() that returns a set
        """
        return set(self.get_descendants())

    def ancestors_set(self):
        """
        Compatibility wrapper for get_ancestors() that returns a set
        """
        return set(self.get_ancestors())

    def distance(self, target):
        """
        Returns the shortest hops count to the target vertex
        """
        return len(self._path_id(target))

    def _path_id(self, target):
        q = self.get_traverse_sql('descendant', select_columns='node.id, path')
        q += """
            WHERE node.id = %s
            AND ARRAY_LENGTH(traverse.path, 1) =
                (SELECT MIN(ARRAY_LENGTH(t2.path, 1)) FROM traverse t2
                 WHERE t2.id = node.id) LIMIT 1
        """
        qs = self.__class__.objects.raw(q, [self.id, target.id])

        try:
            return qs[0].path[1:]

        except self.DoesNotExist:
            raise NodeNotReachableException

    def path(self, target):
        """
        Returns the shortest path
        """
        ids_path = self._path_id(target)
        nodes = self.__class__.objects.in_bulk(ids_path)
        return [nodes[nodeid] for nodeid in ids_path]

    def is_root(self):
        """
        Check if has children and not ancestors
        """
        return bool(self.children.exists() and not self._parents.exists())

    def is_leaf(self):
        """
        Check if has ancestors and not children
        """
        return bool(self._parents.exists() and not self.children.exists())

    def is_island(self):
        """
        Check if has no ancestors nor children
        """
        return bool(not self.children.exists() and not self._parents.exists())

    def is_descendant_of(self, target):
        """
        Check if self is a descendant of target
        """
        q = (self.get_traverse_sql('descendant', select_columns='node.id') +
             "WHERE traverse.id = %s")

        return bool(len(list(self.__class__.objects.raw(q, [target.id,
                                                            self.id]))))

    def is_ancestor_of(self, target):
        """
        Check if self is an ancestor of target
        """
        q = (self.get_traverse_sql('ancestor', select_columns='node.id') +
             "WHERE traverse.id = %s")

        return bool(len(list(self.__class__.objects.raw(q, [target.id,
                                                            self.id]))))

    def get_endpoints(self, direction):
        q = self.get_traverse_sql(direction, select_columns='node.*')
        q += """
            WHERE NOT EXISTS (SELECT * FROM {edge_table}
                              WHERE {from_column} = node.id)
        """.format(edge_table=self.children.through._meta.db_table,
                   **self.get_column_names(direction))

        qs = self.__class__.objects.raw(q, [self.id])

        return set(qs)

    def get_roots(self):
        """
        Returns roots nodes, if any
        """
        return self.get_endpoints('ancestor')

    def get_leaves(self):
        """
        Returns leaves nodes, if any
        """
        return self.get_endpoints('descendant')

    @staticmethod
    def circular_checker(parent, child):
        """
        Checks that the object is not an ancestor or a descendant,
        avoid self links
        """
        if parent == child:
            raise ValidationError('Self links are not allowed')
        if child.is_ancestor_of(parent):
            raise ValidationError('The object is an ancestor.')
        if child.is_descendant_of(parent):
            raise ValidationError('The object is a descendant.')



def edge_factory(node_model, child_to_field = "id", parent_to_field = "id", concrete = True, base_model = models.Model):
    """
    Dag Edge factory
    """
    if isinstance(node_model, str):
        try:
            node_model_name = node_model.split('.')[1]
        except IndexError:
            node_model_name = node_model
    else:
        node_model_name = node_model._meta.module_name

    class Edge(base_model):
        class Meta:
            abstract = not concrete

        parent = models.ForeignKey(node_model, related_name = "%s_child" % node_model_name, to_field = parent_to_field)
        child = models.ForeignKey(node_model, related_name = "%s_parent" % node_model_name, to_field = child_to_field)

        def __unicode__(self):
            return "%s is child of %s" % (self.child, self.parent)

        def save(self, *args, **kwargs):
            self.parent.__class__.circular_checker(self.parent, self.child)
            super(Edge, self).save(*args, **kwargs) # Call the "real" save() method.

    return Edge

def node_factory(edge_model, children_null = True, base_model = models.Model):
    """
    Dag Node factory
    """
    class Node(base_model, NodeBase):
        class Meta:
            abstract        = True

        children  = models.ManyToManyField(
                'self',
                null        = children_null,
                blank       = children_null,
                symmetrical = False,
                through     = edge_model,
                related_name = '_parents') # NodeBase.parents() is a function

    return Node

