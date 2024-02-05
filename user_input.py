from monty.json import MSONable

class MaterialProperty(MSONable):
    def __init__(self, prop, upper_bound, lower_bound):
        self.prop = prop
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

class Material(MSONable):
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties

    def as_dict(self):
        # Leverage recursive_as_dict implicitly by structuring the data as needed
        return {self.name: {p.prop: {"upper_bound": p.upper_bound, "lower_bound": p.lower_bound} for p in self.properties}}

class MixtureProperty(MSONable):
    def __init__(self, prop, desired_prop):
        self.prop = prop
        self.desired_prop = desired_prop

class Mixture(MSONable):
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties

    def as_dict(self):
        # Leverage recursive_as_dict implicitly by structuring the data as needed
        return {self.name: {p.prop: {"desired_prop": p.desired_prop} for p in self.properties}}