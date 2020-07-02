

class InformationCriterion:

    def __init__(self, model):
        self.model = model


class AkaikeInformationCriterion(InformationCriterion):

    def __call__(self, X, *args, **kwargs):
