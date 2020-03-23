from abc import abstractmethod

class AbstractNLPProcessor:

    @abstractmethod
    def extract_named_entities(self, token):
        pass

    @abstractmethod
    def get_named_entity(self, token):
        pass

    @abstractmethod
    def get_named_entity_type(self, token):
        pass

    @abstractmethod
    def is_hyponym(self, first, second):
        pass

    @abstractmethod
    def is_hypernym(self, first, second):
        pass