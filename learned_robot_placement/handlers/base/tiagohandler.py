from abc import ABC, abstractmethod

class TiagoBaseHandler(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
    def get_robot(self):
        pass

    @abstractmethod
    def post_reset(self):
        pass

    @abstractmethod
    def apply_actions(self):
        pass

    # @abstractmethod
    # def get_obs_dict(self):
    #     pass

    @abstractmethod
    def reset(self):
        pass

