class GlobalCounter():
    def __init__(self):
        self.counter = 0
    
    def count(self):
        self.counter += 1
    
    def get_count(self):
        return self.counter