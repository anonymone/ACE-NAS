class Action:
    def __init__(self):
        self.ADD_EDGE = 0
        self.ADD_NODE = 1
    
    def ActionNormlize(self,code):
        return code% (self.ADD_NODE+1)
        
if __name__ == "__main__":
    a = Action()
    print('Hello')