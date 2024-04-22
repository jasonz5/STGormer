class MoEScheduler():
    def __init__(self, top_k_init, current_epoch=0, policy="dense2sparse"):
        super(MoEScheduler, self).__init__()
        self.top_k = top_k_init
        self.policy = policy
        self.current_epoch = current_epoch
        self.milestones = None
        if policy == "dense2sparse":
            self.milestones = [50, 75, 100, 125]
    
    def update_epoch(self, current_epoch):
        self.current_epoch = current_epoch  # 更新当前的epoch
    
    def get_k(self):
        if self.policy == "dense2sparse" and self.current_epoch in self.milestones:
            self.top_k -= 1
        return self.top_k

