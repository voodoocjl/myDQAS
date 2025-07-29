class Arguments:
    def __init__(self, task = 'MNIST'):

        self.device     = 'cpu'        
        self.clr        = 0.005
        self.qlr        = 0.01
        
        if task != 'MOSI':
            # self.n_qubits   = 10
            self.n_qubits = 4

            self.epochs     = 1
            self.batch_size = 256        
            self.sampling = 5

            self.n_layers = 4
            self.base_code = [self.n_layers, 2, 3, 4, 1]
            self.exploration = [0.001, 0.002, 0.003]

            self.task    = 'MNIST'
            # self.task    = 'FASHION'
            # self.task    = 'MOSI'
            self.backend    = 'tq'
            # self.digits_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.digits_of_interest = [0, 1, 2, 3]
            self.train_valid_split_ratio = [0.95, 0.05]
            self.center_crop = 24
            self.resize = 28
        else:
            self.n_qubits   = 7

            self.a_insize   = 74
            self.v_insize   = 35
            self.t_insize   = 300
            self.a_hidsize  = 6
            self.v_hidsize  = 3
            self.t_hidsize  = 12

            self.epochs     = 3
            self.batch_size = 32      
            self.n_layers = 5
            self.base_code = [self.n_layers, 2, 3, 4, 5, 6, 7, 1]
