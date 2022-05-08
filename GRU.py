import torch
from torch import nn
import numpy as np


class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """
    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer10
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer20
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        super(GRUCellV2, self).__init__()
        self.activation = activation

        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)#(60,10)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)#(60,20)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)#(60,1)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)#(60,1)

        
    def forward(self, x, h):
        
        """
        Performs a forward pass through a GRU cell
        


        Returns the current hidden state h_t for every datapoint in batch.
        
        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        #

        #x(5,10)
        #h(5,20)
        #print(x.shape)

        w_ih=torch.matmul(x,self.w_ih.T)+self.b_ih#(5,10),(10,60)=(5,60)
        w_hh=torch.matmul(h,self.w_hh.T)+self.b_hh#(5,20),(20,60)=(5,60)



        w_ir, w_iz, w_in= torch.chunk(w_ih, 3, 1)#(5,20),(5,20),(5,20)
       
        w_hr,w_hz,w_hn=torch.chunk(w_hh,3,1)#(5,20),(5,20),(5,20)

       
        
        r=torch.sigmoid(w_hr+w_ir)#(5,20)
        z=torch.sigmoid(w_hz+w_iz)#(5,20)
        #print(z)
        n=torch.tanh(r*(w_hn)+(w_in))#(5,20)
        #print(n.size())
        h=(1-z)*n+(z*h)

        
        return h








        pass


class GRU2(nn.Module):
    """
    GRU network implementation
    """
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False):
        super(GRU2, self).__init__()
        
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation) # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation) # backward cell
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.
        Takes as input a 3D tensor `x` of dimensionality (B, T, D),
        where B is the batch size;
              T is the sequence length (if sequences have different lengths, they should be padded before being inputted to forward)
              D is the dimensionality of each element in the sequence, e.g. word vector dimensionality

        The method returns a 3-tuple of (outputs, h_fw, h_bw), if self.bidirectional is True,
                           a 2-tuple of (outputs, h_fw), otherwise
        `outputs` is a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class);
                  NOTE: if bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
        `h_fw` is the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence;
        `h_bw` is the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards)
        
        :param      x:    a batch of sequences of dimensionality (B, T, D)
        :type       x:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        #
        torch.autograd.set_detect_anomaly(True)
        batch_size=x.size()[0]
        seq_length=x.size()[1]
        dimension=x.size()[2]
        result=torch.zeros((batch_size,seq_length,self.hidden_size))#(5,3,20)

        #for t in range(seq_length-1,-1,-1):

        for t in range(seq_length):
            if(t==0):
                last_hidden_state=torch.zeros((batch_size,self.hidden_size))#(5,20)
            else:
                last_hidden_state=result.clone()[:,t-1,:]
            result[:,t,:]=self.fw.forward(x[:,t,:],last_hidden_state)

        
        if(self.bidirectional==True):
            result_backward=torch.zeros((batch_size,seq_length,self.hidden_size))#(5,3,20)
            for t in range (seq_length-1,-1,-1):

                if(t==seq_length-1):
                    last_hidden_state=torch.zeros((batch_size,self.hidden_size))#(5,20)
                else:
                    last_hidden_state=result_backward.clone()[:,t+1,:]
                result_backward[:,t,:]=self.bw.forward(x[:,t,:],last_hidden_state)

        
            return result,result[:,seq_length-1,:],result_backward[:,0,:]

        return result,result[:,seq_length-1,:]



                









        pass


def is_identical(a, b):
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"


if __name__ == '__main__':
    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)#creates a 3d tensor of dim (5,3,10)
    gru = nn.GRU(10, 20, bidirectional=False, batch_first=True)
    outputs, h = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)#creates a 3d tensor of dim (5,3,10) with same set of rand numbers
    gru2 = GRU2(10, 20, bidirectional=False)
    outputs, h_fw = gru2(x)
    
    print("Checking the unidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = GRU2(10, 20, bidirectional=True)
    outputs, h_fw, h_bw = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
    outputs, h = gru2(x)
    
    print("Checking the bidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))
    print("Same hidden states of the backward cell?\t{}".format(
        is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
    ))