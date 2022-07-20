import torch
import torch.nn as nn
import math
import numpy as np

class AsymmetricConvolution(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        #self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 1), padding=(0, 0))
        #self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 3), padding=(1, 1), bias=False)
        #self.conv2 = nn.Conv2d(out_cha, out_cha, kernel_size=(3, 3), padding=(1, 1))
        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):
        x = x.unsqueeze(0)
        shortcut = self.shortcut(x)

        #x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x + shortcut
        x = x.squeeze(0)
        return x
        

class InteractionMask(nn.Module):
    def __init__(self, n_conv, spatial_channel, temporal_channel):
        super(InteractionMask, self).__init__()

        self.spatial_output = nn.Sigmoid()
        self.number_asymmetric_conv_layer = n_conv
        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()
        #self.asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(AsymmetricConvolution(temporal_channel,temporal_channel))
            self.temporal_asymmetric_convolutions.append(AsymmetricConvolution(spatial_channel,spatial_channel))
            #self.asymmetric_convolutions.append(AsymmetricConvolution(spatial_channel,spatial_channel))
        
        self.spatial_output = nn.Tanh()
        self.temporal_output = nn.Tanh()
        #self.st_output = nn.Sigmoid()
        #self.st_output = nn.Tanh()



    def forward(self, dense_spatial_interaction,dense_temporal_interaction, threshold):
    #def forward(self, dense_st_interaction, threshold):

        ######################################################################à
        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)
            #dense_st_interaction = self.asymmetric_convolutions[j](dense_st_interaction)
        ######################################################################


        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)
        #st_interaction_mask = self.st_output(dense_st_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask, device='cuda')
        temporal_zero = torch.zeros_like(temporal_interaction_mask, device='cuda')
        #st_zero = torch.zeros_like(st_interaction_mask, device='cuda')
        
        spatial_interaction_mask = torch.where(torch.abs(spatial_interaction_mask) > threshold, spatial_interaction_mask,spatial_zero)
        temporal_interaction_mask = torch.where(torch.abs(temporal_interaction_mask) > threshold, temporal_interaction_mask, temporal_zero)
        #st_interaction_mask = torch.where(torch.abs(dense_st_interaction) > threshold, dense_st_interaction, st_zero)


        return spatial_interaction_mask, temporal_interaction_mask #st_interaction_mask#



class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, K):
        super(depthwise_separable_conv, self).__init__()
        #self.depthwise = nn.Conv2d(nin, K * nin, kernel_size=1, padding=0, groups=nin)
        
        self.depthwise =nn.Sequential(
            nn.Conv2d(nin, K * nin, kernel_size=1, padding=0, groups=nin),
            nn.BatchNorm2d(K * nin),
            nn.Dropout(0.1, inplace=True),
        )
        self.relu6 = nn.ReLU6()

        self.pointwise = nn.Conv2d(K * nin, nout, kernel_size=(1,1), padding=(0,0))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu6(out)
        out = self.pointwise(out)
        return out


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim, in_channels
    ):
        super(ConvTemporalGraphical,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)
        
        self.T=nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        

        #self.interaction_mask = InteractionMask(n_conv=1, spatial_channel=joints_dim, temporal_channel=time_dim)

    def forward(self, x, Amask, Tmask):
      
        Tt = Tmask*self.T
        Aa = Amask*self.A
        x = torch.einsum('nctv,vtq->ncqv', (x, Tt))
        x = torch.einsum('nctv,tvw->nctw', (x, Aa))
        return x.contiguous() 


# In[3]:


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        
        self.gcn=ConvTemporalGraphical(time_dim,joints_dim, in_channels) # the convolution layer
        
        self.dsc=depthwise_separable_conv(in_channels, out_channels, K=1)
        
                


        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()
        self.prelu2 = nn.PReLU()

        

    def forward(self, x, Amask, Tmask):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x, Amask, Tmask) 
        x=self.dsc(x)
        x=x+res
        x=self.prelu(x)
        return x


# In[4]:


class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        
        
        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)] 



            
        
        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):
        
        output= self.block(x)
        return output


# In[11]:


class Model(nn.Module):
    """ 
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 bias=True):
        
        super(Model,self).__init__()



        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.st_gcnns=nn.ModuleList()
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns=nn.ModuleList()
        
      
        self.st_gcnns.append(ST_GCNN_layer(input_channels,66,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(66,66,[1,1],1,input_time_frame,      #3,3
                                               joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(66,66,[1,1],1,input_time_frame,      #3,3
                                               joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(66,66,[1,1],1,input_time_frame,      #3,3
                                               joints_to_consider,st_gcnn_dropout))
                                               
        self.st_gcnns.append(ST_GCNN_layer(66,input_channels,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))                                               
                
                
                # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))
        
            
        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())


        
    
    def forward(self, x, maskA, maskT):
        num = 0
        for gcn in (self.st_gcnns):
            x = gcn(x, maskA[num], maskT[num])
            num+=1
            
        x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)
        
        x=self.prelus[0](self.txcnns[0](x))
        
        for i in range(1,self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) +x # residual connection
            
        return x