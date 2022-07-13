#!/usr/bin/python
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, MLP, TransformerConv, GCNConv
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import add_self_loops, degree

class ColorFunction(MessagePassing):
      def __init__(self):
            super().__init__()

class VertexFunction(MessagePassing):
      def __init__(self):
            super().__init__()


class MessageBlock(torch.nn.Module):
      def __init__(self,
                  vertex_message,
                  color_message
                  ):
            super().__init__()

class SimpleNet(torch.nn.Module):
      def __init__(self, num_features, num_classes):
            super().__init__()
            self.relu_alpha = 0.2
            self.conv = GCNConv(in_channels=num_features, out_channels=64)
            self.voting = MLP([64,64,64,1])
      
      def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv(x, edge_index)
            x = F.leaky_relu(x)
            x = self.voting(x)
            if isinstance(data, Batch):

                  #https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
                  labels = data.batch
                  labels = labels.view(labels.size(0), 1).expand(-1, x.size(1))

                  unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

                  res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, x)
                  x = res / labels_count.float().unsqueeze(1)
            else:
                  x = torch.mean(x,axis=0)
            return torch.sigmoid(x)

class ColorNetwork(torch.nn.Module):
      def __init__(self, max_message_passing=32):
            super().__init__()
            self.max_message_passing = max_message_passing
            
            self.voting = MLP([64,64,64,1])
            self.color_messages = ColorFunction()
            self.vertex_messages = VertexFunction()
            self.color_updater = 0
            self.vertex_updater = 0
            

      def forward(self, data):
            M_v, edge_index, M_c = data.x, data.edge_index, data.col_enc
            for t in range(self.max_message_passing):
                  M_v = self.vertex_updater(
                              torch.matmul(edge_index, M_v),
                              torch.matmul(color_index, self.color_messages(M_c))
                              )
                  M_c = self.color_updater(
                        torch.matmul(color_index, self.vertex_messages(M_v))
                  ) 
            M_v = self.voting(M_v)
            return torch.sigmoid(torch.mean(M_v))