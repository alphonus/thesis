#%%
import os, sys
import os.path as osp
import random
import numpy as np
from functools import reduce
from pathlib import Path
import torch
from torch_geometric.data import   Data, Dataset,InMemoryDataset, download_url

#%%
class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        #return [f'/m0.graph']#[ self.raw_dir + '/' + x for x in os.listdir(self.raw_dir) ]
        return [f for f in os.listdir(self.raw_dir) if osp.isfile(osp.join(self.raw_dir, f))]
    @property
    def processed_file_names(self):
        #return ['not_implemented.pt']#[ self.processed_dir + '/' + x for x in os.listdir(self.processed_dir) ]
        #return [f for f in os.listdir(self.processed_dir) if osp.isfile(osp.join(self.processed_dir, f))]
        return ['data.pt']
    def process(self):
        data_list = []
        idx = 0
        for raw_path in self.raw_paths:
            Ma,chrom_number,diff_edge = read_graph(raw_path)
            col = torch.tensor([[1, chrom_number]], dtype=torch.int)
            #generate initial graph coloring
            M_v = torch.tensor(np.random.choice(chrom_number, np.max(Ma)), dtype=torch.int)
            data = Data(x=M_v, edge_index=Ma, y=col, chromatic_number=chrom_number)
            ##if self.pre_filter is not None and not self.pre_filter(data):
            ##    continue

            ##if self.pre_transform is not None:
            ##    data = self.pre_transform(data)

            ##torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            data_list.append(data)
            #generate adversarial entry
            col = torch.tensor([0, chrom_number], dtype=torch.int)
            #generate initial graph coloring
            Ma_fake = np.vstack((Ma, diff_edge, np.flip(diff_edge)))
            data = Data(x=M_v, edge_index=Ma_fake, y=col, chromatic_number=chrom_number)
            ##torch.save(data, osp.join(self.processed_dir, f'data_{idx}_adversarial.pt'))
            data_list.append(data)


            idx += 1
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    """
    def len(self):
        return self.data.shape[0]
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    """
#%%
class InstanceLoader(object):

    def __init__(self,path):
        self.path = path
        self.filenames = [ path + '/' + x for x in os.listdir(path) ]
        random.shuffle(self.filenames)
        self.reset()
    #end

    def get_instances(self, n_instances):
        for i in range(n_instances):
            # Read graph from file
            Ma,chrom_number,diff_edge = read_graph(self.filenames[self.index])
            f = self.filenames[self.index]
            
            Ma1 = Ma
            Ma2 = Ma.copy()

            if diff_edge is not None:
                # Create a (UNSAT/SAT) pair of instances with one edge of difference
                # The second instance has one edge more (diff_edge) which renders it SAT
                Ma2[diff_edge[0],diff_edge[1]] = Ma2[diff_edge[1],diff_edge[0]] =1
            #end

            # Yield both instances
            yield Ma1,chrom_number,f
            yield Ma2,chrom_number,f

            if self.index + 1 < len(self.filenames):
                self.index += 1
            else:
                self.reset()
        #end
    #end

    def create_batch(instances):

        
        n_instances = len(instances)
        
        # n_vertices[i]: number of vertices in the i-th instance
        n_vertices  = np.array([ x[0].shape[0] for x in instances ])
        # n_edges[i]: number of edges in the i-th instance
        n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
        # n_colors[i]: number of colors in the i-th instance
        n_colors = np.array( [x[1] for x in instances])
        # total_vertices: total number of vertices among all instances
        total_vertices  = sum(n_vertices)
        # total_edges: total number of edges among all instances
        total_edges     = sum(n_edges)
        # total_colors: total number of colors among all instances
        total_colors = sum(n_colors)

        # compute binary adjacency matrix from vertex to vertex
        # Compute matrices M, MC
        # M is the adjacency matrix
        M              = np.zeros((total_vertices,total_vertices))
        #compute binary adjacency matrix from vertex to colors
        # b dsdMC is a matrix connecting each problem nodes to its colors candidates
        MC = np.zeros((total_vertices, total_colors))        

        # Even index instances are SAT, odd are UNSAT
        cn_exists = np.array([ 1-(i%2) for i in range(n_instances) ])

        for (i,(Ma,chrom_number,f)) in enumerate(instances):
            # Get the number of vertices (n) and edges (m) in this graph
            n, m, c = n_vertices[i], n_edges[i], n_colors[i]
            # Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
            n_acc = sum(n_vertices[0:i])
            m_acc = sum(n_edges[0:i])
            c_acc = sum(n_colors[0:i])
            #Populate MC
            #Binary Color matrix
            MC[n_acc:n_acc+n,c_acc:c_acc+c] = 1

            # Get the list of edges in this graph
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Populate M
            for e,(x,y) in enumerate(edges):
                if Ma[x,y] == 1:
                  M[n_acc+x,n_acc+y] = M[n_acc+y,n_acc+x] = 1
                #end if
            #end for
        #end for
        return M, n_colors, MC, cn_exists, n_vertices, n_edges, f
    #end

    def get_batches(self, batch_size):
        for i in range( len(self.filenames) // batch_size ):
            instances = list(self.get_instances(batch_size))
            yield InstanceLoader.create_batch(instances)
        #end
    #end
    
    def get_test_batches(self, batch_size, total_instances):
        for i in range( total_instances ):
            instances = list(self.get_instances(batch_size))
            yield InstanceLoader.create_batch(instances)
        #end
    #end

    def reset(self):
        random.shuffle(self.filenames)
        self.index = 0
    #end
#end

def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        #Ma = np.zeros((n,2),dtype=int)
        Ma = []

        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma.append([i,j])#Ma[i,j] = 1
            line = f.readline()
        #end while

        # Parse diff edge
        while 'DIFF_EDGE' not in line: line = f.readline();
        diff_edge = [ int(x) for x in f.readline().split() ]

        # Parse target cost
        while 'CHROM_NUMBER' not in line: line = f.readline();
        chrom_number = int(f.readline().strip())

    #end
    return np.array(Ma), chrom_number, diff_edge
#end

#%%
#dataset = GraphDataset(root="data")
# %%

# %%
