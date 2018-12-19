import pandas as pd
import datetime
import random
import sys
import pdb

# Every node in the BTree stores a number of item objects
# Implementation of the item class

class Item():
  def __init__(self, key, value):
    self.k = key
    self.v = value

  def __str__(self):
    return "Key: " + str(self.k) + " Value: " + str(self.v)

  def __eq__(self, other):
    return self.k == other.k

  def __gt__(self, other):
    return self.k > other.k

  def __ge__(self, other):
    return self.k >= other.k

  def __lt__(self, other):
    return self.k < other.k

  def __le__(self, other):
    return self.k <= other.k

# Every node in the B-Tree is implemented as a BTreeNode

class BTreeNode:

  # numberOfKeys - number of non-null keys in the node
  # items - keys in a node
  # children - indexes of nodes that are the children of the current node
  # index - position of the node in sorted order

  def __init__(self, keys_per_node = 0):
    self.items = [None] * keys_per_node
    self.children = [None] * (keys_per_node + 1)
    self.isLeaf = True
    self.numberOfKeys = 0
    self.index = None

  def set_index(self, index):
    self.index = index

  def search(self, b_tree, an_item):
    i = 0
    while i < self.numberOfKeys and an_item > self.items[i]:
      i += 1
    if i < self.numberOfKeys and an_item == self.items[i]:
      return {'found': True, 'fileIndex': self.index, 'nodeIndex': i}
    if self.isLeaf:
      return {'found': False, 'fileIndex': self.index, 'nodeIndex': i - 1}
    else:
      return b_tree.get_node(self.children[i]).search(b_tree, an_item)


# B-Tree stores the properties of the required B-Tree

class BTree:

  # keys_per_node - page_size (number of keys in each node)
  # degree - Number of keys to half-fill the node
  # rootNode - root node in the B-Tree
  # nodes - map of all nodes in the B-Tree (index to node mapping)
  # rootIndex - index of the rootNode in the nodes map
  # freeIndex - An incremental counter always pointing to next position to be filled

  def __init__(self, keys_per_node = 0):
    self.keys_per_node = keys_per_node
    self.degree = (self.keys_per_node + 1) / 2
    self.rootIndex = 1
    self.freeIndex = self.rootIndex + 1
    self.rootNode = BTreeNode(self.keys_per_node)
    self.rootNode.set_index(self.rootIndex)
    self.nodes = {}
    self.write_at(self.rootIndex, self.rootNode)
    

  def build(self, keys, values):
    if len(keys) != len(values):
      return
    for ind in range(len(keys)):
      self.insert(Item(keys[ind], values[ind]))

  def write_at(self, index, a_node):
    self.nodes[index] = a_node

  def representation(self):
    return "BTree("+str(self.degree)+",\n function" + str(self.function()) + ","+ str(self.rootIndex)+","+str(self.freeIndex)
  
  def get_free_index(self):
    self.freeIndex += 1
    return self.freeIndex - 1

  def set_root_node(self, r):
    self.rootNode = r
    self.rootIndex = self.rootNode.index

  def get_free_node(self):
    new_node = BTreeNode(self.keys_per_node)
    index = self.get_free_index()
    new_node.set_index(index)
    self.write_at(index, new_node)
    return new_node

  def print_nodes(self):
    s = ''
    for x in self.nodes:
      s = s + str(x) + ' ';
    return s
    
  def function(self):
    return 'BTree Degree:' + str(self.degree) + ' RootIndex:' + str(self.rootIndex)+ ' FreeIndex:' +str(self.freeIndex) + '\nNodes:' + str(self.print_nodes()) 

  def predict(self, key):
    search_result = self.search(Item(key, 0))
    if not search_result['found']:
      return -1
    node = search_result['fileIndex']
    item = search_result['nodeIndex']
    return self.nodes[node].items[item].v

  def get_node(self, index):
    return self.nodes[index]

  def split_child(self, p_node, i, c_node):
    new_node = self.get_free_node()
    new_node.isLeaf = c_node.isLeaf
    new_node.numberOfKeys = self.degree - 1
    for j in range(0, self.degree - 1):
      new_node.items[j] = c_node.items[j + self.degree]
    if not c_node.isLeaf:
      for j in range(0, self.degree):
        new_node.children[j] = c_node.children[j + self.degree]
    c_node.numberOfKeys = self.degree - 1
    j = p_node.numberOfKeys + 1
    while j > i + 1:
      p_node.children[j + 1] = p_node.children[j]
      j -= 1
    p_node.children[j] = new_node.index
    j = p_node.numberOfKeys
    while j > i:
      p_node.items[j + 1] = p_node.items[j]
      j -= 1
    p_node.items[i] = c_node.items[self.degree - 1]
    p_node.numberOfKeys += 1

  def search(self, an_item):
     return self.rootNode.search(self, an_item)

  def insert(self, an_item):
    search_result = self.search(an_item)
    if search_result['found']:
      return None
    r = self.rootNode
    if r.numberOfKeys == 2 * self.degree - 1:
      s = self.get_free_node()
      self.set_root_node(s)
      s.isLeaf = False
      s.numberOfKeys = 0
      s.children[0] = r.index
      self.split_child(s, 0, r)
      self.insert_not_full(s, an_item)
    else:
      self.insert_not_full(r, an_item)

  def insert_not_full(self, inNode, anItem):
    i = inNode.numberOfKeys - 1
    if inNode.isLeaf:
      while i >= 0 and anItem < inNode.items[i]:
        inNode.items[i + 1] = inNode.items[i]
        i -= 1
      inNode.items[i + 1] = anItem
      inNode.numberOfKeys += 1
    else:
      while i >= 0 and anItem < inNode.items[i]:
        i -= 1
      i += 1
      if self.get_node(inNode.children[i]).numberOfKeys == 2 * self.degree - 1:
        self.split_child(inNode, i, self.get_node(inNode.children[i]))
        if anItem > inNode.items[i]:
          i += 1
      self.insert_not_full(self.get_node(inNode.children[i]), anItem)

def b_tree_main(path, page_size):
  data = pd.read_csv(path, header = None)
  btree = BTree(page_size)
  total_data_size = data.shape[0]
  test_data_size = int(0.1 * total_data_size)
  
  # range_min = float('inf')
  # range_max = - float('inf')
  # for i in range(total_data_size):
  #   range_min = min(range_min, data.ix[i, 0])
  #   range_max = max(range_max, data.ix[i, 0])

  print "B-Tree size = " + str(total_data_size)
  print "Page size = " + str(btree.keys_per_node)

  

  model_start_time = datetime.datetime.now()
  for i in range(total_data_size):
    btree.insert(Item(data.ix[i, 0], data.ix[i, 1]))
  model_end_time = datetime.datetime.now()

  print "Lookup in progress!"

  start_time = datetime.datetime.now()
  for x in range(test_data_size):
    # num = random.randint(range_min, range_max)
    num_index = random.randint(0, total_data_size - 1)
    btree.predict(data.ix[num_index, 0])

  end_time = datetime.datetime.now()

  print "Time taken for B-Tree construction:\t" + str(model_end_time - model_start_time) + " seconds"
  print "Time taken for B-Tree lookup:\t" + str(end_time - start_time) + " seconds"

# Pass path of file in argv[1] and page size in argv[2]
if __name__ == '__main__':
  b_tree_main(sys.argv[1], int(sys.argv[2]))

