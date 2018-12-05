import pandas as pd
import datetime

class BTreeNode:
    def __init__(self, degree=2, number_of_keys=0, is_leaf=True, items=None, children=None,
                 index=None):
        self.isLeaf = is_leaf
        self.numberOfKeys = number_of_keys
        self.index = index
        if items is not None:
            self.items = items
        else:
            self.items = [None] * (degree * 2 - 1)
        if children is not None:
            self.children = children
        else:
            self.children = [None] * degree * 2

           
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

class BTree:
    def __init__(self, degree=2, nodes=None, root_index=1, free_index=2):
        if nodes is None:
            nodes = {}
        self.degree = degree

        if len(nodes) == 0:
            self.rootNode = BTreeNode(degree)
            self.nodes = {}
            self.rootNode.set_index(root_index)
            self.write_at(1, self.rootNode)
        else:
            self.nodes = nodes
            self.rootNode = self.nodes[root_index]
        self.rootIndex = root_index
        self.freeIndex = free_index

    def write_at(self, index, a_node):
    	self.nodes[index] = a_node 

	def representation(self):
		print "inside"
		return "BTree("+str(self.degree)+",\n function" + str(self.function()) + ","+ \
	    	str(self.rootIndex)+","+str(self.freeIndex)

    def get_free_index(self):
        self.freeIndex += 1
        return self.freeIndex - 1

    def set_root_node(self, r):
        self.rootNode = r
        self.rootIndex = self.rootNode.index

    def get_free_node(self):
        new_node = BTreeNode(self.degree)
        print "here"
        index = self.get_free_index()
        new_node.set_index(index)
        print "get free node"
        print index
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
		# print 'predict'
		search_result = self.search(key)
		a_node = self.nodes[search_result['fileIndex']]
		print 'a_node' + str(a_node)
		if a_node.items[search_result['nodeIndex']] is None:
		    return -1
		return a_node.items[search_result['nodeIndex']]

    def get_node(self, index):
    	return self.nodes[index]

    def split_child(self, p_node, i, c_node):
        new_node = self.get_free_node()
        new_node.isLeaf = c_node.isLeaf
        new_node.numberOfKeys = self.degree - 1
        for j in range(0, self.degree - 1):
        	new_node.items[j] = c_node.items[j + self.degree]
        	print 'self_degree' + str(self.degree)
        	print c_node.items[j + self.degree]	   
        if c_node.isLeaf is False:
            for j in range(0, self.degree):
                new_node.children[j] = c_node.children[j + self.degree]
        c_node.numberOfKeys = self.degree - 1
        j = p_node.numberOfKeys + 1
        while j > i + 1:
            p_node.children[j + 1] = p_node.children[j]
            j -= 1
        print 'new node index' + str(new_node.index)
        p_node.children[j] = new_node.index
        j = p_node.numberOfKeys
        while j > i:
            p_node.items[j + 1] = p_node.items[j]
            j -= 1
        p_node.items[i] = c_node.items[self.degree - 1]
        print 'key'
        print c_node.items[self.degree - 1]
        p_node.numberOfKeys += 1

    def search(self, an_item):
     	return self.rootNode.search(self, an_item)

    def insert(self, an_item):
        search_result = self.search(an_item)
        if search_result['found']:
            return None
        r = self.rootNode
        print "num keys"
        print r.numberOfKeys
        print "total"
        print 2 * self.degree - 1

        if r.numberOfKeys == 2 * self.degree - 1:
			print "inside"
			s = self.get_free_node()
			self.set_root_node(s)
			s.isLeaf = False
			s.numberOfKeys = 0
			s.children[0] = r.index
			print "insert"
			self.split_child(s, 0, r)
			self.insert_not_full(s, an_item)
        else:
        	print "full"
        	self.insert_not_full(r, an_item)

    def insert_not_full(self, inNode, anItem):
		i = inNode.numberOfKeys - 1
		print "anItem"
		print anItem
		if inNode.isLeaf:
			print "isLeaf"
			while i >= 0 and anItem < inNode.items[i]:
				inNode.items[i + 1] = inNode.items[i]
				i -= 1
			inNode.items[i + 1] = anItem
			inNode.numberOfKeys += 1
		else:
			print "isNotLeaf"
			while i >= 0 and anItem < inNode.items[i]:
				i -= 1
			i += 1
			if self.get_node(inNode.children[i]).numberOfKeys == 2 * self.degree - 1:
				self.split_child(inNode, i, self.get_node(inNode.children[i]))
				if anItem > inNode.items[i]:
					i += 1
			self.insert_not_full(self.get_node(inNode.children[i]), anItem)




def b_tree_main():

	# lst = [10,8,22,14,12,18,2,50,15]
	b = BTree(2)

	path = "lognormal.csv"
	data = pd.read_csv(path)
	b = BTree(2)
    
	for i in range(data.shape[0]):
		print(b.function())
		print '\n***Inserting*** ' + str(i)
		# b.insert(x)
		b.insert(data.ix[i, 0])

	print '\npredicting'
	before = datetime.datetime.now()
	pos = b.predict(30310)
	after = datetime.datetime.now()
	delta = after -  before
	print 'position'
	print(pos)
	print 'delta'
	print(delta)

	
	# for x in lst:
	# 	# print(b.representation())
	# 	print(b.function())
	# 	print '\n***Inserting*** ' + str(x)
	# 	b.insert(x)
    

if __name__ == '__main__':
    b_tree_main()