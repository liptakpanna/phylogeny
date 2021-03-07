from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import numpy as np
import csv
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import math
import itertools
from random import seed
from random import randint

import dendropy
from dendropy.calculate import treecompare

class PhylogenyEnv(py_environment.PyEnvironment):
    def getStateFromCSV(self):
        path = self.getPath(False)
        
        with open(path, 'r') as f:
            reader = csv.reader(f)
            matrix_data = list(reader)
        label = matrix_data[0][1:]
        distance = []
        for x in matrix_data[1:]:
            distance.append(list(map(float, x[1:])))
        state = []
        labels = []
        for x in range(len(distance)):
            for j in range(len(distance[0])):
                if j > x:
                    state.append(distance[x][j])
                    labels.append(label[x]+","+label[j])
        return state, labels, len(label)

    def getPath(self,isTree):
        if isTree:
            end = '.tre'
            mid = 'trees'
        else:
            end = '.csv'
            mid = 'distances'
        path = ""
        firstBreak = int(self.setSize*self.balProportion)
        secondBreak = int(self.setSize*(self.balProportion+self.pecProportion))
        if self._i <= firstBreak:
            path = "train_set/"+mid+"/balanced/dist"+str(int(self._i))+end
        elif self._i <= secondBreak:
            path = "train_set/"+mid+"/pectinate/dist"+str(int(self._i-firstBreak))+end
        else:
            path = "train_set/"+mid+"/random/dist"+str(int(self._i-secondBreak))+end
        return path
  
    def __init__(self, isEval = False, discount = 0.75, setSize = 100, balProportion = 0.25, pecProportion = 0.25):
        self.setSize = setSize
        self.discount = discount
        self.balProportion = balProportion
        self.pecProportion = pecProportion
        self.isEval = isEval
        seed(1)
        self._tns = dendropy.TaxonNamespace()
        self._i = 1
        self._repeat = 1
        self._state, self._labels, self._n = self.getStateFromCSV()
        self._maxstate = int(self._n*(self._n-1)/2)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._maxstate-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,self._maxstate), dtype=np.int32, minimum=-15, name='observation')
        self._episode_ended = False
        self._calculated_tree = ""
        self._tree_pieces = []
        self._goal_tree = dendropy.Tree.get(path= self.getPath(True), schema = "newick", taxon_namespace = self._tns)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def getCurrentState(self):
        return self._labels

    def _reset(self):
        if self.isEval:
            repeatLimit = 1
        else:
            repeatLimit = 10
        if self._repeat > repeatLimit:
            self._i = self._i+1
            if self._i > self.setSize:
                self._i = 1
            self._repeat = 1
        else:
            self._repeat += 1
        self._state, self._labels, self._n = self.getStateFromCSV()
        self._episode_ended = False
        self._calculated_tree = ""
        self._tree_pieces = []
        self._goal_tree = dendropy.Tree.get(path=self.getPath(True), schema = "newick", taxon_namespace = self._tns)
        return ts.restart(np.array([self._state], dtype=np.int32))

    def addNodeToTree(self, first_node, second_node, ):
        firstPiece = ""
        secondPiece = ""
        if len(self._tree_pieces) == 0: self._tree_pieces.append("("+first_node+","+second_node+")")
        else:
            for piece in self._tree_pieces:
                for x in first_node:
                    if x in piece:
                        firstPiece = piece
                for x in second_node:
                    if x in piece:
                        secondPiece = piece
            if firstPiece == "" and secondPiece == "":
                self._tree_pieces.append("("+first_node+","+second_node+")")
            elif firstPiece != "" and secondPiece =="":
                self._tree_pieces.remove(firstPiece)
                self._tree_pieces.append("("+firstPiece+","+second_node+")")
            elif secondPiece != "" and firstPiece =="":
                self._tree_pieces.remove(secondPiece)
                self._tree_pieces.append("("+secondPiece+","+firstPiece+")")
            else:
                self._tree_pieces.remove(firstPiece)
                self._tree_pieces.remove(secondPiece)
                self._tree_pieces.append("("+firstPiece+","+secondPiece+")")

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if self._state[action] == -1:
            reward = -1
            return ts.termination(np.array([self._state], dtype=np.int32), reward)

        else :
            nodes_to_join = self._labels[action].split(",")
            first_node =  nodes_to_join[0]
            second_node = nodes_to_join[1]
            new_node = first_node + second_node

            index_to_delete = []
            index_to_delete.append(action.astype(int))
            new_labels = []
            new_dists = []

            for i, lab in enumerate(self._labels):
                if lab != self._labels[action] and first_node in lab:
                    other_node = lab.replace(first_node, "").replace(",","")
                    new_label =  new_node + "," + other_node

                    for j,x in enumerate(self._labels):
                        if second_node in x and other_node in x:
                            other_index = j

                    new_dist = 1/2*(self._state[i]+self._state[other_index]-self._state[action])

                    index_to_delete.extend([i,other_index])
                    new_labels.append(new_label)
                    new_dists.append(new_dist)
            index_to_delete.sort(reverse=True)
            for i in index_to_delete:
                del self._state[i]
                del self._labels[i]

            self._state.extend(new_dists)
            self._labels.extend(new_labels)

            if len(self._state) < self._maxstate:
                diff = self._maxstate - len(self._state)
                self._state.extend([-1]*diff)
                self._labels.extend([""]*diff)

            self.addNodeToTree(first_node, second_node)

            notEmptyLabels = [e for e in range(len(self._labels)) if self._labels[e] != ""]
            if(len(notEmptyLabels) == 1):
                remainingNodes = self._labels[notEmptyLabels[0]].split(",")
                self.addNodeToTree(remainingNodes[0], remainingNodes[1])
                self._calculated_tree= self._tree_pieces[0] + ";"
                self._episode_ended = True

        if self._episode_ended:
          #print(self._calculated_tree)
          #print(self._goal_tree)
            tree = dendropy.Tree.get(data=self._calculated_tree,schema="newick",taxon_namespace=self._tns)
            reward =  treecompare.symmetric_difference(self._goal_tree,tree)
          #print(treecompare.symmetric_difference(self._goal_tree,tree))
          #print(reward/maxdist*100)
            reward = (2*(6-3)-reward)/(2*(6-3))*10
          #print(reward)
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(np.array([self._state], dtype=np.int32), reward=0.0, discount=self.discount)