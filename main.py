#!/usr/bin/env python
import unittest
from c45 import C45

c1 = C45("./dataset/healthy_train.csv")

class testC45Methods(unittest.TestCase):
	c1 = C45("./dataset/healthy_train.csv")

	def testFoo(self):
		self.assertEqual(True, True)


def main():
	unittest.main()

c1.fetchData()
c1.generateTree()
# c1.printTree()
