{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/pooja/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": [
    "data= pd.read_csv(\"digit/train.csv\").as_matrix()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(42000, 785)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "xtrain=data[0:21000,1:]"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "ytrain=data[0:21000,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 1, ..., 4, 9, 4])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "clf=DecisionTreeClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "xtest=data[21000:,1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n",
       "            max_features=None, max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=1, min_samples_split=2,\n",
       "            min_weight_fraction_leaf=0.0, presort=False, random_state=None,\n",
       "            splitter='best')"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "clf.fit(xtrain,train_label)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fbdb418eb70>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "d=xtest[8]\n",
    "d.shape=(28,28)\n",
    "py.imshow(255-d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[3]\n"
     ]
    }
   ],
   "source": [
    "print(clf.predict([xtest[8]]))\n",
    "py.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy= 83.63333333333334\n"
     ]
    }
   ],
   "source": [
    "p=clf.predict(xtest)\n",
    "count=0\n",
    "for i in range(0,21000):\n",
    "    count+=1 if p[i]==ytest[i] else 0\n",
    "print (\"Accuracy=\", (count/21000)*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
