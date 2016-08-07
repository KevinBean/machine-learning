

<h2>Google的机器学习课程</h2>

<p>地址：<a href="https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal">&nbsp;Google的机器学习课程&nbsp;</a></p>

<h2>笔记</h2>

<h3>1</h3>

<p>通用程序，解决所有问题。帮我们寻找规律。分类。有监督的学习：自动生成规则。</p>

<p>Python机器学习工具包：<a href="http://scikit-learn.org/stable/index.html">scikit-learn</a></p>

<p>训练数据＝》分类器＝》做出预测</p>

<p>Decison Tree 决策树</p>

<pre><code>from sklearn import tree
Features = [[140, 1], [130, 1], [150, 0], [170, 0]]
Labels = [1,1,0,0]
clf =tree.DecisionTreeClassifier()
clf = clf.fit(Features, Labels)
print clf.predict([150, 0])
</code></pre>

<h3>2 可视化决策树</h3>

<p>Iris data</p>

<p>分解训练集 和 测试集</p>

<p>将决策树可视化</p>

<p>Pydotpydotplus工具包：用来绘制图</p>

<p>Graphviz工具包：brew install graphviz</p>

<p>以下程序最后一步无法绘图成功，原因不明。</p>

<pre><code>import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]
test_idx = [0, 50, 100]

# trainning data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

# viz code
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True, rounded=True,
        impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_png(&#39;&quot;Iris.png&quot;&#39;)
</code></pre>

<h3>3 选择好的特征</h3>

<ul>
	<li>Informative</li>
	<li>Independent</li>
	<li><p>Simple</p>

		<p>import numpy as np</p>

		<p>import matplotlib.pyplot as plt</p>

		<p>greyhounds = 500</p>

		<p>labs = 500</p>

		<p>grey_height = 28 + 4 * np.random.randn(greyhounds)</p>

		<p>lab_height = 24 + 4 *np.random.randn(labs)</p>

		<p>plt.hist([grey_height, lab_height],stacked = True, color=[&#39;r&#39;,&#39;b&#39;])</p>

		<p>plt.show()</p></li>
</ul>

<h3>4 pipeline</h3>

<p>f(x)=y</p>

<p>分类器原理</p>

<p>引入tensorflow</p>

<h3>5 classifier</h3>

<p>新建一个class</p>

<p>包含fit和predict动作</p>

<p>第一步作为测试，可使用一个随机分类器（random工具包）</p>

<pre><code>class RandomNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions
</code></pre>

<p>KNN分类器 K-Nearest Neighbors K近邻 根据最近的几个样本进行判断(distance.euclidean 欧几里得距离)</p>

<pre><code>from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist &lt; best_dist:
                best_index = i
            return self.y_train[best_index]
</code></pre>

<p>更多还有神经网络分类器、决策树分类器。</p>

<h3>6 TensorFlow做图片分类器</h3>

<p>深度学习：神经网络</p>

<p>图片分类的特殊性：某些特征无法用简单的数字来表示，难以手动找出明确的有效特征。</p>

<p>Inception：google的图片分类器，开源</p>

<p>分类器效果取决于训练数据的多样性和数量。</p>

<p>代码:<a href="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/">https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/</a></p>


