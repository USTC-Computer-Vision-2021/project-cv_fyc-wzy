[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6626804&assignment_repo_type=AssignmentRepo)

## 基于图像拼接技术实现 A Look Into the Past

成员及分工

- 王子怡 
  - coding

- 冯一纯
  - 设计与分析

### 问题描述

------

图像拼接是一个日益流行的研究领域，他已经成为照相绘图学、计算机视觉、图像处理和计算机图形学研究中的热点。图像拼接解决的问题一般式，通过对齐一系列空间重叠的图像，构成一个无缝的、高清晰的图像，它具有比单个图像更高的分辨率和更大的视野。由图片的某部分匹配到这张图片，可以用此技术从过去的照片中找到图片中景物的地点，对机器人识别技术有着广泛的前景。

### 原理分析

------

基于特征的配准方法是通过像素导出图像的特征，然后以图像特征为标准，对图像重叠部分的对应特征区域进行搜索匹配，该类拼接算法有比较高的健壮性和鲁棒性。

基于特征的配准方法有两个过程：特征抽取和特征配准。首先从两幅图像中提取灰度变化明显的点、线、区域等特征形成特征集合。然后在两幅图像对应的特征集中利用特征匹配算法尽可能地将存在对应关系的特征对选择出来。之后根据这些匹配的特征对计算单应性矩阵，其中包括基于所有对的常规方法、基于RANSAC的鲁棒方法以及基于最小中位数的方法。最后根据单应矩阵进行透视变换和原图覆盖即可。

#### 关键点检测

使用SIFT，SURF，ORB等鲁棒特征描述子

#### 特征匹配

使用BruteForce(BFMatcher)，KNN，K-D Tree等匹配算法

#### 单应性估计

使用基于所有对的常规方法，基于RANSAC的鲁棒方法，基于最小中位数的方法等

### 代码实现

------

代码的任务是先对彩色的全局图和黑白的局部图进行特征点匹配，仿射变换后，将黑白色的局部图片覆盖到原图相应区域，以完成“A Look Into the Past”的效果。

读入图片后，先创建用于实现ORB算法的类的实例orb，而后利用类方法“detectAndCompute”分别检测出两幅图片各自的关键点并计算对应的描述符。

```python
orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
```

再创建用于实现BF算法的类的实例bf，用类方法“match”对上文所得的ORB描述符进行蛮力匹配。

```python
bf = cv.BFMatcher.create()

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
goodPoints =[]
for i in range(len(matches)-1):
    if matches[i].distance < GOOD_POINTS_LIMITED * matches[i+1].distance:
        goodPoints.append(matches[i])
```

随后依据距离法则筛选出可用于特征匹配的“好点”，根据这些好点的对应关系，即可利用“findHomography”函数计算出两张图片对应的最优仿射变换矩阵。

```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)

M, mask = cv.findHomography(dst_pts,src_pts, cv.RHO)
```

此时再将局部图片仿射变换并转化为与全局图同等大小。此时要做的只需将仿射后的局部图中非零的部分覆盖在原图片中即可。

利用“inRange”函数提取出图片掩膜，随后使用“bitwise_not”函数将该掩膜取反，再利用“bitwise_and”函数将取反后的掩膜作用在原图片上，即可将原图片待被局部图覆盖的部分置零。

最后将两幅图片简单相加，即可得到最终的结果。

```python
h1,w1,p1 = img2.shape
h2,w2,p2 = img1.shape

h = np.maximum(h1,h2)
w = np.maximum(w1,w2)

_movedis = int(np.maximum(dst_pts[0][0][0],src_pts[0][0][0]))
imageTransform = cv.warpPerspective(img2,M,(w1+w2-_movedis,h))
mask = cv.inRange(imageTransform,np.array([1,1,1]),np.array([255,255,255]))
mask = cv.bitwise_not(mask)

M1 = np.float32([[1, 0, 0], [0, 1, 0]])
h_1,w_1,p = img1.shape
dst1 = cv.warpAffine(img1,M1,(w1+w2-_movedis, h))
dst1 = cv.bitwise_and(dst1,dst1,mask=mask)

dst = cv.add(dst1,imageTransform)
dst_no = np.copy(dst)
dst_target = np.maximum(dst1,imageTransform)
```

### 效果展示

------



- **Input 1:**

![img1](img\img1.jpg)

- **Input2:**

  ![img2](img\img2.jpg)

- **Output:**

  ![img3](img\output.jpg)

### 运行说明

------



```bash
pip install opencv-python
pip install numpy
pip install matplotlib
python .\cv.py -i1 .\img\img1.jpg -i2 .\img\img2.jpg -o .\img\output.jpg
```

