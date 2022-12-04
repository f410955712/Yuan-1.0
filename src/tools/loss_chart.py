import matplotlib.pyplot as plt
import os
"""损失函数折线图"""

def draw(path, save_dir):
    f = open(path)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法，一次读取一行
    count=0
    index=0
    X=[]
    Y=[]

    while line:
        if (count%200)==0:
            index+=1
            X.append(index)
            Y.append(float(line[0:len(line)-2]))
        count += 1
        line = f.readline()

    f.close()
    fig = plt.figure(figsize=(20, 4), dpi=600)
    plt.plot(X, Y)
    plt.plot()
    plt.show()

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "loss.png"))
