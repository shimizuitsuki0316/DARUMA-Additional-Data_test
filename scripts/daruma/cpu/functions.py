import numpy as np
import sys, os
from datetime import datetime
import time
from struct import unpack


dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)



def load_AAindex():
    key = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X"]

    features_path = os.path.join(os.path.dirname(__file__), 'data', 'AAindex553-Normal-X0.feature')
    with open(features_path,"rb") as f:
        binary_data = f.read()
    data = unpack(">11613f",binary_data)

    Dic = {}
    for a,i in zip(key,range(0,11613,553)):
        Dic[a] = data[i:i+553]
    
    return Dic

def smoothing(window,prob):
    L = len(prob)
    pool = window//2
    if L >= window:
        new_prob = list(map(mean,[prob[:i] for i in range(pool+1,window)]+[prob[i:i+window] for i in range(L-pool)]))
    else:
        new_prob = list(map(mean,[prob[max(0,i):i+window] for i in range(-pool,L-pool)]))
    return new_prob

def mean(l):
    return sum(l)/len(l)

def classify(prob,threshold=0.5):
    return np.where(np.array(prob)>=threshold,1,0).tolist()

def remove_short_idr(pred):
    pred = "".join(map(str,pred))
    pred = pred.replace("10","1-0").replace("01","0-1")
    return list(map(int,list("".join([_replace(region,"1","0") for region in pred.split("-")]))))

def remove_short_stru(pred):
    pred = "".join(map(str,pred))
    pred = pred.replace("10","1-0").replace("01","0-1")
    return list(map(int,list("".join([_replace(region,"0","1") for region in pred.split("-")]))))

def _replace(region,label1,label2):
    l = len(region)
    if l < 10:
        return region.replace(label1,label2)
    else:
        return region


##### layer

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def D1im2col(x, filter_l):
    L, dim = x.shape
    out_h = L - filter_l + 1

    col = np.zeros((filter_l, out_h, dim),dtype="float32")

    for j in range(filter_l):
        col[j, :, :] = x[j:j+out_h, :]

    col = col.transpose(1, 0, 2).reshape(out_h, -1)
    return col


class D1Conv:
    def __init__(self,W,b,active):
        FW,_,FN = W.shape #(FW,dim,FN)

        self.FW = FW
        self.FN = FN
        self.W = W.reshape(-1,FN)
        self.b = b
        self.active = active
        
    def forward(self,x):
        
        L,_ = x.shape
        OL = L - self.FW + 1

        col = D1im2col(x,self.FW)
        x = np.dot(col,self.W) + self.b
        x = x.reshape(OL,self.FN)

        return self.active(x)

class Affine:
    def __init__(self,W,b,active):
        self.W = W
        self.b = b
        self.active = active
        
    def forward(self,x):
        return self.active(np.dot(x,self.W) + self.b)



##### result file

class TimingsManager:
    def __init__(self,output_path):
        output_path = output_path.rsplit(".", 1)[0]

        self.path = f"{output_path}_timings.csv"

        current_time = datetime.now()
        formatted_time = current_time.strftime("%a %b %d %H:%M:%S CET %Y")
        with open(self.path,"w") as f:
            f.write(f"# Running DARUMA, started {formatted_time}\n")
            f.write("sequence,milliseconds\n")

    def start(self):
        self.start_time = time.time()

    def end(self,ac):
        execution_time = int((time.time() - self.start_time) * 1000)
        with open(self.path,"a") as f:
            f.write(f"{ac},{execution_time}\n")

    def append_write(self,out):
        with open(self.path,"a") as f:
            f.write(out)


class ResultFileManager:
    def __init__(self,output_path):
        self.path = output_path

    def append_write(self):
        pass

    def close_manager(self):
        pass


class format_residue_single(ResultFileManager):
    def __init__(self,output_path="."):
        if not output_path.endswith("/"):
            output_path += "/"
        
        self.path = "disorder/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def append_write(self,ac,seq,pred_prob,pred_class):
        with open(f"{self.path}{ac}.daruma","w") as f:
            out = "\n".join([f"{i} {res} {p:.3f} {c}" for i,res,p,c in zip(range(1,len(seq)+1),seq,pred_prob,pred_class)])
            f.write(f">{ac}\n{out}\n")


class format_residue_multi(ResultFileManager):
    def __init__(self,output_path):
        super().__init__(output_path)

        with open(self.path,"w") as f:
            f.write("")

    def append_write(self,ac,seq,pred_prob,pred_class):
        with open(self.path,"a") as f:
            out = "\n".join([f"{i} {res} {p:.3f} {c}" for i,res,p,c in zip(range(1,len(seq)+1),seq,pred_prob,pred_class)])
            f.write(f">{ac}\n{out}\n")


class format_4lines(ResultFileManager):
    def __init__(self,output_path):
        super().__init__(output_path)

        with open(self.path,"w") as f:
            f.write("")
    
    def append_write(self,ac,seq,pred_prob,pred_class):
        with open(self.path,"a") as f:
           f.write(f'>{ac}\n{",".join(seq)}\n{",".join(map(lambda x:f"{x:.3f}",pred_prob))}\n{",".join(map(str,pred_class))}\n')


class format_fast(ResultFileManager):
    def __init__(self, output_path):
        super().__init__(output_path)
        self.out = ""

    def append_write(self,ac,seq,pred_prob,pred_class):
        self.out += f'>{ac}\n{",".join(seq)}\n{",".join(map(lambda x:f"{x:.3f}",pred_prob))}\n{",".join(map(str,pred_class))}\n'

    def close_manager(self):
        with open(self.path,"w") as f:
            f.write(self.out)