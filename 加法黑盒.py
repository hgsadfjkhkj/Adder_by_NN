import torch


def num_to_tensor(n,d):
    k=0
    list=[]
    binary = bin(n)[2:]
    print(n,binary)
    binary = binary[::-1]
    for b in binary:

        t=torch.Tensor([0,int(b),k,d,0])
        list.append(t)
        k=k+1
    return list

def module_feature_extractor(t,d,n,i):#在d维向量中，若第i位是n，则将其转换成1；否则转换成0。其它位都置0
    l1 = torch.nn.Linear(d, d+1, True)
    l1.weight.data = torch.zeros([d+1,d])
    l1.bias.data = torch.zeros([d+1])
    l1.weight.data[i][i] = 1
    l1.weight.data[d][i]=-1
    l1.bias.data[i]=-n
    l1.bias.data[d] = +n
    l2 = torch.nn.Linear(d+1, d, True)
    l2.weight.data = torch.zeros([d,d+1])
    l2.bias.data = torch.zeros([5])
    for j in range(0,d):
        l2.weight.data[j][j] = 1
    l2.weight.data[i][d]=1
    l2.bias.data[i]=0
    l3 = torch.nn.Linear(d, d, True)
    l3.weight.data = torch.eye(d)
    l3.bias.data = torch.zeros([5])
    l3.weight.data[i][i] = -1
    l3.bias.data[i] = 0.5
    l4 = torch.nn.Linear(d, d, True)
    l4.weight.data = torch.eye(d)
    l4.bias.data = torch.zeros([5])
    l4.weight.data[i][i] = 2
    l4.bias.data[i] = 0

    x=l1(t)
    x=torch.relu(x)
    x = l2(x)
    x = torch.relu(x)
    x = l3(x)
    x = torch.relu(x)
    x = l4(x)
    x = torch.relu(x)
    return x



def module1(t,sourse):
    l1=torch.nn.Linear(5,1,True)
    l1.weight.data=torch.Tensor([[0,0,2,0,0]])
    l1.bias.data=torch.Tensor([1])
    l2 = torch.nn.Linear(1, 1, True)
    l2.weight.data = torch.Tensor([[-1]])
    l2.bias.data = torch.Tensor([2])
    l3 = torch.nn.Linear(5, 2, True)
    l3.weight.data = torch.Tensor([[0,1,0,0,0],[0,1,0,0,0]])
    l3.bias.data = torch.Tensor([-1,0])
    l4 = torch.nn.Linear(2, 5, True)
    l4.weight.data = torch.Tensor([[1, 0], [-2, 1],[0,0],[0,0],[0,0]])
    l4.bias.data = torch.Tensor([0, 0,0,0,0])
    q=torch.Tensor([0,0,0,0,0])
    for s in sourse:
        k=(t-s).abs()
        k=l1(k)
        k=torch.relu(k)
        k=l2(k)
        k = torch.relu(k)
        q=q+k*s
    v=l3(q)
    v=torch.relu(v)
    v=l4(v)
    v=torch.relu(v)
    return v

def module2(t,sourse):
    l1=torch.nn.Linear(5,5,True)
    l1.weight.data=torch.Tensor([[1,0,0,0,0],[0,1,0,0,0],[0,0,-1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    l1.bias.data=torch.Tensor([0,0,1.5,0,0])
    l2 = torch.nn.Linear(1, 1, True)
    l2.weight.data = torch.Tensor([[-1]])
    l2.bias.data = torch.Tensor([2])
    l3 = torch.nn.Linear(5, 2, True)
    l3.weight.data = torch.Tensor([[0,1,0,0,0],[0,1,0,0,0]])
    l3.bias.data = torch.Tensor([-1,0])
    l4 = torch.nn.Linear(2, 5, True)
    l4.weight.data = torch.Tensor([[1, 0], [-2, 1],[0,0],[0,0],[0,0]])
    l4.bias.data = torch.Tensor([0, 0,0,0,0])
    l5 = torch.nn.Linear(5, 5, True)
    l5.weight.data = torch.zeros([5,5])
    l5.weight.data[4][4]=-1
    l5.bias.data = torch.Tensor([0, 0, 0, 0, 1])
    l6 = torch.nn.Linear(5, 5, True)
    l6.weight.data = torch.zeros([5, 5])
    l6.weight.data[3][2] = 1
    l6.weight.data[4][4] = 1
    l6.bias.data = torch.Tensor([0, 0, 0, 0, 0])
    l7 = torch.nn.Linear(5, 5, True)
    l7.weight.data = torch.Tensor(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, -100], [0, 0, 0, 1, -100], [0, 0, 0, 0, 0]])
    l7.bias.data = torch.Tensor([0, 0, 0, 0, 0])
    l8 = torch.nn.Linear(5, 5, True)
    l8.weight.data = torch.Tensor(
        [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    l8.bias.data = torch.Tensor([0, 0, 0, 0, 0])
    l9 = torch.nn.Linear(5, 2, True)
    l9.weight.data = torch.Tensor([[1,1,0,0,0],[1,1,0,0,0]])
    l9.bias.data = torch.Tensor([-1, 0])
    l10 = torch.nn.Linear(2, 5, True)
    l10.weight.data = torch.Tensor([[1, 0], [-2, 1],[0,0],[0,0],[0,0]])
    l10.bias.data = torch.Tensor([0, 0,0,0,0])
    q=torch.Tensor([0,0,0,0,0])
    for s in sourse:
        k=(t-s)
        k3=module_feature_extractor(k,5,0,4)
        k1=module_feature_extractor(k,5,1,2)
        k2 = module_feature_extractor(k, 5, 0, 2)
        k3=l5(k3)
        k1=l6(k1)
        k=k1+k2+k3
        k=l7(k)
        k=torch.relu(k)
        k = l8(k)
        k = torch.relu(k)
        q=q+k*s
    v=l9(q)
    v=torch.relu(v)
    v = l10(v)
    v = torch.relu(v)
    return v

class model_rnncell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=torch.nn.Linear(6,1)
        self.l1.weight.data=torch.Tensor([[1,0,0,0,0,1]])
        self.l1.bias.data=torch.Tensor([0])

    def forward(self,input,hidden):
        x=torch.cat((input,hidden),dim=0)
        x=self.l1(x)
        x=torch.relu(x)
        return x


if __name__ == '__main__':
    n1=114
    n2=191
    pool=[]
    pools=[]
    pool=pool+num_to_tensor(n1,1)
    l1=len(pool)
    pools=pools+pool
    pool=[]
    pool = pool + num_to_tensor(n2, -1)
    l2=len(pool)
    pools = pools + pool
    pool=[]
    for i in range(0,max(l1,l2)+1):
        t = torch.Tensor([0, 0, i, 0, 1])
        pool.append(t)
    flag=True
    i=0

    l1 = torch.nn.Linear(5, 5, True)
    l1.weight.data = torch.Tensor([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 1]])
    l1.bias.data = torch.Tensor([0,0,0,0,1])

    model=model_rnncell()
    while(flag):
        i=i+1
        print("第",i,"次推理")
        target=[t for t in pool]
        pools=pools+pool
        sourse=[t for t in pools]
        pool=[]
        for t in target:
            v1 = module1(t, sourse)
            v2 = module2(t, sourse)
            x = l1(t)
            k=module_feature_extractor(t,5,1,4)
            k=k.sum()
            x=x+v1*k+v2*(1-k)
            pool.append(x)
        flag=False
        hidden=torch.Tensor([0])
        for t in pool:
            hidden = model(t, hidden)
        if int(hidden[0])>0.8:
            flag = True
    result=[]
    for t in pool:
        result.append(int(t[1].item()))
    result=result[::-1]
    n=0
    for i in result:
        n=n+i
        n=n*2
    n=int(n/2)
    print(n1,n2,n1+n2,n)

