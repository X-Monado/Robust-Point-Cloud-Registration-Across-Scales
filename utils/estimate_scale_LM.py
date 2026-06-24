import open3d as o3d
import numpy as np

def get_pointcloud(point_x,point_y):

    # point_x = np.asarray(pcd2.points)
    # point_y = np.asarray(pcd1.points)

    # point_x[:,0] = point_y[:,0]*5
    # point_x[:,1]  =point_y[:,1]*3
    # point_x[:,2]  =point_y[:,2]*2.3
    
    #添加噪声
    # noise = np.random.normal(0,0.1,point_x.shape)
    # point_x = point_x+noise
    
    
    # #添加错误匹配点
    # for i in range(int(point_x.shape[0]*0.02)):
    #     # print('i:',int(point_x.shape[0]*0.01))
    #     point_x[i,0]  = np.random.randint(10)
    #     point_x[i,1]  = np.random.randint(10)
    #     point_x[i,2]  = np.random.randint(10)

    # point_y = np.asarray([[1,1,1],[2,2,2],[1,2,3],[1,0,2]])
    # # # # point_x = np.asarray([[1,1,1],[2,2,2],[1,2,3],[1,0,2]])
    # point_x = np.asarray([[2,3,3],[4,6,6],[2,6,9],[2,0,6]])
    # 创建p矩阵
    for i in range(len(point_x)):
        if i+1>=len(point_x):
            x = point_x[i,0]-point_x[0,0]
            y = point_x[i,1]-point_x[0,1]
            z = point_x[i,2]-point_x[0,2]
        else:
            x = point_x[i,0]-point_x[i+1,0]
            y = point_x[i,1]-point_x[i+1,1]
            z = point_x[i,2]-point_x[i+1,2]
    #     print(x,y,z)
        #按照列追加
        x = x**2
        y = y**2
        z = z**2
        if i==0:
            point_p = np.array([x,y,z])
        else:
            point_p = np.column_stack((point_p, [x,y,z]))
    # print('p矩阵[px-px-1]:',point_p)
    # point_t= np.asarray([])
    # 创建p矩阵
    for i in range(len(point_y)):
        if i+1>=len(point_y):
            x = point_y[i,0]-point_y[0,0]
            y = point_y[i,1]-point_y[0,1]
            z = point_y[i,2]-point_y[0,2]
        else:
            x = point_y[i,0]-point_y[i+1,0]
            y = point_y[i,1]-point_y[i+1,1]
            z = point_y[i,2]-point_y[i+1,2]
        x = x**2
        y = y**2
        z = z**2
    #     print(x,y,z)
        #按照列追加
        if i==0:
            point_t = np.array([x,y,z])
        else:
            point_t = np.column_stack((point_t, [x,y,z]))
    # print('t矩阵[Tx-T(x-1)]:',point_t)
    #初始化Scale矩阵
    S = np.asarray([1,1,1]).reshape(1,3)
    T = point_t
    P = point_p

    point_x =point_x.T
    point_y =point_y.T
    return point_x,point_y,P,T


#梯度
def cal_jacobian(S,P,T):
    
    ji=np.array([0,0,0])
    A=[S[0][0],S[1][0],S[2][0]]
    S = A
    print('S:',S)
    #判断梯度是否还增加
    #Scale转换为对角矩阵,方便计算
    # scale=abs(np.diag(S))
    jacobian = []
    for n in range(T.shape[1]):
        ti=np.array(T[:,n]).T
        pi=np.array(P[:,n]).T
        #这个地方梯度是累加


        A = pi[0]+pi[1]+pi[2] - S[0]*S[0]*ti[0]- S[1]*S[1]*ti[1]- S[2]*S[2]*ti[2]
        # 添加最近点距离最近
        
        x_rate = -4*S[0]* ti[0]*A #-2*S[0]*ti[0]
        y_rate = -4*S[1]* ti[1]*A #-2*S[1]*ti[1]
        z_rate = -4*S[2]* ti[2]*A #-2*S[2]*ti[2]


        # if flag_x:
        #     x_rate = 0
        # if flag_y:
        #     y_rate = 0
        # if flag_z:
        #     z_rate = 0
        ji = [x_rate,y_rate,z_rate]
        # print('ji:',ji)
        jacobian.append([x_rate,y_rate,z_rate])
    # if x_error<0:
    #     ji[]
    # print('梯度2：',ji/T.shape[1])
    #梯度裁剪，解决梯度爆炸
    # for i in range(len(ji)):
    #     if ji[i]> 300*1000:
    #         ji[i] = 300*1000
    #     elif ji[i] < -300*1000:
    #         ji[i] = -300*1000
    # print('梯度解决后：',jacobian)
    
    return np.array(jacobian)#ji/T.shape[1],x_error,y_error,z_error
#损失函数
def compute_cost(S,P,T,point_x,point_y):
    cost = 0
    for n in range(P.shape[1]):
        ti=np.array([T[:,n]]).T
        pi=np.array([P[:,n]]).T
        p_or = point_x[:,n]
        t_or = point_y[:,n]

        A = (pi[0]+pi[1]+pi[2] - S[0]*S[0]*ti[0]- S[1]*S[1]*ti[1]- S[2]*S[2]*ti[2])**2
        
        cost = cost + A
        return cost/P.shape[1]

#calculating residual, whose shape is (num_data,1)
def cal_residual(S,P,T):
    #计算误差
    residual =[]
    for n in range(P.shape[1]):
        ti=np.array([T[:,n]]).T
        pi=np.array([P[:,n]]).T

        A = (pi[0]+pi[1]+pi[2] - S[0]*S[0]*ti[0]- S[1]*S[1]*ti[1]- S[2]*S[2]*ti[2])**2
        
        # cost = cost + A
        residual.append(A)
    return np.array(residual)#cost/P.shape[1]

#get the init u, using equation u=tao*max(Aii)
def get_init_u(A,tao):
    m = np.shape(A)[0]
    Aii = []
    for i in range(0,m):
        Aii.append(A[i,i])
    u = tao*max(Aii)
    return u


def newton(S,P,T,P_point,T_point):
    global flag_x
    global flag_y
    global flag_z
    delta =1
    print('初始点为:')
    print(S,'\n')
    i = 1
    imax = 40
    tao = 10**-3
    threshold_stop = 10**-25
    threshold_step = 10**-25
    threshold_residual = 10**-25
    residual_memory = []

    params = S#np.array([[2],[2],[2]])
    lamda = 0.1
    # alpha_list = [0.00001,0.00001,0.00001]#0.00005
    k=0
    J = cal_jacobian(S,P,T)
    # print('J:',J.shape)
    #根据阻尼系数lamda混合得到H矩阵
    A = np.dot(J.T,J)
    # H_lm = H+lamda*np.eye(3,3)
    # H_lm=H+(lamda*eye(Nparams,Nparams));
    # % 计算步长dp，并根据步长计算新的可能的\参数估计值
    # dp=inv(H_lm)*(J'*d(:));
    residual = cal_residual(S,P,T)
    # print('residual:',residual)
    g = np.dot(J.T,residual)
    # dp = np.dot(np.linalg.inv(H_lm) ,)
        
    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop)#set the init stop
    u = get_init_u(A,tao)#set the init u
    # u =0.1
    # print('u:',u)
    v = 1#set the init v=2
    while((not stop) and (k<imax)):
        
        k+=1
        while(1):
            Hessian_LM = A + u*np.eye(3)#calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)#calculating the update step
            # print('step:',step)
            if(np.linalg.norm(step) <= threshold_step):
                stop = True
            else:
                
                new_params = params - step#update params using step
                # print('params:',params)
                # print('step:',step)
                # print('new_params:',new_params)
                # print('k:',k)
                # print('x方向：',P_point[:,0])
                # print('x方向：',T_point[:,0])
                # print('x方向：',P_point[:,0] -new_params[0]*T_point[:,0])
                if np.mean(abs(P_point[:,0] -new_params[0]*T_point[:,0])) <0.00000001:
                    print('x方向最佳：',new_params[0])
                if np.mean(abs(P_point[:,1] -new_params[1]*T_point[:,1])) <0.00000001:
                    print('y方向最佳：',new_params[1])
                if np.mean(abs(P_point[:,2] -new_params[2]*T_point[:,2])) <0.00000001:
                    print('z方向最佳：',new_params[2])
                
                new_residual = cal_residual(new_params,P,T)
                rou = (np.linalg.norm(residual)**2 - np.linalg.norm(new_residual)**2) / (step.T.dot(u*step+g))
                if rou > 0:
                    params = new_params
                    residual =   new_residual
                    residual_memory.append(np.linalg.norm(residual)**2)
                        #print (np.linalg.norm(new_residual)**2)
                        # Jacobian = cal_Jacobian(params,input_data)#recalculating Jacobian matrix with new params
                    Jacobian = cal_jacobian(params,P,T)
                    # print('Jacobian:',Jacobian.shape)
                    A = Jacobian.T.dot(Jacobian)#recalculating A
                    g = Jacobian.T.dot(residual)#recalculating gradient g
                    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop) or (np.linalg.norm(residual)**2 <= threshold_residual)
                    u = u*max(1/3,1-(2*rou-1)**3)
                    v = 2
                else:
                    u = u*v
                    v = 2*v
            if(rou > 0 or stop):
                break
        
    return params

def test(point_x,point_y):
    #读取点云文件
    # pcd1 =o3d.io.read_point_cloud(r'C:\Users\peog\pcd\toilet\toilet_0001.ply')
    # pcd1 =o3d.io.read_point_cloud('C:/Users/peog/Desktop/toilet/bunny_key_points.pcd')
    # pcd1 =o3d.io.read_point_cloud(r'C:\Users\peog\Desktop\scale_icp\door_0001.ply')
    # pcd1 =o3d.io.read_point_cloud(r'C:\Users\peog\Desktop\pcd\2022-09-22 13-27-54.076202.pcd')

    # pcd2 =o3d.io.read_point_cloud(r'C:\Users\peog\Desktop\pcd\2022-09-22 13-27-54.076202.pcd')
    # pcd2 =o3d.io.read_point_cloud(r'C:\Users\peog\Desktop\scale_icp\door_0001.ply')
    # pcd2 =o3d.io.read_point_cloud(r'C:/Users/peog/Desktop/toilet/bunny_5_key_points.pcd')
    # pcd2 =o3d.io.read_point_cloud(r'C:\Users\peog\pcd\toilet\toilet_0001.ply')
    # point_x = np.asarray(pcd1.points)
    # point_y = np.asarray(pcd2.points)

    point_x,point_y,P,T = get_pointcloud(point_y,point_x)
    print('point_x:',point_x)
    flag_x =False
    flag_y =False
    flag_z =False

    #1.8 3.1 2.4
    S = np.array([[1],[1],[1]])
    S = newton(S,P,T,point_x.T,point_y.T)
    print('比例参数为：',S,'\n')
    return S
    