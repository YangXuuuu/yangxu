import tushare as ts
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers

solvers.options['show_progress'] = False  # 关闭cvxopt日志


def optimal_portfolio(returns,A_inv):
    n = len(returns)
    returns = np.asmatrix(returns)
    s=np.cov(returns)

    N = 200
    lambdas = [10 ** (5 * t / N - 2.0) for t in range(N)]  # 原参考代码为mus, 此处为了不引起歧义改名

    # 转化为cvxopt matrices
    S = opt.matrix(np.cov(returns))
    ret_vec = opt.matrix(np.mean(returns, axis=1))  # 收益向量
    rt=np.mean(returns, axis=1)
    # 代数法参数定义
    a0=np.dot(np.dot(ret_vec.T,np.linalg.inv(s)),ret_vec)[0,0]
    b0=np.dot(np.dot(ret_vec.T,np.linalg.inv(s)),np.tile(1,(n,1)))[0,0]
    c0=np.dot(np.dot(np.tile(1,(n,1)).T,np.linalg.inv(s)),np.tile(1,(n,1)))[0,0]

    # 约束条件
    G = -opt.matrix(np.eye(n))  # G*x <= h 用于表示约束条件 w0, w1, ... wn >= 0   G: 值为-1的对角矩阵, x: 列向量 h: 列向量0
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))  # A*x = b 用于表示约束条件 w1 + w2 + ... + wn = 1 dim(A): (1, n), dim(b): 标量1
    b = opt.matrix(1.0)


    # 使用凸优化计算有效前沿
    portfolios = [solvers.qp(lbd * S, -ret_vec, G, h, A, b)['x']
                  for lbd in lambdas]

    ## 计算有效前沿的收益率和风险
    returns = [blas.dot(ret_vec, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    #效用边界散点图
    fig=plt.figure(3)
    plt.scatter(risks,returns,c='blue')
    plt.xlabel('risks')
    plt.ylabel('returns')
    plt.show()

    # 优化法计算投资组合曲线最高点风险以及收益
    wt_high= solvers.qp(opt.matrix(0 * S), -ret_vec, G, h, A, b)['x']
    return_high=np.dot(np.asmatrix(wt_high).T, np.asmatrix(ret_vec))[0,0]
    risk_high=np.sqrt(np.dot(np.dot(np.asmatrix(wt_high).T, s), np.asmatrix(wt_high)))[0,0]
    print(wt_high,return_high,risk_high,'opthigh')

    # # 代数法计算对应权重(不符合卖空限制，存在偏差）
    # tool=np.array(([a0,b0],[b0,c0]))
    # lmd1=2*np.dot(np.linalg.inv(tool),np.vstack((return_high,np.tile(1.0,(1,1)))))[0,0]
    # lmd2=2*np.dot(np.linalg.inv(tool),np.vstack((return_high,np.tile(1.0,(1,1)))))[1,0]
    # wt_high_cal=(np.dot(np.linalg.inv(s),lmd2*np.tile(1.0,(n,1)))+np.dot(np.linalg.inv(s),lmd1*ret_vec))*(1/2)
    # print(wt_high_cal,'wt_high_cal')
    #
    # # 代数法计算收益最高点（全投收益最高股票）
    # return_high_cal=np.dot(rt.T,wt_high_cal)[0,0]
    # risk_high_cal=np.sqrt(np.dot(wt_high_cal.T,np.dot(s,wt_high_cal)))[0,0]
    # print(return_high_cal,risk_high_cal,'calhigh')

    # 通过优化法计算有效前沿最低点权重，收益，风险
    wt_low=solvers.qp(opt.matrix(99999999999999* S), -ret_vec, G, h, A, b)['x']
    return_low=np.dot(np.asmatrix(wt_low).T, np.asmatrix(ret_vec))[0,0]
    risk_low = np.sqrt(np.dot(np.dot(np.asmatrix(wt_low).T, s), np.asmatrix(wt_low)))[0, 0]
    print(wt_low,return_low,risk_low,'optlow')

    # # 代数法求有效前沿最低点收益及风险(不符合卖空限制，存在偏差）
    # return_low_cal=b0/c0
    # risk_low_cal=np.sqrt(1/c0)
    # print(return_low_cal,risk_low_cal,'callow')

    # # 代数法求最低点权重(不符合卖空限制，存在偏差）
    # tool=np.array(([a0,b0],[b0,c0]))
    # lmd1=2*np.dot(np.linalg.inv(tool),np.vstack((return_low,np.tile(1.0,(1,1)))))[0,0]
    # lmd2=2*np.dot(np.linalg.inv(tool),np.vstack((return_low,np.tile(1.0,(1,1)))))[1,0]
    # wt_low_cal=(np.dot(np.linalg.inv(s),lmd2*np.tile(1.0,(n,1)))+np.dot(np.linalg.inv(s),lmd1*ret_vec))*(1/2)
    # print(wt_low_cal,'wt_low_cal')

    # m1为有效前沿方程曲线(拟合法求，当负收益股票权重为0时，拟合出的二次曲线与规划出的有效前沿近似）
    m1 = np.polyfit(returns, risks, 2)

    ## 计算最佳风险资产点
    # 设置参数
    a1=m1[0]
    b1=m1[1]
    c1=m1[2]
    d1=0.04/12 #无风险利率
    # 求有效前沿与资本配置线的切线
    delta=(2*b1+4*a1*d1)**2-4*(b1**2-4*a1*c1)
    k1=(2*b1+4*a1*d1-np.sqrt(delta))/2
    k2=(2*b1+4*a1*d1+np.sqrt(delta))/2
    if k1>0:
        k=k1
    else:
        k=k2
    returnb=(k-b1)/(2*a1)
    riskb=k*(returnb-d1)
    kn=(returnb-d1)/riskb
    if (returnb<=return_high and returnb>=return_low):
        # 计算资本配置线与投资组合切点对应权重
        a2 = np.tile(1.0, (1, n))
        a2 = np.vstack((a2, ret_vec.T))
        # 优化法与代数法有区别是因为优化法认为不能卖空
        A2 = opt.matrix(a2)
        b2 = opt.matrix([1.0, returnb])
        wt2 = solvers.qp(opt.matrix(S), opt.matrix(ret_vec), G, h, A2, b2)['x']
        print(np.dot(rt.T, wt2)[0,0], 'returnmod')
        print(np.sqrt((np.dot(np.dot(wt2.T, s) ,wt2)))[0,0], 'riskmod')
        # 计算资本配置线和效用曲线的切点
        A_ef=A_inv
        # U=(kn**2+2*A_ef*d1)/(2*A_ef)
        # riskp=kn/A_ef
        # returnp=kn*riskp+d1
        ratio1=(returnb-d1)/(A_ef*(riskb**2))
        riskp=ratio1*riskb
        returnp=ratio1*returnb+(1-ratio1)*d1
        U=returnp-0.5*A_ef*(riskp**2)
        ## 作图
        X1 = np.linspace(return_low, return_high, 30)
        # 代数法方程曲线
        #Y1 = np.sqrt((c0 * (X1 ** 2) - 2 * b0 * X1 + a0) / (a0 * c0 - b0 ** 2))

        ##优化法拟合方程曲线
        Y1 = a1 * (X1 ** 2) + b1 * X1 + c1  #还未引入无风险资产,横纵坐标是风险资产投资组合收益和风险
        # 资本配置线
        X2 = np.linspace(d1, return_high, 30)
        Y2= k*(X2-d1)
        #画图
        fig=plt.figure(0)
        plt.plot(Y1,X1,c='blue',label='Effective Frontier')
        plt.plot(Y2,X2,c='red',label='Capital Allocation Line')
        plt.scatter(riskb,returnb,c="black",label='B')
        plt.xlabel('risk')
        plt.ylabel('return')
        plt.ylim(d1, return_high)
        plt.legend(loc=0,edgecolor='black',facecolor='white',shadow='True',fontsize=10)
        plt.show()

        ##资本配置线和效用曲线
        X3=np.linspace(0,risk_high*ratio1*2,30)
        Y3=0.5*A_ef*(X3**2)+U
        X4=np.linspace(0,risk_high,30)
        Y4=kn*X4+d1
        #画图
        fig=plt.figure(1)
        plt.plot(X3,Y3,c='blue',label='Utility Curve')
        plt.plot(X4,Y4,c='red',label='Capital Allocation Line')
        plt.scatter(riskp,returnp,c="brown",label='A')
        plt.xlabel('risk')
        plt.ylabel('return')
        plt.ylim(0,max(returnp*2,return_high*2))
        plt.legend(loc=0,edgecolor='black',facecolor='white',shadow='True',fontsize=10)
        plt.show()

        ##三线合一
        Y5=a1*(Y4**2)+b1*Y4+c1
        #画图
        fig=plt.figure(2)
        plt.plot(X3,Y3,c='blue',label='Utility Curve')
        plt.plot(X4, Y4,c='red',label='Capital Allocation Line')
        plt.plot(Y5, Y4,c='green',label='Effective Frontier')
        plt.scatter(riskb,returnb,c="black",label='B')
        plt.scatter(riskp,returnp,c="brown",label='A')
        plt.xlabel('risk')
        plt.ylabel('return')
        plt.ylim(0,max(returnp*2,return_high*2))
        plt.legend(loc=0,edgecolor='black',facecolor='white',shadow='True',fontsize=10)
        plt.show()
    else:
        ratio1=0
        returnp=0
        riskp=0
        wt2=np.array([])
        print("找不到切点")

    # #代数式法计算对应权重
    # # tool=np.array(([a0,b0],[b0,c0]))
    # # lmd1=2*np.dot(np.linalg.inv(tool),np.vstack((returnb,np.tile(1.0,(1,1)))))[0,0]
    # # lmd2=2*np.dot(np.linalg.inv(tool),np.vstack((returnb,np.tile(1.0,(1,1)))))[1,0]
    # # wt2=(np.dot(np.linalg.inv(s),lmd2*np.tile(1.0,(n,1)))+np.dot(np.linalg.inv(s),lmd1*ret_vec))*(1/2)
    # # print(wt2)
    return [returnp,riskp], [returnb, riskb],[ratio1,(np.array(wt2)).tolist()]


if __name__ == '__main__':
    file='test'+'.txt'
    df = DataFrame()
    dff=DataFrame()
    ts.set_token("80971bcc1578fc62b852426e78bfaf94effb8d69598ac6467ca41d9e")
    with open(file, 'r', encoding='utf-8') as f:
        potfolios =f.readlines()
    for i in potfolios:
        data = ts.pro_bar(ts_code=i.rstrip('\n'), start_date='20181201', end_date='20211231', freq='M', adj='qfq')
        # dff=dff.append(DataFrame(data))
        # dff.to_excel('result.xlsx')
        df = pd.concat([df,DataFrame(data['close']).pct_change(-1).fillna(0)], axis=1, ignore_index=True)
        df.drop([len(df) - 1], inplace=True)
    test=optimal_portfolio(df.T,4)
    st=''
    for index in range(len(potfolios)):
        st=st+potfolios[index].rstrip('\n')+(':%0.2f' % (test[2][1][index][0]*100))+'%'+'\n'
    print('无风险资产和风险资产最优组合的收益为%0.2f%%，风险为%0.2f。风险资产最优组合的收益为%0.2f%%，风险为%0.2f。投资策略为，应当买入%0.2f%%的风险资产,买入%0.2f%%的无风险资产。其中，风险资产的权重分别为\n' % (test[0][0]*100,test[0][1],test[1][0]*100,test[1][1],test[2][0]*100,(100-test[2][0]*100))+st)

