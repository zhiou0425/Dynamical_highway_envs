import gymnasium
import highway_env
import numpy as np
import matplotlib.pyplot as plt
# from sac_att import SAC_Agent


# m=1200,a=3*0.45，b=3*0.55,wid=1.6,Rw=0.35,g=9.81,hg=0.54,Iz=800



env=gymnasium.make('doublelane-v0',render_mode='human')#simulate frequency=20
state=env.reset()[0].reshape(-1)

# breakpoint()
beta_list=[]
yawrate_list=[]
vx_list=[]
vy_list=[]
rel_tra_x=[]
rel_tra_y=[]
heading_list=[]
action_list=[]
alpha_list=[]
vx=5
vy=0
yawrate=0
beta=0
heading=0
str_f=0
str_r=0
a_x=0
a_y=0
dyawrate=0
flag=0
for i in range(600):
    if i>=30 and i<80:
        str=(np.pi/4)*np.sin(0.04*np.pi*(i-30))
    elif i>=80 and i<=130:
        str=-(np.pi/4)*np.sin(0.04*np.pi*(i-80))
    else:
        str=0
    str=np.pi/6
    # action=[7.5*(5-vx),7.5*(5-vx),7.5*(5-vx),7.5*(5-vx),str,0]
    action=[150,150,150,150,str,0]
    next_state,reward,done,truncated,info=env.step(action)
    action_list.append(action)
    vx=info['vx']
    vy=info['vy']
    yawrate=info['yawrate']
    beta=info['beta']
    heading=info['heading']
    heading_list.append(heading)
    a_x=info['a_x']
    a_y=info['a_y']
    dyawrate=info['d_yawrate']
    beta_list.append(beta)
    yawrate_list.append(yawrate)
    vx_list.append(vx)
    vy_list.append(vy)
    state=next_state.reshape(-1)
    rel_tra_x.append(state[0])
    rel_tra_y.append(state[1])
    str_f=action[-2]
    str_r=0
    alpha_list.append([info['alpha_f'],info['alpha_r']])
t=np.linspace(0,0.05*len(vx_list),len(vx_list))
action_list=np.array(action_list)
alpha_list=np.array(alpha_list)
env.close()

plt.figure()
plt.subplots_adjust(top=0.970,bottom=0.060,left=0.050,right=0.990,hspace=0.350,wspace=0.170)
plt.subplot(3,3,1)
plt.title(r'Beta $\beta$',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(t,beta_list,linewidth=2.5,color='r')
plt.xlabel(r't/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$\beta$/($rad$)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.subplot(3,3,2)
plt.title(r'Yawrate $\omega$',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(t,yawrate_list,linewidth=2.5,color='r')
plt.xlabel('t/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$\omega$/(rad/s)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.subplot(3,3,3)
plt.title(r'$v_x$',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(t,vx_list,linewidth=2.5,color='r')
plt.xlabel('t/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$v_x$/(m/s)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.subplot(3,3,4)
plt.title(r'$v_y$',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(t,vy_list,linewidth=2.5,color='r')
plt.xlabel(r't/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$v_y$/(m/s)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.subplot(3,3,5)
plt.title(r'$\varphi$',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(t,heading_list,linewidth=2.5,color='r')
plt.xlabel(r't/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$\varphi$/(rad)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.subplot(3,3,6)
plt.title(r'Trajectory',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(rel_tra_x,rel_tra_y,linewidth=2.5,color='r')
plt.xlabel(r'x/(m) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'y/(m)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.subplot(3,3,7)
plt.title(r'Cornering angle $\alpha$',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
alpha_list[0,0]=alpha_list[1,0]
plt.plot(t,alpha_list[:,0],linewidth=2.5,label=r'$\alpha_f$')
plt.plot(t,alpha_list[:,1],linewidth=2.5,label=r'$\alpha_r$')
plt.xlabel(r't/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$\alpha$/($\circ$)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.legend(prop={'family':'Times New Roman'})
plt.subplot(3,3,8)
plt.title(r'Torque',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(t,action_list[:,0],linewidth=2.5,label='$Q_{fl}$')
plt.plot(t,action_list[:,1],linewidth=2.5,label='$Q_{fr}$')
plt.plot(t,action_list[:,2],linewidth=2.5,label='$Q_{rl}$')
plt.plot(t,action_list[:,3],linewidth=2.5,label='$Q_{rr}$')
plt.xlabel(r't/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$Q$/(N$\cdot$m)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.legend(prop={'family':'Times New Roman'})
plt.subplot(3,3,9)
plt.title('Steer angle',fontdict={'family':'Times New Roman'},fontweight='bold',size=17)
plt.plot(t,action_list[:,4],linewidth=2.5,label='$\delta_f$')
plt.plot(t,action_list[:,5],linewidth=2.5,label='$\delta_r$')
plt.xlabel(r't/(s) ',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.ylabel(r'$\delta$/(rad)',fontdict={'family':'Times New Roman'},fontweight='bold',size=15)
plt.legend(prop={'family':'Times New Roman'})
plt.show()