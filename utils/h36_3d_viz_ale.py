#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from utils import h36motion3d as datasets
from utils.loss_funcs import mpjpe_error
from utils.data_utils import define_actions



# In[10]:


def create_pose(ax,plots,vals,pred=True,update=False):

            
    # [16, 20, 23, 24, 28, 31] IGNORE
    # [13, 19, 22, 13, 27, 30] EQUAL
    # h36m 32 joints(full)
    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (0, 6),
            (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (24, 25), (24, 17),
            (24, 14), (14, 15)
    ]
    LR = [
            False, True, True, True,
            True, True, False, False,
             False, False,
            False, True, True, True, True, True, True, 
            False, False, False, False, False, False, False, True,
            False, True, True, True, True,
            True, True
    ]  


# Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])
# Left / right indicator
    LR  = np.array([LR[a] or LR[b] for a,b in connect])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        if not update:
            if i ==0:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))
        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)
    
    return plots
   # ax.legend(loc='lower left')


# In[11]:
def update_pred(num,data_gt,data_pred,data_tr,plots_gt,plots_pred,plots_tr,fig,ax):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    # tr_vals=data_tr[1]
    # plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
    plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True) ########à
    #plots_tr=create_pose(ax,plots_pred,pred_vals,pred=False,update=True)
    
    r = 0.75
    xroot, zroot, yroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_pred ##################

def update_gt(num,data_gt,data_pred,data_tr,plots_gt,plots_pred,plots_tr,fig,ax):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    # tr_vals=data_tr[1] 

    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
    # plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True) ########à
    # plots_tr=create_pose(ax,plots_pred,pred_vals,pred=False,update=True)
    
    r = 0.75
    xroot, zroot, yroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_gt #plots_pred ##################

def update_tr(num,data_gt,data_pred,data_tr,plots_gt,plots_pred,plots_tr,fig,ax):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    tr_vals=data_tr[num]
    # plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
    # plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True) ########à
    plots_tr=create_pose(ax,plots_tr,tr_vals,pred=False,update=True)
    

    
    
    r = 0.75
    xroot, zroot, yroot = tr_vals[0,0], tr_vals[0,1], tr_vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_tr #plots_gt plots_pred ##################


    
# In[12]:


def visualize(modello, path, maskA, maskT,input_n,output_n,visualize_from,device,n_viz,skip_rate,actions):
    
    actions=define_actions(actions)
    
    for action in actions:
    
        if visualize_from=='train':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, split=0,actions=[action])
        elif visualize_from=='validation':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, split=1,actions=[action])
        elif visualize_from=='test':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, split=2,actions=[action])
            
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                        46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                        75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
      # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
            
            
        loader = DataLoader(
        loader,
        batch_size=1,
        shuffle = True,
        num_workers=0)       
        
            
    
        for cnt,batch in enumerate(loader): 
            batch = batch.to(device) 
            
            all_joints_seq=batch.clone()[:, input_n:input_n+output_n,:]
            
            sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
            sequences_gt=batch[:, input_n:input_n+output_n, :]
          
            sequences_predict=modello(sequences_train, maskA, maskT).permute(0,1,3,2).contiguous().view(-1,output_n,len(dim_used)) ####
            all_joints_seq[:,:,dim_used] = sequences_predict
            all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]
            
            all_joints_seq=all_joints_seq.view(-1,output_n,32,3)
            
            sequences_gt=sequences_gt.view(-1,output_n,32,3)
            
            loss=mpjpe_error(all_joints_seq,sequences_gt)# # both must have format (batch,T,V,C)
            
            tr = batch.clone()[:, 0:input_n,:]
            tr[:,:,dim_used] = batch[:, 0:input_n, dim_used]
            tr[:,:,index_to_ignore] = tr[:,:,index_to_equal]
            tr=tr.view(-1,input_n,32,3)

    
            data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()/1000 # in meters
            data_gt=torch.squeeze(sequences_gt,0).cpu().data.numpy()/1000
            data_tr=torch.squeeze(tr,0).cpu().data.numpy()/1000


            data_pred_ = []
            data_gt_ = []
            data_tr_=[]
            '''
            for i in range(25):
                if i % 3 == 0:
                    data_pred_.append(data_pred[i,:,:])
                    data_gt_.append(data_gt[i,:,:])
            '''
            for h in range(10):
                data_tr_.append(data_tr[h,:,:])

            #data_tr_.append(data_tr[7,:,:])
            #data_tr_.append(data_tr[9,:,:])
            
            data_tr=data_tr_
            #data_pred = data_pred_
            #data_gt=data_gt_

            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            ax1.view_init(elev=20, azim=-40)
            

            fig2 = plt.figure()
            ax2 = Axes3D(fig2)
            ax2.view_init(elev=20, azim=-40)
            

            fig3 = plt.figure()
            ax3 = Axes3D(fig3)
            ax3.view_init(elev=20, azim=-40)
            


            vals = np.zeros((32, 3)) # or joints_to_consider
            gt_plots=[]
            pred_plots=[]
            tr_plots=[]
           
    
            gt_plots=create_pose(ax1,gt_plots,vals,pred=False,update=False)
            pred_plots=create_pose(ax2,pred_plots,vals,pred=True,update=False)
            tr_plots=create_pose(ax3,tr_plots,vals,pred=False,update=False)

            
            
            
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            ax1.set_axis_off()
            # ax1.legend(loc='lower left')
            ax1.set_xlim3d([-1, 1.5])
            ax1.set_xlabel('X')    
            ax1.set_ylim3d([-1, 1.5])
            ax1.set_ylabel('Y')   
            ax1.set_zlim3d([0.0, 1.5])
            ax1.set_zlabel('Z')

            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("z")
            ax2.set_axis_off()
            # ax2.legend(loc='lower left')    
            ax2.set_xlim3d([-1, 1.5])
            ax2.set_xlabel('X')    
            ax2.set_ylim3d([-1, 1.5])
            ax2.set_ylabel('Y')    
            ax2.set_zlim3d([0.0, 1.5])
            ax2.set_zlabel('Z')

            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_zlabel("z")
            ax3.set_axis_off()
            # ax3.legend(loc='lower left') 
            ax3.set_xlim3d([-1, 1.5])
            ax3.set_xlabel('X')    
            ax3.set_ylim3d([-1, 1.5])
            ax3.set_ylabel('Y')
            ax3.set_zlim3d([0.0, 1.5])
            ax3.set_zlabel('Z')
            
           
           # ax.set_title('loss in mm is: '+str(round(loss.item(),4))+' for action : '+str(action)+' for '+str(output_n)+' frames')
            
            plt.rcParams['grid.color'] = "white" # COMMENT FOR GRID
            
            line_anim_gt = animation.FuncAnimation(fig1, update_gt, output_n, fargs=(data_gt,data_pred,data_tr, gt_plots,pred_plots, tr_plots,
                                                                       fig1,ax1),interval=70, blit=False)
            
            line_anim_pred = animation.FuncAnimation(fig2, update_pred, output_n, fargs=(data_gt,data_pred,data_tr, gt_plots,pred_plots, tr_plots,
                                                                       fig2,ax2),interval=70, blit=False)
            
            line_anim_tr = animation.FuncAnimation(fig3, update_tr, 10, fargs=(data_gt,data_pred,data_tr, gt_plots,pred_plots, tr_plots,
                                                                       fig3,ax3),interval=70, blit=False)
          
            
            plt.show()

            
        
            line_anim_gt.save(str(actions)+'_gt.gif',writer='pillow')
            line_anim_pred.save(str(actions)+'_pred.gif',writer='pillow')
            line_anim_tr.save(str(actions)+'_tr.gif',writer='pillow')

            '''
            # PUNTI EMBEDDING
            vals=data_pred[0,:,:]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=10)
            for i in np.arange( 22 ):
                x = np.array( [vals[i, 0], vals[i, 0]] )
                z = np.array( [vals[i, 1], vals[i, 1]] )
                y = np.array( [vals[i, 2], vals[i, 2]] )
                ax.scatter(x, y, z)
            plt.show()
            '''
            if cnt==n_viz-1:
                break

