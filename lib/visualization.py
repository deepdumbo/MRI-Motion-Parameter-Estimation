import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns

def get_outputs(train_generator,val_generator):
    out_shape = train_generator[0][1].shape
    train_outs = np.empty((0,out_shape[1],out_shape[2]))
    for i in range(len(train_generator)):
        train_outs = np.append(train_outs,train_generator[i][1],axis=0)

    val_outs = np.empty((0,out_shape[1],out_shape[2]))
    for i in range(len(val_generator)):
        val_outs = np.append(val_outs,val_generator[i][1],axis=0)
    
    return(train_outs,val_outs)

def get_model_outputs(model,train_generator,val_generator):
    train_model_outs = model.predict_generator(train_generator)
    val_model_outs = model.predict_generator(val_generator)
    return(train_model_outs,val_model_outs)

def generate_error_kde(train_outs,train_model_outs,val_outs,val_model_outs,filename):      
    num_scatter = -1
    fig, axes = plt.subplots(2, 2, figsize=(15,15))  

    sns.kdeplot(train_outs[:,:,0].flatten()[:num_scatter],train_model_outs[:,:,0].flatten()[:num_scatter], shade=True, cbar=True, ax=axes[0][0], xlim=[0,10], ylim=[0,10])
    axes[0][0].set(xlabel='True Translation (px) - Train', ylabel='Estimated Translation (px) - Train')

    sns.kdeplot(val_outs[:,:,0].flatten()[:num_scatter],val_model_outs[:,:,0].flatten()[:num_scatter], shade=True, cbar=True, ax=axes[0][1], xlim=[0,10], ylim=[0,10])
    axes[0][1].set(xlabel='True Translation (px) - Val', ylabel='Estimated Translation (px) - Val')

    sns.kdeplot(train_outs[:,:,2].flatten()[:num_scatter],train_model_outs[:,:,2].flatten()[:num_scatter], shade=True, cbar=True, ax=axes[1][0], xlim=[-45,45], ylim=[-45,45])
    axes[1][0].set(xlabel='True Rotation (degrees) - Train', ylabel='Estimated Rotation (degrees) - Train')

    sns.kdeplot(val_outs[:,:,2].flatten()[:num_scatter],val_model_outs[:,:,2].flatten()[:num_scatter], shade=True, cbar=True, ax=axes[1][1], xlim=[-45,45], ylim=[-45,45])
    axes[1][1].set(xlabel='True Rotation (degrees) - Val', ylabel='Estimated Rotation (degrees) - Val')
    
    plt.savefig(filename)  

def generate_error_scatter(train_outs,train_model_outs,val_outs,val_model_outs,filename):
    num_scatter = -1
    fig, axes = plt.subplots(2, 2, figsize=(15,15))

    sns.scatterplot(train_outs[:,:,0].flatten()[:num_scatter],train_model_outs[:,:,0].flatten()[:num_scatter], ax=axes[0][0])
    axes[0][0].set(xlabel='True Translation (px) - Train', ylabel='Estimated Translation (px) - Train', xlim=[0,10], ylim=[0,10])

    sns.scatterplot(val_outs[:,:,0].flatten()[:num_scatter],val_model_outs[:,:,0].flatten()[:num_scatter], ax=axes[0][1])
    axes[0][1].set(xlabel='True Translation (px) - Val', ylabel='Estimated Translation (px) - Val', xlim=[0,10], ylim=[0,10])

    sns.scatterplot(train_outs[:,:,2].flatten()[:num_scatter],train_model_outs[:,:,2].flatten()[:num_scatter], ax=axes[1][0])
    axes[1][0].set(xlabel='True Rotation (degrees) - Train', ylabel='Estimated Rotation (degrees) - Train', xlim=[-45,45], ylim=[-45,45])

    sns.scatterplot(val_outs[:,:,2].flatten()[:num_scatter],val_model_outs[:,:,2].flatten()[:num_scatter], ax=axes[1][1])
    axes[1][1].set(xlabel='True Rotation (degrees) - Val', ylabel='Estimated Rotation (degrees) - Val', xlim=[-45,45], ylim=[-45,45])

    plt.savefig(filename)
