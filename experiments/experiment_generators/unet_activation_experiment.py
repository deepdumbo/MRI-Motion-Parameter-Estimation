activations = ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','exponential','linear']

experiment_file = '../experiments/unet-activations.txt'
e = open(experiment_file, 'w+')

for activation in activations:
    filename = 'brain64unetall-'+activation+'.ini'
    
    f = open('../configs/'+filename, 'w+')
    f.writelines([
        '[DATA]\n',
        'n = 64\n',
        'dataset = BRAIN\n',
        'corruption = ALL\n',
        '\n',
        '[MODEL]\n',
        'architecture = UNET\n',
        'output_domain = FREQUENCY\n',
        'nonlinearity = '+activation+'\n',
        '\n',
        '[TRAINING]\n',
        'pretrain = False\n',
        'num_epochs = 500\n'
    ])
    f.close()

    e.write(filename+'\n')

e.close()

