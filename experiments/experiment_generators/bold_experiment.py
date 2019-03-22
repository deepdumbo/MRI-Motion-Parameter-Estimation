input_domains = ['FREQUENCY','IMAGE']
architectures = ['STANDARD','UNET']
experiment_file = '../bold-baselines.txt'
e = open(experiment_file, 'w+')

for i in range(len(input_domains)):
    input_domain = input_domains[i]
    architecture = architectures[i]

    filename = 'bold-'+input_domain+'.ini'
    
    f = open('../../configs/'+filename, 'w+')
    f.writelines([
        '[DATA]\n',
        'n = 64\n',
        'dataset = BOLD\n',
        'corruption = SEQUENTIAL\n',
        '\n',
        '[MODEL]\n',
        'architecture = '+architecture+'\n',
        'input_domain = '+input_domain+'\n'
        'output_domain = IMAGE\n',
        'nonlinearity = relu\n',
        '\n',
        '[TRAINING]\n',
        'pretrain = False\n',
        'num_epochs = 500\n'
    ])
    f.close()

    e.write(filename+'\n')

e.close()

