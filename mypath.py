class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'Kvasir':
            return 'data/Kvasir/'
        
        elif dataset == 'ISIC2018':
            return 'data/ISIC2018/'
        
        elif dataset == 'BUSI':
            return 'data/BUSI/'

        elif dataset == 'PH2':
            return 'data/PH2/'

        elif dataset == 'GLAS':
            return 'data/GLAS/'
        
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
