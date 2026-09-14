from dataloaders.datasets import Kvasir, ISIC2018, PH2, BUSI, GLAS
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from mypath import Path


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def make_data_loader(args, **kwargs):
    if args.dataset == 'Kvasir':
        train_set = Kvasir.Segmentation(args, split='train')
        val_set = Kvasir.Segmentation(args, split='val')
        test_set = Kvasir.Segmentation(args, split='test')

        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader
    
    elif args.dataset == 'ISIC2018':
        train_set = ISIC2018.Segmentation(args, split='train')
        val_set = ISIC2018.Segmentation(args, split='val')
        test_set = ISIC2018.Segmentation(args, split='test')

        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader
    
    elif args.dataset == 'BUSI':
        train_set = BUSI.Segmentation(args, split='train')
        val_set = BUSI.Segmentation(args, split='val')
        test_set = BUSI.Segmentation(args, split='test')

        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader
    
    elif args.dataset == 'PH2':
        train_set = PH2.Segmentation(args, split='test')
        val_set = PH2.Segmentation(args, split='test')
        test_set = PH2.Segmentation(args, split='test')

        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

    elif args.dataset == 'GLAS':
        train_set = GLAS.Segmentation(args, split='train')
        val_set = GLAS.Segmentation(args, split='val')
        test_set = GLAS.Segmentation(args, split='test')

        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader
  
    else:
        raise NotImplementedError

