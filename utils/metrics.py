import numpy as np
import torch
from sklearn.metrics import confusion_matrix

class Evaluator(object):
    def __init__(self, num_class, cuda=False, alpha=0.5):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.total_e_measure = 0.0  # 累积的 E_measure 值
        self.total_s_measure = 0.0  # 累积的 S_measure 值
        self.batch_count = 0        # 批次数量
        self.cuda = cuda            # 是否使用 CUDA 加速
        self.smeasure_calculator = SMeasure(alpha=alpha, cuda=cuda)  # 初始化 SMeasure 类

    def Specificity(self):
        TN = self.confusion_matrix[0, 0]
        FP = self.confusion_matrix[0, 1]
        SP = TN / (TN + FP + 1e-10)
        return SP

    def Sensitivity(self):
        TP = self.confusion_matrix[1, 1]
        FN = self.confusion_matrix[1, 0]
        SE = TP / (TP + FN + 1e-10)
        return SE

    def DSC(self):
        TP = self.confusion_matrix[1, 1]
        FP = self.confusion_matrix[0, 1]
        FN = self.confusion_matrix[1, 0]
        DSC = (2 * TP) / (2 * TP + FP + FN + 1e-10)
        return DSC

    def Mean_Dice_Score(self):
        dice_scores = []
        for i in range(self.num_class):
            TP = self.confusion_matrix[i, i]
            FP = np.sum(self.confusion_matrix[:, i]) - TP
            FN = np.sum(self.confusion_matrix[i, :]) - TP
            Dice = (2 * TP) / (2 * TP + FP + FN + 1e-10)
            dice_scores.append(Dice)
        mDice = np.mean(dice_scores)
        return mDice

    def Accuracy(self):
        TP = self.confusion_matrix[1, 1]
        TN = self.confusion_matrix[0, 0]
        total = self.confusion_matrix.sum()
        Acc = (TP + TN) / (total + 1e-10)
        return Acc

    def Precision(self):
        TP = self.confusion_matrix[1, 1]
        FP = self.confusion_matrix[0, 1]
        precision = TP / (TP + FP + 1e-10)
        return precision

    def Recall(self):
        TP = self.confusion_matrix[1, 1]
        FN = self.confusion_matrix[1, 0]
        recall = TP / (TP + FN + 1e-10)
        return recall

    def Weighted_F_measure(self, beta=1):
        P = self.Precision()
        R = self.Recall()
        Fwβ = (1 + beta ** 2) * R * P / ((beta ** 2) * P + R + 1e-10)
        return Fwβ

    def E_measure(self):
        if self.batch_count == 0:
            return 0.0
        mean_E = self.total_e_measure / self.batch_count
        return mean_E

    def S_measure(self):
        if self.batch_count == 0:
            return 0.0
        mean_S = self.total_s_measure / self.batch_count
        return mean_S

    def _calculate_e_measure(self, y_pred, y_true):
        # 将 numpy 数组转换为 PyTorch 张量
        y_pred = torch.from_numpy(y_pred).float()
        y_true = torch.from_numpy(y_true).float()

        # 如果启用 CUDA，加速计算
        if self.cuda:
            y_pred = y_pred.cuda()
            y_true = y_true.cuda()

        # 计算 E-measure
        fm = y_pred - torch.mean(y_pred)
        gt = y_true - torch.mean(y_true)
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-10)
        enhanced = ((align_matrix + 1) ** 2) / 4
        num_elements = y_true.numel()
        score = torch.sum(enhanced) / (num_elements - 1 + 1e-10)

        # 返回 E-measure 分数
        return score.item()

    def Intersection_over_Union(self):
        TP = self.confusion_matrix[1, 1]
        FP = self.confusion_matrix[0, 1]
        FN = self.confusion_matrix[1, 0]
        IoU = TP / (TP + FP + FN + 1e-10)
        return IoU

    def Mean_Intersection_over_Union(self):
        IoUs = []
        for i in range(self.num_class):
            TP = self.confusion_matrix[i, i]
            FP = np.sum(self.confusion_matrix[:, i]) - TP
            FN = np.sum(self.confusion_matrix[i, :]) - TP
            IoU = TP / (TP + FP + FN + 1e-10)
            IoUs.append(IoU)
        mIoU = np.mean(IoUs)
        return mIoU

    def _generate_matrix(self, y_true, y_pred, num_classes=2):
        # 将输入展平为一维数组
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        # 将标签转换为整数类型
        y_true = y_true.astype(np.int32)
        y_pred = y_pred.astype(np.int32)
        # 检查标签范围
        if np.any(y_true < 0) or np.any(y_true >= num_classes) \
           or np.any(y_pred < 0) or np.any(y_pred >= num_classes):
            raise ValueError("标签必须在 [0, num_classes-1] 范围内")
        # 计算混淆矩阵
        CM = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        return CM

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        
        b, c, H, W = gt_image.shape
        for i in range(b):
            # 计算当前批次的 E-measure
            E = self._calculate_e_measure(pre_image[i], gt_image[i])
            # 计算当前批次的 S-measure，使用 SMeasure 类
            S = self.smeasure_calculator.calculate(pre_image[i], gt_image[i])
            # 累积 E-measure 和 S-measure
            self.total_e_measure += E
            self.total_s_measure += S
            # 更新批次数量
            self.batch_count += 1

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.total_e_measure = 0.0
        self.total_s_measure = 0.0
        self.batch_count = 0
        

class SMeasure(object):
    def __init__(self, alpha=0.5, cuda=False):
        self.alpha = alpha  # S-measure 中的参数 alpha
        self.cuda = cuda    # 是否使用 CUDA 加速

    def calculate(self, pred, gt):
        # 将 numpy 数组转换为 PyTorch 张量
        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)

        if self.cuda:
            pred = pred.cuda()
            gt = gt.cuda()

        # 计算 S-measure
        Q = self._cal_s_measure(pred, gt)
        return Q

    def _cal_s_measure(self, pred, gt):
        gt = gt.float()
        pred = pred.float()

        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            alpha = self.alpha
            Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.tensor(0.0)
        return Q.item()

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        return score

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.float().mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _centroid(self, gt):
        h, w = gt.size()[-2:]
        gt = gt.view(h, w)

        if gt.sum() == 0:
            X = torch.tensor(round(w / 2))
            Y = torch.tensor(round(h / 2))
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.arange(0, w).cuda().float()
                j = torch.arange(0, h).cuda().float()
            else:
                i = torch.arange(0, w).float()
                j = torch.arange(0, h).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]

        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-10)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-10)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-10)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-10)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0.0

        return Q

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

        return Q

