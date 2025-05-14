import numpy as np
import os
import pywt
 
class CSDA:
    #初始化
    def __init__(self):
        self.subject_id = 0 

    def subject_DWT_DA_ML(self, subject_id, level):
        load_path_dir = os.path.join(os.getcwd(),"Data", "Source")
        self.x = np.load(os.path.join(load_path_dir,"x.npy"))
        self.y = np.load(os.path.join(load_path_dir,"y.npy"))            
        self.subject_id = subject_id
        #对x数据进行处理
        Xs_DA = []
        Ys_DA = []
        for day_id in [1,2]:
            Xs = self.x[self.subject_id - 1,day_id - 1]
            Ys = self.y[self.subject_id - 1,day_id - 1]
            #数据集中的其他被试数据
            for subject_id in [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                if subject_id != self.subject_id:
                    Xt = self.x[subject_id - 1,day_id - 1]
                    Ys_Chn = []
                    Xs_Chn = []
                    for chn  in range(Xs.shape[1]):
                        #选择小波变换的模式
                        wavename = 'db4'
                        #进行小波变换
                        Cs = pywt.wavedec(Xs[:,chn], wavename, level=level)
                        #ScA, ScD2, ScD1 = Cs[0], Cs[1], Cs[2]  # Level 2 (Optional)
                        Ct = pywt.wavedec(Xt[:,chn], wavename, level=level)
                        #TcA, TcD2, TcD1 = Ct[0], Ct[1], Ct[2]  # Level 2 (Optional)
                        #进行逆小波变换
                        Xs_coeffs = [Cs[0]] + Ct[1:]
                        # 目标近似系数 + 源全部细节系数
                        Xt_coeffs = [Ct[0]] + Cs[1:]  
                        Xs_aug = pywt.waverec(Xs_coeffs, wavename, 'smooth')  # Src approximated component + Tar detailed component
                        Xt_aug = pywt.waverec(Xt_coeffs, wavename, 'smooth')  # Tar approximated component + Src detailed component
                        #将交换生成的数据链接
                        Xs_Chn.append(np.concatenate(( Xs[:,chn], Xt_aug, Xs_aug), axis=0))
                    Xs_Chn = np.stack(Xs_Chn, axis=1)
                    Ys_Chn = np.concatenate(( Ys, Ys, Ys), axis=0)
            Xs_DA.append(Xs_Chn)
            Ys_DA.append(Ys_Chn)
        Xs_DA = np.stack(Xs_DA, axis=0)
        Ys_DA = np.stack(Ys_DA, axis=0)
        #存储数据 x的数据格式为 (受试者,天数,采样点,通道数) y的数据格式为(受试者,天数,标签)
        save_path_dir = os.path.join(os.getcwd(),"Data", f"DWT_{level}L", f"Sub{self.subject_id:>02d}")
        os.makedirs(save_path_dir, exist_ok=True)
        np.save(os.path.join(save_path_dir,"x.npy"), Xs_DA)
        np.save(os.path.join(save_path_dir,"y.npy"), Ys_DA)