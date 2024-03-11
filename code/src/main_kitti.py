import os
import shutil
import numpy as np
from collections import namedtuple
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict
from dataset import BaseDataset
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from train_torch_filter import train_filter
from utils_plot import results_filter


def launch(args):
    if args.read_data:
        args.dataset_class.read_data(args)
    dataset = args.dataset_class(args)   #此处，感觉又重复了一次，把上一步的dataset传入进来就可以

    if args.train_filter:
        train_filter(args, dataset)

    if args.test_filter:
        test_filter(args, dataset)

    if args.results_filter:
        results_filter(args, dataset)


class KITTIParameters(IEKF.Parameters):  #先定义好协方差初始的参数，然后再调节
    # gravity vector
    g = np.array([0, 0, -9.80655])
    a = 2
    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(KITTIParameters) if
                     not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))


class KITTIDataset(BaseDataset):   #这是每个IMU/OXTS的数据格式
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
                                                                       '' + 'ax, ay, az, af, al, '
                                                                            'au, ' + 'wx, wy, wz, '
                                                                                     'wf, wl, wu, '
                                                                                     '' +
                            'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
                                                                                  'velmode, '
                                                                                  'orimode')

    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
    min_seq_dim = 25 * 10  # 60 s
    datasets_fake = ['11', '12', '13']
    """
    '2011_09_30_drive_0028_extract' has trouble at N = [6000, 14000] -> test data
    '2011_10_03_drive_0027_extract' has trouble at N = 29481
    '2011_10_03_drive_0034_extract' has trouble at N = [33500, 34000]
    """

    # training set to the raw data of the KITTI dataset.
    # The following dict lists the name and end frame of each sequence that
    # has been used to extract the visual odometry / SLAM training set
    odometry_benchmark = OrderedDict()
    # odometry_benchmark["2011_10_03_drive_0027_extract"] = [0, 45692]  #seq-00
    # odometry_benchmark["2011_10_03_drive_0042_extract"] = [0, 12180]  #seq-01
    # odometry_benchmark["2011_10_03_drive_0034_extract"] = [0, 47935]  #seq-02
    # odometry_benchmark["2011_09_26_drive_0067_extract"] = [0, 8000]  #seq-03
    # odometry_benchmark["2011_09_30_drive_0016_extract"] = [0, 2950]  #seq-04
    # odometry_benchmark["2011_09_30_drive_0018_extract"] = [0, 28659]  #seq-05
    # odometry_benchmark["2011_09_30_drive_0020_extract"] = [0, 11347]  #seq-06
    # odometry_benchmark["2011_09_30_drive_0027_extract"] = [0, 11545]  #seq-07
    # odometry_benchmark["2011_09_30_drive_0028_extract"] = [11231, 53650]  #seq-08
    # odometry_benchmark["2011_09_30_drive_0033_extract"] = [0, 16589]  #seq-09
    # odometry_benchmark["2011_09_30_drive_0034_extract"] = [0, 12744]  #seq-10

    # 10Hz
    odometry_benchmark["00"] = [0, 4540]  # seq-00
    odometry_benchmark["01"] = [0, 1100]  # seq-01
    odometry_benchmark["02"] = [0, 4660]  # seq-02
    odometry_benchmark["04"] = [0, 270]  # seq-04
    odometry_benchmark["05"] = [0, 2760]  # seq-05
    odometry_benchmark["06"] = [0, 1100]  # seq-06
    odometry_benchmark["07"] = [0, 1100]  # seq-07
    odometry_benchmark["08"] = [0, 4070]  # seq-08
    odometry_benchmark["09"] = [0, 1590]  # seq-09
    odometry_benchmark["10"] = [0, 1200]  # seq-10

    odometry_benchmark_img = OrderedDict()
    odometry_benchmark_img["2011_10_03_drive_0027_extract"] = [0, 45400]
    odometry_benchmark_img["2011_10_03_drive_0042_extract"] = [0, 11000]
    odometry_benchmark_img["2011_10_03_drive_0034_extract"] = [0, 46600]
    odometry_benchmark_img["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark_img["2011_09_30_drive_0016_extract"] = [0, 2700]
    odometry_benchmark_img["2011_09_30_drive_0018_extract"] = [0, 27600]
    odometry_benchmark_img["2011_09_30_drive_0020_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0027_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0028_extract"] = [11000, 51700]
    odometry_benchmark_img["2011_09_30_drive_0033_extract"] = [0, 15900]
    odometry_benchmark_img["2011_09_30_drive_0034_extract"] = [0, 12000]

    # Args是上一步 args = KITTIArgs()  #先初始化好了协方差参数与数据集参数
    # 此时self只有上面的值，在super后，会继承BaseDataset里面的初始值
    def __init__(self, args):
        super(KITTIDataset, self).__init__(args)
        #super是调用父类，在构造class时，如果有继承，一般会有super.
        #此处，父类是BaseDataset，当前类是KITTIDataset，然后传入参数args调用父类的初始化函数.__init__.赋给当前的self.
        #如果没有.__init__(args)就只继承，不执行初始化

        # self.datasets_validatation_filter['2011_09_30_drive_0028_extract'] = [11231, 53650]  #seq-08
        # self.datasets_train_filter["2011_10_03_drive_0042_extract"] = [0, None]
        # self.datasets_train_filter["2011_09_30_drive_0018_extract"] = [0, 15000]
        # self.datasets_train_filter["2011_09_30_drive_0020_extract"] = [0, None]
        # self.datasets_train_filter["2011_09_30_drive_0027_extract"] = [0, None]
        # self.datasets_train_filter["2011_09_30_drive_0033_extract"] = [0, None]
        # self.datasets_train_filter["2011_10_03_drive_0027_extract"] = [0, 18000]
        # self.datasets_train_filter["2011_10_03_drive_0034_extract"] = [0, 31000]
        # self.datasets_train_filter["2011_09_30_drive_0034_extract"] = [0, None]

        self.datasets_validatation_filter["08"] = [0, 4070]  # seq-08
        self.datasets_train_filter["00"] = [0, 4540]  # seq-00
        self.datasets_train_filter["01"] = [0, 1100]  # seq-01
        self.datasets_train_filter["02"] = [0, 4660]  # seq-02
        self.datasets_train_filter["04"] = [0, 270]  # seq-04
        self.datasets_train_filter["05"] = [0, 2760]  # seq-05
        self.datasets_train_filter["06"] = [0, 1100]  # seq-06
        self.datasets_train_filter["07"] = [0, 1100]  # seq-07
        self.datasets_train_filter["09"] = [0, 1590]  # seq-09
        self.datasets_train_filter["10"] = [0, 1200]  # seq-10

        for dataset_fake in KITTIDataset.datasets_fake:
            if dataset_fake in self.datasets:
                self.datasets.remove(dataset_fake)
            if dataset_fake in self.datasets_train:
                self.datasets_train.remove(dataset_fake)

    @staticmethod
    def read_data(args):
        """
        Read the data from the KITTI dataset

        :param args:
        :return:
        """

        print("Start read_data")
        t_tot = 0  # sum of times for the all dataset
        date_dirs = os.listdir(args.path_data_base)
        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                # read data
                oxts_files = sorted(glob.glob(os.path.join(path2, 'data', '*.txt')))
                if date_dir in ['00', '02', '05', '06', '07', '08', '09']:
                    pose_index = '/media/wuhan2021/ZX1-grey/3.Results/1.multi-sensor/ATE-RTE/' + date_dir + '/time_index.txt'
                    pose_index = np.fromfile(pose_index, dtype=int, sep='\n')
                    pose_index = pose_index.tolist()
                    oxts_files_new = np.array(oxts_files)
                    oxts_files_new = oxts_files_new[pose_index]
                    oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files_new)

                else:
                    oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files)
                    pose_index = [0]



                """ Note on difference between ground truth and oxts solution:
                    - orientation is the same
                    - north and east axis are inverted
                    - position are closed to but different
                    => oxts solution is not loaded
                """

                print("\n Sequence name : " + date_dir)
                if len(oxts) < KITTIDataset.min_seq_dim:  #  sequence shorter than 30 s are rejected
                    cprint("Dataset is too short ({:.2f} s)".format(len(oxts) / 100), 'yellow')
                    continue
                lat_oxts = np.zeros(len(oxts))
                lon_oxts = np.zeros(len(oxts))
                alt_oxts = np.zeros(len(oxts))
                roll_oxts = np.zeros(len(oxts))
                pitch_oxts = np.zeros(len(oxts))
                yaw_oxts = np.zeros(len(oxts))
                roll_gt = np.zeros(len(oxts))
                pitch_gt = np.zeros(len(oxts))
                yaw_gt = np.zeros(len(oxts))
                t = KITTIDataset.load_timestamps(path2, pose_index)
                acc = np.zeros((len(oxts), 3))
                acc_bis = np.zeros((len(oxts), 3))
                gyro = np.zeros((len(oxts), 3))
                gyro_bis = np.zeros((len(oxts), 3))
                p_gt = np.zeros((len(oxts), 3))
                v_gt = np.zeros((len(oxts), 3))
                v_rob_gt = np.zeros((len(oxts), 3))

                k_max = len(oxts)
                for k in range(k_max):
                    oxts_k = oxts[k]
                    t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[
                        k].microsecond / 1e6
                    lat_oxts[k] = oxts_k[0].lat
                    lon_oxts[k] = oxts_k[0].lon
                    alt_oxts[k] = oxts_k[0].alt
                    acc[k, 0] = oxts_k[0].af
                    acc[k, 1] = oxts_k[0].al
                    acc[k, 2] = oxts_k[0].au
                    acc_bis[k, 0] = oxts_k[0].ax
                    acc_bis[k, 1] = oxts_k[0].ay
                    acc_bis[k, 2] = oxts_k[0].az
                    gyro[k, 0] = oxts_k[0].wf
                    gyro[k, 1] = oxts_k[0].wl
                    gyro[k, 2] = oxts_k[0].wu
                    gyro_bis[k, 0] = oxts_k[0].wx
                    gyro_bis[k, 1] = oxts_k[0].wy
                    gyro_bis[k, 2] = oxts_k[0].wz
                    roll_oxts[k] = oxts_k[0].roll
                    pitch_oxts[k] = oxts_k[0].pitch
                    yaw_oxts[k] = oxts_k[0].yaw
                    v_gt[k, 0] = oxts_k[0].ve
                    v_gt[k, 1] = oxts_k[0].vn
                    v_gt[k, 2] = oxts_k[0].vu
                    v_rob_gt[k, 0] = oxts_k[0].vf
                    v_rob_gt[k, 1] = oxts_k[0].vl
                    v_rob_gt[k, 2] = oxts_k[0].vu
                    p_gt[k] = oxts_k[1][:3, 3]
                    Rot_gt_k = oxts_k[1][:3, :3]
                    roll_gt[k], pitch_gt[k], yaw_gt[k] = IEKF.to_rpy(Rot_gt_k)

                t0 = t[0]
                # t = np.array(t) - t[0]
                t = np.array(t)
                # some data can have gps out
                if np.max(t[:-1] - t[1:]) > 0.2:
                    cprint(date_dir2 + " has time problem", 'yellow')
                ang_gt = np.zeros((roll_gt.shape[0], 3))
                ang_gt[:, 0] = roll_gt
                ang_gt[:, 1] = pitch_gt
                ang_gt[:, 2] = yaw_gt

                p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0],
                                 alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
                p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]]  # see note

                # take correct imu measurements
                u = np.concatenate((gyro_bis, acc_bis), -1)
                # convert from numpy
                t = torch.from_numpy(t)
                p_gt = torch.from_numpy(p_gt)
                v_gt = torch.from_numpy(v_gt)
                ang_gt = torch.from_numpy(ang_gt)
                u = torch.from_numpy(u)

                # convert to float
                t = t.float()
                u = u.float()
                p_gt = p_gt.float()
                ang_gt = ang_gt.float()
                v_gt = v_gt.float()

                mondict = {
                    't': t, 'p_gt': p_gt, 'ang_gt': ang_gt, 'v_gt': v_gt,
                    'u': u, 'name': date_dir, 't0': t0
                    }

                t_tot += t[-1] - t[0]
                KITTIDataset.dump(mondict, args.path_data_save, date_dir)
        print("\n Total dataset duration : {:.2f} s".format(t_tot))

    @staticmethod
    def prune_unused_data(args):
        """
        Deleting image and velodyne
        Returns:

        """

        unused_list = ['image_00', 'image_01', 'image_02', 'image_03', 'velodyne_points']
        date_dirs = ['2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

        for date_dir in date_dirs:
            path1 = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                print(path2)
                for folder in unused_list:
                    path3 = os.path.join(path2, folder)
                    if os.path.isdir(path3):
                        print(path3)
                        shutil.rmtree(path3)

    @staticmethod
    def subselect_files(files, indices):
        try:
            files = [files[i] for i in indices]
        except:
            pass
        return files

    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = KITTIDataset.rotx(packet.roll)
        Ry = KITTIDataset.roty(packet.pitch)
        Rz = KITTIDataset.rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @staticmethod
    def load_oxts_packets_and_poses(oxts_files):
        """Generator to read OXTS ground truth data.
           Poses are given in an East-North-Up coordinate system
           whose origin is the first GPS position.
        """
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []

        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts 最后五个条目是标志和计数
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    packet = KITTIDataset.OxtsPacket(*line)   #生成一个包，将格式与数据对应起来，像字典一样引用packet.lat

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.)

                    R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)   #从OXTS packet中读取真值的pose

                    if origin is None:  #每个序列的第一个为none,然后origin都以此P0为基准
                        origin = t

                    T_w_imu = KITTIDataset.transform_from_rot_trans(R, t - origin)   # 将t和R合到4*4的矩阵中

                    oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))   #30个原始数据和4*4的矩阵
        return oxts

    @staticmethod
    def load_timestamps(data_path, pose_index):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        if len(pose_index) > 1:
            timestamps_new = np.array(timestamps)[pose_index]
            return timestamps_new.tolist()
        return timestamps

    def load_timestamps_img(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'image_00', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps


def test_filter(args, dataset):  #初始化的KF参数和数据集参数
    iekf = IEKF()  #实例化一个IEKF类的对象，并初始化__init__
    torch_iekf = TORCHIEKF()  #实例化一个TORCHIEKF类的对象(但是继承了IEKF)，并初始化__init__

    # put Kitti parameters
    iekf.filter_parameters = KITTIParameters()
    iekf.set_param_attr()  #调用的KITTIParameters的set_param_attr()
    torch_iekf.filter_parameters = KITTIParameters()
    torch_iekf.set_param_attr()  #调用的torch_iekf自己的set_param_attr()

    torch_iekf.load(args, dataset)  # KF参数 和 数据集参数
    iekf.set_learned_covariance(torch_iekf)  #将torch_iekf训练的参数给iekf，注意只有Q

    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)

        # if dataset_name not in dataset.odometry_benchmark.keys():
        #     continue
        print("Test filter on sequence: " + dataset_name)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i,
                                                       to_numpy=True)
        N = None
        u_t = torch.from_numpy(u).double()
        measurements_covs = torch_iekf.forward_nets(u_t)   # u是输入，3轴加速度，3轴陀螺仪
        measurements_covs = measurements_covs.detach().numpy()  # IMU帧数 * 2
        start_time = time.time()
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs,
                                                                   v_gt, p_gt, N,
                                                                   ang_gt[0], dataset_name)
        diff_time = time.time() - start_time
        print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time,
                                                                          t[-1] - t[0]))
        mondict = {  #保存结果
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
            }
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")


class KITTIArgs():   #只有参数，没有方法
        path_data_base = "/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_drive/oxts_10Hz"
        path_data_save = "../data_360"
        path_results = "../result_360"
        path_temp = "../temp"

        epochs = 400
        seq_dim = None #又重新定义了

        # training, cross-validation and test dataset  在seq-08上验证
        cross_validation_sequences = ['08']
        test_sequences = ['08']
        continue_training = True

        # choose what to do
        read_data = 0
        train_filter = 0
        test_filter = 1
        results_filter = 1
        dataset_class = KITTIDataset   #只引用类，不会初始化，也没有类的成员函数
        parameter_class = KITTIParameters


if __name__ == '__main__':
    args = KITTIArgs()  #先初始化好了协方差参数与数据集参数
    dataset = KITTIDataset(args)  #将参数代入，然后制作训练/测试数据集
    launch(KITTIArgs)

