import time
import datetime
import os
import numpy as np
    #
    # timestamp_file = os.path.join(data_path, 'timestamps.txt')
    #
    # # Read and parse the timestamps
    # timestamps = []
    # with open(timestamp_file, 'r') as f:
    #     for line in f.readlines():
    #         # NB: datetime only supports microseconds, but KITTI timestamps
    #         # give nanoseconds, so need to truncate last 4 characters to
    #         # get rid of \n (counts as 1) and extra 3 digits
    #         t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
    #         timestamps.append(t)
    # if len(pose_index) > 1:
    #     timestamps_new = np.array(timestamps)[pose_index]
    #     return timestamps_new.tolist()


##############################时间的索引##################################
#   先对齐
# odom_time_path = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-times.txt'
# gt_time_path = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/times-gt-poses.txt'
# odom_poses = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-poses.txt'
# gt_poses = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/gt-poses-KITTI.txt'
#
# odom_time = np.fromfile(odom_time_path, dtype=float, sep='\n')
# gt_time = np.fromfile(gt_time_path, dtype=float, sep='\n')
# odom_poses = np.fromfile(odom_poses, dtype=float, sep='\n').reshape(-1, 12)
# gt_poses = np.fromfile(gt_poses, dtype=float, sep='\n').reshape(-1, 12)
# # B_timestamps = []
# # with open(B_path, 'r') as f:
# #     for line in f.readlines():
# #         t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
# #         t = float(t.strftime("%s.%f"))
# #         B_timestamps.append(t)
#
# odom_time = np.around(odom_time, 3)
# # A_file = np.around(A_file, 3)
# # B_file = np.around(B_file, 3)
# index = [i for i, num in enumerate(odom_time) if num in gt_time]
# # for j in range(0, 5674):
# #     if j != index[j]:
# #         print(j+1)
# #         break
# odom_time_new = odom_time[index]
# odom_poses_new = odom_poses[index]
# # pose_gt_new = gt_poses[index]
# np.savetxt('/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-times-new.txt', odom_time_new, fmt='%.4f')
# np.savetxt('/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-poses-new.txt', odom_poses_new, fmt='%.4f')
# # np.savetxt('/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0004_odom/time_index.txt', index, fmt="%d")
# a = 1

# 再索引
odom_time_path = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-times.txt'
gt_time_path = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/timestamps.txt'
odom_poses = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-poses.txt'
gt_poses = '/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/gt-poses-KITTI.txt'

odom_time = np.fromfile(odom_time_path, dtype=float, sep='\n')
# gt_time = np.fromfile(gt_time_path, dtype=float, sep='\n')
odom_poses = np.fromfile(odom_poses, dtype=float, sep='\n').reshape(-1, 12)
gt_poses = np.fromfile(gt_poses, dtype=float, sep='\n').reshape(-1, 12)
B_timestamps = []
with open(gt_time_path, 'r') as f:
    for line in f.readlines():
        t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
        t = float(t.strftime("%s.%f"))
        B_timestamps.append(t)

odom_time = np.around(odom_time, 3)
B_timestamps = np.around(B_timestamps, 3)
# A_file = np.around(A_file, 3)
# B_file = np.around(B_file, 3)
index = [i for i, num in enumerate(B_timestamps) if num in odom_time]
# for j in range(0, 5674):
#     if j != index[j]:
#         print(j+1)
#         break
time_index = index
# pose_gt_new = gt_poses[index]
# np.savetxt('/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-times-new.txt', odom_time_new, fmt='%.4f')
# np.savetxt('/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/odom-poses-new.txt', odom_poses_new, fmt='%.4f')
np.savetxt('/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0005_odom/time_index.txt', time_index, fmt="%d")
# np.savetxt('/media/wuhan2021/ZX1-grey/AI-IMU data/kitti_360_drive/oxts_10Hz/10Hz_duiqi/0004_odom/gt-poses-new.txt', pose_gt_new, fmt='%.4f')
a = 1