import pandas as pd

# 1. 读取你之前那两个表现最好的文件（文件名需对应你保存的名字）
sub1 = pd.read_csv("submission_0.95381.csv") # 那个高分文件
sub2 = pd.read_csv("submission_0.95105.csv") # 那个略低的文件

# 2. 暴力融合：高分占 70% 权重，低分占 30%
sub1["Heart Disease"] = sub1["Heart Disease"] * 0.7 + sub2["Heart Disease"] * 0.3

# 3. 保存并再次提交
sub1.to_csv("blended_submission.csv", index=False)