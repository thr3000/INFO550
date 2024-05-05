import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

csv_file_path = './scores.csv'
df = pd.read_csv(csv_file_path)

start_time = datetime.strptime('5/5/24 5:41', '%m/%d/%y %H:%M')
df['Time'] = pd.to_datetime(df['Time'])
start_time = df['Time'].min()
df['Minutes'] = (df['Time'] - start_time).dt.total_seconds() / 60

plt.figure(figsize=(10, 6))
plt.plot(df['Minutes'], df['Q-Learning'], label='Q-Learning')
plt.plot(df['Minutes'], df['Deep Q-Learning'], label='Deep Q-Learning')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Q-Learning vs Deep Q-Learning Pong Scores')
plt.legend()
plt.grid(True)

plt.savefig('./scores.png')
print("Graph saved")
