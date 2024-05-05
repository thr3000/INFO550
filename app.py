from flask import Flask, request, jsonify, send_from_directory
import torch
import time
import os
import threading
import numpy as np
from q_learning import QLearningAgent
from deep_q_learning import DeepQLearningAgent
import csv

app = Flask(__name__, static_folder='static')

# Score
@app.route('/update_score', methods=['POST'])
def update_score():
    data = request.json
    q_score = data['q_score']
    dql_score = data['dql_score']
    with open('scores.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if not os.path.exists('scores.csv') or os.stat('scores.csv').st_size == 0:
            writer.writerow(['Time', 'Q-Learning', 'Deep Q-Learning'])
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), q_score, dql_score])
        print("Scores saved to CSV at", time.strftime('%Y-%m-%d %H:%M:%S'))
    return jsonify({
        'message': "Score Updated"
    })

# Agents set up
alpha = 0.1
gamma = 0.99
epsilon = 0.1
ql_agent = QLearningAgent(alpha, gamma, epsilon)
dql_agent = DeepQLearningAgent(0.01, gamma, epsilon)

q_table_filename = 'q_table.npy'
if os.path.exists(q_table_filename):
    ql_agent.load_q_table(q_table_filename)
else:
    print("No existing Q-table found. Starting fresh.")
dql_filename = 'policy_net.pth'
if os.path.exists(dql_filename):
    dql_agent.load_model(dql_filename)
else:
    print("No existing DQL model found. Starting fresh.")

def prepare_dql_state(ball_x, ball_delta_x, ball_y, ball_delta_y, right_paddle_y):
    return torch.tensor([[ball_x, ball_delta_x, ball_y, ball_delta_y, right_paddle_y]], dtype=torch.float32)

@app.route('/update_game', methods=['POST'])
def update_game():
    data = request.json
    balls = data['balls']
    left_paddle_y = data['leftPaddleY']
    right_paddle_y = data['rightPaddleY']
    paddle_height = data.get('paddleHeight', 100)

    left_rewards = []
    right_rewards = []

    # Iterate over each ball to update positions and calculate rewards
    for ball in balls:
        ball_x = ball['x']
        ball_y = ball['y']
        ball_delta_x = ball['deltaX']
        ball_delta_y = ball['deltaY']

        # Update Left Paddle using Q-Learning
        current_state_ql = ql_agent.get_state([ball_y, ball_delta_y, left_paddle_y])
        action_ql = ql_agent.choose_action(current_state_ql)
        movement_map_ql = {0: -10, 1: 0, 2: 10}
        new_left_paddle_y = max(0, min(left_paddle_y + movement_map_ql[action_ql], 400 - paddle_height))

        # Calculate reward for left paddle
        reward_ql = 1 if ball_y >= new_left_paddle_y and ball_y <= new_left_paddle_y + paddle_height else -1
        distance_before_ql = abs(ball_y - left_paddle_y)
        distance_after_ql = abs(ball_y - new_left_paddle_y)
        reward_ql += 0.5 if distance_after_ql < distance_before_ql else -0.5
        left_rewards.append(reward_ql)

        # Update Q-table for left paddle
        new_state_ql = ql_agent.get_state([ball_y, ball_delta_y, new_left_paddle_y])
        ql_agent.update_q_table(current_state_ql, action_ql, reward_ql, new_state_ql)

        # Update Right Paddle using Deep Q-Learning
        current_state_dql = prepare_dql_state(ball_x, ball_delta_x, ball_y, ball_delta_y, right_paddle_y)
        action_dql = dql_agent.select_action(current_state_dql).item()
        movement_map_dql = {0: -10, 1: 0, 2: 10}
        new_right_paddle_y = max(0, min(right_paddle_y + movement_map_dql[action_dql], 400 - paddle_height))

        # Calculate reward for right paddle
        reward_dql = 1 if ball_y >= new_right_paddle_y and ball_y <= new_right_paddle_y + paddle_height else -1
        distance_before_dql = abs(ball_y - right_paddle_y)
        distance_after_dql = abs(ball_y - new_right_paddle_y)
        reward_dql += 0.5 if distance_after_dql < distance_before_dql else -0.5
        right_rewards.append(reward_dql)

        # Update memory and optimize model for right paddle
        next_state_dql = prepare_dql_state(ball_x, ball_delta_x, ball_y, ball_delta_y, new_right_paddle_y)
        dql_agent.memory.push(current_state_dql, torch.tensor([[action_dql]], dtype=torch.long), next_state_dql, torch.tensor([reward_dql], dtype=torch.float))
        torch.autograd.set_detect_anomaly(True)
        dql_agent.optimize_model()

    # Update paddle positions to the last processed ball's new positions
    return jsonify({
        'leftPaddleY': new_left_paddle_y,
        'rightPaddleY': new_right_paddle_y
    })


def save_periodically():
    while True:
        time.sleep(300)  # Save every 5 minutes
        ql_agent.save_q_table(q_table_filename)
        torch.save(dql_agent.save_model, dql_filename)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'interface.html')

if __name__ == "__main__":
    save_thread = threading.Thread(target=save_periodically)
    save_thread.start()
    app.run(debug=True)