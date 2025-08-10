"""
Vlastní implementace RL algoritmů od nuly bez framworku

Reinforcement Learning

Klíčové funkce

3 RL algoritmy: Q-Table, DQN (Deep Q-Network), REINFORCE
GridWorld prostředí: Konfigurovatelné překážky nebo bažiny
Interaktivní interface: Volba parametrů přes terminál
BFS analýza optimality: Porovnání s teoreticky nejkratšími cestami
Detailní vizualizace: ASCII mapy s očíslovanými kroky
Kompletní statistiky: Rozložení délek cest, četnost tras
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class GridWorld:
    def __init__(self, grid_size=6):
        self.size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        self.obstacles = []
        self.swamps = []
        self.terrain_type = "obstacles"
        self.generate_terrain()
        self.reset()
        
    def generate_terrain(self, num_terrain=None):
        if num_terrain is None:
            num_terrain = max(2, self.size // 3)
        
        positions = []
        max_attempts = num_terrain * 5  # Více pokusů pro větší gridy
        
        for _ in range(max_attempts):
            x = np.random.randint(1, self.size-1)
            y = np.random.randint(1, self.size-1)
            if (x, y) not in [self.start, self.goal] and (x, y) not in positions:
                positions.append((x, y))
                if len(positions) >= num_terrain:
                    break
        
        if self.terrain_type == "obstacles":
            self.obstacles = positions
            self.swamps = []
        else:
            self.swamps = positions
            self.obstacles = []
        
    def reset(self):
        self.pos = self.start
        return self._get_state()
    
    def _get_state(self):
        # Pro neural networks - flatten state
        state = np.zeros((self.size, self.size))
        state[self.pos] = 1  # Agent position
        state[self.goal] = 0.5  # Goal
        for obs in self.obstacles:
            state[obs] = -1  # Obstacle
        for swamp in self.swamps:
            state[swamp] = -0.5  # Swamp
        return state.flatten()
    
    def step(self, action):
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        dx, dy = moves[action]
        new_x = max(0, min(self.size-1, self.pos[0] + dx))
        new_y = max(0, min(self.size-1, self.pos[1] + dy))
        
        if (new_x, new_y) not in self.obstacles:
            self.pos = (new_x, new_y)
        
        reward = self._get_reward()
        done = self.pos == self.goal
        return self._get_state(), reward, done
    
    def _get_reward(self):
        if self.pos == self.goal:
            return 10
        elif self.pos in self.obstacles:
            return -5
        elif self.pos in self.swamps:
            return -2
        else:
            return -0.1
    
    def print_grid(self, path=None):
        print("\n" + "="*15)
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if path and (i, j) in path:
                    # Najdi pozici v cestě
                    step = path.index((i, j))
                    if step == 0:
                        row += "S "  # Start
                    elif step == len(path) - 1:
                        row += "G "  # Goal
                    else:
                        row += f"{step} " if step < 10 else "* "
                elif (i, j) == self.pos and not path:
                    row += "A "  # Agent (pouze pokud nezobrazujeme cestu)
                elif (i, j) == self.goal and not path:
                    row += "G "  # Goal
                elif (i, j) in self.obstacles:
                    row += "# "  # Obstacle
                elif (i, j) in self.swamps:
                    row += "~ "  # Swamp
                else:
                    row += ". "  # Empty
            print(row)
        print("="*15)
    
    def print_big_path(self, path):
        print("\n=== ZVĚTŠENÁ MAPA S CESTOU ===")
        
        grid = [[" " for _ in range(self.size)] for _ in range(self.size)]
        
        # Označ překážky a bažiny
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = "###"
        for swamp in self.swamps:
            grid[swamp[0]][swamp[1]] = "~~~"
        
        # Označ cestu s pořadovými čísly
        for i, (x, y) in enumerate(path):
            if i == 0:
                grid[x][y] = "STA"  # Start
            elif i == len(path) - 1:
                grid[x][y] = "CÍL"  # Cíl
            else:
                grid[x][y] = f"{i:2d} "  # Pořadové číslo kroku
        
        # Vytiskni s velkými políčky
        print("┌" + "────┬" * (self.size-1) + "────┐")
        for i, row in enumerate(grid):
            print("│", end="")
            for cell in row:
                print(f"{cell:^4}│", end="")
            print()
            if i < self.size - 1:
                print("├" + "────┼" * (self.size-1) + "────┤")
        print("└" + "────┴" * (self.size-1) + "────┘")
        
        terrain_legend = "###=Překážka" if self.obstacles else "~~~=Bažina"
        print(f"\nLegenda: STA=Start, CÍL=Cíl, {terrain_legend}, čísla=kroky")
        print(f"Celkem kroků: {len(path)-1}")

def find_all_shortest_paths_bfs(env):
    """Najde všechny nejkratší cesty pomocí BFS"""
    from collections import deque
    
    start = env.start
    goal = env.goal
    obstacles = set(env.obstacles)
    
    # BFS pro najítí nejkratší vzdálenosti
    queue = deque([(start, 0)])
    distances = {start: 0}
    
    while queue:
        pos, dist = queue.popleft()
        
        if pos == goal:
            continue
            
        for action, (dx, dy) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
            new_x = max(0, min(env.size-1, pos[0] + dx))
            new_y = max(0, min(env.size-1, pos[1] + dy))
            new_pos = (new_x, new_y)
            
            # Přeskočit překážky nebo mimo hranice
            if new_pos in obstacles or new_pos == pos:
                continue
                
            if new_pos not in distances:
                distances[new_pos] = dist + 1
                queue.append((new_pos, dist + 1))
    
    if goal not in distances:
        return [], float('inf')
    
    optimal_distance = distances[goal]
    
    # Najdi všechny optimální cesty rekurzivně
    all_paths = []
    
    def find_paths(current_pos, current_path, remaining_steps):
        if remaining_steps < 0:
            return
            
        if current_pos == goal and remaining_steps == 0:
            all_paths.append(current_path[:])
            return
            
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x = max(0, min(env.size-1, current_pos[0] + dx))
            new_y = max(0, min(env.size-1, current_pos[1] + dy))
            new_pos = (new_x, new_y)
            
            if (new_pos not in obstacles and 
                new_pos not in current_path and 
                new_pos in distances and
                distances[new_pos] == remaining_steps):
                
                current_path.append(new_pos)
                find_paths(new_pos, current_path, remaining_steps - 1)
                current_path.pop()
    
    find_paths(start, [start], optimal_distance)
    return all_paths, optimal_distance

def analyze_agent_path_optimality(env, agent_path, all_optimal_paths):
    """Analyzuje jak dobrá je cesta agenta ve srovnání s optimálními"""
    agent_length = len(agent_path) - 1
    optimal_length = len(all_optimal_paths[0]) - 1 if all_optimal_paths else float('inf')
    
    print(f"\n=== ANALÝZA OPTIMALITY ===")
    print(f"Cesta agenta: {agent_length} kroků")
    print(f"Optimální délka: {optimal_length} kroků")
    print(f"Počet optimálních cest: {len(all_optimal_paths)}")
    
    if agent_length == optimal_length:
        # Zkontroluj, jestli je cesta agenta mezi optimálními
        agent_tuple = tuple(agent_path)
        is_optimal = any(tuple(path) == agent_tuple for path in all_optimal_paths)
        
        if is_optimal:
            print("✅ Agent našel OPTIMÁLNÍ cestu!")
            path_index = next(i for i, path in enumerate(all_optimal_paths) if tuple(path) == agent_tuple)
            print(f"   (Varianta {path_index + 1} z {len(all_optimal_paths)})")
        else:
            print("✅ Agent má optimální DÉLKU, ale jinou cestu")
            print("   (Stejný počet kroků, ale jiná trasa)")
    else:
        efficiency = (optimal_length / agent_length) * 100
        print(f"❌ Agent není optimální - efektivita {efficiency:.1f}%")
        extra_steps = agent_length - optimal_length
        print(f"   (O {extra_steps} kroků více než nutné)")
    
    return agent_length == optimal_length

# 1. Q-TABLE ALGORITHM
class QTableAgent:
    def __init__(self, grid_size):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.rewards_history = []
        self.successful_paths = []  # Všechny úspěšné cesty
        self.algorithm = "Q-Table"
        
    def get_state_key(self, state):
        return tuple(state)
    
    def get_q_value(self, state, action):
        key = self.get_state_key(state)
        return self.q_table.get((key, action), 0.0)
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        
        q_values = [self.get_q_value(state, a) for a in range(4)]
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        old_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            next_max = max([self.get_q_value(next_state, a) for a in range(4)])
            target = reward + self.gamma * next_max
        
        key = self.get_state_key(state)
        self.q_table[(key, action)] = old_q + self.lr * (target - old_q)

# 2. DQN (DEEP Q-NETWORK)
class DQN(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)  # 4 actions
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 0.001
        self.gamma = 0.95
        self.batch_size = 32
        self.rewards_history = []
        self.successful_paths = []  # Všechny úspěšné cesty
        self.algorithm = "DQN"
        
        self.q_network = DQN(state_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self._replay()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 3. REINFORCE ALGORITHM
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class REINFORCEAgent:
    def __init__(self, state_size):
        self.policy_network = PolicyNetwork(state_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.gamma = 0.99
        self.episode_rewards = []
        self.episode_log_probs = []
        self.rewards_history = []
        self.successful_paths = []  # Všechny úspěšné cesty
        self.algorithm = "REINFORCE"
        
    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        self.episode_log_probs.append(action_dist.log_prob(action))
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        self.episode_rewards.append(reward)
        
        if done:
            self._update_policy()
            
    def _update_policy(self):
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative = 0
        for reward in reversed(self.episode_rewards):
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)
        
        # Normalize rewards
        rewards_tensor = torch.FloatTensor(discounted_rewards)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, reward in zip(self.episode_log_probs, rewards_tensor):
            policy_loss.append(-log_prob * reward)
        
        # Update network
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Reset episode data
        self.episode_rewards = []
        self.episode_log_probs = []

def get_user_preferences():
    print("=== POKROČILÉ RL ALGORITMY ===\n")
    
    # Velikost gridu
    while True:
        try:
            size = int(input("Velikost gridu (6-10): "))
            if 6 <= size <= 10:
                break
            print("Zadejte číslo mezi 6 a 10")
        except ValueError:
            print("Zadejte platné číslo")
    
    # Typ terénu
    print("\nTypy terénu:")
    print("1. Překážky (nelze projít)")
    print("2. Bažiny (pomalé, -2 odměna)")
    while True:
        choice = input("Vyberte (1/2): ").strip()
        if choice == "1":
            terrain = "obstacles"
            break
        elif choice == "2":
            terrain = "swamps"
            break
        print("Zadejte 1 nebo 2")
    
    # Počet překážek/bažin
    max_terrain = (size-2) * (size-2) // 3  # Max cca třetina dostupných políček
    while True:
        try:
            num_terrain = int(input(f"Počet {'překážek' if terrain == 'obstacles' else 'bažin'} (1-{max_terrain}): "))
            if 1 <= num_terrain <= max_terrain:
                break
            print(f"Zadejte číslo mezi 1 a {max_terrain}")
        except ValueError:
            print("Zadejte platné číslo")
    
    # Algoritmus
    print("\nRL Algoritmy:")
    print("1. Q-Table (klasická tabulka)")
    print("2. DQN (Deep Q-Network)")
    print("3. REINFORCE (Policy Gradient)")
    while True:
        choice = input("Vyberte (1/2/3): ").strip()
        if choice == "1":
            algorithm = "qtable"
            break
        elif choice == "2":
            algorithm = "dqn"
            break
        elif choice == "3":
            algorithm = "reinforce"
            break
        print("Zadejte 1, 2 nebo 3")
    
    # Počet epizod
    episodes = 500 if algorithm == "qtable" else 1000
    print(f"Počet epizod: {episodes} (optimální pro {algorithm.upper()})")
    
    return size, terrain, algorithm, episodes, num_terrain

def plot_results(agent, episodes):
    if len(agent.rewards_history) < 2:
        print("Nedostatek dat pro graf")
        return
    
    plt.figure(figsize=(12, 4))
    
    # Graf odměn
    plt.subplot(1, 2, 1)
    plt.plot(agent.rewards_history, alpha=0.7, linewidth=1)
    if len(agent.rewards_history) > 50:
        window = 50
        moving_avg = [np.mean(agent.rewards_history[max(0, i-window):i+1]) 
                     for i in range(len(agent.rewards_history))]
        plt.plot(moving_avg, 'r-', linewidth=2, label=f'Průměr ({window})')
        plt.legend()
    
    plt.title(f'Vývoj odměn - {agent.algorithm}')
    plt.xlabel('Epizoda')
    plt.ylabel('Celková odměna')
    plt.grid(True, alpha=0.3)
    
    # Statistiky
    plt.subplot(1, 2, 2)
    recent_rewards = agent.rewards_history[-100:] if len(agent.rewards_history) > 100 else agent.rewards_history
    plt.hist(recent_rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribuce odměn (posledních 100 epizod)')
    plt.xlabel('Odměna')
    plt.ylabel('Četnost')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== STATISTIKY {agent.algorithm.upper()} ===")
    print(f"Průměrná odměna: {np.mean(agent.rewards_history):.2f}")
    print(f"Nejlepší odměna: {np.max(agent.rewards_history):.2f}")
    if len(recent_rewards) > 0:
        print(f"Průměr posledních 100: {np.mean(recent_rewards):.2f}")

# Hlavní program
if __name__ == "__main__":
    try:
        grid_size, terrain_type, algorithm, num_episodes, num_terrain = get_user_preferences()
        
        # Vytvoř prostředí
        env = GridWorld(grid_size)
        env.terrain_type = terrain_type
        env.generate_terrain(num_terrain)
        
        # Vytvoř agenta podle výběru
        state_size = grid_size * grid_size
        if algorithm == "qtable":
            agent = QTableAgent(grid_size)
        elif algorithm == "dqn":
            agent = DQNAgent(state_size)
        elif algorithm == "reinforce":
            agent = REINFORCEAgent(state_size)
        
        print(f"\n=== ZAČÍNÁ TRÉNINK ===")
        terrain_name = "překážky" if terrain_type == "obstacles" else "bažiny"
        print(f"Grid: {grid_size}x{grid_size}, {num_terrain} {terrain_name}, Algoritmus: {agent.algorithm}")
        env.print_grid()
        
        # Trénink
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            episode_path = [env.pos]  # Sleduj cestu této epizody
            
            while steps < 200:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                episode_path.append(env.pos)  # Přidej pozici
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    # Uložit úspěšnou cestu
                    agent.successful_paths.append({
                        'episode': episode,
                        'path': episode_path.copy(),
                        'steps': len(episode_path) - 1,
                        'reward': total_reward
                    })
                    break
            
            agent.rewards_history.append(total_reward)
            
            if episode % (num_episodes // 10) == 0:
                print(f"Epizoda {episode}: {total_reward:.1f} odměna, {steps} kroků")
        
        print(f"\n=== TRÉNINK DOKONČEN ===")
        
        # Test nejlepší cesty
        print("\n=== TEST NEJLEPŠÍ CESTY ===")
        
        # Vypni exploraci pro čistě optimální cestu
        if hasattr(agent, 'epsilon'):
            old_epsilon = agent.epsilon
            agent.epsilon = 0
        
        state = env.reset()
        path_positions = [env.pos]  # Začni s pozicí, ne stavem
        total_reward = 0
        
        for step in range(50):  # Max 50 kroků
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            path_positions.append(env.pos)
            total_reward += reward
            
            if done:
                break
        
        # Obnov epsilon pokud existuje
        if hasattr(agent, 'epsilon'):
            agent.epsilon = old_epsilon
        
        print(f"Nejkratší cesta: {len(path_positions)-1} kroků")
        print(f"Celková odměna: {total_reward:.1f}")
        print(f"Cesta: {path_positions}")
        
        # Analyzuj všechny cesty které agent našel
        print(f"\n=== VŠECHNY CESTY AGENTA BĚHEM TRÉNINKU ===")
        print(f"DEBUG: agent.successful_paths má {len(agent.successful_paths)} cest")
        if agent.successful_paths:
            print(f"Agent našel cestu {len(agent.successful_paths)}x během tréninku")
            
            # Seřaď podle délky (nejkratší první)
            sorted_paths = sorted(agent.successful_paths, key=lambda x: x['steps'])
            shortest_length = sorted_paths[0]['steps']
            
            # Najdi všechny nejkratší cesty
            shortest_paths = [p for p in sorted_paths if p['steps'] == shortest_length]
            
            # Najdi unikátní nejkratší cesty (stejná trasa = duplicita)
            unique_shortest = {}
            for path_info in shortest_paths:
                path_tuple = tuple(path_info['path'])  # Převeď na tuple pro hashování
                if path_tuple not in unique_shortest:
                    unique_shortest[path_tuple] = path_info
                # Jinak je duplicita - ignoruj
            
            unique_paths_list = list(unique_shortest.values())
            
            print(f"\n=== NEJKRATŠÍ CESTY AGENTA ({shortest_length} kroků) ===")
            print(f"Celkem pokusů s nejkratší délkou: {len(shortest_paths)}")
            print(f"Unikátní nejkratší trasy: {len(unique_paths_list)}")
            print(f"Duplicity: {len(shortest_paths) - len(unique_paths_list)}")
            
            print(f"\n=== UNIKÁTNÍ NEJKRATŠÍ TRASY ===")
            for i, path_info in enumerate(unique_paths_list[:5]):  # Zobraz max 5
                print(f"\nUnikátní trasa {i+1} (první výskyt: epizoda {path_info['episode']}):")
                print(f"  Kroky: {path_info['steps']}, Odměna: {path_info['reward']:.1f}")
                print(f"  Trasa: {path_info['path']}")
            
            if len(unique_paths_list) > 5:
                print(f"... a dalších {len(unique_paths_list)-5} unikátních tras")
            
            # Vizualizace všech unikátních nejkratších cest
            print(f"\n=== VIZUALIZACE UNIKÁTNÍCH NEJKRATŠÍCH TRAS ===")
            for i, path_info in enumerate(unique_paths_list[:3]):  # Max 3 mapy
                duplicates_count = len([p for p in shortest_paths if tuple(p['path']) == tuple(path_info['path'])])
                print(f"\n--- UNIKÁTNÍ TRASA {i+1} (Objevena {duplicates_count}x, první v epizodě {path_info['episode']}) ---")
                env.print_big_path(path_info['path'])
            
            if len(unique_paths_list) > 3:
                print(f"\n(Zbývajících {len(unique_paths_list)-3} unikátních tras není vizualizováno)")
            
            # Analýza četnosti tras
            if len(unique_paths_list) > 1:
                print(f"\n=== ČETNOST JEDNOTLIVÝCH TRAS ===")
                path_counts = {}
                for path_info in shortest_paths:
                    path_tuple = tuple(path_info['path'])
                    path_counts[path_tuple] = path_counts.get(path_tuple, 0) + 1
                
                # Seřaď podle četnosti
                sorted_counts = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
                for i, (path_tuple, count) in enumerate(sorted_counts[:5]):
                    percentage = (count / len(shortest_paths)) * 100
                    print(f"Trasa {i+1}: {count}x ({percentage:.1f}%) - {list(path_tuple)}")
            
            # Statistiky všech cest
            all_lengths = [p['steps'] for p in agent.successful_paths]
            print(f"\n=== STATISTIKY VŠECH ÚSPĚŠNÝCH CEST ===")
            print(f"Nejkratší: {min(all_lengths)} kroků")
            print(f"Nejdelší: {max(all_lengths)} kroků")
            print(f"Průměr: {np.mean(all_lengths):.1f} kroků")
            
            # Zobraz rozložení délek
            length_counts = {}
            for length in all_lengths:
                length_counts[length] = length_counts.get(length, 0) + 1
            
            print(f"\nRozložení délek cest:")
            for length in sorted(length_counts.keys()):
                count = length_counts[length]
                percentage = (count / len(all_lengths)) * 100
                print(f"  {length} kroků: {count}x ({percentage:.1f}%)")
        else:
            print("Agent nenašel cestu k cíli ani jednou!")
        
        # Porovnání s teoreticky optimálními cestami (pouze pokud agent našel cestu)
        if agent.successful_paths:
            print("\n=== POROVNÁNÍ S TEORETICKY OPTIMÁLNÍMI CESTAMI ===")
            all_optimal_paths, optimal_distance = find_all_shortest_paths_bfs(env)
            
            if all_optimal_paths:
                print(f"Teoreticky existuje {len(all_optimal_paths)} optimálních cest s délkou {optimal_distance} kroků")
                
                agent_best = min(p['steps'] for p in agent.successful_paths)
                if agent_best == optimal_distance:
                    print("✅ Agent našel optimální cestu!")
                else:
                    efficiency = (optimal_distance / agent_best) * 100
                    print(f"❌ Agent nenašel optimální cestu - efektivita {efficiency:.1f}%")
                    print(f"   Agent nejlepší: {agent_best}, Optimální: {optimal_distance}")
        
        # Zobraz cestu v malém gridu
        print("\n=== MAPA S CESTOU AGENTA ===")
        env.print_grid(path_positions)
        
        # Zobraz zvětšenou cestu
        env.print_big_path(path_positions)
        
        # Zobraz grafy
        plot_results(agent, num_episodes)
        
    except KeyboardInterrupt:
        print("\nTrénink přerušen uživatelem")
    except Exception as e:
        print(f"Chyba: {e}")
        print("Ujistěte se, že máte nainstalované: torch matplotlib numpy")
