"""
GridWorld DQN - Framwork

Kombinuje jednoduchost původního rl_algorithms.py s "frameworkem" z gymnasium. Používá Stable-Baselines3 pro robustní DQN.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from collections import deque


class GridWorldEnv(gym.Env):
    """GridWorld prostředí kompatibilní s Gymnasium pro SB3."""
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, size=8, terrain_type="obstacles", num_terrain=None, max_steps=500):
        super().__init__()
        
        self.size = size
        self.terrain_type = terrain_type
        self.num_terrain = num_terrain or max(2, size // 3)
        self.max_steps = max_steps
        
        # Pozice
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        
        # Terén
        self.obstacles = []
        self.swamps = []
        self._generate_terrain()
        
        # Stav
        self.agent_pos = None
        self.step_count = 0
        
        # Gymnasium spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        obs_dim = size * size
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _generate_terrain(self):
        """Generuj překážky/bažiny jako v původním kódu."""
        positions = []
        max_attempts = self.num_terrain * 5
        
        for _ in range(max_attempts):
            x = np.random.randint(1, self.size-1)
            y = np.random.randint(1, self.size-1)
            if (x, y) not in [self.start_pos, self.goal_pos] and (x, y) not in positions:
                positions.append((x, y))
                if len(positions) >= self.num_terrain:
                    break
        
        if self.terrain_type == "obstacles":
            self.obstacles = positions
            self.swamps = []
        else:
            self.swamps = positions
            self.obstacles = []
    
    def _get_obs(self):
        """Pozorování jako v původním kódu."""
        state = np.zeros((self.size, self.size), dtype=np.float32)
        state[self.agent_pos] = 1.0  # Agent
        state[self.goal_pos] = 0.5   # Goal
        
        for obs in self.obstacles:
            state[obs] = -1.0  # Obstacle
        for swamp in self.swamps:
            state[swamp] = -0.5  # Swamp
            
        state[self.agent_pos] = 1.0  # Agent overrides
        return state.flatten()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_pos = self.start_pos
        self.step_count = 0
        
        # Možnost regenerovat terén
        if options and options.get("regenerate_terrain", False):
            self._generate_terrain()
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        
        # Pohyb jako v původním kódu
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        dx, dy = moves[action]
        new_x = max(0, min(self.size-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size-1, self.agent_pos[1] + dy))
        
        # Kontrola překážek
        if (new_x, new_y) not in self.obstacles:
            self.agent_pos = (new_x, new_y)
        
        # Odměny jako v původním kódu
        reward = self._get_reward()
        terminated = self.agent_pos == self.goal_pos
        truncated = self.step_count >= self.max_steps
        
        info = {
            "agent_pos": self.agent_pos,
            "step_count": self.step_count,
            "is_success": terminated
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_reward(self):
        """Vylepšené odměny s reward shaping pro lepší learning."""
        if self.agent_pos == self.goal_pos:
            return 100.0  # Větší odměna za cíl
        elif self.agent_pos in self.obstacles:
            return -5.0
        elif self.agent_pos in self.swamps:
            return -2.0
        else:
            # Reward shaping - odměna na základě vzdálenosti k cíli
            distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            max_distance = 2 * (self.size - 1)
            distance_reward = (max_distance - distance) / max_distance  # 0 až 1
            return distance_reward * 0.1 - 0.05  # -0.05 až +0.05
    
    def render(self, mode="ansi"):
        """Vykreslení jako v původním kódu."""
        if mode == "ansi":
            return self._render_ansi()
        return None
    
    def _render_ansi(self):
        """ASCII vykreslení zachovává původní styl."""
        print("\n" + "="*15)
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if (i, j) == self.agent_pos:
                    row += "A "  # Agent
                elif (i, j) == self.goal_pos:
                    row += "G "  # Goal
                elif (i, j) in self.obstacles:
                    row += "# "  # Obstacle
                elif (i, j) in self.swamps:
                    row += "~ "  # Swamp
                else:
                    row += ". "  # Empty
            print(row)
        print("="*15)
        print(f"Step: {self.step_count}, Pos: {self.agent_pos}")
    
    def print_path(self, path):
        """Velká mapa s cestou - OPRAVENÁ verze."""
        print("\n=== ZVĚTŠENÁ MAPA S CESTOU ===")
        
        # Validate path
        if not path or len(path) == 0:
            print("❌ Prázdná cesta!")
            return
            
        # Debug info
        print(f"DEBUG: Cesta má {len(path)} pozic: {path[:5]}...{path[-5:] if len(path) > 10 else path}")
        
        # Initialize grid
        grid = [[" " for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark obstacles and swamps
        for obs in self.obstacles:
            if 0 <= obs[0] < self.size and 0 <= obs[1] < self.size:
                grid[obs[0]][obs[1]] = "###"
        for swamp in self.swamps:
            if 0 <= swamp[0] < self.size and 0 <= swamp[1] < self.size:
                grid[swamp[0]][swamp[1]] = "~~~"
        
        # Mark path with step numbers - CAREFUL with bounds checking
        for i, (x, y) in enumerate(path):
            # Bounds check
            if not (0 <= x < self.size and 0 <= y < self.size):
                print(f"⚠️  WARNING: Position ({x}, {y}) out of bounds!")
                continue
                
            if i == 0:
                grid[x][y] = "STA"  # Start
            elif i == len(path) - 1:
                grid[x][y] = "CÍL"  # Goal
            else:
                # Format step number properly
                if i < 10:
                    grid[x][y] = f" {i} "  # Single digit with spaces
                elif i < 100:
                    grid[x][y] = f"{i} "   # Two digits with space
                else:
                    grid[x][y] = "***"     # Too many steps
        
        # Print table with proper formatting
        print("┌" + "────┬" * (self.size-1) + "────┐")
        for i, row in enumerate(grid):
            print("│", end="")
            for cell in row:
                print(f"{cell:^4}│", end="")
            print()  # New line after each row
            
            # Print horizontal separator (except after last row)
            if i < self.size - 1:
                print("├" + "────┼" * (self.size-1) + "────┤")
        print("└" + "────┴" * (self.size-1) + "────┘")
        
        # Legend
        terrain_legend = "###=Překážka" if self.obstacles else "~~~=Bažina"  
        print(f"\nLegenda: STA=Start, CÍL=Cíl, {terrain_legend}, čísla=kroky")
        print(f"Celkem kroků: {len(path)-1}")
        
        # Additional debug
        print(f"Grid size: {self.size}x{self.size}, Path length: {len(path)}")


def find_all_shortest_paths_bfs(env):
    """Najde všechny nejkratší cesty pomocí BFS - s debug výpisem"""
    start = env.start_pos
    goal = env.goal_pos
    obstacles = set(env.obstacles)
    size = env.size
    
    # DEBUG informace
    print(f"DEBUG BFS: Start {start}, Goal {goal}, Size {size}")
    print(f"DEBUG BFS: {len(obstacles)} obstacles: {sorted(obstacles)[:10]}...")
    
    # BFS pro najítí nejkratší vzdálenosti
    queue = deque([(start, 0)])
    distances = {start: 0}
    
    while queue:
        pos, dist = queue.popleft()
        
        if pos == goal:
            continue
            
        for action, (dx, dy) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
            new_x = max(0, min(size-1, pos[0] + dx))
            new_y = max(0, min(size-1, pos[1] + dy))
            new_pos = (new_x, new_y)
            
            # Přeskočit překážky nebo mimo hranice
            if new_pos in obstacles or new_pos == pos:
                continue
                
            if new_pos not in distances:
                distances[new_pos] = dist + 1
                queue.append((new_pos, dist + 1))
    
    if goal not in distances:
        print(f"DEBUG BFS: Goal {goal} NOT reachable from start {start}")
        print(f"DEBUG BFS: Reachable positions: {len(distances)}")
        print(f"DEBUG BFS: Sample reachable: {list(distances.keys())[:10]}")
        return [], float('inf')
    
    optimal_distance = distances[goal]
    print(f"DEBUG BFS: Goal {goal} IS reachable! Distance: {optimal_distance}")
    
    # Najdi všechny optimální cesty rekurzivně - s debug výpisem
    all_paths = []
    
    def find_paths(current_pos, current_path, remaining_steps):
        if remaining_steps < 0:
            return
            
        if current_pos == goal and remaining_steps == 0:
            all_paths.append(current_path[:])
            print(f"DEBUG: Nalezena optimální cesta: {current_path}")
            return
            
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x = max(0, min(size-1, current_pos[0] + dx))
            new_y = max(0, min(size-1, current_pos[1] + dy))
            new_pos = (new_x, new_y)
            
            if (new_pos not in obstacles and 
                new_pos not in current_path and 
                new_pos in distances and
                distances[new_pos] == remaining_steps):
                
                current_path.append(new_pos)
                find_paths(new_pos, current_path, remaining_steps - 1)
                current_path.pop()
    
    print(f"DEBUG: Hledám cesty ze start {start} s distance {optimal_distance}")
    print(f"DEBUG: Distances obsahuje {len(distances)} pozic")
    print(f"DEBUG: První 10 distances: {dict(list(distances.items())[:10])}")
    
    find_paths(start, [start], optimal_distance)
    
    print(f"DEBUG: Celkem nalezeno {len(all_paths)} optimálních cest")
    
    # Pokud žádné cesty nenašlo, zkus jednoduchší přístup
    if not all_paths:
        print("DEBUG: Rekurzivní hledání selhalo, zkouším přímou cestu...")
        # Jednoduchý backtrack z cíle
        current = goal
        simple_path = [current]
        
        while current != start and len(simple_path) < optimal_distance + 5:
            found_prev = False
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                prev_x = current[0] + dx
                prev_y = current[1] + dy
                prev_pos = (prev_x, prev_y)
                
                if (prev_pos in distances and 
                    distances[prev_pos] == distances[current] - 1):
                    simple_path.insert(0, prev_pos)
                    current = prev_pos
                    found_prev = True
                    break
            
            if not found_prev:
                break
        
        if current == start:
            all_paths = [simple_path]
            print(f"DEBUG: Nalezena jednoduchá cesta: {simple_path}")
    return all_paths, optimal_distance


def analyze_agent_path_optimality(env, agent_paths, all_optimal_paths):
    """Analyzuje jak dobrá je cesta agenta ve srovnání s optimálními - opravená verze"""
    if not agent_paths:
        print("Agent nemá žádné úspěšné cesty k analýze")
        return
    
    # Najdi nejkratší cesty agenta
    agent_lengths = [p['steps'] for p in agent_paths]
    min_agent_length = min(agent_lengths)
    
    optimal_length = len(all_optimal_paths[0]) - 1 if all_optimal_paths else float('inf')
    
    print(f"\n=== ANALÝZA OPTIMALITY ===")
    print(f"Nejkratší cesta agenta: {min_agent_length} kroků")
    print(f"Teoreticky optimální délka: {optimal_length} kroků")
    print(f"Počet teoreticky optimálních cest: {len(all_optimal_paths)}")
    
    if min_agent_length == optimal_length:
        print("✅ Agent našel OPTIMÁLNÍ cestu!")
        
        # Zkontroluj počet optimálních cest od agenta
        agent_shortest = [p for p in agent_paths if p['steps'] == min_agent_length]
        print(f"Agent našel {len(agent_shortest)} cest s optimální délkou")
        
    else:
        if optimal_length == float('inf'):
            print("⚠️  Žádná teoreticky optimální cesta neexistuje (cíl nedosažitelný)")
        else:
            efficiency = (optimal_length / min_agent_length) * 100
            print(f"❌ Agent není optimální - efektivita {efficiency:.1f}%")
            extra_steps = min_agent_length - optimal_length
            print(f"   (O {extra_steps} kroků více než nutné)")
    
    # Zobraz některé teoreticky optimální cesty
    if all_optimal_paths:
        paths_to_show = min(3, len(all_optimal_paths))
        print(f"\n=== TEORETICKY OPTIMÁLNÍ CESTY (zobrazeno {paths_to_show} z {len(all_optimal_paths)}) ===")
        for i, path in enumerate(all_optimal_paths[:paths_to_show]):
            print(f"\nOptimální cesta {i+1} ({len(path)-1} kroků):")
            print(f"  Pozice: {path}")
            
            # Zobraz mapu pro první 2
            if i < 2:
                env.print_path(path)
    
    return min_agent_length == optimal_length


class GridWorldDQNCallback(BaseCallback):
    """Opravený callback pro sledování tréninku."""
    
    def __init__(self, check_freq=1000, save_path="./logs/", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.successful_paths = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Vytvoř složku pro logy
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self):
        # Sleduj odměny a délky epizod
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Zkontroluj jestli skončila epizoda
        done = self.locals.get('dones', [False])[0]
        if done:
            # Epizoda skončila
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # DEBUG: Zkontroluj proč není detekována jako úspěšná
            info = self.locals.get('infos', [{}])[0]
            is_success = info.get('is_success', False)
            
            # OPRAVENO: Používej reward jako indikátor úspěchu
            # Pokud reward je pozitivní, agent dosáhl cíle (reward 100 - malé penalty)
            success_by_reward = self.current_episode_reward > 50  # Threshold pro úspěch
            
            if is_success or success_by_reward:
                # Úspěšná epizoda
                self.successful_paths.append({
                    'episode': len(self.successful_paths),
                    'reward': self.current_episode_reward,
                    'steps': self.current_episode_length,
                    'timestep': self.num_timesteps,
                    'detection_method': 'info_flag' if is_success else 'reward_threshold'
                })
                
                if self.verbose > 1:
                    method = 'info_flag' if is_success else 'reward_threshold'
                    print(f"DEBUG: Úspěšná cesta detekována ({method}): {self.current_episode_reward:.1f} reward, {self.current_episode_length} kroků")
            
            # Reset pro novou epizodu
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        if self.num_timesteps % self.check_freq == 0 and self.verbose > 0:
            print(f"Timestep {self.num_timesteps}: {len(self.successful_paths)} úspěšných cest")
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-10:]
                print(f"  Průměrná odměna (10 posledních): {np.mean(recent_rewards):.2f}")
        
        return True


def get_user_preferences():
    """Zachováváme původní interaktivní rozhraní."""
    print("=== GRIDWORLD DQN - FRAMEWORK STYLE ===\n")
    
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
    max_terrain = (size-2) * (size-2) // 3
    while True:
        try:
            num_terrain = int(input(f"Počet {'překážek' if terrain == 'obstacles' else 'bažin'} (1-{max_terrain}): "))
            if 1 <= num_terrain <= max_terrain:
                break
            print(f"Zadejte číslo mezi 1 a {max_terrain}")
        except ValueError:
            print("Zadejte platné číslo")
    
    # Počet kroků - automaticky na základě složitosti
    base_timesteps = 50000  # Základní počet pro 6x6
    complexity_factor = (size * size * num_terrain) / (6 * 6 * 2)  # Relativní složitost
    recommended_timesteps = int(base_timesteps * complexity_factor)
    
    print(f"\nDoporučený počet kroků pro {size}x{size} s {num_terrain} překážkami/bažinami: {recommended_timesteps}")
    
    while True:
        try:
            default_choice = input(f"Použít doporučené? (y/n, default=y): ").strip().lower()
            if default_choice in ['', 'y', 'yes']:
                timesteps = recommended_timesteps
                break
            elif default_choice in ['n', 'no']:
                timesteps = int(input("Vlastní počet kroků (10000-200000): "))
                if 10000 <= timesteps <= 200000:
                    break
                print("Zadejte číslo mezi 10000 a 200000")
            else:
                print("Zadejte y nebo n")
        except ValueError:
            print("Zadejte platné číslo")
    
    return size, terrain, num_terrain, timesteps


def analyze_training_results(callback, env):
    """Analyzuj výsledky s detailní diagnostikou - BEZ cest (callback je nesleduje)."""
    print(f"\n=== ANALÝZA VÝSLEDKŮ ===")
    
    # Debug info
    print(f"Celkem epizod během tréninku: {len(callback.episode_rewards)}")
    if callback.episode_rewards:
        print(f"Průměrná odměna: {np.mean(callback.episode_rewards):.2f}")
        print(f"Nejlepší odměna: {max(callback.episode_rewards):.2f}")
        print(f"Průměrná délka epizody: {np.mean(callback.episode_lengths):.1f}")
    
    if not callback.successful_paths:
        print("❌ Agent nenašel cestu k cíli!")
        
        # Diagnostika proč se to nepovedlo
        if callback.episode_rewards:
            best_rewards = sorted(callback.episode_rewards, reverse=True)[:5]
            print(f"5 nejlepších odměn: {[f'{r:.1f}' for r in best_rewards]}")
            
            success_threshold = 80  # Odměna blízká úspěchu (100 - penalty za kroky)
            near_success = [r for r in callback.episode_rewards if r > success_threshold]
            print(f"Epizody blízko úspěchu (>{success_threshold}): {len(near_success)}")
        
        return
    
    print(f"✅ Agent našel cestu k cíli {len(callback.successful_paths)}x během tréninku!")
    
    # Najdi nejkratší cesty (podle kroků/délky epizody)
    min_steps = min(path['steps'] for path in callback.successful_paths)
    shortest_paths = [p for p in callback.successful_paths if p['steps'] == min_steps]
    
    print(f"\n=== STATISTIKY ÚSPĚŠNÝCH CEST ===")
    print(f"Nejkratší délka: {min_steps} kroků")
    print(f"Počet nejkratších cest: {len(shortest_paths)}")
    
    # Zobraz statistiky (bez vizualizace cest, kterou nemáme)
    all_steps = [p['steps'] for p in callback.successful_paths]
    all_rewards = [p['reward'] for p in callback.successful_paths]
    
    print(f"Průměrná délka úspěšných cest: {np.mean(all_steps):.1f} kroků")
    print(f"Nejdelší úspěšná cesta: {max(all_steps)} kroků")
    print(f"Průměrná odměna úspěšných cest: {np.mean(all_rewards):.1f}")
    
    # Zobraz první 3 nejkratší cesty (bez map)
    print(f"\n=== NEJKRATŠÍ CESTY Z TRÉNINKU ({min_steps} kroků) ===")
    for i, path_info in enumerate(shortest_paths[:5]):
        detection = path_info.get('detection_method', 'unknown')
        print(f"Cesta {i+1}: {path_info['steps']} kroků, odměna {path_info['reward']:.1f}, "
              f"timestep {path_info['timestep']}, detekce: {detection}")
    
    # Rozložení délek
    from collections import Counter
    length_counts = Counter(all_steps)
    print(f"\n=== ROZLOŽENÍ DÉLEK ÚSPĚŠNÝCH CEST ===")
    for length in sorted(length_counts.keys())[:10]:  # Prvních 10
        count = length_counts[length]
        percentage = (count / len(all_steps)) * 100
        print(f"  {length} kroků: {count}x ({percentage:.1f}%)")


def plot_training_progress(model, callback):
    """Vykreslení pokroku tréninku."""
    if len(callback.successful_paths) < 2:
        print("Nedostatek dat pro graf")
        return
    
    # Příprava dat
    timesteps = [p['timestep'] for p in callback.successful_paths]
    steps = [p['steps'] for p in callback.successful_paths]
    
    plt.figure(figsize=(12, 4))
    
    # Graf délek cest v čase
    plt.subplot(1, 2, 1)
    plt.scatter(timesteps, steps, alpha=0.6, s=20)
    plt.xlabel('Tréninkové kroky')
    plt.ylabel('Délka cesty')
    plt.title('Vývoj délky cest během tréninku')
    plt.grid(True, alpha=0.3)
    
    # Histogram délek
    plt.subplot(1, 2, 2)
    plt.hist(steps, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Délka cesty')
    plt.ylabel('Četnost')
    plt.title('Distribuce délek cest')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_trained_agent(model, env, n_episodes=5):
    """Test natrénovaného agenta s detailním sledováním cest."""
    print(f"\n=== TEST NATRÉNOVANÉHO AGENTA ===")
    
    successful_paths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        path = [env.agent_pos]
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Předpověz akci (deterministicky)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            path.append(env.agent_pos)
            total_reward += reward
        
        if done:  # Úspěšně dosáhl cíle
            successful_paths.append({
                'episode': episode,
                'path': path,
                'steps': len(path) - 1,
                'reward': total_reward
            })
            
            print(f"Epizoda {episode}: {len(path)-1} kroků, odměna {total_reward:.1f}")
            print(f"  Cesta: {path}")
            if episode < 2:  # Zobraz první 2 mapy
                env.print_path(path)
        else:
            print(f"Epizoda {episode}: NEÚSPĚŠNÁ (timeout)")
    
    if successful_paths:
        # Analýza úspěšných cest
        all_lengths = [p['steps'] for p in successful_paths]
        min_length = min(all_lengths)
        
        # Najdi nejkratší cesty
        shortest_paths = [p for p in successful_paths if p['steps'] == min_length]
        
        print(f"\nÚspěšnost: {len(successful_paths)}/{n_episodes}")
        print(f"Průměrná délka: {np.mean(all_lengths):.1f} kroků")
        print(f"Nejkratší: {min_length} kroků")
        print(f"Počet nejkratších cest: {len(shortest_paths)}")
        
        # Zobraz všechny nejkratší cesty
        if len(shortest_paths) > 1:
            print(f"\n=== VŠECHNY NEJKRATŠÍ CESTY AGENTA ({min_length} kroků) ===")
            for i, path_info in enumerate(shortest_paths):
                print(f"\nNejkratší cesta {i+1}:")
                print(f"  Cesta: {path_info['path']}")
                print(f"  Odměna: {path_info['reward']:.1f}")
                
                # Zkontroluj jestli jsou cesty unikátní
                path_tuple = tuple(path_info['path'])
                duplicates = sum(1 for p in shortest_paths if tuple(p['path']) == path_tuple)
                if duplicates > 1:
                    print(f"  (Duplikát - objevena {duplicates}x)")
    
    return successful_paths


# Hlavní program
if __name__ == "__main__":
    try:
        # Získej parametry od uživatele
        grid_size, terrain_type, num_terrain, timesteps = get_user_preferences()
        
        # Vytvoř prostředí
        # Adaptivní max_steps na základě velikosti gridu
        max_steps = grid_size * grid_size * 3  # Více kroků pro větší gridy
        
        env = GridWorldEnv(
            size=grid_size, 
            terrain_type=terrain_type, 
            num_terrain=num_terrain,
            max_steps=max_steps
        )
        
        # Zkontroluj prostředí
        print("\n=== KONTROLA PROSTŘEDÍ ===")
        check_env(env, warn=True)
        print("✅ Prostředí je kompatibilní s Gymnasium")
        
        # Zobraz prostředí
        print(f"\n=== PROSTŘEDÍ {grid_size}x{grid_size} ===")
        terrain_name = "překážky" if terrain_type == "obstacles" else "bažiny"
        print(f"Terén: {num_terrain} {terrain_name}")
        env.reset()
        env.render()
        
        # RYCHLÝ TEST: Zkus manuálně dojít k cíli
        print(f"\n=== RYCHLÝ TEST PROSTŘEDÍ ===")
        test_obs, _ = env.reset()
        print(f"Start pozice: {env.agent_pos}")
        print(f"Cíl pozice: {env.goal_pos}")
        
        # Zkus jednoduchý path: doprava → dolů
        test_steps = 0
        while env.agent_pos != env.goal_pos and test_steps < 50:
            # Jednoduchá heuristika: jdi doprava, pokud nemůžeš, jdi dolů
            if env.agent_pos[1] < env.goal_pos[1]:  # Jdi doprava
                action = 3  # Right
            else:
                action = 1  # Down
            
            test_obs, test_reward, done, truncated, info = env.step(action)
            test_steps += 1
            
            if done:
                print(f"✅ MANUAL TEST: Dosáhl cíle za {test_steps} kroků! Reward: {test_reward:.1f}")
                print(f"   Info: {info}")
                break
            elif test_steps >= 50:
                print(f"❌ MANUAL TEST: Nedosáhl cíle za 50 kroků")
                break
        
        # Reset pro skutečný trénink
        env.reset()
        
        # Vytvoř DQN model (Stable-Baselines3)
        print(f"\n=== VYTVÁŘENÍ DQN MODELU ===")
        vec_env = DummyVecEnv([lambda: env])
        
        # Adaptivní hyperparametry na základě složitosti prostředí
        complexity_factor = (grid_size * grid_size * num_terrain) / (6 * 6 * 2)
        
        # Škálování parametrů podle složitosti
        buffer_size = max(10000, int(50000 * complexity_factor))
        learning_starts = max(1000, int(2000 * complexity_factor))
        exploration_fraction = min(0.7, 0.4 + 0.1 * complexity_factor)  # Delší explorace pro složitější prostředí
        
        print(f"Adaptivní hyperparametry:")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Learning starts: {learning_starts}")
        print(f"  Exploration fraction: {exploration_fraction:.2f}")
        
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=5e-4,  # Vyšší learning rate
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=64,  # Větší batch
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,  # Vyšší minimální explorace
            target_update_interval=500,  # Častější updates
            policy_kwargs=dict(net_arch=[256, 256]),  # Větší síť
            verbose=1,
            device="auto"
        )
        
        print("✅ DQN model vytvořen s professional parametry")
        
        # Vytvoř callback pro monitoring
        callback = GridWorldDQNCallback(
            check_freq=1000,
            save_path="./gridworld_logs/",
            verbose=2  # Zvýšit pro debug
        )
        
        # Trénink
        print(f"\n=== ZAČÍNÁ TRÉNINK ({timesteps} kroků) ===")
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("✅ Trénink dokončen")
        
        # Analýza výsledků
        analyze_training_results(callback, env)
        
        # Najdi všechny teoreticky optimální cesty
        print(f"\n=== HLEDÁNÍ TEORETICKY OPTIMÁLNÍCH CEST ===")
        all_optimal_paths, optimal_distance = find_all_shortest_paths_bfs(env)
        
        if all_optimal_paths:
            print(f"✅ Nalezeno {len(all_optimal_paths)} teoreticky optimálních cest s délkou {optimal_distance} kroků")
            
            # Porovnej s agentovými cestami
            if callback.successful_paths:
                analyze_agent_path_optimality(env, callback.successful_paths, all_optimal_paths)
        else:
            print("❌ Žádná cesta k cíli neexistuje (cíl je nedosažitelný)")
        
        # Test natrénovaného modelu
        test_paths = test_trained_agent(model, env, n_episodes=5)
        
        # Pokud test našel cesty, porovnej i ty s optimálními
        if test_paths and all_optimal_paths:
            print(f"\n=== POROVNÁNÍ TESTOVACÍCH CEST S OPTIMÁLNÍMI ===")
            analyze_agent_path_optimality(env, test_paths, all_optimal_paths)
        
        # Zobraz grafy
        plot_training_progress(model, callback)
        
        # Uložení modelu
        model_path = f"gridworld_dqn_{grid_size}x{grid_size}.zip"
        model.save(model_path)
        print(f"\n✅ Model uložen: {model_path}")
        
        # Uložení dat o cestách
        with open(f"paths_{grid_size}x{grid_size}.pkl", "wb") as f:
            pickle.dump({
                'training_paths': callback.successful_paths,
                'test_paths': test_paths,
                'env_config': {
                    'size': grid_size,
                    'terrain_type': terrain_type,
                    'num_terrain': num_terrain,
                    'obstacles': env.obstacles,
                    'swamps': env.swamps
                }
            }, f)
        
        print("✅ Data o cestách uložena")
        
    except KeyboardInterrupt:
        print("\n⚠️ Trénink přerušen uživatelem")
    except Exception as e:
        print(f"❌ Chyba: {e}")
        print("Ujistěte se, že máte nainstalované: pip install stable-baselines3[extra] gymnasium matplotlib")
