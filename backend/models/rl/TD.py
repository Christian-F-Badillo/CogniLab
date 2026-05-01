import numpy as np
from typing import Dict, List, Any
from collections import deque

from ...core.baseAgent import Agent
from ...schemas.data import ModelConfig
from .policies import BasePolicy, GreedyPolicy


class QLearningAgent(Agent):
    """
    Control Off-Policy TD(0).
    Aproxima Q* asumiendo una política objetivo puramente Greedy (max Q).
    """
    def __init__(self, config: ModelConfig, policy: BasePolicy):
        super().__init__(config)
        self.Q = np.zeros(shape=(self._nstates, self._nactions), dtype=np.float64)
        self.policy = policy
        self.lr = self._params.get("lr", 0.1)
        self.gamma = self._params.get("gamma", 0.9)

    def update(self, input: Dict[str, Any]) -> None:
        current_state = input["current_state"]
        next_state = input["next_state"]
        reward = input["reward"]
        action = input["action"]

        is_done = input.get("done", False)
        best_action_q = 0.0 if is_done else np.max(self.Q[next_state, :])

        self.Q[current_state, action] += self.lr * (
            reward + (self.gamma * best_action_q) - self.Q[current_state, action]
        )

    def act(self, input: Dict[str, Any]) -> int | float | List[Any]:
        state = input["current_state"]
        return self.policy.select_action(self.Q[state, :])


class SarsaAgent(Agent):
    """
    Control On-Policy TD(0).
    Actualiza Q asumiendo que la siguiente acción será dictada por la política actual.
    """
    def __init__(self, config: ModelConfig, policy: BasePolicy):
        super().__init__(config)
        self.Q = np.zeros(shape=(self._nstates, self._nactions), dtype=np.float64)
        self.policy = policy
        self.lr = self._params.get("lr", 0.1)
        self.gamma = self._params.get("gamma", 0.9)

    def update(self, input: Dict[str, Any]) -> None:
        current_state = input["current_state"]
        next_state = input["next_state"]
        reward = input["reward"]
        action = input["action"]
        is_done = input.get("done", False)

        if is_done:
            q_next_action = 0.0
        else:
            next_action = self.act({"current_state": next_state})
            q_next_action = self.Q[next_state, next_action]

        td_target = reward + (self.gamma * q_next_action)
        self.Q[current_state, action] += self.lr * (td_target - self.Q[current_state, action])

    def act(self, input: Dict[str, Any]) -> int | float | List[Any]:
        state = input["current_state"]
        return self.policy.select_action(self.Q[state, :])


class NStepSarsaAgent(Agent):
    """
    Control On-Policy a n-pasos.
    Retrasa la actualización n pasos para acumular recompensas empíricas antes de hacer bootstrapping.
    """
    def __init__(self, config: ModelConfig, policy: BasePolicy):
        super().__init__(config)
        self.Q = np.zeros(shape=(self._nstates, self._nactions), dtype=np.float64)
        self.policy = policy
        
        self.lr = self._params.get("lr", 0.1)
        self.gamma = self._params.get("gamma", 0.9)
        self.n = self._params.get("n_steps", 3)
        
        # Caché FIFO para preservar la Propiedad de Markov Extendida
        self.memory = deque(maxlen=self.n)

    def update(self, input: Dict[str, Any]) -> None:
        current_state = input["current_state"]
        action = input["action"]
        reward = input["reward"]
        next_state = input["next_state"]
        is_done = input.get("done", False)

        # Almacenamos la tupla de transición completa
        self.memory.append((current_state, action, reward, next_state, is_done))

        # Actualización diferida: Solo procesamos cuando el caché alcanza horizonte n
        if len(self.memory) == self.n:
            self._process_n_step_update()

        # Condición de vaciado (Flush): Resolver estados residuales al terminar el episodio
        if is_done:
            while len(self.memory) > 0:
                self._process_n_step_update()

    def _process_n_step_update(self):
        tau_state, tau_action, _, _, _ = self.memory[0]
        _, _, _, final_next_state, final_is_done = self.memory[-1]

        # Acumulación de recompensas empíricas
        G = 0.0
        for i, (_, _, r, _, _) in enumerate(self.memory):
            G += (self.gamma ** i) * r

        # Bootstrapping condicional al nodo final del horizonte
        if not final_is_done:
            next_action = self.act({"current_state": final_next_state})
            G += (self.gamma ** len(self.memory)) * self.Q[final_next_state, next_action]

        # Actualización TD Clásica sobre el estado tau (el más antiguo del buffer)
        self.Q[tau_state, tau_action] += self.lr * (G - self.Q[tau_state, tau_action])
        self.memory.popleft()

    def act(self, input: Dict[str, Any]) -> int | float | List[Any]:
        state = input["current_state"]
        return self.policy.select_action(self.Q[state, :])


class TreeBackupAgent(Agent):
    """
    Control Off-Policy a n-pasos (Tree-Backup / Q(sigma)).
    Evita la inestabilidad del Muestreo por Importancia ponderando los valores esperados.
    Permite aprender la política óptima (Greedy) mientras sigue una política exploratoria (Behavior).
    """
    def __init__(self, config: ModelConfig, policy: BasePolicy):
        super().__init__(config)
        self.Q = np.zeros(shape=(self._nstates, self._nactions), dtype=np.float64)
        
        self.behavior_policy = policy # Política generadora de datos (ej. Epsilon-Greedy)
        self.target_policy = GreedyPolicy() # Política a optimizar (Off-Policy -> Greedy)
        
        self.lr = self._params.get("lr", 0.1)
        self.gamma = self._params.get("gamma", 0.9)
        self.n = self._params.get("n_steps", 3)
        
        self.memory = deque(maxlen=self.n)

    def update(self, input: Dict[str, Any]) -> None:
        current_state = input["current_state"]
        action = input["action"]
        reward = input["reward"]
        next_state = input["next_state"]
        is_done = input.get("done", False)

        self.memory.append((current_state, action, reward, next_state, is_done))

        if len(self.memory) == self.n:
            self._process_tree_backup_update()

        if is_done:
            while len(self.memory) > 0:
                self._process_tree_backup_update()

    def _process_tree_backup_update(self):
        tau_state, tau_action, _, _, _ = self.memory[0]
        _, _, _, final_next_state, final_is_done = self.memory[-1]

        # 1. Caso Base (Horizonte final de evaluación)
        if final_is_done:
            G = 0.0
        else:
            q_final = self.Q[final_next_state, :]
            probs_final = self.target_policy.get_probabilities(q_final)
            G = np.sum(probs_final * q_final) # V(S_{t+n})

        # 2. Iteración Recursiva hacia atrás (desde k=t+n-1 hasta k=tau)
        for i in reversed(range(len(self.memory))):
            state, action, reward, next_state, done = self.memory[i]
            
            if done:
                # Nodo terminal colapsa a la recompensa obtenida
                G = reward 
            else:
                if i == len(self.memory) - 1:
                    # El último elemento no tiene una "siguiente acción tomada" dentro de la memoria.
                    # G ya contiene V(S_{k+1}) del bloque base.
                    G = reward + self.gamma * G
                else:
                    # Acción A_{k+1} que el agente SÍ tomó empíricamente en el siguiente paso
                    next_action_taken = self.memory[i+1][1]
                    
                    q_vals = self.Q[next_state, :]
                    probs = self.target_policy.get_probabilities(q_vals)
                    
                    # Valor esperado puro: V(S_{k+1})
                    v_next = np.sum(probs * q_vals)
                    
                    # Probabilidad objetivo de la acción empírica
                    prob_a_next = probs[next_action_taken]
                    
                    # Formulación robusta de Tree-Backup:
                    # R + γV(s') + γπ(a'|s')[G - Q(s',a')]
                    G = reward + self.gamma * v_next + self.gamma * prob_a_next * (G - q_vals[next_action_taken])

        # 3. Actualización TD 
        self.Q[tau_state, tau_action] += self.lr * (G - self.Q[tau_state, tau_action])
        self.memory.popleft()

    def act(self, input: Dict[str, Any]) -> int | float | List[Any]:
        # El comportamiento siempre lo rige la política behavior (off-policy)
        state = input["current_state"]
        return self.behavior_policy.select_action(self.Q[state, :])