# backend/core/monitors.py
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..types.data import ExperimentResultRL


class BaseResultMonitor(ABC):
    """
    Monitor base abstracto para la recolección, almacenamiento y procesamiento
    básico de resultados de experimentos.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.results: List[Any] = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @abstractmethod
    def add_result(self, result: Any) -> None:
        """Añade un resultado a la memoria en caché."""
        pass

    @abstractmethod
    def save(self, filename: Optional[str] = None) -> str:
        """Persiste los resultados recolectados en el sistema de archivos o BD."""
        pass

    def clear(self) -> None:
        """Limpia la memoria del monitor."""
        self.results = []


class RLResultMonitor(BaseResultMonitor):
    """
    Monitor específico para recolectar resultados de experimentos de
    Aprendizaje por Refuerzo (RL).
    """

    def __init__(self, output_dir: str = "results/rl"):
        super().__init__(output_dir=output_dir)
        self.results: List[ExperimentResultRL] = []

    def add_result(self, result: ExperimentResultRL) -> None:
        """
        Almacena el resultado de una corrida de RL.
        Validamos implícitamente mediante el tipado (Pydantic si se usara).
        """
        self.results.append(result)

    def save(self, filename: Optional[str] = None) -> str:
        """
        Guarda los resultados como JSON.
        Nota: Para experimentos masivos (Monte Carlo con N elevado),
        se recomienda usar formatos columnares (Parquet) u HDF5.
        """
        if not self.results:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename:
            model_name = self.results[0].model_id.replace(" ", "_").lower()
            filename = f"rl_experiment_{model_name}_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        data_to_save = []
        for res in self.results:
            if hasattr(res, "model_dump"):
                data_to_save.append(res.model_dump())
            elif hasattr(res, "__dict__"):
                data_to_save.append(res.__dict__)
            else:
                data_to_save.append(res)

        with open(filepath, "w") as f:
            json.dump(data_to_save, f, indent=2)

        return filepath

    def get_summary(self) -> Dict[str, Any]:
        """
        Calcula estadísticas rápidas sobre las corridas almacenadas.
        Útil para responder a la API sin devolver todo el tensor de resultados.
        """
        if not self.results:
            return {"error": "No results available"}

        summary = {
            "total_runs": len(self.results),
            "models_tested": list(set(r.model_id for r in self.results)),
            "average_final_rewards": {},
        }

        for res in self.results:
            if len(res.rewards_history) > 10:
                avg_last_10 = sum(res.rewards_history[-10:]) / 10.0
            else:
                avg_last_10 = sum(res.rewards_history) / max(
                    1, len(res.rewards_history)
                )

            if res.model_id not in summary["average_final_rewards"]:
                summary["average_final_rewards"][res.model_id] = []

            summary["average_final_rewards"][res.model_id].append(avg_last_10)

        return summary
