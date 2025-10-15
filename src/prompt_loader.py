"""
Cargador de prompts desde archivos externos.

Permite gestionar diferentes templates de prompts para el sistema RAG.
"""

import os
from pathlib import Path


class PromptLoader:
    """
    Gestiona la carga de prompts desde archivos .txt en la carpeta prompts/.

    Attributes:
        prompts_dir: Ruta al directorio de prompts
        _cache: Cache de prompts cargados

    Example:
        >>> loader = PromptLoader()
        >>> prompt = loader.load("detailed")
        >>> print(prompt)
    """

    def __init__(self, prompts_dir: str = None):
        """
        Inicializa el cargador de prompts.

        Args:
            prompts_dir: Ruta al directorio de prompts (default: ./prompts/)
        """
        if prompts_dir is None:
            # Obtener directorio raíz del proyecto (2 niveles arriba de src/)
            project_root = Path(__file__).parent.parent
            prompts_dir = project_root / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self._cache = {}

        # Verificar que existe el directorio
        if not self.prompts_dir.exists():
            raise FileNotFoundError(
                f"Directorio de prompts no encontrado: {self.prompts_dir}"
            )

    def load(self, prompt_name: str = "default") -> str:
        """
        Carga un prompt desde archivo.

        Args:
            prompt_name: Nombre del prompt (sin extensión .txt)

        Returns:
            Contenido del prompt como string

        Raises:
            FileNotFoundError: Si el prompt no existe

        Example:
            >>> loader = PromptLoader()
            >>> prompt = loader.load("spanish")
        """
        # Revisar cache primero
        if prompt_name in self._cache:
            return self._cache[prompt_name]

        # Construir ruta al archivo
        prompt_path = self.prompts_dir / f"{prompt_name}.txt"

        if not prompt_path.exists():
            available = self.list_prompts()
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' no encontrado.\n"
                f"Prompts disponibles: {', '.join(available)}"
            )

        # Leer contenido
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Guardar en cache
        self._cache[prompt_name] = content

        return content

    def list_prompts(self) -> list[str]:
        """
        Lista todos los prompts disponibles.

        Returns:
            Lista de nombres de prompts (sin extensión)

        Example:
            >>> loader = PromptLoader()
            >>> prompts = loader.list_prompts()
            >>> print(prompts)
            ['default', 'detailed', 'concise', 'spanish']
        """
        if not self.prompts_dir.exists():
            return []

        prompts = []
        for file in self.prompts_dir.glob("*.txt"):
            prompts.append(file.stem)

        return sorted(prompts)

    def reload(self, prompt_name: str = None):
        """
        Recarga un prompt desde disco (invalida cache).

        Args:
            prompt_name: Nombre del prompt a recargar (None = todos)

        Example:
            >>> loader = PromptLoader()
            >>> loader.reload("default")  # Recarga solo default
            >>> loader.reload()           # Recarga todos
        """
        if prompt_name is None:
            self._cache.clear()
        elif prompt_name in self._cache:
            del self._cache[prompt_name]

    def get_prompt_info(self, prompt_name: str) -> dict:
        """
        Obtiene información sobre un prompt.

        Args:
            prompt_name: Nombre del prompt

        Returns:
            Diccionario con info del prompt

        Example:
            >>> loader = PromptLoader()
            >>> info = loader.get_prompt_info("detailed")
            >>> print(info['length'])
        """
        prompt_path = self.prompts_dir / f"{prompt_name}.txt"

        if not prompt_path.exists():
            return {"exists": False}

        content = self.load(prompt_name)
        lines = content.split('\n')

        return {
            "exists": True,
            "name": prompt_name,
            "path": str(prompt_path),
            "length": len(content),
            "lines": len(lines),
            "first_line": lines[0] if lines else ""
        }


# Instancia global para reutilizar
_default_loader = None


def get_prompt_loader() -> PromptLoader:
    """
    Obtiene la instancia global del PromptLoader (singleton).

    Returns:
        Instancia de PromptLoader

    Example:
        >>> loader = get_prompt_loader()
        >>> prompt = loader.load("default")
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader
