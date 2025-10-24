"""
Módulo de chat interactivo para RAG.

Permite conversaciones multi-turno manteniendo contexto conversacional
cuando el provider LLM lo soporta (ej: OpenAI).
"""

from typing import Optional
from dataclasses import dataclass
from src.inserter import ChromaCollection
from src.config_loader import RAGConfig


@dataclass
class ChatState:
    """Estado de la conversación durante la sesión de chat."""
    conversation_id: Optional[str] = None
    turn_count: int = 0
    provider_supports_history: bool = True


class ChatInterface:
    """
    Interfaz de chat interactivo para consultas RAG.

    Maneja el loop de conversación, comandos especiales y mantenimiento
    de estado conversacional.
    """

    def __init__(self, chroma: ChromaCollection, config: RAGConfig):
        """
        Inicializa la interfaz de chat.

        Args:
            chroma: Instancia de ChromaCollection ya inicializada e indexada
            config: Configuración RAG activa
        """
        self.chroma = chroma
        self.config = config
        self.state = ChatState()

        # Detectar si el provider soporta historial conversacional
        self.state.provider_supports_history = self._check_provider_support()

    def _check_provider_support(self) -> bool:
        """
        Verifica si el provider LLM soporta conversation_id.

        Returns:
            True si soporta historial, False en caso contrario
        """
        provider = self.config.model.provider.lower()
        # Por ahora solo OpenAI soporta conversation_id
        return provider == "openai"

    def start(self):
        """
        Inicia el loop de chat interactivo.

        Mantiene el loop hasta que el usuario ejecute /exit o presione Ctrl+D.
        """
        self._print_welcome()

        while True:
            try:
                user_input = input("Q: ").strip()

                # Ignorar entradas vacías
                if not user_input:
                    continue

                # Manejar comandos especiales
                if user_input.startswith("/"):
                    should_exit = self._handle_command(user_input)
                    if should_exit:
                        break
                    continue

                # Procesar query normal
                self._process_query(user_input)

            except KeyboardInterrupt:
                print("\n\nSesión interrumpida. Usa /exit para salir limpiamente.")
                continue
            except EOFError:
                print("\n\nHasta pronto!")
                break

    def _print_welcome(self):
        """Muestra mensaje de bienvenida al iniciar el chat."""
        print("\n" + "="*60)
        print("  MODO CHAT INTERACTIVO")
        print("="*60)
        print(f"\nConfig: {self.config.name}")
        print(f"Modelo: {self.config.model.name}")

        if not self.state.provider_supports_history:
            print(f"\nNOTA: {self.config.model.provider} no mantiene historial conversacional")
            print("      Cada pregunta será independiente.")

        print("\nComandos disponibles:")
        print("  /help   - Mostrar ayuda")
        print("  /clear  - Reiniciar conversación")
        print("  /exit   - Salir del chat")
        print("\n" + "="*60 + "\n")

    def _handle_command(self, cmd: str) -> bool:
        """
        Maneja comandos especiales del chat.

        Args:
            cmd: Comando a ejecutar (debe empezar con /)

        Returns:
            True si debe salir del chat, False en caso contrario
        """
        cmd = cmd.lower().strip()

        if cmd in ["/exit", "/quit"]:
            print("\nHasta pronto!")
            return True

        elif cmd == "/clear":
            self.state.conversation_id = None
            self.state.turn_count = 0
            print("\nConversación reiniciada\n")
            return False

        elif cmd == "/help":
            self._print_help()
            return False

        else:
            print(f"Comando desconocido: {cmd}")
            print("Usa /help para ver comandos disponibles")
            return False

    def _print_help(self):
        """Muestra ayuda de comandos disponibles."""
        print("\n" + "="*60)
        print("  COMANDOS DISPONIBLES")
        print("="*60)
        print("\n/help   - Mostrar esta ayuda")
        print("/clear  - Reiniciar conversación (nuevo conversation_id)")
        print("/exit   - Salir del chat")
        print("\nEscribe tu pregunta normalmente para consultar el código.")
        print("="*60 + "\n")

    def _process_query(self, query: str):
        """
        Procesa una query del usuario.

        Ejecuta el pipeline RAG completo (retrieval + reranking + generation)
        y muestra la respuesta.

        Args:
            query: Pregunta del usuario sobre el código
        """
        # Indicador de búsqueda
        print("Buscando...", end="", flush=True)

        try:
            # Ejecutar RAG con conversation_id si el provider lo soporta
            response_text, response_obj = self.chroma.rag(
                query=query,
                verbose=False,  # Output simplificado en modo chat
                conversation_id=self.state.conversation_id if self.state.provider_supports_history else None
            )

            # Limpiar línea de "Buscando..."
            print("\r" + " "*20 + "\r", end="")

            # Mostrar respuesta
            print(f"A: {response_text}\n")

            # Actualizar estado conversacional
            self.state.turn_count += 1
            if self.state.provider_supports_history and response_obj.conversation_id:
                self.state.conversation_id = response_obj.conversation_id

        except Exception as e:
            # Limpiar línea de "Buscando..."
            print("\r" + " "*20 + "\r", end="")
            print(f"Error: {e}\n")


def start_chat(chroma: ChromaCollection, config: RAGConfig):
    """
    Inicia el modo chat interactivo.

    Esta es la función principal que se llama desde main.py cuando
    no se proporciona el argumento -p.

    Args:
        chroma: Instancia de ChromaCollection ya inicializada e indexada
        config: Configuración RAG
    """
    chat = ChatInterface(chroma, config)
    chat.start()
