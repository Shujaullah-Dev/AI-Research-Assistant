from ollama import Client


class OllamaLLM:
    """Client wrapper for a locally running Ollama model."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.client = Client(host=host)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
        )

        return response["response"]