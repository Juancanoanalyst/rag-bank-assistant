"""HTTP client the Streamlit UI uses to talk to the API.

Kept free of Streamlit imports so it can be unit-tested on its own.
"""

import requests


class ApiError(Exception):
    """The API could not be reached or returned an error; the message is user-facing."""


class ApiClient:
    def __init__(self, base_url: str, timeout: float, session: requests.Session | None = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()

    def chat(self, session_id: str, question: str) -> dict:
        return self._request("POST", "/chat", json={"session_id": session_id, "question": question})

    def history(self, session_id: str) -> list[dict]:
        return self._request("GET", f"/sessions/{session_id}/history")["messages"]

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._session.request(
                method, f"{self._base_url}{path}", timeout=self._timeout, **kwargs
            )
        except requests.Timeout as exc:
            raise ApiError("El asistente tardó demasiado en responder. Intenta de nuevo.") from exc
        except requests.RequestException as exc:
            raise ApiError("No se pudo conectar con la API del asistente.") from exc

        if response.status_code == 422:
            raise ApiError("La pregunta o el ID de sesión no son válidos.")
        if response.status_code != 200:
            raise ApiError(_detail(response) or f"La API respondió HTTP {response.status_code}.")
        return response.json()


def _detail(response: requests.Response) -> str | None:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail if isinstance(detail, str) else None
