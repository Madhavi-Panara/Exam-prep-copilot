"""Application errors. main.py turns every AppError into a JSON {"detail": ...} response."""


class AppError(Exception):
    status_code = 400

    def __init__(self, detail: str, status_code: int | None = None):
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class IngestError(AppError):
    status_code = 422


class LLMError(AppError):
    status_code = 502


class NoMaterialError(AppError):
    status_code = 400


class NotFoundError(AppError):
    status_code = 404
